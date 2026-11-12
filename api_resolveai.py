"""API acadêmica ResolveAI. Executar com Python 3.13 e requirements.txt."""
from pathlib import Path
import json
import math
import warnings
from importlib.metadata import version

import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.exceptions import InconsistentVersionWarning
from werkzeug.exceptions import HTTPException

BASE_DIR = Path(__file__).resolve().parent
FEATURES = ["categoria", "regiao", "dia_semana", "hora_abertura",
            "moradores_ativos_7d", "taxa_apoio_anterior_28d", "ocorrencias_regiao_7d"]
CATEGORIAS = ["iluminacao", "limpeza", "vias", "acessibilidade"]
REGIOES = ["norte", "sul", "leste", "oeste"]
MODEL_NAMES = ("regressao_logistica", "random_forest")
NUMERIC_FEATURES = ("moradores_ativos_7d", "taxa_apoio_anterior_28d", "ocorrencias_regiao_7d")


def validate_payload(payload, training_ranges=None):
    training_ranges = training_ranges or {}
    if not isinstance(payload, dict):
        return None, "O corpo deve ser um objeto JSON."
    if set(payload) != {"modelo", "caracteristicas"}:
        return None, "Informe exatamente modelo e caracteristicas."
    name, values = payload["modelo"], payload["caracteristicas"]
    if not isinstance(name, str) or name not in MODEL_NAMES:
        return None, "modelo deve ser regressao_logistica ou random_forest."
    if not isinstance(values, dict) or set(values) != set(FEATURES):
        return None, "caracteristicas deve conter exatamente: " + ", ".join(FEATURES)
    for field, allowed in (("categoria", CATEGORIAS), ("regiao", REGIOES)):
        if not isinstance(values[field], str) or values[field] not in allowed:
            return None, field + " deve ser um de: " + ", ".join(allowed)
    # bool é subclasse de int em Python: rejeitar explicitamente.
    for field, maximum in (("dia_semana", 6), ("hora_abertura", 23),
                           ("moradores_ativos_7d", 200), ("ocorrencias_regiao_7d", 1000)):
        minimum = 0
        if field in training_ranges:
            minimum, maximum = training_ranges[field]["min"], training_ranges[field]["max"]
        value = values[field]
        if type(value) is not int or not minimum <= value <= maximum:
            suffix = " (faixa observada no treino)." if field in training_ranges else "."
            return None, f"{field} deve ser inteiro entre {minimum} e {maximum}" + suffix
    rate = values["taxa_apoio_anterior_28d"]
    minimum = training_ranges.get("taxa_apoio_anterior_28d", {}).get("min", 0)
    maximum = training_ranges.get("taxa_apoio_anterior_28d", {}).get("max", 1)
    if type(rate) not in (int, float) or not minimum <= rate <= maximum or not math.isfinite(rate):
        suffix = " (faixa observada no treino; use proporção, não porcentagem)." if training_ranges else "."
        return None, f"taxa_apoio_anterior_28d deve ser número finito entre {minimum} e {maximum}" + suffix
    return (name, pd.DataFrame([values], columns=FEATURES)), None


def create_app(model_path=None):
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"),
                static_folder=str(BASE_DIR / "static"))
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    app.json.ensure_ascii = False

    @app.after_request
    def utf8_json(response):
        # PowerShell 5.1 precisa do charset explícito para exibir acentos.
        if response.is_json:
            response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    bundle, load_error, training_ranges = None, None, None
    try:
        if version("scikit-learn") != "1.6.1":
            raise ValueError("scikit-learn incompatível; instale requirements.txt")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            bundle = joblib.load(model_path or BASE_DIR / "modelos_resolveai.joblib")
        if bundle["metadados"]["schema"] != 1 or bundle["entradas"] != FEATURES:
            raise ValueError("estrutura ou entradas incompatíveis")
        training_ranges = bundle["metadados"]["faixas_treino"]
        if not isinstance(training_ranges, dict) or set(training_ranges) != set(NUMERIC_FEATURES):
            raise ValueError("faixas de treinamento ausentes ou incompatíveis; use o joblib atualizado")
        for field, bounds in training_ranges.items():
            if not isinstance(bounds, dict) or set(bounds) != {"min", "max"}:
                raise ValueError("limites de treinamento incompatíveis")
            if any(type(bounds[key]) not in (int, float) or not math.isfinite(bounds[key]) for key in bounds):
                raise ValueError("limites de treinamento não numéricos")
            if not 0 <= bounds["min"] <= bounds["max"]:
                raise ValueError("limites de treinamento incoerentes")
            if field == "taxa_apoio_anterior_28d" and bounds["max"] > 1:
                raise ValueError("limite de proporção incoerente")
            if field != "taxa_apoio_anterior_28d" and any(type(v) is not int for v in bounds.values()):
                raise ValueError("limites de contadores devem ser inteiros")
        for lib, trained in bundle["metadados"]["versoes"].items():
            if version(lib) != trained:
                raise ValueError(f"versão incompatível de {lib}; instale requirements.txt")
        probe = pd.DataFrame([dict(categoria="vias", regiao="norte", dia_semana=2,
                                  hora_abertura=14, moradores_ativos_7d=40,
                                  taxa_apoio_anterior_28d=0.5, ocorrencias_regiao_7d=20)])
        for name in MODEL_NAMES:
            if list(bundle[name].feature_names_in_) != FEATURES:
                raise ValueError("entradas do pipeline incompatíveis")
            if list(bundle[name].classes_) != [0, 1]:
                raise ValueError("classes incompatíveis")
            probabilities = bundle[name].predict_proba(probe[FEATURES])
            if probabilities.shape != (1, 2) or not all(math.isfinite(x) for x in probabilities[0]):
                raise ValueError("pipeline não produziu probabilidades válidas")
    except FileNotFoundError:
        bundle, training_ranges, load_error = None, None, "Arquivo modelos_resolveai.joblib ausente. Coloque-o junto da API."
    except Exception as exc:
        bundle, training_ranges, load_error = None, None, "Não foi possível carregar os modelos e seus limites. Use o joblib atualizado e requirements.txt."
        app.logger.error("Falha ao carregar modelos: %s: %s", type(exc).__name__, exc)
    app.config["MODEL_BUNDLE"] = bundle
    app.config["TRAINING_RANGES"] = training_ranges

    @app.get("/")
    def interface():
        return render_template("index.html", faixas_treino=training_ranges or {})

    @app.get("/health")
    def health():
        ready = bundle is not None
        response = dict(status="ok" if ready else "indisponivel", api_disponivel=True,
                        modelos={name: ready for name in MODEL_NAMES}, dados_sinteticos=True)
        if load_error:
            response["erro"] = load_error
        return jsonify(response), 200 if ready else 503

    @app.post("/predict")
    def predict():
        if not request.is_json:
            return jsonify(erro="Use Content-Type: application/json."), 415
        # O decodificador estrito rejeita NaN, Infinity e chaves repetidas.
        def reject_constant(value):
            raise ValueError(f"Constante JSON inválida: {value}")
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Chave JSON repetida")
                result[key] = value
            return result
        try:
            payload = json.loads(request.get_data(), parse_constant=reject_constant,
                                 object_pairs_hook=unique_keys)
        except (ValueError, UnicodeError):
            return jsonify(erro="JSON inválido; não repita chaves nem use NaN/Infinity."), 400
        validated, error = validate_payload(payload, training_ranges)
        if error:
            return jsonify(erro=error), 400
        if bundle is None:
            return jsonify(erro=load_error), 503
        name, frame = validated
        try:
            probability = float(bundle[name].predict_proba(frame)[0, 1])
            prediction = int(bundle[name].predict(frame)[0])
            if not math.isfinite(probability):
                raise ValueError("Probabilidade não finita")
        except Exception:
            app.logger.exception("Falha na inferência")
            return jsonify(erro="Falha ao executar o modelo. Verifique os artefatos e dependências."), 500
        guidance = ("Considere compartilhar o relato com os vizinhos para ampliar sua divulgação."
                    if prediction == 0 else "Acompanhe a participação da comunidade nesta ocorrência.")
        return jsonify(modelo=name, previsao_apoio_72h=prediction,
                       probabilidade_apoio_72h=probability, orientacao=guidance,
                       dados_sinteticos=True,
                       aviso="Simulação acadêmica; não comprova apoio real nem resolução do problema.")

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(erro=error.description), error.code

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
