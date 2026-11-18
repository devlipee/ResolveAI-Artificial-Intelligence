"""API REST acadêmica ResolveAI com modelos treinados em dados reais do SP156.

Executar localmente: py -3.13 api_resolveai.py
Instalar previamente as versões indicadas em requirements.txt. A aplicação lê o
arquivo joblib uma única vez na inicialização; nenhuma rota treina modelos.
Em produção: gunicorn api_resolveai:app --bind 0.0.0.0:$PORT --workers 1.
"""
from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import math
import warnings

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.exceptions import InconsistentVersionWarning
from werkzeug.exceptions import HTTPException

BASE_DIR = Path(__file__).resolve().parent
FEATURES = ["servico", "distrito", "canal", "dia_semana"]
CATEGORICAL_FEATURES = ("servico", "distrito", "canal")
MODEL_NAMES = ("regressao_logistica", "random_forest")
CORE_LIBRARIES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
OBJECTIVE = "encerramento_7d"
HORIZON_DAYS = 7
CHANCE_BANDS = {"baixa_max": .30, "alta_min": .70}
SOURCE = {
    "nome": "SP156 - Prefeitura de São Paulo",
    "url": "https://dados.prefeitura.sp.gov.br/dataset/dados-do-sp156",
}
NOTICE = (
    "Estimativa acadêmica baseada no histórico público do SP156. "
    "Encerramento administrativo em até 7 dias corridos, pela diferença entre "
    "datas, não comprova a resolução física do problema e não é garantia de prazo."
)


def scenario_key(values):
    return tuple(values[field] for field in FEATURES)


def validate_payload(payload, categories=None, scenarios=None):
    """Validate the complete external contract before constructing a DataFrame."""
    if not isinstance(payload, dict):
        return None, "O corpo deve ser um objeto JSON."
    if set(payload) != {"modelo", "caracteristicas"}:
        return None, "Informe exatamente modelo e caracteristicas."
    name = payload["modelo"]
    if not isinstance(name, str) or name not in MODEL_NAMES:
        return None, "modelo deve ser regressao_logistica ou random_forest."
    values = payload["caracteristicas"]
    if not isinstance(values, dict) or set(values) != set(FEATURES):
        return None, "caracteristicas deve conter exatamente: " + ", ".join(FEATURES)
    for field in CATEGORICAL_FEATURES:
        value = values[field]
        if not isinstance(value, str) or not value.strip() or len(value) > 500:
            return None, f"{field} deve ser texto não vazio com até 500 caracteres."
        if categories is not None and value not in categories[field]:
            return None, f"{field} não pertence aos cenários disponíveis. Selecione um cenário em GET /metadata."
    # bool subclasses int in Python; exact type is intentional.
    day = values["dia_semana"]
    if type(day) is not int or not 0 <= day <= 6:
        return None, "dia_semana deve ser inteiro entre 0 (segunda) e 6 (domingo)."
    if scenarios is not None and scenario_key(values) not in scenarios:
        return None, (
            "Combinação fora do catálogo com cobertura verificada. "
            "Selecione um cenário completo em GET /metadata; não combine campos de cenários diferentes."
        )
    return (name, pd.DataFrame([values], columns=FEATURES)), None


def load_bundle(path):
    """Fail closed when the exported pipelines or their contract are incompatible."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("entradas") != FEATURES:
        raise ValueError("estrutura ou entradas do artefato incompatíveis")
    metadata = bundle.get("metadados")
    if not isinstance(metadata, dict) or metadata.get("schema") != 2:
        raise ValueError("use o artefato de dados reais com schema 2")
    if metadata.get("objetivo") != OBJECTIVE or metadata.get("prazo_dias") != HORIZON_DAYS:
        raise ValueError("objetivo ou prazo do artefato incompatível")
    if metadata.get("dados_sinteticos") is not False:
        raise ValueError("o artefato deve identificar dados reais")
    versions = metadata.get("versoes")
    if not isinstance(versions, dict) or not set(CORE_LIBRARIES).issubset(versions):
        raise ValueError("versões das bibliotecas de treinamento ausentes")
    for library in CORE_LIBRARIES:
        if not isinstance(versions[library], str) or version(library) != versions[library]:
            raise ValueError(f"versão incompatível de {library}; instale requirements.txt")
    categories = metadata.get("categorias")
    if not isinstance(categories, dict) or set(categories) != set(CATEGORICAL_FEATURES):
        raise ValueError("opções das entradas ausentes ou incompatíveis")
    for field in CATEGORICAL_FEATURES:
        values = categories[field]
        if (not isinstance(values, list) or not values
                or any(not isinstance(v, str) or not v.strip() or len(v) > 500 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"opções de {field} inválidas")
    if metadata.get("escolhido") not in MODEL_NAMES:
        raise ValueError("modelo selecionado pela validação ausente")
    source = metadata.get("fonte")
    if (not isinstance(source, dict) or not isinstance(source.get("nome"), str)
            or not isinstance(source.get("url"), str)):
        raise ValueError("identificação da fonte real ausente")
    examples = metadata.get("exemplos", [])
    if not isinstance(examples, list):
        raise ValueError("exemplos devem formar uma lista")
    for example in examples:
        if not isinstance(example, dict) or not isinstance(example.get("nome"), str):
            raise ValueError("exemplo incompatível")
        _, error = validate_payload({
            "modelo": metadata["escolhido"],
            "caracteristicas": example.get("caracteristicas"),
        }, categories)
        if error:
            raise ValueError("exemplo fora do contrato: " + error)
    probe = pd.DataFrame([{
        **{field: categories[field][0] for field in CATEGORICAL_FEATURES},
        "dia_semana": 0,
    }], columns=FEATURES)
    for name in MODEL_NAMES:
        model = bundle[name]
        if list(model.feature_names_in_) != FEATURES or list(model.classes_) != [0, 1]:
            raise ValueError("entradas ou classes do pipeline incompatíveis")
        probabilities = model.predict_proba(probe)
        if (probabilities.shape != (1, 2)
                or any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1
                       for x in probabilities[0])
                or not math.isclose(float(probabilities[0].sum()), 1, abs_tol=1e-6)):
            raise ValueError("pipeline não produziu probabilidades válidas")
        if int(model.predict(probe)[0]) not in (0, 1):
            raise ValueError("pipeline não produziu classe válida")
    return bundle


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _metric(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def load_catalog(path, bundle, model_path):
    """Load a frozen catalog tied to the exact model and training CSV version.

    Counts and validation scores are verified against the declared operational
    policy. These safeguards establish coverage, never certainty of prediction.
    """
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(catalog, dict) or type(catalog.get("schema")) is not int
            or catalog["schema"] != 1 or catalog.get("entradas") != FEATURES):
        raise ValueError("estrutura do catálogo incompatível")
    model_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    csv_hash = bundle["metadados"].get("csv_sha256")
    if not isinstance(csv_hash, str) or len(csv_hash) != 64:
        raise ValueError("hash do CSV de treinamento ausente")
    if catalog.get("modelo_sha256") != model_hash or catalog.get("csv_sha256") != csv_hash:
        raise ValueError("catálogo não corresponde aos modelos ou ao CSV; gere-o novamente")
    policy = catalog.get("politica")
    if not isinstance(policy, dict):
        raise ValueError("política de cobertura ausente")
    floors = {"min_treino_servico": 200, "min_treino_combinacao": 8,
              "min_validacao_servico": 50, "min_validacao_combinacao": 2,
              "min_treino_contexto": 20}
    for field, floor in floors.items():
        if not _integer(policy.get(field), floor):
            raise ValueError("mínimo operacional inválido: " + field)
    accuracy_floor = policy.get("min_acuracia_validacao_ambos")
    if not _metric(accuracy_floor) or accuracy_floor < .75:
        raise ValueError("piso de validação dos dois modelos inválido")
    if type(policy.get("exemplos_por_faixa")) is not int or policy["exemplos_por_faixa"] != 5:
        raise ValueError("a demonstração exige exatamente cinco exemplos por faixa")
    scenarios = catalog.get("cenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("catálogo sem cenários disponíveis")
    ids, keys = set(), set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("cenário incompatível")
        identifier, label = scenario.get("id"), scenario.get("nome")
        if (not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 128
                or not isinstance(label, str) or not label.strip() or len(label) > 1000):
            raise ValueError("identificação de cenário inválida")
        values = scenario.get("caracteristicas")
        _, error = validate_payload({"modelo": MODEL_NAMES[0], "caracteristicas": values},
                                    bundle["metadados"]["categorias"])
        if error:
            raise ValueError("cenário fora do domínio treinado: " + error)
        key = scenario_key(values)
        if identifier in ids or key in keys:
            raise ValueError("catálogo contém ID ou combinação duplicados")
        ids.add(identifier)
        keys.add(key)
        coverage = scenario.get("cobertura")
        if not isinstance(coverage, dict):
            raise ValueError("contagens de cobertura ausentes")
        minima = {"treino_combinacao": policy["min_treino_combinacao"],
                  "treino_servico": policy["min_treino_servico"],
                  "validacao_servico": policy["min_validacao_servico"],
                  "validacao_combinacao": policy["min_validacao_combinacao"],
                  "treino_contexto": policy["min_treino_contexto"],
                  "validacao_contexto": policy["min_validacao_combinacao"]}
        if any(not _integer(coverage.get(field), minimum) for field, minimum in minima.items()):
            raise ValueError("cenário não atende às contagens mínimas de cobertura")
        if (not coverage["treino_combinacao"] <= coverage["treino_contexto"] <= coverage["treino_servico"]
                or not coverage["validacao_combinacao"] <= coverage["validacao_contexto"] <= coverage["validacao_servico"]):
            raise ValueError("contagens de cobertura incoerentes")
        scores = scenario.get("metricas_validacao_servico")
        if not isinstance(scores, dict):
            raise ValueError("métricas de validação do serviço ausentes")
        for name in MODEL_NAMES:
            metrics = scores.get(name)
            if (not isinstance(metrics, dict)
                    or any(not _metric(metrics.get(field)) for field in ("acuracia", "f1_macro", "brier"))
                    or metrics["acuracia"] < accuracy_floor):
                raise ValueError("serviço não atende à validação dos dois modelos")
        counts = scenario.get("classes_validacao_servico")
        if (not isinstance(counts, dict) or set(counts) != {"0", "1"}
                or any(not _integer(value) for value in counts.values())
                or sum(counts.values()) != coverage["validacao_servico"]):
            raise ValueError("distribuição das classes de validação incompatível")
    evaluation = catalog.get("avaliacao")
    if not isinstance(evaluation, dict):
        raise ValueError("avaliação do recorte do catálogo ausente")
    for split in ("validacao", "teste"):
        record = evaluation.get(split)
        if (not isinstance(record, dict) or not _integer(record.get("n"))
                or not _metric(record.get("cobertura")) or not isinstance(record.get("metricas"), dict)):
            raise ValueError("avaliação do catálogo incompatível: " + split)
    # Verify the selected demonstration examples against actual model estimates.
    # The five-per-band quota never changes models or their global evaluation.
    frame = pd.DataFrame([s["caracteristicas"] for s in scenarios], columns=FEATURES)
    probabilities = {name: bundle[name].predict_proba(frame)[:, 1] for name in MODEL_NAMES}
    for i, scenario in enumerate(scenarios):
        values = {name: float(probabilities[name][i]) for name in MODEL_NAMES}
        if not all(_metric(value) for value in values.values()):
            raise ValueError("probabilidades do catálogo incompatíveis")
        scenario["probabilidades_estimadas"] = values
        scenario["faixa_chance"] = (
            "baixa" if max(values.values()) <= CHANCE_BANDS["baixa_max"] else
            "alta" if min(values.values()) >= CHANCE_BANDS["alta_min"] else "intermediaria"
        )
    catalog["faixas_chance"] = dict(CHANCE_BANDS)
    counts = {band: sum(s["faixa_chance"] == band for s in scenarios)
              for band in ("baixa", "alta", "intermediaria")}
    if counts != {"baixa": 5, "alta": 5, "intermediaria": 0}:
        raise ValueError("catálogo deve conter somente cinco exemplos baixos e cinco altos")
    return catalog


def create_app(model_path=None, catalog_path=None):
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"),
                static_folder=str(BASE_DIR / "static"))
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    app.json.ensure_ascii = False

    @app.after_request
    def utf8_json(response):
        if response.is_json:
            response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    bundle, catalog, load_error = None, None, None
    artifact_path = Path(model_path or BASE_DIR / "modelos_resolveai.joblib")
    coverage_path = Path(catalog_path or artifact_path.parent / "catalogo_cenarios.json")
    try:
        bundle = load_bundle(artifact_path)
        catalog = load_catalog(coverage_path, bundle, artifact_path)
    except FileNotFoundError:
        bundle, catalog = None, None
        load_error = "Modelos ou catalogo_cenarios.json ausentes. Execute o notebook e gere o catálogo antes de usar a API."
    except Exception as exc:
        bundle, catalog = None, None
        load_error = "Não foi possível carregar modelos e catálogo compatíveis. Confira o joblib, catalogo_cenarios.json e requirements.txt."
        app.logger.error("Falha ao carregar modelos: %s: %s", type(exc).__name__, exc)
    metadata = bundle["metadados"] if bundle is not None else {}
    scenarios = catalog["cenarios"] if catalog is not None else []
    by_values = {scenario_key(s["caracteristicas"]): s for s in scenarios}
    categories = ({field: sorted({s["caracteristicas"][field] for s in scenarios})
                   for field in CATEGORICAL_FEATURES} if scenarios else None)
    examples = [{"nome": s["nome"], "caracteristicas": s["caracteristicas"], "cenario_id": s["id"]}
                for s in scenarios]
    app.config["MODEL_BUNDLE"] = bundle
    app.config["MODEL_METADATA"] = metadata
    app.config["SCENARIO_CATALOG"] = catalog
    app.config["SCENARIOS_BY_VALUES"] = by_values
    app.config["ALLOWED_CATEGORIES"] = categories

    @app.get("/")
    def interface():
        return render_template("index.html")

    @app.get("/health")
    def health():
        ready = bundle is not None
        response = dict(
            status="ok" if ready else "indisponivel", api_disponivel=True,
            modelos={name: ready for name in MODEL_NAMES}, dados_sinteticos=False,
            modelo_recomendado=metadata.get("escolhido"), objetivo=OBJECTIVE,
            prazo_dias=HORIZON_DAYS, fonte=metadata.get("fonte", SOURCE),
            cenarios_disponiveis=len(scenarios), modo_interacao="cenarios_verificados",
        )
        if load_error:
            response["erro"] = load_error
        return jsonify(response), 200 if ready else 503

    @app.get("/metadata")
    def model_metadata():
        if bundle is None:
            return jsonify(erro=load_error), 503
        return jsonify(
            schema=2, entradas=FEATURES, categorias=categories,
            exemplos=examples, cenarios=scenarios,
            politica_cenarios=catalog["politica"], avaliacao_cenarios=catalog["avaliacao"],
            faixas_chance=catalog["faixas_chance"],
            selecao_demonstracao=catalog.get("selecao_demonstracao", {}),
            modo_interacao="cenarios_verificados",
            modelo_recomendado=metadata["escolhido"],
            metricas_validacao=metadata.get("metricas_validacao", {}),
            metricas_teste=metadata.get("metricas_teste", {}),
            objetivo=OBJECTIVE, prazo_dias=HORIZON_DAYS, dados_sinteticos=False,
            fonte=metadata["fonte"], aviso=NOTICE,
            limitacoes=metadata.get("limitacoes", []),
        )

    @app.post("/predict")
    def predict():
        if not request.is_json:
            return jsonify(erro="Use Content-Type: application/json."), 415

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
        validated, error = validate_payload(payload, categories, by_values if bundle is not None else None)
        if error:
            return jsonify(erro=error), 400
        if bundle is None:
            return jsonify(erro=load_error), 503
        name, frame = validated
        scenario = by_values[scenario_key(payload["caracteristicas"])]
        try:
            probability = float(bundle[name].predict_proba(frame)[0, 1])
            prediction = int(bundle[name].predict(frame)[0])
            if not math.isfinite(probability) or not 0 <= probability <= 1 or prediction not in (0, 1):
                raise ValueError("Saída fora do contrato")
        except Exception:
            app.logger.exception("Falha na inferência")
            return jsonify(erro="Falha ao executar o modelo. Confira o artefato e as dependências."), 500
        guidance = (
            "O modelo estima encerramento em até 7 dias. Acompanhe o protocolo nos canais oficiais."
            if prediction == 1 else
            "O modelo estima que o protocolo não será encerrado em até 7 dias. Acompanhe o andamento nos canais oficiais."
        )
        return jsonify(
            modelo=name, previsao_encerramento_7d=prediction,
            probabilidade_encerramento_7d=probability, prazo_dias=HORIZON_DAYS,
            dados_sinteticos=False, orientacao=guidance, aviso=NOTICE,
            cenario_id=scenario["id"], cobertura=scenario["cobertura"],
        )

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(erro=error.description), error.code

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
