"use strict";

(() => {
  const form = document.querySelector("#prediction-form");
  const button = document.querySelector("#predict-button");
  const panel = document.querySelector(".results-panel");
  const results = document.querySelector("#prediction-results");
  const empty = document.querySelector("#empty-state");
  const status = document.querySelector("#result-status");
  const error = document.querySelector("#form-error");
  const details = document.querySelector("#technical-details");
  const healthButton = document.querySelector("#health-button");
  const healthError = document.querySelector("#health-error");
  const boundedFields = ["moradores_ativos_7d", "taxa_apoio_anterior_28d", "ocorrencias_regiao_7d"]
    .map(id => document.getElementById(id));
  const limitNumber = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 10 });
  const labels = { regressao_logistica: "Regressão Logística", random_forest: "Random Forest" };
  const percent = new Intl.NumberFormat("pt-BR", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const scenarios = {
    iluminacao: { categoria: "iluminacao", regiao: "norte", dia_semana: 2, hora_abertura: 19, moradores_ativos_7d: 40, taxa_apoio_anterior_28d: 50, ocorrencias_regiao_7d: 20 },
    baixa: { categoria: "acessibilidade", regiao: "sul", dia_semana: 1, hora_abertura: 8, moradores_ativos_7d: 28, taxa_apoio_anterior_28d: 35, ocorrencias_regiao_7d: 30 },
    alta: { categoria: "iluminacao", regiao: "oeste", dia_semana: 5, hora_abertura: 18, moradores_ativos_7d: 110, taxa_apoio_anterior_28d: 60, ocorrencias_regiao_7d: 24 }
  };
  let busy = false;
  let report = null;

  function validateTrainingRange(field) {
    field.setCustomValidity("");
    if (field.disabled || field.value === "") return;
    const value = field.valueAsNumber;
    const minimum = Number(field.min);
    const maximum = Number(field.max);
    if (!Number.isFinite(value) || value < minimum || value > maximum || (field.step === "1" && !Number.isInteger(value))) {
      const label = form.querySelector(`label[for="${field.id}"]`).textContent;
      const unit = field.id === "taxa_apoio_anterior_28d" ? "%" : "";
      const kind = field.step === "1" ? "um número inteiro" : "um número";
      field.setCustomValidity(`${label}: informe ${kind} entre ${limitNumber.format(minimum)}${unit} e ${limitNumber.format(maximum)}${unit}, dentro da faixa observada no treino.`);
    }
  }

  boundedFields.forEach(field => field.addEventListener("input", () => validateTrainingRange(field)));
  form.addEventListener("invalid", event => {
    if (boundedFields.includes(event.target)) validateTrainingRange(event.target);
    error.textContent = event.target.validationMessage;
    error.hidden = false;
  }, true);

  async function requestJson(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal, cache: "no-store" });
      const body = await response.json();
      return { status: response.status, ok: response.ok, body };
    } catch (err) {
      throw new Error(err.name === "AbortError"
        ? "A consulta demorou mais que o esperado. Confira o terminal do Flask e tente novamente."
        : "Não foi possível conectar à API. Mantenha o servidor Flask aberto e tente novamente.");
    } finally { clearTimeout(timer); }
  }

  async function checkHealth() {
    healthButton.disabled = true;
    document.querySelector("#health-label").textContent = "Verificando conexão";
    try {
      const response = await requestJson("/health");
      const ready = response.ok && response.body.status === "ok" && response.body.modelos?.regressao_logistica === true && response.body.modelos?.random_forest === true;
      healthButton.dataset.state = ready ? "ok" : "error";
      document.querySelector("#health-label").textContent = ready ? "API e modelos disponíveis" : "Modelos indisponíveis";
      healthButton.title = ready ? "Os dois modelos estão prontos. Clique para verificar novamente." : response.body.erro || "Confira os arquivos dos modelos e o terminal do Flask.";
      if (!ready) {
        healthError.textContent = response.body.erro || "Os modelos estão indisponíveis. Confira o arquivo joblib junto da API e tente novamente.";
        healthError.hidden = false;
      } else { healthError.hidden = true; }
    } catch (err) {
      healthButton.dataset.state = "error";
      document.querySelector("#health-label").textContent = "Sem conexão com a API";
      healthButton.title = err.message;
      healthError.textContent = err.message;
      healthError.hidden = false;
    } finally { healthButton.disabled = false; }
  }

  function clearResult() {
    report = null;
    results.replaceChildren();
    results.hidden = true;
    empty.hidden = false;
    details.hidden = true;
    details.open = false;
    status.textContent = "Preencha a ocorrência para começar.";
    delete status.dataset.state;
    error.hidden = true;
  }

  function edited() {
    error.hidden = true;
    if (report) {
      // Nunca deixar uma resposta anterior parecer relativa às entradas novas.
      clearResult();
      status.textContent = "Cenário alterado. Faça uma nova estimativa.";
      status.dataset.state = "stale";
    }
    document.querySelectorAll("[data-scenario]").forEach(el => el.setAttribute("aria-pressed", "false"));
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderPrediction(item) {
    const body = item.body;
    const article = element("article", "model-result");
    article.dataset.prediction = String(body.previsao_apoio_72h);
    const heading = element("div", "model-result-header");
    const name = element("div");
    name.append(element("h3", "", labels[body.modelo]));
    if (body.modelo === "regressao_logistica") name.append(element("p", "model-tag", "Selecionado na validação"));
    const probability = element("div", "probability", percent.format(body.probabilidade_apoio_72h));
    heading.append(name, probability);
    article.append(heading, element("p", "probability-caption", "Probabilidade estimada de receber apoio"));
    const track = element("div", "probability-track");
    track.setAttribute("role", "meter");
    track.setAttribute("aria-label", `Probabilidade de apoio segundo ${labels[body.modelo]}`);
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String(body.probabilidade_apoio_72h * 100));
    track.setAttribute("aria-valuetext", percent.format(body.probabilidade_apoio_72h));
    const fill = element("div", "probability-fill");
    fill.style.width = `${body.probabilidade_apoio_72h * 100}%`;
    track.append(fill, element("span", "threshold-mark"));
    const decision = element("p", "decision-label", body.previsao_apoio_72h === 1 ? "Previsão: receberá apoio em 72h" : "Previsão: não receberá apoio em 72h");
    article.append(track, decision, element("p", "guidance", body.orientacao));
    return article;
  }

  function setBusy(value) {
    busy = value;
    panel.setAttribute("aria-busy", String(value));
    // Bloquear mudanças durante a consulta para manter entrada e resultado coerentes.
    form.querySelectorAll("input, select, button").forEach(el => { el.disabled = value; });
    document.querySelectorAll("[data-scenario]").forEach(el => { el.disabled = value; });
    button.querySelector("span").textContent = value ? "Consultando os modelos…" : "Estimar apoio em 72h";
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    boundedFields.forEach(validateTrainingRange);
    if (busy || !form.reportValidity()) return;
    const input = new FormData(form);
    const characteristics = {
      categoria: input.get("categoria"), regiao: input.get("regiao"),
      dia_semana: Number(input.get("dia_semana")), hora_abertura: Number(input.get("hora_abertura")),
      moradores_ativos_7d: Number(input.get("moradores_ativos_7d")),
      taxa_apoio_anterior_28d: Number(input.get("taxa_apoio_anterior_28d")) / 100,
      ocorrencias_regiao_7d: Number(input.get("ocorrencias_regiao_7d"))
    };
    const names = input.get("modelo") === "ambos" ? ["regressao_logistica", "random_forest"] : [input.get("modelo")];
    const payloads = names.map(modelo => ({ modelo, caracteristicas: characteristics }));
    clearResult();
    status.textContent = names.length === 2 ? "Consultando os dois modelos…" : "Consultando o modelo…";
    setBusy(true);
    try {
      const responses = await Promise.all(payloads.map(payload => requestJson("/predict", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
      })));
      report = { entradas: payloads, respostas: responses.map(({ status, body }) => ({ http_status: status, ...body })), dados_sinteticos: true };
      document.querySelector("#request-json").textContent = JSON.stringify(payloads.length === 1 ? payloads[0] : payloads, null, 2);
      document.querySelector("#response-json").textContent = JSON.stringify(report.respostas.length === 1 ? report.respostas[0] : report.respostas, null, 2);
      document.querySelector("#request-description").textContent = `${payloads.length} ${payloads.length === 1 ? "requisição POST /predict. O valor" : "requisições POST /predict, uma por modelo. O valor"} de histórico é convertido de porcentagem para proporção.`;
      details.hidden = false;
      empty.hidden = true;
      results.hidden = false;
      const valid = responses.filter(response => response.ok);
      for (const response of responses) {
        if (response.ok) {
          const body = response.body;
          if (!Object.hasOwn(labels, body.modelo) || ![0, 1].includes(body.previsao_apoio_72h) || !Number.isFinite(body.probabilidade_apoio_72h) || body.probabilidade_apoio_72h < 0 || body.probabilidade_apoio_72h > 1) throw new Error("A API retornou uma resposta inesperada. Confira o terminal e tente novamente.");
          results.append(renderPrediction(response));
        } else {
          const message = element("p", "error-message", response.body.erro || "Não foi possível obter a previsão deste modelo.");
          results.append(message);
        }
      }
      if (valid.length === 2) {
        const disagree = valid[0].body.previsao_apoio_72h !== valid[1].body.previsao_apoio_72h;
        const note = element("p", "comparison-note", disagree
          ? "Os modelos discordaram neste cenário. Isso evidencia a incerteza da previsão; nenhum resultado é garantia de participação."
          : "Os modelos chegaram à mesma classe neste cenário. Mesmo com concordância, as probabilidades não garantem participação.");
        results.append(note);
      }
      if (valid.length === responses.length) {
        status.textContent = `${valid.length === 2 ? "Comparação concluída" : "Estimativa concluída"}. Dados sintéticos.`;
        delete status.dataset.state;
      } else {
        status.textContent = "A consulta apresentou um erro. Confira a mensagem abaixo e tente novamente.";
        status.dataset.state = "error";
        void checkHealth();
      }
      if (window.matchMedia("(max-width: 760px)").matches) panel.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
    } catch (err) {
      clearResult();
      error.textContent = err.message;
      error.hidden = false;
      status.textContent = "Não foi possível concluir a estimativa.";
      status.dataset.state = "error";
      healthButton.dataset.state = "error";
      document.querySelector("#health-label").textContent = "Verificar conexão com a API";
    } finally { setBusy(false); }
  });

  form.addEventListener("input", edited);
  form.addEventListener("change", edited);
  form.addEventListener("reset", () => {
    clearResult();
    boundedFields.forEach(field => field.setCustomValidity(""));
    document.querySelectorAll("[data-scenario]").forEach(el => el.setAttribute("aria-pressed", "false"));
  });
  document.querySelectorAll("[data-scenario]").forEach(el => {
    el.setAttribute("aria-pressed", "false");
    el.addEventListener("click", () => {
      clearResult();
      for (const [field, value] of Object.entries(scenarios[el.dataset.scenario])) form.elements.namedItem(field).value = value;
      boundedFields.forEach(validateTrainingRange);
      document.querySelectorAll("[data-scenario]").forEach(other => other.setAttribute("aria-pressed", String(other === el)));
      status.textContent = "Exemplo preenchido. Faça uma estimativa ou altere as informações.";
    });
  });
  document.querySelector("#download-result").addEventListener("click", () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "resolveai_estimativa.json";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  healthButton.addEventListener("click", checkHealth);
  void checkHealth();
})();
