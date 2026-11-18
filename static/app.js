"use strict";

(() => {
  const $ = selector => document.querySelector(selector);
  const form = $("#prediction-form");
  const panel = $(".results-panel");
  const results = $("#prediction-results");
  const details = $("#technical-details");
  const empty = $("#empty-state");
  const status = $("#result-status");
  const error = $("#form-error");
  const healthError = $("#health-error");
  const healthButton = $("#health-button");
  const predictButton = $("#predict-button");
  const labels = Object.freeze({ regressao_logistica: "Regressão Logística", random_forest: "Random Forest" });
  const bands = Object.freeze({ baixa: "Chance baixa", alta: "Chance alta", intermediaria: "Chance intermediária" });
  const percent = new Intl.NumberFormat("pt-BR", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const metricNumber = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
  let metadata = null;
  let scenarios = new Map();
  const selectedScenario = () => scenarios.get($("#cenario").value);
  const scenarioForBand = band => [...scenarios.values()]
    .filter(s => s.faixa_chance === band)
    .sort((a, b) => b.cobertura.validacao_combinacao - a.cobertura.validacao_combinacao
      || b.cobertura.treino_combinacao - a.cobertura.treino_combinacao
      || a.id.localeCompare(b.id))[0];
  let ready = false;
  let busy = false;
  let initializing = false;
  let report = null;

  function element(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }

  function updateControls() {
    const disabled = !ready || busy || initializing;
    form.querySelectorAll("input, select, button").forEach(el => { el.disabled = disabled; });
    ["baixa", "alta"].forEach(band => {
      const button = $("#example-" + band);
      button.disabled = disabled || !scenarioForBand(band);
      button.setAttribute("aria-pressed", String(selectedScenario()?.faixa_chance === band));
    });
    predictButton.disabled = disabled || !selectedScenario();
    healthButton.disabled = busy || initializing;
    predictButton.querySelector("span").textContent = busy ? "Consultando os modelos…" : "Estimar encerramento em 7 dias";
    panel.setAttribute("aria-busy", String(busy));
  }

  async function requestJson(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 45000);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal, cache: "no-store" });
      let body;
      try { body = await response.json(); }
      catch { throw new Error("O servidor não retornou uma resposta válida. Verifique a conexão e tente novamente."); }
      if (!body || typeof body !== "object" || Array.isArray(body)) throw new Error("O servidor retornou uma resposta incompatível. Verifique a conexão e tente novamente.");
      return { status: response.status, ok: response.ok, body };
    } catch (err) {
      if (err.name === "AbortError") throw new Error("A consulta demorou mais que o esperado. Aguarde alguns instantes e verifique a conexão novamente.");
      if (err instanceof TypeError) throw new Error("Não foi possível conectar à API. Confira o servidor e use o botão de conexão para tentar novamente.");
      throw err;
    } finally { clearTimeout(timer); }
  }

  function clearResult() {
    report = null;
    results.replaceChildren();
    results.hidden = true;
    empty.hidden = false;
    details.hidden = true;
    details.open = false;
    error.hidden = true;
    status.textContent = ready ? "Selecione um exemplo e consulte os modelos." : "Aguardando os exemplos da demonstração.";
    delete status.dataset.state;
  }

  function updateServiceText() {
    const scenario = selectedScenario();
    $("#scenario-details").hidden = !scenario;
    if (!scenario) return;
    const c = scenario.caracteristicas, n = scenario.cobertura;
    $("#scenario-band").textContent = scenario.faixa_chance === "intermediaria"
      ? "Estimativas em faixa intermediária ou diferentes entre os modelos"
      : `${bands[scenario.faixa_chance]} nos dois modelos`;
    $("#scenario-band").dataset.band = scenario.faixa_chance;
    const days = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"];
    $("#scenario-service").textContent = c.servico;
    $("#scenario-district").textContent = c.distrito;
    $("#scenario-channel").textContent = c.canal;
    $("#scenario-day").textContent = days[c.dia_semana];
    $("#training-coverage").textContent = `Combinação exata: ${n.treino_combinacao} registros no treino e ${n.validacao_combinacao} na validação. O contexto de serviço, distrito e canal tem ${n.treino_contexto} no treino e ${n.validacao_contexto} na validação.`;
    $("#validation-coverage").textContent = `O serviço reúne ${n.treino_servico} registros no treino e foi avaliado em ${n.validacao_servico} registros de validação.`;
  }

  function edited() {
    if (busy || initializing) return;
    const hadResult = Boolean(report);
    clearResult();
    if (hadResult) {
      status.textContent = "Informações alteradas. Faça uma nova estimativa.";
      status.dataset.state = "stale";
    }
    updateServiceText();
    updateControls();
  }

  function validScenario(s) {
    const c = s?.caracteristicas, n = s?.cobertura, p = s?.probabilidades_estimadas;
    return typeof s?.id === "string" && s.id.length > 0 && typeof s.nome === "string"
      && c && Object.keys(c).length === 4
      && ["servico", "distrito", "canal"].every(f => typeof c[f] === "string" && c[f].trim())
      && Number.isInteger(c.dia_semana) && c.dia_semana >= 0 && c.dia_semana <= 6
      && n && ["treino_combinacao", "validacao_combinacao", "treino_contexto", "validacao_contexto", "treino_servico", "validacao_servico"].every(f => Number.isInteger(n[f]) && n[f] >= 0)
      && Object.hasOwn(bands, s.faixa_chance)
      && p && Object.keys(labels).every(name => Number.isFinite(p[name]) && p[name] >= 0 && p[name] <= 1);
  }

  function populateMetadata(body) {
    if (!Array.isArray(body.cenarios) || body.cenarios.length !== 10 || !body.cenarios.every(validScenario)
      || !Object.hasOwn(labels, body.modelo_recomendado) || body.prazo_dias !== 7 || body.dados_sinteticos !== false
      || body.modo_interacao !== "cenarios_verificados"
      || body.faixas_chance?.baixa_max !== .3 || body.faixas_chance?.alta_min !== .7) throw new Error("Catálogo indisponível ou incompatível. Gere o catálogo e tente novamente.");
    if (new Set(body.cenarios.map(s => s.id)).size !== body.cenarios.length) throw new Error("O catálogo retornou cenários duplicados.");
    body.cenarios.forEach(s => {
      const values = Object.keys(labels).map(name => s.probabilidades_estimadas[name]);
      const expected = values.every(p => p <= body.faixas_chance.baixa_max) ? "baixa"
        : values.every(p => p >= body.faixas_chance.alta_min) ? "alta" : "intermediaria";
      if (s.faixa_chance !== expected) throw new Error("A faixa do cenário não corresponde às estimativas dos modelos. Verifique a conexão novamente.");
    });
    if (["baixa", "alta"].some(band => body.cenarios.filter(s => s.faixa_chance === band).length !== 5)) {
      throw new Error("A demonstração precisa conter cinco exemplos de chance baixa e cinco de chance alta. Atualize o catálogo e tente novamente.");
    }
    metadata = body;
    scenarios = new Map(body.cenarios.map(s => [s.id, Object.freeze({ ...s,
      caracteristicas: Object.freeze({ ...s.caracteristicas }),
      probabilidades_estimadas: Object.freeze({ ...s.probabilidades_estimadas }) })]));
    const select = $("#cenario");
    select.replaceChildren(new Option("Selecione um exemplo", "", true, true));
    const counts = {};
    Object.keys(bands).forEach(band => {
      const members = [...scenarios.values()].filter(s => s.faixa_chance === band);
      counts[band] = members.length;
      if (!members.length) return;
      const group = document.createElement("optgroup");
      group.label = `${bands[band]} (${members.length})`;
      members.forEach(s => group.append(new Option(s.nome, s.id)));
      select.append(group);
    });
    ["baixa", "alta"].forEach(band => {
      $("#example-" + band).textContent = `${bands[band]} (${counts[band]})`;
    });
    const distribution = `${counts.baixa} de chance baixa e ${counts.alta} de chance alta`
      + (counts.intermediaria ? `; ${counts.intermediaria} em faixa intermediária` : "");
    $("#scenario-help").textContent = `${scenarios.size} exemplos: ${distribution}. Escolha um dos cinco exemplos de cada grupo; as quatro informações ficam fixas.`;
    $("#chance-help").textContent = "Baixa: até 30% nos dois modelos. Alta: a partir de 70% nos dois. São estimativas, não certezas de encerramento. Escolha um exemplo e use Estimar para consultar os modelos.";
    $("#selection-description").textContent = `${labels[body.modelo_recomendado]} foi selecionada na validação. Os dois modelos continuam disponíveis para comparação.`;
    if (typeof body.fonte?.nome === "string") $("#source-link").textContent = body.fonte.nome;
    try { const u = new URL(body.fonte?.url); if (u.protocol === "https:") $("#source-link").href = u.href; } catch { /* Keep official link. */ }
    $("#source-period").textContent = typeof body.fonte?.periodo === "string" ? body.fonte.periodo : "";
    const p = body.politica_cenarios;
    if (p) $("#catalog-policy").textContent = `São ${p.exemplos_por_faixa} exemplos por faixa de chance, priorizados pelo volume observado no treino e na validação. Cada combinação exige pelo menos ${p.min_treino_combinacao} registros de treino e ${p.min_validacao_combinacao} de validação; o contexto de serviço, distrito e canal exige ${p.min_treino_contexto} no treino. O serviço exige ${p.min_treino_servico} no treino, ${p.min_validacao_servico} na validação e acurácia mínima de ${percent.format(p.min_acuracia_validacao_ambos)} em cada modelo. O teste não escolhe os exemplos. A seleção serve à demonstração e não representa todas as solicitações; esses mínimos não garantem acerto individual.`;
    const tbody = $("#metrics-body"); tbody.replaceChildren();
    ["referencia_dummy", ...Object.keys(labels)].forEach(name => {
      const m = body.metricas_teste?.[name];
      if (!m || !Number.isFinite(m.acuracia) || !Number.isFinite(m.f1_macro)) return;
      const row = document.createElement("tr");
      const h = element("th", "", name === "referencia_dummy" ? "Referência simples (Dummy)" : labels[name]); h.scope = "row";
      row.append(h, element("td", "", percent.format(m.acuracia)), element("td", "", metricNumber.format(m.f1_macro))); tbody.append(row);
    });
    $("#evaluation-details").hidden = !tbody.children.length;
    updateServiceText();
  }

  async function initialize() {
    if (initializing || busy) return;
    initializing = true;
    ready = false;
    scenarios = new Map();
    $("#cenario").replaceChildren(new Option("Carregando os exemplos…", ""));
    $("#scenario-help").textContent = "Carregando cinco exemplos de chance baixa e cinco de chance alta.";
    ["baixa", "alta"].forEach(band => { $("#example-" + band).textContent = bands[band]; });
    updateServiceText();
    clearResult();
    updateControls();
    $("#health-label").textContent = "Verificando conexão";
    healthError.hidden = true;
    try {
      const [health, config] = await Promise.all([requestJson("/health"), requestJson("/metadata")]);
      if (!health.ok || health.body.status !== "ok" || !Object.keys(labels).every(name => health.body.modelos?.[name] === true)) {
        throw new Error(health.body.erro || "Os modelos não estão disponíveis. Confira o servidor e o arquivo de modelos antes de tentar novamente.");
      }
      if (!config.ok) throw new Error(config.body.erro || "Não foi possível carregar as opções dos modelos. Tente novamente.");
      populateMetadata(config.body);
      ready = true;
      clearResult();
      healthButton.dataset.state = "ok";
      $("#health-label").textContent = "API e modelos disponíveis";
      healthButton.title = "Verificar novamente a conexão com a API e os dois modelos";
    } catch (err) {
      scenarios = new Map();
      $("#cenario").replaceChildren(new Option("Catálogo indisponível", ""));
      $("#scenario-help").textContent = "Catálogo indisponível. Use o botão de conexão para tentar novamente.";
      updateServiceText();
      healthButton.dataset.state = "error";
      $("#health-label").textContent = "Verificar conexão";
      healthButton.title = "Tentar carregar novamente a API e as opções dos modelos";
      healthError.textContent = err.message;
      healthError.hidden = false;
      status.textContent = "Não foi possível carregar modelos e catálogo. Use o botão de conexão para tentar novamente.";
      status.dataset.state = "error";
    } finally {
      initializing = false;
      updateControls();
    }
  }

  function validPrediction(body, expectedModel) {
    return body?.modelo === expectedModel && body.cenario_id === selectedScenario()?.id && [0, 1].includes(body.previsao_encerramento_7d)
      && Number.isFinite(body.probabilidade_encerramento_7d)
      && body.probabilidade_encerramento_7d >= 0 && body.probabilidade_encerramento_7d <= 1
      && body.prazo_dias === 7 && body.dados_sinteticos === false;
  }

  function renderPrediction(body) {
    const article = element("article", "model-result");
    article.dataset.prediction = String(body.previsao_encerramento_7d);
    const heading = element("div", "model-result-header");
    const name = document.createElement("div");
    name.append(element("h3", "", labels[body.modelo]));
    if (body.modelo === metadata.modelo_recomendado) name.append(element("p", "model-tag", "Selecionado na validação"));
    heading.append(name, element("div", "probability", percent.format(body.probabilidade_encerramento_7d)));
    article.append(heading, element("p", "probability-caption", "Chance estimada de encerramento em até 7 dias"));
    const track = element("div", "probability-track");
    track.setAttribute("role", "meter");
    track.setAttribute("aria-label", `Chance de encerramento segundo ${labels[body.modelo]}`);
    track.setAttribute("aria-valuemin", "0");track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String(body.probabilidade_encerramento_7d * 100));
    track.setAttribute("aria-valuetext", percent.format(body.probabilidade_encerramento_7d));
    const fill = element("div", "probability-fill");fill.style.width = `${body.probabilidade_encerramento_7d * 100}%`;
    track.append(fill, element("span", "threshold-mark"));
    const decision = body.previsao_encerramento_7d === 1 ? "Previsão: encerramento em até 7 dias" : "Previsão: sem encerramento nos primeiros 7 dias";
    article.append(track, element("p", "decision-label", decision), element("p", "guidance", typeof body.orientacao === "string" ? body.orientacao : "Acompanhe a solicitação pelos canais oficiais."));
    return article;
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (!ready || busy || !form.reportValidity()) return;
    const input = new FormData(form);
    const scenario = selectedScenario();
    if (!scenario) { error.textContent = "Selecione um cenário disponível no catálogo."; error.hidden = false; updateControls(); return; }
    // Four fixed values come only from the frozen catalog, never editable DOM fields.
    const characteristics = { ...scenario.caracteristicas };
    const selected = input.get("modelo");
    const names = selected === "ambos" ? Object.keys(labels) : [selected];
    if (!names.every(name => Object.hasOwn(labels, name))) return;
    const payloads = names.map(modelo => ({ modelo, caracteristicas: characteristics }));
    clearResult();
    busy = true;
    updateControls();
    status.textContent = names.length === 2 ? "Consultando os dois modelos…" : "Consultando o modelo…";
    try {
      const responses = await Promise.all(payloads.map(payload => requestJson("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })));
      responses.forEach((response, index) => {
        if (response.ok && !validPrediction(response.body, names[index])) throw new Error("A API retornou uma previsão incompatível. Confira o servidor e tente novamente.");
      });
      report = { cenario: scenario, entradas: payloads, respostas: responses.map(({ status: httpStatus, body }) => ({ http_status: httpStatus, ...body })), dados_sinteticos: false, fonte: metadata.fonte, objetivo: "encerramento_7d" };
      $("#request-json").textContent = JSON.stringify(payloads.length === 1 ? payloads[0] : payloads, null, 2);
      $("#response-json").textContent = JSON.stringify(report.respostas.length === 1 ? report.respostas[0] : report.respostas, null, 2);
      $("#request-description").textContent = names.length === 2 ? "Duas requisições POST /predict, uma por modelo, com as mesmas quatro informações." : "Uma requisição POST /predict com as quatro informações da solicitação.";
      details.hidden = false;empty.hidden = true;results.hidden = false;
      responses.forEach(response => results.append(response.ok ? renderPrediction(response.body) : element("p", "error-message", response.body.erro || "Não foi possível obter a previsão deste modelo.")));
      const valid = responses.filter(response => response.ok);
      if (valid.length === 2) results.append(element("p", "comparison-note", valid[0].body.previsao_encerramento_7d === valid[1].body.previsao_encerramento_7d
        ? "Os modelos indicaram a mesma classe. A concordância não garante o prazo de encerramento."
        : "Os modelos indicaram classes diferentes. Isso mostra a incerteza da estimativa; o prazo não é garantido."));
      status.textContent = valid.length === responses.length ? "Estimativa concluída com modelos treinados em dados reais do SP156." : "A consulta apresentou um erro. Confira a mensagem e tente novamente.";
      if (valid.length !== responses.length) status.dataset.state = "error";
      if (window.matchMedia("(max-width: 760px)").matches) panel.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
    } catch (err) {
      clearResult();error.textContent = err.message;error.hidden = false;
      status.textContent = "Não foi possível concluir a estimativa.";status.dataset.state = "error";
    } finally { busy = false;updateControls(); }
  });

  form.addEventListener("input", edited);form.addEventListener("change", edited);
  ["baixa", "alta"].forEach(band => {
    $("#example-" + band).addEventListener("click", () => {
      if (!ready || busy || initializing) return;
      const scenario = scenarioForBand(band);
      if (!scenario) return;
      $("#cenario").value = scenario.id;
      edited();
    });
  });
  form.addEventListener("reset", () => {
    clearResult();
    setTimeout(() => { $("#cenario").value = ""; updateServiceText(); updateControls(); }, 0);
  });
  $("#download-result").addEventListener("click", () => {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json;charset=utf-8" }));
    const a = document.createElement("a");a.href = url;a.download = "resolveai_estimativa_sp156.json";
    document.body.append(a);a.click();a.remove();setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  healthButton.addEventListener("click", initialize);
  void initialize();
})();
