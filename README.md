# ResolveAI — Machine Learning com dados reais do SP156

Projeto de Artificial Intelligence & Chatbot · Global Solution 2026/2 · CitizenTech · FIAP.

## Site e API publicados

**Site de demonstração:** [Abrir o ResolveAI](https://resolveai-artificial-intelligence.onrender.com/).

- **Disponibilidade dos modelos:** [GET /health](https://resolveai-artificial-intelligence.onrender.com/health).
- **Exemplos aceitos e métricas:** [GET /metadata](https://resolveai-artificial-intelligence.onrender.com/metadata).
- **Previsão por JSON:** `POST https://resolveai-artificial-intelligence.onrender.com/predict`.

O mesmo serviço Flask disponibiliza o site e a API. Para usar a versão publicada na apresentação, abra o link; a execução local abaixo é uma alternativa.

## Objetivo

Estimar **encerramento administrativo em até 7 dias corridos**, usando serviço, distrito, canal e dia da abertura, para apoiar o acompanhamento do protocolo.

Encerramento pode incluir cancelamento ou indeferimento; não comprova resolução física. O prazo usa diferença entre datas de calendário, não exatamente 168 horas.

## Estrutura do projeto

```text
ResolveAI_Entrega/
├── ResolveAI_Machine_Learning.ipynb
├── dados_resolveai.csv / modelos_resolveai.joblib
├── api_resolveai.py
├── catalogo_cenarios.json / gerar_catalogo_cenarios.py
├── fonte_dados.json / preparar_dados_sp156.py
├── requirements.txt / requirements-prod.txt / .python-version
├── templates/index.html
├── static/  (CSS, JavaScript, marca e favicon)
├── ResolveAI_Apresentacao_Presencial.pdf
├── integrantes_e_explicacoes.txt
└── README.md
```

O notebook treina e exporta; Flask carrega joblib e catálogo uma vez, sem treinar nas requisições.

## Execução local

Use Python **3.13**. Na pasta do projeto, no PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe api_resolveai.py
```

Abra **http://127.0.0.1:5000/** e mantenha o terminal aberto. `Ctrl+C` encerra o servidor. Os arquivos pré-treinados já permitem usar a aplicação sem executar o notebook antes.

Para abrir o notebook:

```powershell
.venv\Scripts\python.exe -m notebook ResolveAI_Machine_Learning.ipynb
```

Se executar todas as células e exportar novamente o joblib, gere o catálogo **depois da exportação** e reinicie a API:

```powershell
.venv\Scripts\python.exe gerar_catalogo_cenarios.py
.venv\Scripts\python.exe api_resolveai.py
```

O catálogo referencia os hashes do modelo e do CSV. A API recusa artefatos incompatíveis; não cria modelos ou exemplos automaticamente.

## Dados e preparação

Fonte: [SP156 — Prefeitura de São Paulo](https://dados.prefeitura.sp.gov.br/dataset/dados-do-sp156), aberturas no primeiro trimestre de 2025. URLs oficiais, hashes e auditoria ficam em `fonte_dados.json`.

Usamos os temas **Rua e bairro** e **Lixo e limpeza**, filtramos datas incoerentes e situações sem observação suficiente e extraímos uma amostra uniforme de **60.000 registros**, sem reposição ou balanceamento pelo alvo, com semente `20261117`. Os filtros e contagens estão no notebook e na auditoria.

A amostra tem 64,36% de encerramentos no prazo e 35,64% sem encerramento. O CSV omite endereços completos e campos pessoais; `id_registro` identifica a linha da fonte, não o protocolo.

Download: 17/11/2026; a fonte inclui fechamentos posteriores ao trimestre. Na exploração do treino, entulho teve 98,09% de encerramentos no prazo, contra 51,73% em Tapa-buraco: serviços têm ritmos diferentes.

Para reproduzir a preparação, opcionalmente:

```powershell
.venv\Scripts\python.exe preparar_dados_sp156.py --fonte caminho\arquivo-original.csv
```

Sem `--fonte`, baixa cerca de 155 MB para `.cache_sp156/`. Depois, execute notebook, geração do catálogo e API.

## Pipeline, modelos e avaliação

As entradas são `servico`, `distrito`, `canal` e `dia_semana` (segunda = 0; domingo = 6). **Status, fechamento e duração não são entradas.** Cada pipeline inclui `OneHotEncoder`, ajustado somente no treino.

Os dois modelos são de classificação:

- **Regressão Logística:** alternativa simples para estimar probabilidades.
- **Random Forest:** árvores que captam interações entre as características.

O Dummy é uma referência. Selecionamos configurações pelo F1 macro na validação, com limiar fixo de 0,5. A Logística venceu; ambos os modelos ficam na API.

| Etapa | Abertura das solicitações | Registros |
|---|---|---:|
| Treino | 01/01 a 14/02/2025 | 30.714 |
| Validação | 22/02 a 08/03/2025 | 9.744 |
| Teste | 16/03 a 31/03/2025 | 9.695 |

Os intervalos entre etapas permitem observar os 7 dias do alvo. Outros 9.847 registros ficam nesses intervalos e não entram no ajuste ou na avaliação.

| Modelo | Acurácia no teste | F1 macro | Brier |
|---|---:|---:|---:|
| Dummy | 62,76% | 0,3856 | 0,2343 |
| Regressão Logística | **83,87%** | **0,8264** | **0,1082** |
| Random Forest | 82,45% | 0,8035 | 0,1190 |

Acurácia mede acertos; F1 macro combina precisão e recall com peso igual às classes (maior é melhor); Brier mede erro das probabilidades (menor é melhor). Precisão, recall e matrizes de confusão estão no notebook. Os resultados também ficam no joblib. Probabilidade individual não é acurácia.

## Exemplos da interface e contrato da API

A interface oferece **10 exemplos: 5 baixos e 5 altos**. Baixa: até 30% nos dois modelos; alta: pelo menos 70%. Os atalhos preenchem entradas fixas; clique em **Estimar encerramento em 7 dias** para consultar a API.

O catálogo exige pelo menos:

- 8 registros da combinação exata no treino e 2 na validação;
- 20 registros de treino do contexto serviço/distrito/canal, sem considerar o dia;
- 200 registros de treino e 50 de validação do serviço;
- 75% de acurácia na validação do serviço para cada modelo.

O piso anterior de 20 registros idênticos fornecia só um baixo. A política foi ajustada para a demonstração. O ranking usa volume de treino exato, depois validação exata e ID; não usa acertos no teste. Avaliar o serviço não comprova desempenho da combinação exata nem garante acerto individual.

| Método e rota | Uso |
|---|---|
| `GET /` | Interface visual |
| `GET /health` | Disponibilidade dos modelos e quantidade de exemplos |
| `GET /metadata` | Dez exemplos, cobertura, faixas e métricas |
| `POST /predict` | Previsão de um modelo com corpo JSON |

Exemplo aceito em `POST http://127.0.0.1:5000/predict`, com `Content-Type: application/json`:

```json
{
  "modelo": "regressao_logistica",
  "caracteristicas": {
    "servico": "Reclamação de poluição sonora - PSIU",
    "distrito": "SAUDE",
    "canal": "PORTAL",
    "dia_semana": 5
  }
}
```

Troque apenas `modelo` por `random_forest` para comparar: cerca de **1,68% / 7,79%**, respectivamente. A resposta traz classe, probabilidade, `cenario_id`, cobertura e orientações. Combinações fora de `/metadata` retornam 400; artefatos indisponíveis, 503.

## Limites dos resultados

O serviço pode ter sido reclassificado após a abertura. O relatório não reconstrói completamente o cadastro inicial: os resultados são **retrospectivos**, sem comprovação prospectiva para novos protocolos, períodos ou cidades.

Os dez exemplos são demonstrativos, não representativos. Seu recorte no teste tem só 44 registros; as métricas em `/metadata` são descritivas, sem validação independente, pois a política mudou após a análise do teste geral. Não substituem as métricas globais nem demonstram melhora dos modelos.

## Verificação e deploy

Confira `/health` e `/metadata` pelos links publicados acima. Para verificar a inferência, use a interface ou envie o JSON do exemplo para `POST /predict`.

No Render, mantenha os arquivos na raiz, branch `main` e Python definido em `.python-version`:

```text
Build: python -m pip install --only-binary=:all: -r requirements-prod.txt
Start: gunicorn api_resolveai:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

O mesmo serviço Flask entrega o site e a API. `requirements-prod.txt` mantém as versões de ML do joblib e evita instalar o notebook no servidor.

## Evidências da disciplina e integrantes

Evidências: CSV real CitizenTech; notebook com exploração → preparação → treinamento → avaliação → exportação; dois classificadores no joblib; API Flask consumindo os modelos por JSON. A apresentação exige demonstração ao vivo e domínio técnico de todos. Apoio: `ResolveAI_Apresentacao_Presencial.pdf`.

- Andrei Oliveira de Sousa — RM 573373
- Felipe Coelho da Fonseca — RM 572133
- Henrique Soares Pereira — RM 569898
- Luiz Felipe Lourenço da Silva — RM 572576
