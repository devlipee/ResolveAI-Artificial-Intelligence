diff --git a/C:\Users\sluiz\Downloads\ResolveAI_Entrega_Com_Limites\ResolveAI_Entrega\README.md b/C:\Users\sluiz\Downloads\ResolveAI_Entrega_Com_Limites\ResolveAI_Entrega\README.md
new file mode 100644
--- /dev/null
+++ b/C:\Users\sluiz\Downloads\ResolveAI_Entrega_Com_Limites\ResolveAI_Entrega\README.md
@@ -0,0 +1,310 @@
+# ResolveAI — apoio comunitário com Machine Learning
+
+Projeto acadêmico CitizenTech que estima, no momento da abertura de uma ocorrência urbana, se ela receberá **pelo menos um apoio válido de outra conta nas próximas 72 horas**. Todos os dados são sintéticos: foram produzidos por uma simulação, sem registros reais de moradores.
+
+## Resumo do projeto
+
+A proposta do ResolveAI envolve registrar, apoiar e acompanhar relatos sobre iluminação, limpeza, vias e acessibilidade. Esta entrega implementa o experimento de Machine Learning e uma interface de demonstração para prever a participação da comunidade.
+
+O fluxo da demonstração é:
+
+1. O usuário informa as características da ocorrência e o contexto anterior da comunidade.
+2. A interface envia os dados para uma API Flask.
+3. Um dos dois modelos treinados estima a probabilidade de apoio em até 72 horas e retorna uma classe.
+4. Uma regra textual sugere compartilhar o relato quando a previsão é ausência de apoio, ou acompanhar a participação quando a previsão é apoio.
+
+A orientação é uma regra baseada na previsão, sem um terceiro modelo. A demonstração não grava ocorrências ou apoios, não altera status e não prevê a resolução do problema. Não há integração com prefeitura ou execução de serviços públicos.
+
+## Arquivos principais
+
+| Arquivo | Função |
+|---|---|
+| [ResolveAI_Machine_Learning.ipynb](ResolveAI_Machine_Learning.ipynb) | Geração e auditoria dos dados, exploração, treinamento, avaliação e exportação dos modelos. Contém as saídas e os gráficos da execução entregue. |
+| [dados_resolveai.csv](dados_resolveai.csv) | Base sintética efetivamente utilizada no experimento. |
+| [modelos_resolveai.joblib](modelos_resolveai.joblib) | Dois pipelines treinados, com pré-processamento, modelos e metadados. |
+| [api_resolveai.py](api_resolveai.py) | Carrega os modelos e disponibiliza a interface e as rotas de previsão. |
+| [templates/index.html](templates/index.html) e [static/](static/) | Interface para simular cenários e comparar as previsões. |
+| [requirements.txt](requirements.txt) | Dependências fixadas para o ambiente de notebook e execução local no Windows. |
+| [requirements-prod.txt](requirements-prod.txt) | Dependências do serviço Flask no Linux, incluindo Gunicorn. |
+| [integrantes_e_explicacoes.txt](integrantes_e_explicacoes.txt) | Instruções detalhadas, contrato da API e registro das verificações da entrega. |
+
+## Dados e significado das entradas
+
+Cada linha do CSV representa uma ocorrência. A base final tem **3.458 linhas e 11 colunas**, com aberturas de janeiro a julho de 2026. São 1.980 ocorrências sem apoio e 1.478 com apoio. As datas representam o horário local de São Paulo.
+
+Os dois modelos recebem as mesmas sete entradas:
+
+| Entrada | O que representa | Como é utilizada |
+|---|---|---|
+| `categoria` | `iluminacao`, `limpeza`, `vias` ou `acessibilidade`. | Contexto do tipo de relato. |
+| `regiao` | `norte`, `sul`, `leste` ou `oeste`, todas fictícias. | Contexto regional da comunidade simulada. |
+| `dia_semana` | Dia da abertura: segunda = 0, domingo = 6. | Padrões de participação por dia. |
+| `hora_abertura` | Hora da abertura, de 0 a 23. | Padrões de participação por horário. |
+| `moradores_ativos_7d` | Quantidade de contas distintas com sessões na região nos sete dias anteriores. | Indicador de atividade recente da comunidade. |
+| `taxa_apoio_anterior_28d` | Proporção suavizada de ocorrências semelhantes que receberam apoio, com resultados já conhecidos antes da abertura atual. | Indicador histórico de participação na mesma região e categoria. |
+| `ocorrencias_regiao_7d` | Quantidade de ocorrências abertas na região nos sete dias anteriores, excluindo a atual. | Indicador de volume recente de relatos e possível competição por atenção. |
+
+As outras quatro colunas têm funções diferentes:
+
+| Coluna | Função |
+|---|---|
+| `ocorrencia_id` | Identificador sintético para auditoria; não entra no modelo. |
+| `criada_em` | Data e hora usadas na ordenação, nos históricos e na separação temporal; não entra diretamente no modelo. |
+| `observacao_ate` | Abertura + 72 horas; permite verificar se a janela do resultado já terminou. Não entra no modelo. |
+| `apoio_72h` | Alvo: **1** para ao menos um apoio válido em 72 horas; **0** para nenhum. Nunca é uma entrada da previsão. |
+
+Um apoio válido deve vir de uma conta habilitada, diferente da autora e que ainda não tenha apoiado aquela ocorrência. O alvo só pode ser conhecido após o encerramento da janela de 72 horas.
+
+## Como os dados sintéticos foram gerados
+
+A geração é feita pela função `simular_ocorrencias()` no notebook, com a semente fixa **20261112**. Não foi usado um modelo generativo treinado em dados reais: o simulador combina regras inventadas, distribuições de probabilidade e eventos de uso.
+
+São sorteadas 4.000 aberturas entre 01/12/2025 e 31/07/2026. Dezembro serve para formar históricos anteriores às primeiras linhas de janeiro e não entra no CSV. A extração simulada ocorre em 04/08/2026, quando todas as 3.458 ocorrências mantidas já têm a janela completa de observação.
+
+### Parâmetros e distribuições da simulação
+
+Estes valores são **hipóteses do gerador**, sem medição empírica que os sustente. Não são métricas de qualidade dos modelos.
+
+| Elemento | Regra usada | Finalidade |
+|---|---|---|
+| Comunidade | 200 contas por região; cada conta tem 96% de chance de estar habilitada. | Formar o universo fictício de participantes. |
+| Aberturas | 4.000 instantes distintos sorteados no período; região e categoria sorteadas uniformemente. | Criar ocorrências ao longo do tempo. |
+| Sessões por região | Quantidade diária por Poisson, com bases 5, 9, 14 e 20 para norte, sul, leste e oeste. | Produzir regiões com diferentes níveis de atividade. |
+| Oscilação das sessões | Multiplicador lognormal com média do log = −0,08 e desvio do log = 0,4. | Variar a atividade entre os dias. |
+| Atenção futura | Multiplicador lognormal com média do log = −0,245 e desvio do log = 0,7. | Representar circulação e interesse futuros que não são observados na abertura. |
+| Categoria | Multiplicadores: iluminação 1,30; limpeza 0,85; vias 1,05; acessibilidade 0,75. | Alterar a quantidade esperada de exposições. |
+| Horário | Multiplicador 1,25 das 17h às 22h; 0,80 antes das 7h; 1,00 nos demais horários. | Simular variação de exposição por horário. |
+| Fim de semana | Multiplicador 1,15 no sábado e domingo; 1,00 nos demais dias. | Simular variação de exposição por dia. |
+| Tentativa de apoio | Cada exposição tem 72% de chance de gerar uma tentativa. | Separar visualizar um relato de efetivamente tentar apoiá-lo. |
+| Instante da tentativa | Sorteado entre 1 segundo e 96 horas após a abertura. | Permitir apoios dentro e fora da janela do alvo. |
+
+Poisson sorteia contagens a partir de uma quantidade esperada. A lognormal produz multiplicadores positivos e permite oscilações ocasionais maiores. Os parâmetros de média e desvio indicados para a lognormal se referem ao logaritmo da variável.
+
+A quantidade de exposições de cada ocorrência é sorteada por Poisson com a seguinte média:
+
+```text
+lambda = (0,65 + moradores_ativos_7d / 40)
+         × fator_categoria × fator_horario × fator_fim_semana
+         × atencao_futura / (1 + ocorrencias_regiao_7d / 45)
+```
+
+No gerador, mais moradores ativos aumentam a exposição esperada; mais relatos recentes reduzem a atenção disponível para cada ocorrência. Esses efeitos são hipóteses programadas, não conclusões sobre comunidades reais. O histórico de apoio é calculado como entrada, mas não aparece diretamente nessa fórmula.
+
+As tentativas são processadas em ordem de tempo. São descartados apoios da própria autora, de contas não habilitadas e apoios repetidos da mesma conta. Somente apoios válidos até 72 horas tornam `apoio_72h = 1`; uma ocorrência pode receber apoio depois e ainda ter alvo 0.
+
+As classes surgem desses eventos: não foram geradas separadamente, não houve troca de rótulos e o simulador não foi ajustado para alcançar determinada acurácia ou favorecer um modelo. Atenção futura, exposições, contas e eventos de apoio são internos ao simulador e ficam fora das entradas dos classificadores.
+
+### Indicadores históricos calculados para sintetizar o contexto
+
+Os três indicadores numéricos resumem apenas informações disponíveis antes da abertura:
+
+- **Moradores ativos:** número de contas distintas com sessões no intervalo `[abertura − 7 dias, abertura)`. Várias sessões da mesma conta contam como um morador ativo.
+- **Ocorrências recentes:** número de aberturas da mesma região nesse intervalo de sete dias. Não depende de terem recebido apoio ou sido resolvidas.
+- **Taxa histórica de apoio:** considera a mesma região e categoria, com abertura a partir de 31 dias antes e janela de 72 horas encerrada estritamente antes da ocorrência atual. Isso corresponde a aproximadamente 28 dias de aberturas com resultados já conhecidos.
+
+A taxa histórica usa suavização:
+
+```text
+taxa_apoio_anterior_28d = (2 + ocorrencias_anteriores_com_apoio)
+                        / (4 + total_de_ocorrencias_anteriores_elegiveis)
+```
+
+Os valores 2 e 4 funcionam como uma referência inicial de 50%, reduzindo taxas extremas quando há poucos exemplos. Sem histórico, a taxa é `2/4 = 0,5`. Com 3 ocorrências com apoio em 6 elegíveis, ela é `(2 + 3)/(4 + 6) = 0,5`. A API recebe essa taxa como proporção: `0.5` significa 50%.
+
+### Medidas usadas para resumir e conferir os dados
+
+O notebook usa contagens, proporções por classe, tabelas por categoria e mês, histogramas, média, desvio padrão, mínimo, máximo e quartis. A média resume o nível geral; o desvio padrão mede a dispersão; a mediana divide os valores ao meio; os quartis de 25% e 75% delimitam a metade central da distribuição.
+
+Exemplo do resumo do **CSV completo**, com valores arredondados:
+
+| Indicador | Média | Desvio padrão | Mínimo | Mediana | Máximo |
+|---|---:|---:|---:|---:|---:|
+| Moradores ativos em 7 dias | 65,423 | 26,963 | 14 | 64 | 129 |
+| Taxa histórica de apoio | 0,439 | 0,124 | 0,143 | 0,436 | 0,778 |
+| Ocorrências na região em 7 dias | 28,480 | 5,050 | 12 | 28 | 46 |
+
+Essas medidas descrevem os dados gerados. **Não foi calculada uma métrica de fidelidade contra dados reais**, pois não há uma base real de referência. Acurácia, F1 e Brier são calculados depois do treinamento para avaliar as previsões.
+
+O notebook também verifica ausência de valores faltantes e duplicidades, identificadores únicos, tipos, limites, ordenação temporal e maturação dos históricos. Reconstrói o alvo pelos eventos de apoio para conferir os rótulos. A reprodução do CSV é comparada com a geração de semente fixa, e seu hash SHA-256 é registrado nos metadados.
+
+Na exploração do treino, foram encontrados 252 casos de alta atividade sem apoio e 177 de baixa atividade com apoio, usando os quartis superior e inferior. Isso mostra sobreposição entre as classes dentro da simulação; não comprova realismo.
+
+## O que é cada modelo e o que ele avalia
+
+**Os dois modelos avaliam a mesma pergunta e usam as mesmas entradas.** A diferença está na maneira de aprender a relação entre o contexto e o apoio futuro. Nenhum avalia a veracidade, a gravidade ou a resolução de uma ocorrência.
+
+| Modelo | Como funciona | Papel no projeto |
+|---|---|---|
+| **Regressão Logística** (`regressao_logistica`) | Aprende pesos para as entradas transformadas e converte a soma ponderada em uma probabilidade pela função logística. Apesar do nome, é usada aqui como classificador binário. | Candidato mais simples, selecionado pelo maior F1 macro na validação. |
+| **Random Forest** (`random_forest`) | Combina várias árvores de decisão treinadas com aleatoriedade de amostras e de atributos. Cada árvore aprende regras de divisão; as probabilidades das árvores são combinadas por média. Pode representar relações não lineares e interações. | Segundo candidato, disponível para comparação com a Regressão Logística. |
+| **DummyClassifier** (`referencia_dummy`) | Ignora as características e prevê a classe majoritária do treino. A probabilidade de apoio é sempre a proporção de classe 1 no treino, aproximadamente 42,57%. | Referência para saber se os candidatos superam uma regra simples. Não é exportado nem disponibilizado na API. |
+
+Os candidatos retornam a classe 0 ou 1 com limiar fixo de 0,5, sem ajuste desse limiar. A probabilidade se refere à classe 1, apoio em 72 horas. No empate exato de probabilidades, os classificadores utilizados escolhem a primeira classe, 0.
+
+## Como os modelos foram treinados
+
+### 1. Separação temporal
+
+Os registros foram separados pela data de abertura, preservando a ordem dos acontecimentos:
+
+| Conjunto | Período de abertura em 2026 | Linhas | Proporção com apoio | Uso |
+|---|---|---:|---:|---|
+| Treino | 01/01 a 14/05 | 2.215 | 42,57% | Aprender o pré-processamento e os classificadores. |
+| Validação | 18/05 a 24/06 | 598 | 43,31% | Escolher entre os dois candidatos. |
+| Teste | 28/06 a 31/07 | 555 | 44,14% | Avaliar o resultado final após a escolha. |
+
+As outras **90 linhas**, abertas nos intervalos de 15–17/05 e 25–27/06, continuam no CSV, mas ficam fora do ajuste e da pontuação. Os intervalos permitem que os rótulos de 72 horas do conjunto anterior já sejam conhecidos antes do próximo conjunto.
+
+Os históricos são atualizados sequencialmente com eventos passados já disponíveis, inclusive durante validação e teste. Isso simula o uso contínuo da aplicação, sem retreinar os modelos nem consultar resultados futuros para formar uma entrada.
+
+### 2. Pré-processamento dentro dos pipelines
+
+- `categoria`, `regiao`, `dia_semana` e `hora_abertura` passam por `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. Cada valor vira um indicador, sem presumir que a numeração de dia ou hora tenha efeito linear.
+- `moradores_ativos_7d`, `taxa_apoio_anterior_28d` e `ocorrencias_regiao_7d` passam por `StandardScaler`: subtrai a média e divide pelo desvio padrão calculados no treino. A floresta usa o mesmo transformador para manter a preparação consistente.
+- Não é necessária imputação, porque a auditoria confirma que não há valores ausentes.
+- `ColumnTransformer` e `Pipeline` mantêm a preparação e o classificador juntos. Todos os ajustes (`fit`) usam exclusivamente o treino; validação, teste e API apenas aplicam as transformações aprendidas.
+
+### 3. Configurações fixadas antes da avaliação
+
+| Modelo | Configuração explícita no notebook |
+|---|---|
+| Regressão Logística | `C=1.0`, `max_iter=1000`, `random_state=20261112`. |
+| Random Forest | `n_estimators=160`, `max_depth=7`, `min_samples_leaf=20`, `random_state=20261112`, `n_jobs=1`. |
+| Referência | `DummyClassifier(strategy="prior")`. |
+
+Na Regressão Logística, `C` controla inversamente a força da regularização e `max_iter` limita as iterações de otimização. Na floresta, 160 é o número de árvores; profundidade 7 e pelo menos 20 amostras por folha limitam a complexidade. Os demais parâmetros usam os padrões do scikit-learn 1.6.1.
+
+Não houve busca de hiperparâmetros, validação cruzada, ajuste do limiar ou balanceamento artificial das classes. O critério prévio foi **maior F1 macro na validação**; empate exato favoreceria a Regressão Logística pela simplicidade.
+
+### 4. Exportação e uso
+
+Os dois candidatos foram exportados **sem retreino após a seleção**, preservando os pipelines avaliados. O `joblib` também guarda entradas, classes, versões, semente, hash do CSV, métricas, períodos e mínimos/máximos numéricos do treino.
+
+A API carrega esse arquivo uma vez na inicialização. Abrir a interface ou pedir uma previsão não treina os modelos. Não é necessário executar o notebook nem ler o CSV para usar a demonstração.
+
+## Métricas de avaliação das previsões
+
+Para explicar as métricas da classe **com apoio**, considere:
+
+- **VP:** previu apoio e houve apoio.
+- **VN:** previu ausência e não houve apoio.
+- **FP:** previu apoio, mas não houve.
+- **FN:** previu ausência, mas houve apoio.
+
+| Métrica | Cálculo ou leitura | O que informa |
+|---|---|---|
+| **Acurácia** | `(VP + VN) / total`. Maior é melhor. | Proporção de previsões corretas. Sozinha pode favorecer a classe mais frequente. |
+| **Precisão** | `VP / (VP + FP)`. Maior é melhor. | Entre as previsões de apoio, quantas se confirmaram. |
+| **Recall / sensibilidade** | `VP / (VP + FN)`. Maior é melhor. | Entre as ocorrências que tiveram apoio, quantas foram identificadas. |
+| **F1 por classe** | `2 × precisão × recall / (precisão + recall)`. Maior é melhor. | Equilíbrio entre precisão e recall. |
+| **F1 macro** | Média simples de F1 da classe 0 e F1 da classe 1. Maior é melhor. | Dá o mesmo peso às duas classes; é o critério de seleção do projeto. |
+| **Suporte** | Número de exemplos reais de cada classe no conjunto avaliado. | Mostra quantos casos sustentam as métricas; não é uma medida de acerto. |
+| **Brier score** | Média de `(probabilidade_de_apoio − alvo)²`. Menor é melhor; 0 representa previsões probabilísticas perfeitas. | Mede o erro das probabilidades, penalizando previsões muito confiantes e erradas. Isoladamente não comprova calibração. |
+
+Precisão, recall e F1 também são apresentados para a classe 0, tratando ausência de apoio como a classe de interesse. Quando uma divisão é indefinida, o relatório do notebook usa `zero_division=0`. Ele também mostra médias ponderadas, que dão a cada classe um peso proporcional ao suporte.
+
+Outras ferramentas de avaliação e interpretação usadas:
+
+- **Matriz de confusão:** mostra os acertos e erros por classe; linhas são resultados observados e colunas são previsões.
+- **Curva de calibração:** compara a probabilidade média prevista com a frequência de apoio observada em seis grupos definidos por quantis. Ajuda a inspecionar a confiabilidade das probabilidades; não houve recalibração.
+- **Importância por permutação:** embaralha uma entrada na validação e mede a queda de F1 macro, em cinco repetições. A média indica quanto o modelo depende daquele sinal, e o desvio mostra a variação entre repetições. Valores próximos de zero ou negativos não indicam contribuição estável nesse procedimento. Não demonstra causalidade e não foi usada para ajustar o modelo.
+
+## Resultados obtidos
+
+Valores das saídas salvas no notebook, arredondados a quatro casas decimais:
+
+| Modelo | Acurácia na validação | F1 macro na validação | Brier na validação | Acurácia no teste | F1 macro no teste | Brier no teste |
+|---|---:|---:|---:|---:|---:|---:|
+| Referência Dummy | 0,5669 | 0,3618 | 0,2456 | 0,5586 | 0,3584 | 0,2468 |
+| Regressão Logística | 0,6171 | **0,5715** | 0,2350 | 0,6126 | 0,5879 | 0,2360 |
+| Random Forest | 0,5853 | 0,5398 | 0,2364 | **0,6360** | **0,5948** | **0,2327** |
+
+A **Regressão Logística foi selecionada pela validação**, com diferença de aproximadamente 0,0317 em F1 macro. No teste, a Random Forest teve F1 macro cerca de 0,0070 maior; isso não altera a escolha feita antes. Não foi realizado teste de significância, e esses resultados não demonstram superioridade geral de um algoritmo.
+
+Ambos superaram o Dummy em F1 macro no teste. Ainda assim, identificar apoio continua difícil: para a classe 1, a precisão/recall da Regressão Logística foram 0,5862/0,4163; os da Random Forest foram 0,6617/0,3592. A floresta acertou uma proporção maior das previsões de apoio, mas encontrou uma proporção menor dos apoios que de fato ocorreram.
+
+Para a Regressão Logística, a matriz no teste foi:
+
+| Resultado observado | Previu sem apoio (0) | Previu com apoio (1) |
+|---|---:|---:|
+| Sem apoio (0) | 238 | 72 |
+| Com apoio (1) | 143 | 102 |
+
+Um falso positivo pode omitir uma sugestão útil de divulgação; um falso negativo pode gerar uma sugestão desnecessária. Esses efeitos são interpretações qualitativas, sem custo social ou econômico medido.
+
+Na importância por permutação da Regressão Logística, moradores ativos, categoria e horário apresentaram as maiores quedas médias de F1 macro na validação: aproximadamente 0,0617, 0,0401 e 0,0350, respectivamente. Isso descreve o comportamento desse modelo na simulação.
+
+## Como executar a demonstração
+
+O ambiente documentado da entrega é **Python 3.13.9, Windows de 64 bits**. Com Python 3.13 instalado e o launcher `py` disponível, abra o PowerShell na pasta do projeto:
+
+```powershell
+py -3.13 --version
+py -3.13 -m pip install --only-binary=:all: -r requirements.txt
+py -3.13 -m pip check
+py -3.13 api_resolveai.py
+```
+
+Acesse **http://127.0.0.1:5000**. Mantenha o terminal aberto; `Ctrl+C` encerra o servidor. A interface permite escolher um modelo ou comparar os dois. A comparação faz duas previsões independentes com as mesmas entradas, sem média ou votação entre os modelos.
+
+Mantenha o `joblib`, `templates/` e `static/` junto da API. Ela confere as versões de NumPy, pandas, SciPy, scikit-learn e joblib registradas no treinamento; o scikit-learn exigido é **1.6.1**.
+
+### Rotas da API
+
+| Rota | Função |
+|---|---|
+| `GET /` | Abre a interface. |
+| `GET /health` | Informa a disponibilidade da API e dos dois modelos. |
+| `POST /predict` | Recebe o modelo e as sete características; retorna classe, probabilidade e orientação. |
+
+Exemplo de corpo JSON para `POST /predict`, com `Content-Type: application/json`:
+
+```json
+{
+  "modelo": "regressao_logistica",
+  "caracteristicas": {
+    "categoria": "vias",
+    "regiao": "norte",
+    "dia_semana": 2,
+    "hora_abertura": 14,
+    "moradores_ativos_7d": 40,
+    "taxa_apoio_anterior_28d": 0.5,
+    "ocorrencias_regiao_7d": 20
+  }
+}
+```
+
+Para consultar a floresta, use `"modelo": "random_forest"`. A resposta contém `previsao_apoio_72h`, `probabilidade_apoio_72h`, `orientacao` e a identificação de dados sintéticos.
+
+A API exige exatamente os campos previstos e rejeita entradas numéricas fora das faixas de treino: **14 a 129 moradores ativos**, **12 a 46 ocorrências recentes** e taxa histórica de **0.142857142857 a 0.75**. A interface mostra a taxa em porcentagem e a converte em proporção para a requisição. O máximo de aproximadamente 0,778 no resumo do CSV completo não amplia o limite de 0,75 obtido apenas no treino.
+
+Esses filtros são regras adicionais de admissibilidade, sem recalibração ou nova avaliação dos modelos. As métricas apresentadas correspondem ao teste original completo. Estar dentro dos limites individuais não garante que uma combinação tenha sido frequente no treino.
+
+### Como reproduzir o treinamento
+
+Após instalar as dependências, execute na pasta do projeto:
+
+```powershell
+py -3.13 -m ipykernel install --user --name resolveai-313 --display-name "ResolveAI Python 3.13"
+py -3.13 -m notebook ResolveAI_Machine_Learning.ipynb
+```
+
+Selecione o kernel **ResolveAI Python 3.13**, reinicie o kernel e execute todas as células em ordem. O notebook reproduz a geração, confere o CSV existente, treina, avalia e sobrescreve `modelos_resolveai.joblib`. Se o CSV não existir, ele é recriado. Reinicie a API após exportar novamente, pois os modelos são carregados na inicialização.
+
+## Limitações
+
+- Os parâmetros são inventados. As métricas medem desempenho neste mundo simulado e não comprovam qualidade com moradores reais.
+- Uma semente e uma separação temporal não demonstram estabilidade em outros períodos ou comunidades. As ocorrências compartilham sessões e históricos, portanto não são observações independentes.
+- Probabilidade não é garantia de apoio. A inspeção de calibração e o Brier não validam probabilidades para uso real.
+- Região e atividade podem reproduzir diferenças de participação digital. Não receber apoio não indica que um relato seja irrelevante ou falso.
+- A orientação de compartilhar não teve seu efeito testado causalmente. A demonstração não decide prioridade de serviço público nem confirma resolução.
+- O servidor Flask é local, demonstrativo e sem autenticação. Para uso real seriam necessários dados com governança, avaliação de vieses, novos períodos de teste e monitoramento.
+
+## Integrantes
+
+- Andrei Oliveira de Sousa — RM 573373.
+- Felipe Coelho da Fonseca — RM 572133.
+- Henrique Soares Pereira — RM 569898.
+- Luiz Felipe Lourenço da Silva — RM 572576.
+
+Este README descreve o código e as saídas salvas no notebook da entrega. Consulte o notebook para os cálculos e gráficos completos e `integrantes_e_explicacoes.txt` para o detalhamento operacional da API.
