# Sistema de Previsão de Risco de Defasagem Escolar
## Associação Passos Mágicos – Datathon

---

## Visão Geral do Projeto

A **Associação Passos Mágicos** atende centenas de crianças e jovens em vulnerabilidade socioeconômica, acompanhando seu desenvolvimento escolar por meio do programa **PEDE** (Programa de Educação e Desenvolvimento Escolar). A cada ano, a equipe pedagógica dispõe de uma Base de Dados com indicadores de desempenho, engajamento e contexto de cada aluno.

O objetivo deste projeto é **identificar preventivamente os alunos com maior risco de piorar sua defasagem escolar no ciclo seguinte**, permitindo que coordenadores e professores priorizem intervenções antes que o problema se agrave.

O sistema entrega uma **lista de alerta stratificada por Fase**, onde a equipe pedagógica pode visualizar quais alunos merecem atenção prioritária, quais são os principais fatores de risco de cada um, e como o modelo está se comportando ao longo do tempo.

**Definição formal do alvo:**
> `target = 1` se `defasagem_t1 > defasagem_t` — ou seja, o aluno piorou de nível de defasagem no ano seguinte.

---

## Solução Proposta

### Formulação do Problema

O problema é tratado como **classificação binária supervisionada com split temporal**:

- **Instância**: par `(aluno, ano)` em que o mesmo RA aparece em dois anos consecutivos.
- **Features**: indicadores do ano corrente `T` (notas, engajamento, contexto, fase).
- **Alvo**: ocorrência de piora de defasagem em `T+1`.
- **Inferência**: dado um conjunto de alunos no ano atual, o modelo estima a probabilidade de piora e gera uma lista de alerta Top-K% por Fase.

### Por que CatBoost?

- Lida nativamente com variáveis categóricas (fase, turma, gênero, instituição) sem necessidade de encoding manual.
- Robusto a dados tabulares de médio porte com desequilíbrio de classes.
- Suporte a `auto_class_weights='Balanced'`, essencial pois menos de 40% dos alunos pioram a cada ciclo.
- Alta interpretabilidade via SHAP values, gerando os **top-3 fatores de risco** por aluno.

### Estratégia de Alerta: Top-K% Estratificado

Em vez de um único threshold global, a política usa um **percentual K por Fase**. Isso garante equidade: fases menores não são ignoradas por terem poucos alunos no Top-K global.

O coordenador escolhe `k_pct ∈ {10, 15, 20, 25}` conforme a capacidade de atendimento.

---

## Regras de Negócio Vigentes

- O modelo prevê **risco de piora de defasagem** (T -> T+1), não diagnóstico atual.
- O target usa alunos pareados entre anos adjacentes e compara estado atual vs próximo ano.
- A política operacional de alerta continua sendo preventiva (Top-K por fase).
- Política de alerta é **Top-K% estratificado por fase**.
- Inglês só é considerado quando obrigatório por fase.
- Nas fases iniciais/equivalentes (`ALFA/F0`, `F1`, `F2`, `F8`), inglês não influencia o risco.

---

## Stack Tecnológica

| Camada | Tecnologia | Uso |
|--------|-----------|-----|
| Linguagem | Python 3.12 | Toda a implementação |
| ML | CatBoost 1.2 | Classificador de risco |
| Explicabilidade | SHAP | Top-3 fatores de risco por aluno |
| API | FastAPI + Uvicorn | Endpoints REST de predição e monitoramento |
| Dashboard | Plotly Dash | Visualização pedagógica interativa |
| Dados | Pandas + OpenPyXL | Processamento do PEDE (.xlsx) |
| Estatística | NumPy + SciPy | PSI, cálculos de drift |
| Testes | Pytest + Coverage | Cobertura ≥ 80% (atual: ~96%) |
| Qualidade | Ruff | Lint e formatação |
| Containerização | Docker | Deploy reproduzível |
| Build | Makefile | Automação de tarefas |

---

## Estrutura do Projeto

```
datathon/
│
├── src/                          # Pacote principal (lógica de negócio)
│   ├── train.py                  # Script de treinamento (pipeline completo)
│   ├── preprocessing.py          # Carregamento e padronização dos dados PEDE
│   ├── feature_engineering.py    # Engenharia de features com anti-leakage
│   ├── model_training.py         # CatBoost: treino, carga, predict_proba
│   ├── evaluation.py             # AUC, Recall@TopK, Precision@TopK, Lift@TopK
│   ├── inference.py              # Scoring em lote + SHAP top-3 fatores
│   └── utils.py                  # Logging, I/O, monitoramento de drift (PSI)
│
├── api/                          # Serviço REST (FastAPI)
│   ├── main.py                   # Inicialização da aplicação e startup
│   ├── routes.py                 # Handlers de todos os endpoints
│   └── schemas.py                # Modelos Pydantic de request/response
│
├── dashboard/
│   └── dashapp.py                # Dashboard Dash (visualização pedagógica)
│
├── models/                       # Artefatos gerados pelo treinamento
│   ├── model/
│   │   ├── catboost_model.cbm    # Modelo serializado
│   │   ├── lookup_tables.pkl     # Estatísticas de grupo para inferência
│   │   ├── model_meta.json       # Metadados (features, best iteration)
│   │   └── cohort_summary.json   # Resumo da amostragem
│   ├── evaluation/
│   │   ├── evaluation_results.json  # Métricas de validação
│   │   ├── valid_scored.csv         # Scores do conjunto de validação
│   │   └── scored_history.csv       # Histórico multi-ano para o dashboard
│   └── monitoring/
│       ├── drift_history.jsonl   # Log de eventos PSI (append-only)
│       └── retrain_metadata.json # Metadados do último retreinamento
│
├── tests/                        # Testes unitários
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   ├── test_inference.py
│   ├── test_evaluation.py
│   ├── test_utils.py
│   ├── test_train_script.py
│   └── test_api.py
│
├── notebooks/                    # Material didático (Colab)
├── data/                         # Dados de entrada e uploads temporários
├── docs/                         # Documentação técnica complementar
├── Dockerfile                    # Imagem de produção
├── Makefile                      # Automação de tarefas
├── requirements.txt              # Dependências Python
└── pytest.ini                    # Configuração dos testes
```

---

## Regras Anti-Data Leakage

Variáveis **excluídas** do modelo por serem sintéticas ou diagnósticas:

| Variável | Motivo |
|----------|--------|
| IAN | Diagnóstico institucional |
| IDA, INDE | Indicadores compostos |
| Pedra (Quartzo, Ágata, etc.) | Síntese do INDE |
| IPV, Atingiu PV | Derivado da decisão |
| Fase ideal | Resultado do diagnóstico |
| Cg, Cf, Ct | Rankings do INDE |
| Rec Av, Rec Psicologia | Recomendações posteriores |

---

## Features Utilizadas

### Acadêmicas
- `matem`, `portug`, `ingles`
- `media_provas`, `disp_provas`, `fez_ingles`

Regra de negócio aplicada:
- Inglês só é considerado quando obrigatório por fase.
- Nas fases equivalentes a `ALFA/F0`, `F1`, `F2` e `F8`, inglês não entra no cálculo de risco.

### Engajamento
- `ieg` (Indicador de Engajamento)

### Psicossocial
- `iaa`, `iaa_participou` (Autoavaliação)
- `ips`, `ipp` (Vulnerabilidade psicossocial)

### Contexto
- `fase`, `turma`, `instituicao`, `genero`
- `ano_ingresso`, `tempo_casa`

### Estatísticas de Grupo (por Turma e Fase)
- `turma_mean_X`, `turma_std_X`, `turma_p25_X`, `turma_p75_X`
- `delta_turma_X`, `z_turma_X`
- `fase_mean_X`, `fase_std_X`
- `abaixo_p25_turma_*` (flags binários)

---

## Split Temporal (sem shuffle)

| Split | Features | Target |
|-------|----------|--------|
| **Treino** | Pares temporais anteriores (ex.: 2022→2023) | Piora em T+1 (`defasagem_t1 > defasagem_t`) |
| **Validação** | Par temporal mais recente (ex.: 2023→2024) | Piora em T+1 (`defasagem_t1 > defasagem_t`) |

Regra central do target:
- Entram no dataset alunos pareados (`RA`) entre anos adjacentes.
- O rótulo é calculado por comparação de estado entre `T` e `T+1`.

---

## Modelo

**CatBoostClassifier** com:
- `iterations=2500`, `learning_rate=0.03`, `depth=8`
- `auto_class_weights='Balanced'` (desbalanceamento de classes)
- `early_stopping_rounds=200`
- `bootstrap_type='Bernoulli'`, `subsample=0.8`, `rsm=0.8`

---

## Política de Alerta

**Top-K% estratificado por Fase** – o coordenador escolhe K ∈ {10%, 15%, 20%, 25%}.

Para cada Fase, seleciona os K% com maior score. Isso garante equidade entre fases de diferentes tamanhos.

---

## Instruções de Deploy

### Pré-requisitos

- Python 3.12+
- Arquivo PEDE em `.xlsx` com abas `PEDE2022`, `PEDE2023`, `PEDE2024` (ou mais anos)

---

### 1. Ambiente Local (Linux / macOS)

```bash
# 1. Clone ou descompacte o projeto
cd datathon

# 2. Crie e ative o ambiente virtual
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Treine o modelo (gera artefatos em models/)
make pipeline
# ou equivalente:
python -m src.train --xls "data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" --model-dir models

# 5. Inicie a API REST
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. (Opcional) Inicie o dashboard em outro terminal
source .venv/bin/activate
python dashboard/dashapp.py --host 0.0.0.0 --port 8502
```

Acesse:
- API + docs interativas: `http://localhost:8000/docs`
- Dashboard pedagógico: `http://localhost:8502`

---

### 2. Ambiente Local (Windows / PowerShell)

```powershell
# 1. Crie e ative o ambiente virtual
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Treine o modelo
.\.venv\Scripts\python -m src.train --xls "data\BASE DE DADOS PEDE 2024 - DATATHON.xlsx" --model-dir models

# 4. Inicie a API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 5. (Opcional) Dashboard
.\.venv\Scripts\python dashboard\dashapp.py --host 0.0.0.0 --port 8502
```

---

### 3. Docker

```bash
# Build da imagem
docker build -t passos-magicos-risk .

# Execute montando o arquivo de dados
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  passos-magicos-risk
```

O container treina o modelo automaticamente e em seguida sobe a API na porta `8000`.

---

### 4. Makefile (atalhos úteis)

```bash
make pipeline          # Executa o treinamento completo
make test              # Roda todos os testes
make test-cov          # Testes + relatório de cobertura (mínimo 80%)
make test-cov-html     # Gera relatório HTML em htmlcov/index.html
make lint              # Verifica estilo com Ruff
make format            # Formata código com Ruff
make clean             # Remove caches e artefatos temporários
```

---

## Etapas do Pipeline de Machine Learning

O script `src/train.py` executa as etapas abaixo em sequência. Cada etapa é implementada em um módulo independente do pacote `src/`.

### 1. Carregamento e Padronização (`src/preprocessing.py`)

- Lê o arquivo `.xlsx` com uma aba por ano (`PEDE2022`, `PEDE2023`, `PEDE2024`, …).
- Normaliza nomes de colunas (remove acentos, espaços, letras maiúsculas, underscores extras).
- Aplica renomeações canônicas para garantir consistência entre anos (ex.: `nota_mat` → `matem`).
- Padroniza tipos: converte notas para `float`, fase para `int`, texto para `str`.
- Remove duplicatas de RA dentro de cada ano.
- Constrói o dataset longitudinal: pareia alunos entre anos consecutivos (`RA` presente em `T` e `T+1`) e calcula o target `piora_defasagem`.

### 2. Engenharia de Features (`src/feature_engineering.py`)

Anti-leakage é garantido: **nenhuma variável derivada do diagnóstico institucional** (IAN, IDA, INDE, Pedra, IPV, Fase Ideal) é usada como feature.

Features criadas:
- **Nota de inglês ajustada**: zerificada nas fases onde inglês não é obrigatório (`ALFA/F0`, `F1`, `F2`, `F8`), evitando viés de fase.
- **Média e dispersão de provas**: `media_provas`, `disp_provas`.
- **Flag de participação**: `fez_ingles`, `iaa_participou`.
- **Tempo de casa**: `tempo_casa = ano_base - ano_ingresso`.
- **Estatísticas de grupo por Turma**: média, desvio-padrão, P25, P75 de `matem`, `portug`, `ieg` dentro de cada turma.
- **Estatísticas de grupo por Fase**: análogos ao nível de fase.
- **Deltas e Z-scores**: `delta_turma_matem = matem - turma_mean_matem`, `z_turma_matem`, etc.
- **Flags de quartil inferior**: `abaixo_p25_turma_matem`, etc.

Em inferência (`is_train=False`), as estatísticas de grupo são recuperadas das `lookup_tables` salvas no treinamento, evitando data leakage.

### 3. Treinamento com Split Temporal (`src/model_training.py`)

- O split é **temporal e sem shuffle**: os pares mais antigos formam o treino, o par mais recente forma a validação.
- Ex.: anos 2022, 2023, 2024 → treino em pares `2022→2023`, validação em pares `2023→2024`.
- **CatBoostClassifier** com `auto_class_weights='Balanced'` e `early_stopping_rounds=200`.
- Variáveis categóricas (`fase`, `turma`, `genero`, `instituicao`) passadas diretamente ao CatBoost.
- Artefatos salvos: `catboost_model.cbm`, `lookup_tables.pkl`, `model_meta.json`.

### 4. Avaliação (`src/evaluation.py`)

Métricas calculadas no conjunto de validação:

| Métrica | Descrição |
|---------|-----------|
| **AUC** | Área sob a curva ROC |
| **Recall@TopK** | Fração de alunos realmente em risco capturados na lista (métrica principal) |
| **Precision@TopK** | Precisão dentro da lista de alerta |
| **Lift@TopK** | Ganho vs. seleção aleatória |

Resultados salvos em `models/evaluation/evaluation_results.json`.

### 5. Geração de Scores e Explicações (`src/inference.py`)

- `score_students()`: aplica o modelo treinado e retorna `score ∈ [0, 100]`.
- `alert_list()`: aplica a política Top-K% estratificada por Fase.
- `_add_shap_explanations()`: calcula SHAP values (quando disponível) e extrai os **top-3 fatores de risco** por aluno, com seus valores numéricos.
- Scores e SHAP salvos em `valid_scored.csv` e `scored_history.csv`.

### 6. Monitoramento de Drift (`src/utils.py`)

- Calcula o **PSI (Population Stability Index)** entre a distribuição de scores da validação (baseline) e de lotes novos enviados via API.
- Classifica a severidade: `ok` (PSI < 0.1), `warning` (0.1–0.2), `critical` (> 0.2).
- Eventos de drift são gravados em `models/monitoring/drift_history.jsonl` a cada chamada ao `/predict`.

---

## Treinamento

```bash
python -m src.train \
  --xls "data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" \
  --model-dir models/ \
  --log-level INFO
```

Ou via Makefile:

```bash
make pipeline
# Para especificar outro arquivo:
make pipeline XLS="data/outro_arquivo.xlsx" MODEL_DIR="models/v2"
```

Saídas em `models/`:
- `catboost_model.cbm` – modelo CatBoost
- `lookup_tables.pkl` – estatísticas de grupo para inferência
- `model_meta.json` – metadados (features, best iteration)
- `evaluation_results.json` – métricas de validação
- `valid_scored.csv` – scores do conjunto de validação
- `scored_history.csv` – histórico de scores multi-ano para o dashboard
- `cohort_summary.json` – resumo de amostragem (bruto, elegível, pareado, usado)

---

## API REST

Após o treinamento, inicie a API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Documentação interativa (Swagger UI): `http://localhost:8000/docs`

### Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Status da API e modelos carregados |
| POST | `/predict` | Scoring de lote de alunos em tempo real |
| GET | `/alert?k_pct=15` | Lista de alerta Top-K% do conjunto de validação |
| GET | `/explain/{ra}` | Explicação SHAP de um aluno pelo RA |
| GET | `/metrics/drift` | Relatório PSI de drift atual vs. baseline |
| GET | `/metrics/drift/history` | Histórico de eventos de monitoramento |

---

### `GET /health`

Verifica se a API está operacional e se o modelo foi carregado.

```bash
curl http://localhost:8000/health
```

**Resposta esperada:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "lookup_tables_loaded": true
}
```

---

### `POST /predict`

Recebe um lote de alunos e retorna scores de risco, flags de alerta e top-3 fatores de risco por aluno.

**Input:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {
        "ra": "RA-001",
        "turma": "FASE3-A",
        "fase": "FASE3",
        "instituicao": "PUBLICA",
        "genero": "F",
        "ano_ingresso": 2020,
        "ieg": 6.5,
        "iaa": 7.0,
        "ips": 5.5,
        "ipp": 6.0,
        "matem": 5.0,
        "portug": 6.0,
        "ingles": 0.0,
        "ano_base": 2024
      },
      {
        "ra": "RA-002",
        "turma": "FASE5-B",
        "fase": "FASE5",
        "instituicao": "PUBLICA",
        "genero": "M",
        "ano_ingresso": 2019,
        "ieg": 4.2,
        "iaa": 5.0,
        "ips": 7.8,
        "ipp": 7.0,
        "matem": 3.5,
        "portug": 4.0,
        "ingles": 3.0,
        "ano_base": 2024
      }
    ],
    "k_pct": 15
  }'
```

**Resposta esperada:**
```json
{
  "n_students": 2,
  "n_alerta": 1,
  "k_pct": 15.0,
  "students": [
    {
      "ra": "RA-001",
      "score": 32.7,
      "fase": "FASE3",
      "turma": "FASE3-A",
      "alerta": false,
      "top3_factors": ["ieg", "matem", "ips"],
      "top3_values": [6.5, 5.0, 5.5]
    },
    {
      "ra": "RA-002",
      "score": 74.1,
      "fase": "FASE5",
      "turma": "FASE5-B",
      "alerta": true,
      "top3_factors": ["matem", "ips", "ieg"],
      "top3_values": [3.5, 7.8, 4.2]
    }
  ]
}
```

Campos da resposta:
- `score`: probabilidade de piora de defasagem em [0, 100].
- `alerta`: `true` se o aluno está no Top-K% da sua Fase.
- `top3_factors`: nomes dos 3 indicadores com maior impacto SHAP.
- `top3_values`: valores originais desses indicadores para o aluno.

**Exemplo com Python (requests):**

```python
import requests

payload = {
    "students": [{
        "ra": "RA-042",
        "turma": "FASE4-C",
        "fase": "FASE4",
        "instituicao": "PUBLICA",
        "genero": "M",
        "ano_ingresso": 2021,
        "ieg": 5.0,
        "iaa": 6.0,
        "ips": 6.5,
        "ipp": 5.5,
        "matem": 4.5,
        "portug": 5.0,
        "ingles": 4.0,
        "ano_base": 2024
    }],
    "k_pct": 15
}

resp = requests.post("http://localhost:8000/predict", json=payload)
data = resp.json()

for aluno in data["students"]:
    print(f"RA {aluno['ra']} | Score: {aluno['score']:.1f} | Alerta: {aluno['alerta']}")
    print(f"  Fatores: {list(zip(aluno['top3_factors'], aluno['top3_values']))}")
```

---

### `GET /alert?k_pct=15`

Retorna a lista de alerta Top-K% estratificada por Fase a partir do conjunto de validação (`valid_scored.csv`). Útil para demonstração sem envio de novos dados.

```bash
curl "http://localhost:8000/alert?k_pct=15"
```

**Resposta esperada:**
```json
{
  "k_pct": 15.0,
  "n_alerta": 47,
  "n_total": 312,
  "students": [
    { "ra": "RA-128", "fase": "FASE3", "turma": "FASE3-B", "score": 88.4 },
    { "ra": "RA-207", "fase": "FASE4", "turma": "FASE4-A", "score": 82.1 },
    ...
  ]
}
```

O parâmetro `k_pct` aceita valores de `5.0` a `50.0` (padrão: `15.0`).

---

### `GET /explain/{ra}`

Retorna a explicação SHAP de um aluno específico pelo seu RA, buscando no conjunto de validação.

```bash
curl "http://localhost:8000/explain/RA-128"
```

**Resposta esperada:**
```json
{
  "ra": "RA-128",
  "score": 88.4,
  "fase": "FASE3",
  "top3_factors": ["matem", "ieg", "ips"],
  "top3_values": [2.5, 4.0, 8.1],
  "shap_values": {
    "matem": 0.312,
    "ieg": 0.198,
    "ips": 0.175,
    "portug": 0.091
  }
}
```

Se o RA não for encontrado, retorna HTTP 404:
```json
{ "detail": "RA RA-999 not found in validation set." }
```

---

### `GET /metrics/drift`

Calcula o PSI (Population Stability Index) comparando os scores atuais do conjunto de validação com o baseline estabelecido no treinamento. Também retorna a análise de drift por Fase.

```bash
curl "http://localhost:8000/metrics/drift"
```

**Resposta esperada:**
```json
{
  "psi": 0.012,
  "severity": "ok",
  "n_baseline": 312,
  "n_current": 312,
  "phase_drift": [
    { "fase": "FASE5", "psi": 0.031, "severity": "ok", "n_current": 48, "n_baseline": 48 },
    { "fase": "FASE3", "psi": 0.009, "severity": "ok", "n_current": 76, "n_baseline": 76 }
  ],
  "history_file": "models/monitoring/drift_history.jsonl"
}
```

Interpretação:
- `psi < 0.1` → `"ok"` (distribuição estável)
- `0.1 ≤ psi < 0.2` → `"warning"` (avaliar necessidade de retreinamento)
- `psi ≥ 0.2` → `"critical"` (retreinamento recomendado)

---

### `GET /metrics/drift/history?limit=10`

Retorna os últimos N eventos de monitoramento gravados no log JSONL.

```bash
curl "http://localhost:8000/metrics/drift/history?limit=3"
```

**Resposta esperada:**
```json
{
  "n_events": 15,
  "events": [
    {
      "timestamp": "2024-11-02T14:32:11",
      "event_type": "predict_batch",
      "psi": 0.027,
      "severity": "ok",
      "n_students": 2,
      "k_pct": 15,
      "n_alerta": 1
    },
    {
      "timestamp": "2024-11-01T09:10:44",
      "event_type": "predict_batch",
      "psi": 0.018,
      "severity": "ok",
      "n_students": 150,
      "k_pct": 20,
      "n_alerta": 30
    },
    {
      "timestamp": "2024-10-31T08:00:00",
      "event_type": "baseline_snapshot",
      "psi": 0.0,
      "severity": "ok",
      "n_students": 312,
      "k_pct": null,
      "n_alerta": null
    }
  ]
}
```

O parâmetro `limit` aceita valores de `1` a `1000` (padrão: `100`).

---

## Dashboard

```powershell
.\start-dashboard.ps1
```

API no PowerShell:

```powershell
.\start-api.ps1
```

API + dashboard no PowerShell:

```powershell
.\start-all.ps1
```

Porta alternativa:

```powershell
.\start-dashboard.ps1 -Port 8503
```

Features:
- Filtros globais: **Ano-base**, **Fase**, **Turma**, **RA** e **Top-K% por fase**
- Aba **Início** com KPIs (total, alertas, % alerta, AUC) e visão de alertas por fase
- Aba **Alertas** com tabela operacional (RA, Fase, Turma, Score, Motivos, link de detalhe)
- Aba **Distribuição por Fase** com `% em alerta` e `score médio`
- Aba **Saúde do Modelo** com métricas Top-K e histórico de drift (quando disponível)
- Rota de detalhe por aluno (`/aluno/<ra>`) com histórico de score e indicadores
- Aba **Dados e Retreinamento** com upload `.xlsx` e execução do `train.py`

### Atualização automática com novo ano (ex.: 2025)

1. Abra o dashboard e use o bloco **"Atualização de dados (XLS)"** na barra lateral.
2. Carregue um arquivo `.xlsx` contendo abas `PEDE2022`, `PEDE2023`, `PEDE2024` e `PEDE2025`.
3. Clique em **"Executar retreinamento"** (chama `python -m src.train` internamente).
4. Após sucesso, o dashboard recarrega os artefatos em `models/`.

Arquivo de exemplo sintético para teste:
- `data/BASE DE DADOS PEDE 2025 - SINTETICA.xlsx`

---

## Notebooks (Colab)

A pasta `notebooks/` contém material didático pronto para Google Colab e agora funciona como trilha **independente**:

- `01_setup_colab_e_dados.ipynb` – monta Drive, cria pasta independente e instala dependências.
- `02_treinamento_e_avaliacao.ipynb` – executa treino temporal sem `src/`/`train.py`, salvando artefatos em `models/`.
- `03_api_dashboard_e_monitoramento.ipynb` – sobe API local mínima no próprio notebook e mostra drift com visualização inline.

Os notebooks não exigem cópia da estrutura de código do projeto para o Drive.
Quando não há arquivo de dados, o notebook de treinamento gera dataset sintético (`PEDE2022`..`PEDE2025`) automaticamente.

---

## Arquivos auxiliares (não usados no fluxo padrão)

Para evitar dúvidas:

- A pasta `notebooks/` é material didático para Colab e **não é utilizada** pela execução normal do projeto (`train.py`, API e dashboard local).
- Arquivos `data/pede_upload_*.xlsx` são uploads temporários gerados no dashboard para retreino e **não fazem parte** do pipeline base.
- O arquivo `data/BASE DE DADOS PEDE 2025 - SINTETICA.xlsx` é somente para teste/demonstração e **não é obrigatório** para uso em produção.

Fluxo padrão do projeto usa diretamente:
- `src/train.py` (via `python -m src.train` ou `make pipeline`)
- `api/main.py`
- `dashboard/dashapp.py`
- artefatos em `models/`

---

## Testes

```bash
# Roda todos os testes (silencioso)
make test

# Testes com saída detalhada
make test-v

# Testes + cobertura no terminal (mínimo 80%)
make test-cov

# Testes + relatório HTML em htmlcov/index.html
make test-cov-html

# Equivalente direto (sem Makefile)
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80 -q
```

**Cobertura atual: ~96%**

---

## Docker

```bash
# Build da imagem
docker build -t passos-magicos-risk .

# Execute montando o diretório de dados
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  passos-magicos-risk

# Com variáveis de ambiente (opcional)
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  passos-magicos-risk
```

O container executa `python -m src.train` e depois sobe `uvicorn` automaticamente.

---

## Métricas Oficiais

Avaliadas na validação (2023→2024):

| Métrica | Descrição |
|---------|-----------|
| AUC | Área sob a curva ROC |
| Recall@TopK | Fração de em-risco capturados (métrica principal) |
| Precision@TopK | Precisão na lista de alerta |
| Lift@TopK | Ganho vs. seleção aleatória |

---

## Justificativa Técnica

Este sistema:
- ✅ Não replica o diagnóstico IAN existente
- ✅ Não usa indicadores sintéticos (IDA, IPV, INDE, Pedra)
- ✅ Trabalha com sinais brutos (provas, engajamento, psicossocial)
- ✅ Respeita a lógica de progressão institucional
- ✅ É **preditivo** (T+1), não diagnóstico (T)
- ✅ Equidade na política de alerta (estratificado por Fase)

### Por que não usar todos os RAs comuns?

Para o objetivo de **piora de defasagem**, usar dados sem pareamento temporal distorce o rótulo.

A escolha atual (pares `T->T+1`, com `target = 1` quando `defasagem_t1 > defasagem_t`) melhora:
- consistência temporal do alvo,
- rastreabilidade da evolução do aluno,
- interpretabilidade das métricas e da lista de alerta.

---

## FAQ (Coordenação Pedagógica)

**1) Por que o total de alunos no dashboard pode ser menor que o total da planilha?**  
Porque o treino/validação usa pares entre anos adjacentes; alunos sem pareamento (`RA`) no ano seguinte não entram no dataset de modelagem.

**2) "RA comuns" é igual a "RA usados no modelo"?**  
Nem sempre. "RA comuns" é interseção bruta; o pipeline ainda aplica padronizações, deduplicação e regras de consistência.

**3) Qual tela mostra risco futuro?**  
Principalmente `Lista de Alertas` e `Análise Individual`.

**4) Inglês entra para todas as fases?**  
Não. Em fases equivalentes a `0`, `1`, `2` e `8`, inglês não é obrigatório e não influencia o risco.

**5) Como atualizar com dados de 2025?**  
No dashboard: upload do XLS na barra lateral e clique em `Executar retreinamento`.
