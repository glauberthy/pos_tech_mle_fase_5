# Sistema de Previsão de Risco de Defasagem Escolar
## Associação Passos Mágicos – Datathon

---

## Visão Geral

Sistema preditivo que identifica **alunos em risco de piora de defasagem escolar** no próximo ciclo, permitindo intervenções pedagógicas preventivas.

**Alvo (estratégia atual):**
`target = 1` se houve **piora de defasagem** no ano seguinte (`defasagem_t1 > defasagem_t`).

---

## Regras de Negócio Vigentes

- O modelo prevê **risco de piora de defasagem** (T -> T+1), não diagnóstico atual.
- O target usa alunos pareados entre anos adjacentes e compara estado atual vs próximo ano.
- A política operacional de alerta continua sendo preventiva (Top-K por fase).
- Política de alerta é **Top-K% estratificado por fase**.
- Inglês só é considerado quando obrigatório por fase.
- Nas fases iniciais/equivalentes (`ALFA/F0`, `F1`, `F2`, `F8`), inglês não influencia o risco.

---

## Arquitetura

```
datathon/
├── src/
│   ├── preprocessing.py       # Carregamento e padronização dos dados PEDE
│   ├── feature_engineering.py # Engenharia de features com anti-leakage
│   ├── model_training.py      # CatBoost + split temporal
│   ├── evaluation.py          # AUC, Recall@TopK, Precision@TopK, Lift@TopK
│   ├── inference.py           # Scoring + SHAP top-3 fatores de risco
│   └── utils.py               # Logging, persistência, monitoramento de drift
├── api/
│   └── main.py                # FastAPI REST API
├── dashboard/
│   └── dashapp.py             # Dash dashboard
├── tests/                     # Testes unitários (≥80% cobertura)
├── train.py                   # Script principal de treinamento
├── start-api.ps1              # Atalho PowerShell para subir API
├── start-dashboard.ps1        # Atalho PowerShell para subir dashboard
├── start-all.ps1              # Sobe API + dashboard juntos (PowerShell)
├── Dockerfile
└── requirements.txt
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

## Instalação

```bash
# Clone / descompacte o projeto
cd datathon

# Instalar dependências
pip install -r requirements.txt

# Treinar modelo
python train.py

# Iniciar API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Iniciar dashboard
python dashboard/dashapp.py --host 0.0.0.0 --port 8502
```

### Execução recomendada no Windows (PowerShell)

```powershell
cd c:\datathon

# Criar ambiente virtual (uma vez)
py -3.12 -m venv .venv

# Instalar dependências no ambiente virtual
.\.venv\Scripts\python -m pip install -r requirements.txt

# Treinar modelo
.\.venv\Scripts\python train.py

# Iniciar API (script recomendado)
.\start-api.ps1

# Iniciar dashboard
.\start-dashboard.ps1

# Iniciar API + dashboard juntos
.\start-all.ps1
```

---

## Treinamento

```bash
python train.py \
  --xls "BASE DE DADOS PEDE 2024 - DATATHON.xlsx" \
  --model-dir models/ \
  --log-level INFO
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

Após `python train.py`, iniciar com:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Documentação interativa: `http://localhost:8000/docs`

### Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Status da API |
| POST | `/predict` | Scoring de lote de alunos |
| GET | `/alert?k_pct=15` | Lista de alerta Top-K% |
| GET | `/explain/{ra}` | Explicação SHAP do aluno |
| GET | `/metrics/drift` | Relatório PSI de drift |
| GET | `/metrics/drift/history` | Histórico de monitoramento de drift |

### Exemplo `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "students": [{
      "ra": "RA-001",
      "turma": "FASE2-A",
      "fase": "FASE2",
      "instituicao": "PUBLICA",
      "genero": "F",
      "ano_ingresso": 2020,
      "ieg": 6.5,
      "iaa": 7.0,
      "ips": 5.5,
      "matem": 5.0,
      "portug": 6.0,
      "ano_base": 2024
    }],
    "k_pct": 15
  }'
```

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
3. Clique em **"Executar retreinamento"**.
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
- `train.py`
- `api/main.py`
- `dashboard/dashapp.py`
- artefatos em `models/`

---

## Testes

```bash
# Rodar todos os testes
pytest tests/ -v

# Com relatório de cobertura
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Docker

```bash
# Build
docker build -t passos-magicos-risk .

# Run (montar o XLS)
docker run -p 8000:8000 \
  -v "$(pwd)/BASE DE DADOS PEDE 2024 - DATATHON.xlsx:/app/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" \
  passos-magicos-risk
```

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
