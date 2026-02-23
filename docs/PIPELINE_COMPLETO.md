# Documento Completo do Pipeline
## Sistema de Previsão de Risco de Defasagem Escolar
### Associação Passos Mágicos – Datathon

---

## Regras de Negócio Vigentes

- O problema é preditivo: estimar **piora de defasagem** no ciclo seguinte.
- O target é calculado por comparação temporal do mesmo aluno:
  - `target = 1` quando `defasagem_t1 > defasagem_t`
  - `target = 0` caso contrário
- O split é temporal dinâmico: treino com pares anteriores e validação no par mais recente, sem shuffle.
- A lista operacional usa **Top-K% estratificado por fase**.
- Inglês só entra quando obrigatório; nas fases equivalentes a `0`, `1`, `2` e `8` não contribui para risco.

---

## 1) Objetivo do sistema

O pipeline foi desenhado para prever, de forma **preditiva** (e não diagnóstica), quais alunos têm maior chance de piorar a defasagem no ciclo seguinte.

Definição formal do alvo:

- `target = 1` quando há piora (`defasagem_t1 > defasagem_t`).
- `target = 0` quando mantém ou melhora (`defasagem_t1 <= defasagem_t`).

Essa definição foca em **prevenção e progressão de risco**: identificar evolução negativa antes de consolidação de casos críticos.

---

## 2) Visão geral do pipeline ponta a ponta

O fluxo completo executado em `train.py` é:

1. Carregar e padronizar as abas `PEDE####` disponíveis (ex.: `PEDE2022`..`PEDE2025`).
2. Construir dataset longitudinal com split temporal dinâmico:
   - Treino: concatenação dos pares anteriores (ex.: `2022->2023`)
   - Validação: par mais recente (ex.: `2023->2024`)
3. Fazer engenharia de features com regras anti-leakage.
4. Treinar `CatBoostClassifier` (sem shuffle, com early stopping).
5. Avaliar em validação com `AUC`, `Recall@TopK`, `Precision@TopK`, `Lift@TopK`.
6. Salvar artefatos:
   - `models/catboost_model.cbm`
   - `models/model_meta.json`
   - `models/lookup_tables.pkl`
   - `models/evaluation_results.json`
   - `models/evaluation_report.txt`
   - `models/valid_scored.csv`
   - `models/scored_history.csv`
   - `models/cohort_summary.json`
7. Consumir esses artefatos via API (`api/main.py`) e Dashboard (`dashboard/dashapp.py`).

---

## 3) Decisões estratégicas e justificativas

## 3.1 Definição de problema e target

**Decisão:** Modelar risco de **piora** de defasagem (evento futuro), e não classificação estática de defasagem atual.  
**Por quê:** Isso suporta ação pedagógica antecipada e reduz abordagem reativa.

**Decisão (enfatizada):** Usar pares `T->T+1` por RA e comparar estado atual vs próximo ano.  
**Por quê:** Interseção sem alinhamento temporal/consistência de colunas reduz qualidade do rótulo e pode enviesar a leitura da evolução.

---

## 3.2 Split temporal em vez de split aleatório

**Decisão:** Treino com pares temporais anteriores e validação no par mais recente, sem embaralhar.  
**Por quê:** Simula produção real (prever o próximo ano com dados do ano corrente), reduzindo otimismo artificial do split aleatório e permitindo evolução anual (incluindo novos anos como 2025).

**Decisão:** Manter separação rígida entre períodos no treinamento e validação.  
**Por quê:** Diminui risco de vazamento de padrões do futuro.

---

## 3.3 Política anti-data leakage

**Decisão:** Remover variáveis sintéticas/diagnósticas e proxies diretos de decisão institucional (ex.: `IAN`, `IDA`, `INDE`, `Pedra`, `IPV`, `Fase ideal`, `Rec Av*`, etc.).  
**Por quê:** Essas variáveis carregam informação já consolidada do resultado ou decisão institucional, gerando atalho indevido no modelo e baixa generalização.

**Decisão:** Engenharia de estatísticas de grupo (`turma`/`fase`) baseada em tabela de lookup treinada no passado.  
**Por quê:** Garante consistência entre treino e inferência sem recalcular estatísticas com dados que não existiriam no momento da decisão.

---

## 3.4 Padronização robusta de dados históricos

**Decisão:** Implementar renomeação com matching flexível e normalização de fase entre anos (2022, 2023, 2024).  
**Por quê:** O dataset muda de nomenclatura/formato entre anos (colunas, fase, caracteres), e o pipeline precisa ser resiliente para manter comparabilidade longitudinal.

**Decisão:** Tratar ausências estruturais (ex.: `ipp` ausente em 2022, `ingles` em formato diferente).  
**Por quê:** Permite construir matriz de features unificada sem quebrar o fluxo.

---

## 3.5 Engenharia de features orientada ao contexto escolar

**Decisão:** Combinar sinais acadêmicos, engajamento, psicossocial e contexto institucional.  
**Por quê:** Defasagem escolar é fenômeno multifatorial; usar só nota tende a ser insuficiente.

**Principais grupos de feature:**
- Acadêmicas: `matem`, `portug`, `ingles`, `media_provas`, `disp_provas`, `fez_ingles`
- Engajamento: `ieg`
- Psicossociais: `iaa`, `iaa_participou`, `ips`, `ipp`
- Contexto: `fase`, `turma`, `instituicao`, `genero`, `ano_ingresso`, `tempo_casa`
- Estatísticas de grupo (turma/fase): média, desvio, p25, p75, deltas e z-scores
- Flags de risco relativo: abaixo de p25 da turma em variáveis-chave

**Regra de negócio adicional (inglês):**
- Inglês só influencia o modelo em fases onde a prova é obrigatória.
- Nas fases equivalentes a `0`, `1`, `2` e `8`, o campo `ingles` é neutralizado (não contribui para o risco).

**Decisão:** Em validação/inferência, preencher ausentes numéricos com `0` quando necessário.  
**Por quê:** Mantém robustez operacional; evita falhas por dados incompletos no consumo.

---

## 3.6 Escolha do algoritmo (CatBoost)

**Decisão:** Usar `CatBoostClassifier` com mistura de features numéricas e categóricas.  
**Por quê:** CatBoost lida bem com categóricas, costuma ser forte em tabular e tem boa performance com pouco pré-processamento manual de encoding.

**Parâmetros estratégicos adotados:**
- `auto_class_weights="Balanced"`  
  **Por quê:** corrige desbalanceamento de classe sem downsampling agressivo.
- `early_stopping_rounds=200` + validação temporal  
  **Por quê:** controla overfitting monitorando generalização real.
- `bootstrap_type='Bernoulli'`, `subsample=0.8`, `rsm=0.8`  
  **Por quê:** regularização por subamostragem de linhas/colunas para reduzir variância.

---

## 3.7 Métricas de negócio e política de alerta

**Decisão:** Não depender apenas de AUC.  
**Por quê:** AUC mede ranking global, mas a operação precisa escolher quem entra na lista de intervenção.

**Métricas priorizadas para ação:**
- `Recall@TopK` (principal): captura de casos em risco dentro da capacidade de atendimento.
- `Precision@TopK`: qualidade da lista.
- `Lift@TopK`: ganho sobre seleção aleatória.

**Decisão:** Aplicar alerta `Top-K%` **estratificado por fase**.  
**Por quê:** Garante distribuição de atenção entre fases de tamanhos diferentes, reduzindo concentração indevida em grupos maiores.

---

## 3.8 Explicabilidade e uso operacional

**Decisão:** Incluir explicações SHAP Top-3 por aluno na inferência (quando disponível).  
**Por quê:** Ajuda coordenação pedagógica a entender fatores predominantes de risco e desenhar intervenção.

**Decisão:** Persistir `valid_scored.csv` para dashboard e monitoramento.  
**Por quê:** Separa camada analítica/visual da camada de treinamento, simplificando consumo.

---

## 3.9 Entrega em camadas (treino, API, dashboard)

**Decisão:** Arquitetura modular em `src/`, `api/` e `dashboard/`.  
**Por quê:** Facilita manutenção, testes e evolução independente de cada componente.

**Decisão:** API com endpoints de saúde, predição, alerta, explicação e drift.  
**Por quê:** Permite integração com outros sistemas e automação de uso.

---

## 4) Artefatos e como interpretar

- `catboost_model.cbm`: modelo treinado.
- `model_meta.json`: lista de features numéricas/categóricas e melhores iterações.
- `lookup_tables.pkl`: estatísticas históricas de turma/fase usadas na engenharia.
- `evaluation_results.json`: métricas oficiais de validação.
- `valid_scored.csv`: base com score por aluno e colunas úteis ao dashboard.
- `scored_history.csv`: histórico de scores por ano-base usado no seletor de ano do dashboard.
- `cohort_summary.json`: transparência da amostragem por ano (bruto, pareado, usado).

Leitura prática:
- Se `AUC` cresce mas `Recall@Top15` cai, o ranking global melhorou porém piorou no ponto de corte operacional.
- Se `Lift@Top10` > 1 de forma estável, há valor real sobre priorização aleatória.
- No detalhe do aluno (`/aluno/<ra>`), use os indicadores acadêmicos/psicossociais para contextualizar o score da fase.
- A diferença entre RA comum bruto e RA usado no modelo é esperada: o pipeline usa pares temporais válidos e regras de consistência.

---

## 5) Riscos conhecidos e próximos passos

1. Dependência de consistência anual do preenchimento de dados (especialmente colunas psicossociais).
2. Possível drift de perfil por turma/fase entre anos.
3. Opportunity: calibrar score (Platt/Isotonic) para probabilidade mais interpretável.
4. Opportunity: validação temporal rolling (mais janelas históricas).
5. Opportunity: testes automatizados de qualidade dos dados antes de treinar.

---

## 6) Tutorial do dashboard (arquivo separado)

O tutorial detalhado de análise foi movido para:

- `TUTORIAL_ANALISE_DASHBOARD.md`

Esse material separado inclui:
- passo a passo de leitura;
- explicação de cada técnica analítica usada no dashboard;
- checklist operacional para rotina de monitoramento.

---

## 7) Resumo executivo

O pipeline foi estrategicamente desenhado para:

- prever risco futuro com split temporal realista;
- evitar atalhos de leakage;
- combinar múltiplas dimensões do aluno;
- transformar score em política operacional justa por fase;
- oferecer leitura executiva e tática no dashboard.

O resultado é um sistema orientado a decisão pedagógica preventiva, com métricas e visualizações adequadas para uso no dia a dia.
