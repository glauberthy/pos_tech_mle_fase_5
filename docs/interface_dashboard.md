

---

## 1) Objetivo do sistema (visão do produto)

Construir um **painel de monitoramento pedagógico** que permita a professores e coordenação:

* **Identificar rapidamente alunos em risco de defasagem** (por ano, fase e turma).
* **Entender os fatores que explicam o risco** (indicadores acadêmicos e psicossociais).
* **Acompanhar evolução/histórico do aluno** e evidências que sustentam a decisão.
* **Monitorar a qualidade e saúde do modelo** (AUC, drift, latência, uso de memória).
* **Operacionalizar re-treinamento com novos dados** de forma segura e rastreável.

---

## 2) Perfis de usuário e linguagem da interface

**Usuários principais**

* **Coordenação pedagógica:** visão macro (tendências, distribuição por fase, metas, alertas).
* **Professor:** visão operacional (lista de alunos, turma/fase, detalhes do aluno e intervenção).

**Diretriz de UX**

* Linguagem simples para ação: “**Risco alto**”, “**Em alerta**”, “**Evolução**”, “**Fatores principais**”.
* Termos técnicos (AUC, drift, latência) ficam em uma área de “**Saúde do Modelo**”, com tooltips.
* Priorizar fluxo: **Visão geral → Lista de alertas → Detalhe do aluno → Ação pedagógica**.

---

## 3) Estrutura de navegação (menu)

1. **Início (Visão Geral)**
2. **Alertas (Lista de Alunos)**
3. **Distribuição por Fase**
4. **Saúde do Modelo**

   * Métricas e Desempenho (AUC etc.)
   * Drift (dados e modelo)
   * Performance (tempo/memória)
5. **Dados e Retreinamento**
6. **Logs e Auditoria** (mais técnico; opcionalmente restrito por perfil)

---

## 4) Filtros globais (presentes no topo das telas principais)

Criar um painel fixo de filtros, consistente em todas as telas:

* **Ano-base:** 2022 / 2023 / 2024 / (novos anos inseridos)
* **Fase:** (multi-seleção) F0…F7
* **Turma:** (dependente do ano/fase)
* **Estratégia de Alerta (Top-K):**

  * seletor de percentual (ex.: 10%, 15%, 20%, 25%)
  * regra aplicada **por Fase** (Top-K dentro de cada fase)
* **Busca por aluno:** RA / nome (se existir)

> Importante: os filtros devem atualizar KPIs e listas em tempo real (com indicador de carregamento leve).

---

## 5) Tela 1 — Início (Visão Geral)

**Objetivo:** dar visão executiva e acionável.

### KPIs (cards)

* **Total de alunos**
* **Quantidade em alerta**
* **% em alerta**
* **AUC (modelo atual)**

### Blocos principais

1. **Resumo por fase**

   * mini visão: total por fase, alertas por fase, % alerta por fase
2. **Tendência temporal (quando houver histórico por ano)**

   * comparação ano a ano (ex.: 2022 vs 2023 vs 2024)
3. **Atalhos de ação**

   * “Ver lista de alunos em alerta”
   * “Ver distribuição por fase”
   * “Ver saúde do modelo”

---

## 6) Tela 2 — Alertas (Lista de Alunos)

**Objetivo:** triagem operacional.

### Tabela (com paginação e ordenação)

Colunas recomendadas:

* **Aluno (RA / Nome)**
* **Fase**
* **Turma**
* **Risco de defasagem (%)** *(0–100)*
* **Status:** Em alerta (sim/não)
* **Principais indicadores** *(top 3 razões, em linguagem clara)*
* **Ação:** **“Ver detalhes”** (link/botão)

### Funcionalidades críticas

* Ordenar por **maior risco**.
* Aplicar Top-K por fase automaticamente.
* Exportar CSV (opcional, perfil coordenação).
* O item “Ver detalhes” abre a **Tela do Aluno**.

---

## 7) Tela 3 — Detalhe do Aluno (visão 360º)

**Objetivo:** explicabilidade + histórico + evidências.

### Cabeçalho do aluno

* Identificação: **RA, ano-base, fase, turma**
* **Risco atual de defasagem (%)** com faixa (baixo/médio/alto)
* Selo: “Em alerta” ou “Monitoramento”

### Seções (em abas ou cards)

1. **Evolução / Histórico**

   * gráfico de evolução do risco ao longo do tempo (se houver)
   * evolução de notas e indicadores (MAT/POR e índices)
2. **Indicadores acadêmicos**

   * notas (Matemática, Português)
   * indicadores derivados (IEG, IDA, etc. conforme projeto)
3. **Indicadores psicossociais e psicológicos**

   * IAA, IPS, IPV e correlatos (com explicação simples)
4. **Por que este aluno está em risco? (Explicação do modelo)**

   * “Top fatores que aumentaram o risco”
   * “Fatores protetores”
   * Exibir contribuição/impacto de forma interpretável (ex.: ranking + setas ↑/↓)
   * Mostrar valores do aluno vs referência da fase/turma (comparativo)

> Diretriz: tudo aqui deve sustentar tomada de decisão pedagógica. Explicação técnica (SHAP, importâncias) pode ficar “expandível”.

---

## 8) Tela 4 — Distribuição por Fase

**Objetivo:** entender onde o risco se concentra e priorizar intervenção.

Conteúdos:

* Distribuição de alunos e alertas **por fase** (quantidade e %)
* Possibilidade de drill-down: fase → turmas → lista de alunos
* Comparação entre anos (quando selecionados)

KPIs persistentes:

* Total, alertas, % alertas, AUC

---

## 9) Tela 5 — Saúde do Modelo (Métricas, Desempenho e Drift)

**Objetivo:** garantir confiabilidade do modelo e transparência do sistema.

### 9.1 Métricas do modelo

* **AUC** (principal)
* (opcionais, se fizer sentido) Precision@TopK, Recall@TopK, matriz confusão por fase
* Estabilidade por fase: AUC por fase (se amostra permitir)

### 9.2 Drift (dados e modelo)

**Painel de Drift** com:

* Drift por variáveis-chave (notas, indicadores, psicossociais)
* Drift do score (distribuição do risco ao longo do tempo)
* Alertas automáticos (ex.: drift acima do limiar)

> Mostrar a mensagem do tipo: “Mudança relevante detectada nas notas de Matemática na Fase 2 — revisar coleta ou retreinar.”

### 9.3 Performance técnica (infra)

Na “Tela de Performance”, apresentar:

* **Tempo médio de processamento** (inference e treinamento)
* **Uso de memória** (pico e médio)
* (opcional) CPU, throughput, tempo por etapa do pipeline

---

## 10) Tela 6 — Dados e Retreinamento (com governança)

**Objetivo:** permitir atualização do modelo sem quebrar rastreabilidade.

### Botão principal

* **“Adicionar novo arquivo para retreinar”**

### Fluxo recomendado (seguro)

1. Upload do arquivo (validação de schema/colunas)
2. Prévia: número de registros, anos/fases, missing, consistência
3. Confirmação e execução do pipeline
4. Registro de versão:

   * dataset_version
   * model_version
   * métricas do treino (AUC etc.)
   * data de treino
5. Publicação do modelo “ativo” (com rollback)

> UX: mostrar progresso do pipeline e mensagens claras (“validando”, “treinando”, “avaliando”, “publicando”).

---

## 11) Logging e auditoria (requisitos técnicos)

Para suportar confiabilidade e prestação de contas, registrar logs estruturados:

**Eventos mínimos**

* Inference: timestamp, model_version, dataset_version, ano/fase/turma, latency, score, status alerta
* Retreinamento: usuário que acionou, dataset usado, validações, métricas, duração, memória pico, resultado (sucesso/erro)
* Drift: quando detectado, em quais variáveis, qual limiar, ações recomendadas
* Acesso: auditoria de telas sensíveis (detalhe do aluno)

**Observabilidade**

* Logs estruturados (JSON), correlação por request_id
* Painel de erros e falhas do pipeline (com stacktrace em área técnica)

---

## 12) Padrões de apresentação (para professores)

* Sempre mostrar **o que fazer a seguir**: “Priorize estes 10 alunos”, “Fase X concentra 45% dos alertas”.
* Sempre permitir **comparação contextual**: aluno vs turma/fase (sem expor dados sensíveis indevidamente).
* Explicações com frases curtas e indicadores com significado (“queda em MAT”, “baixo IPS”, “instabilidade em POR”).

---

