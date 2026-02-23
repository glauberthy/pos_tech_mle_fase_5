# Variáveis e Tratamentos do Treinamento
## Sistema de Previsão de Risco de Defasagem Escolar

---

## Regras de Negócio Vigentes

- O target modela **piora de defasagem** entre T e T+1.
- Regra do target: `target = 1` quando `defasagem_t1 > defasagem_t`.
- Interseção bruta de RA entre anos não é suficiente: o pipeline usa pares temporais consistentes e normalizados.
- Variáveis sintéticas/diagnósticas são removidas por anti-leakage.
- Inglês só influencia em fases com obrigatoriedade da prova.
- Em fases equivalentes a `0`, `1`, `2` e `8`, `ingles` é neutralizado.

---

## 1) Escopo deste documento

Este documento descreve:

1. As variáveis originais utilizadas no pipeline de treinamento.
2. Os tratamentos/transformações aplicados.
3. As flags e features derivadas produzidas.
4. A justificativa da escolha desse conjunto de dados.

Referência de implementação: `src/preprocessing.py`, `src/feature_engineering.py`, `train.py`.

---

## 2) Variáveis originais usadas (após padronização)

As abas históricas disponíveis (`PEDE2022`, `PEDE2023`, `PEDE2024`, `PEDE2025`, etc.) são normalizadas para o mesmo esquema canônico.

### 2.1 Variáveis base carregadas

- `ra`
- `ano_base`
- `turma`
- `fase` (numérica, derivada de `fase_raw`)
- `genero`
- `instituicao`
- `ano_ingresso`
- `ieg`
- `iaa`
- `ips`
- `ipp`
- `matem`
- `portug`
- `ingles`
- `defasagem_t` (estado no ano-base após pareamento)

### 2.2 Variáveis de supervisão longitudinal (target)

Na montagem longitudinal (`T -> T+1`), é criada:

- `defasagem_t1` (vinda do ano seguinte)
- `target = (defasagem_t1 > defasagem_t)`

Observação importante:
- `target` é variável-resposta (rótulo), **não** entra como feature.
- `defasagem_t1` é usada para construir o target e depois excluída do conjunto de treino.
- A coorte usada em cada par temporal é formada por alunos com `RA` presente em `T` e `T+1`.

---

## 3) Tratamento aplicado por variável original

## 3.1 Identificação e tempo

- `ra`
  - tratamento: convertido para `string`, trim de espaços.
  - uso: chave de junção longitudinal e identificação no dashboard/API.
  - entra como feature: **não**.

- `ano_base`
  - tratamento: atribuído por aba (2022, 2023, 2024).
  - uso: contexto temporal e cálculo de `tempo_casa`.
  - entra como feature: **não** (mantido como coluna de suporte).

- `ano_ingresso`
  - tratamento: coerção numérica.
  - derivação: `tempo_casa = ano_base - ano_ingresso`.
  - entra como feature: **sim** (`ano_ingresso`, `tempo_casa`).

## 3.2 Contexto escolar

- `fase_raw -> fase`
  - tratamento: normalização para fase numérica entre anos:
    - `ALFA` -> `0`
    - `FASE 3` -> `3`
    - `3A` -> `3`
  - entra como feature: **sim** (categórica).

- `turma`
  - tratamento: padronização de coluna.
  - entra como feature: **sim** (categórica).
  - também é base para estatísticas de grupo.

- `instituicao`
  - tratamento: padronização textual.
  - entra como feature: **sim** (categórica).

- `genero`
  - tratamento: normalização para `F`, `M`, `NA` a partir de variações textuais.
  - entra como feature: **sim** (categórica).

## 3.3 Acadêmicas

- `matem`, `portug`, `ingles`
  - tratamento: coerção numérica.
  - regra de obrigatoriedade de inglês:
    - fases equivalentes a `0`, `1`, `2` e `8`: inglês não obrigatório.
    - nessas fases, `ingles` é neutralizado (`NaN`) para não influenciar o risco.
  - ausência:
    - quando ausente, permanece `NaN` nas fases sem obrigatoriedade.
  - entra como feature: **sim**.
  - derivações:
    - `fez_ingles`
    - `media_provas`
    - `disp_provas`
    - estatísticas de grupo e flags de posição relativa.

## 3.4 Engajamento e psicossocial

- `ieg`, `ips`, `ipp`
  - tratamento: coerção numérica.
  - `ipp`: criado como `NaN` quando inexistente na origem (ex.: 2022), com imputação posterior.
  - entra como feature: **sim**.

- `iaa`
  - tratamento:
    - coerção numérica;
    - criação da flag `iaa_participou`;
    - preenchimento de `iaa` com `0` quando ausente.
  - entra como feature: **sim** (`iaa` e `iaa_participou`).

---

## 4) Features derivadas e flags produzidas

## 4.1 Features derivadas diretas

- `fez_ingles` = indicador binário de presença de nota de inglês.
- `media_provas` = média de `matem`, `portug`, `ingles` (ignorando ausentes).
- `disp_provas` = desvio padrão das provas disponíveis.
- `tempo_casa` = `ano_base - ano_ingresso`.
- `iaa_participou` = flag de participação no IAA.

## 4.2 Estatísticas por grupo (turma e fase)

Para cada variável em:
- `media_provas`, `matem`, `portug`, `ieg`, `ips`, `iaa`

São criadas (por `turma` e por `fase`):

- `{prefix}_mean_{X}`
- `{prefix}_std_{X}`
- `{prefix}_p25_{X}`
- `{prefix}_p75_{X}`
- `delta_{prefix}_{X}` = valor individual - média do grupo
- `z_{prefix}_{X}` = delta / (std + epsilon)

Onde `{prefix}` é `turma` ou `fase`.

## 4.3 Flags de risco relativo

- `abaixo_p25_turma_media_provas`
- `abaixo_p25_turma_matem`
- `abaixo_p25_turma_ieg`

Essas flags marcam alunos abaixo do 1o quartil da própria turma em dimensões-chave.

---

## 5) Variáveis finais usadas no modelo

## 5.1 Categóricas (CatBoost)

- `fase`
- `turma`
- `instituicao`
- `genero`

## 5.2 Numéricas

Incluem:
- variáveis originais numéricas tratadas (`ano_ingresso`, `ieg`, `iaa`, `ips`, `ipp`, `matem`, `portug`, `ingles`);
- derivadas diretas (`fez_ingles`, `media_provas`, `disp_provas`, `tempo_casa`, `iaa_participou`);
- estatísticas de grupo e seus deltas/z-scores;
- flags binárias `abaixo_p25_*`.

Colunas excluídas da modelagem:
- `ra`, `ano_base`, `target`, `defasagem_t1`.

Observação de negócio:
- embora `ingles` exista no conjunto numérico, seu efeito é desativado nas fases sem obrigatoriedade da prova (equivalentes a `0`, `1`, `2`, `8`).

---

## 6) Variáveis originais removidas por risco de leakage

Exemplos removidos no preprocessing:

- `IAN`
- `IDA`
- `INDE` (e variações por ano)
- `Pedra` (e variações por ano)
- `Fase ideal`
- `IPV`
- `Atingiu PV`
- `Cg`, `Cf`, `Ct`
- `Rec Av*`, `Rec Psicologia`
- `Indicado`
- `Destaque*`

Motivo: são variáveis sintéticas/diagnósticas ou posteriores à decisão, que podem vazar informação do desfecho.

---

## 7) Regra de imputação e robustez

Durante a engenharia:

- treino:
  - numéricas recebem mediana da coluna;
  - se mediana for indefinida (coluna toda vazia), usa `0`.
- validação/inferência:
  - numéricas ausentes recebem `0`.
- exceção:
  - `ingles` não é imputado artificialmente; pode permanecer ausente em fases sem obrigatoriedade.
- categóricas:
  - convertidas para string.

Objetivo: estabilidade operacional, inclusive com dados incompletos na entrada.

---

## 8) Por que este conjunto de dados foi escolhido?

A escolha foi estratégica por quatro motivos:

1. **Aderência ao problema real**
   - O conjunto combina desempenho acadêmico, engajamento, psicossocial e contexto escolar, que são fatores plausíveis de risco de defasagem.

2. **Capacidade preditiva com baixa dependência de proxies**
   - Foram priorizados sinais "brutos" e removidos indicadores sintéticos/diagnósticos para reduzir atalhos e melhorar generalização.

3. **Consistência temporal**
   - Há histórico em múltiplos anos, permitindo modelagem longitudinal (`T -> T+1`) com validação temporal realista.

4. **Acionabilidade pedagógica**
   - O conjunto final permite traduzir score em ação:
     - priorização por fase (Top-K estratificado),
     - leitura por aluno,
     - suporte à intervenção preventiva.

Em resumo, o dataset foi escolhido para equilibrar **valor preditivo**, **integridade metodológica (anti-leakage)** e **utilidade prática para tomada de decisão educacional**.

## 9) Nota metodológica importante (ênfase)

**Pergunta comum:** por que não usar todos os RAs comuns entre anos?  
**Resposta:** porque o problema modelado é “piora de defasagem”, não “qualquer defasagem”.

Se usarmos todos os RAs comuns sem controle:
- aumentamos inconsistências temporais,
- diluímos o sinal de evolução,
- pioramos a interpretabilidade da lista de alerta.

Por isso, os valores de “RA comuns bruto” são naturalmente maiores que “RA usados no modelo”.
