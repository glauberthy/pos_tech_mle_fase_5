# Tutorial de Análise no Dashboard
## Sistema de Monitoramento de Risco Escolar

---

## Regras de Negócio Vigentes

- O dashboard exibe risco de **piora de defasagem** (não diagnóstico retrospectivo).
- O target do modelo é `defasagem_t1 > defasagem_t`.
- O filtro e a lista seguem Top-K% estratificado por fase.
- A leitura deve priorizar `Recall@TopK` e `Lift@TopK` para decisão operacional.
- Em fases equivalentes a `0`, `1`, `2` e `8`, inglês não é obrigatório e não aparece como sinal crítico.

---

## 1) Pré-requisitos

Antes de analisar o dashboard, garanta que o treino gerou:

- `models/valid_scored.csv`
- `models/evaluation_results.json`

Para abrir no Windows (PowerShell):

```powershell
cd c:\datathon
.\start-dashboard.ps1 -Port 8503
```

Acesse: `http://localhost:8503`

Atualização de ano (ex.: 2025):
- use o uploader na aba **Dados e Retreinamento**,
- envie o arquivo com a aba `PEDE2025`,
- clique em **Executar retreinamento** para atualizar automaticamente.

---

## 2) Ordem recomendada de análise

1. Selecionar o **ano-base** de visualização.
2. Definir `Top-K%` e filtros de `Fase`/`Turma`/`RA`.
3. Ler KPIs gerais e cards de transparência da amostra.
4. Priorizar alunos na lista de alertas.
5. Avaliar distribuição por fase.
6. Validar qualidade do modelo (métricas e curva).
7. Fechar análise no nível individual e sinais de alerta.

---

## 3) Técnicas de análise usadas no dashboard (com explicação)

## 3.1 Segmentação por filtros (análise estratificada)

**Onde aparece:** barra lateral (`Ano de visualização`, `Top-K%` e filtro de fase).  
**Técnica:** segmentação de população por recorte operacional.  
**Para que serve:** comparar comportamento de risco entre subgrupos (fases) e simular diferentes capacidades de atendimento.

**Leitura prática:**
- `Top 10%`: alta prioridade, lista mais curta.
- `Top 20-25%`: maior cobertura, menor foco.
- ano-base selecionado `Y` sempre representa risco previsto para `Y+1`.

---

## 3.2 Regra de priorização Top-K% estratificada por fase

**Onde aparece:** cálculo de `alerta` e abas de lista/distribuição.  
**Técnica:** ranking por score com seleção do topo em cada fase (não no total geral).  
**Para que serve:** garantir equidade entre fases de tamanhos diferentes.

**Leitura prática:**
- evita que fases maiores "dominem" toda a lista.
- mantém representatividade na triagem pedagógica.

---

## 3.3 KPIs (análise descritiva executiva)

**Onde aparece:** cartões no topo (`Total`, `Em alerta`, `% em alerta`, `AUC`).  
**Técnica:** sumarização de indicadores-chave.  
**Para que serve:** visão rápida de volume, intensidade de alerta e qualidade global do modelo.

**Como interpretar:**
- `% em alerta` deve acompanhar a política Top-K configurada.
- `AUC` resume capacidade de ordenação geral do modelo.

---

## 3.4 Tabela de alertas ordenada por score (priorização operacional)

**Onde aparece:** aba **Lista de Alertas**.  
**Técnica:** ranking tabular para tomada de decisão por prioridade.  
**Para que serve:** transformar análise em ação (quem atender primeiro).

**Como interpretar:**
- maior score = maior prioridade de acompanhamento.
- use junto com fase/turma para montar plano de intervenção.
- o score já representa o risco previsto no recorte atual.
- use a coluna de motivos para apoiar a priorização pedagógica.

---

## 3.5 Boxplot por fase (análise de distribuição)

**Onde aparece:** aba **Distribuição por Fase** (gráfico de caixa).  
**Técnica:** estatística descritiva visual por grupo (mediana, quartis, dispersão e outliers).  
**Para que serve:** comparar perfis de risco entre fases.

**Como interpretar:**
- mediana alta: fase com risco central maior.
- caixa larga: heterogeneidade maior dentro da fase.
- muitos outliers altos: subgrupo de risco crítico.

---

## 3.6 Barra de `% em alerta` por fase (análise comparativa entre grupos)

**Onde aparece:** aba **Distribuição por Fase** (bar chart).  
**Técnica:** comparação de taxa entre categorias com anotação de volume (`n_alerta`).  
**Para que serve:** identificar onde a política de alerta está concentrando mais casos.

**Como interpretar:**
- fase com `% em alerta` acima das demais merece investigação direcionada.
- comparar `%` com `n_alerta` evita leitura enviesada por tamanho de fase.

---

## 3.7 Métricas Top-K (análise de performance orientada à decisão)

**Onde aparece:** aba **Métricas de Desempenho**.  
**Técnica:** avaliação por ponto de corte operacional.

Métricas:
- `Recall@TopK`: quanto dos casos reais de risco foi capturado.
- `Precision@TopK`: quão "limpa" está a lista de alerta.
- `Lift@TopK`: ganho sobre seleção aleatória.

**Para que serve:** calibrar K conforme capacidade da equipe e objetivo da intervenção.

---

## 3.8 Curva de Recall vs linha aleatória (análise de ganho incremental)

**Onde aparece:** aba **Métricas de Desempenho** (gráfico de linhas).  
**Técnica:** curva de desempenho por cobertura (`K`) com baseline aleatório.

**Para que serve:** visualizar quanto o modelo captura acima do acaso ao aumentar o tamanho da lista.

**Como interpretar:**
- curva do modelo acima da aleatória = modelo agrega valor.
- distância maior entre curvas = melhor ganho prático.

---

## 3.9 Análise individual com gauge (microanálise de caso)

**Onde aparece:** aba **Análise Individual**.  
**Técnica:** análise de caso com score, status e indicadores-chave.

Componentes:
- score numérico (`0-100`);
- gauge com faixas de risco;
- indicadores do aluno (`matem`, `portug`, `ingles`, `ieg`, `iaa`, `ips`).

Regra importante:
- nas fases equivalentes a `0`, `1`, `2` e `8`, inglês não é obrigatório e não deve ser interpretado como alerta.

**Para que serve:** apoiar decisão pedagógica personalizada.

**Como interpretar:**
- score alto + fragilidade acadêmica/engajamento = ação imediata.
- comparar casos da mesma fase ajuda a separar ação individual x coletiva.

---

## 3.10 Detalhe do aluno (rota `/aluno/<ra>`)

**Onde aparece:** ao clicar em **Abrir** na lista de alertas.  
**Técnica:** análise individual com histórico de score e indicadores do aluno.

O que a tela mostra:
- identificação do aluno (RA, fase, turma);
- histórico de score por ano-base;
- tabela de indicadores acadêmicos e psicossociais;
- texto de motivos para contexto da decisão.

Como interpretar:
- score alto + fragilidade em indicadores-chave sugere prioridade de intervenção;
- histórico crescente de score reforça necessidade de acompanhamento contínuo.

Regra de negócio aplicada:
- em fases equivalentes a `0`, `1`, `2` e `8`, inglês não é obrigatório e deve ser interpretado com cautela.

---

## 3.11 Transparência da amostra (bruto vs usado no modelo)

**Onde aparece:** cards após os KPIs (quando há ano selecionado).  
**Técnica:** funil de elegibilidade da coorte.

Os cards mostram:
- total bruto no ano;
- pareados no ano seguinte;
- usados no modelo.

Leitura prática:
- é esperado que “usados no modelo” seja menor que o total bruto;
- isso decorre de pareamento temporal e validações de consistência do pipeline.

---

## 4) Roteiro prático de decisão (30 minutos)

1. Defina `K=15%`.
2. Veja KPIs para noção global.
3. Abra Lista de Alertas e identifique Top 20 alunos por fase.
4. Na distribuição, cheque fases com mediana de score mais alta.
5. Valide se `Lift` está > 1 no K escolhido.
6. Faça leitura individual dos casos críticos.
7. Registre plano de intervenção com responsável e prazo.

---

## 5) Boas práticas de uso

- Não usar apenas `AUC` para decidir operação.
- Reavaliar `K` conforme capacidade da equipe no mês.
- Monitorar estabilidade por fase ao longo do tempo.
- Documentar o resultado das intervenções para retroalimentar o processo.

---

## 6) Resumo

O dashboard combina técnicas de:
- segmentação;
- distribuição estatística;
- comparação entre grupos;
- avaliação de ranking;
- priorização operacional;
- análise de caso individual.

O ganho é transformar score em decisão pedagógica estruturada, com equilíbrio entre visão executiva e ação no nível do aluno.

---

## 7) FAQ rápido de uso

**"Defasou (T+1)" é risco futuro?**  
Não. É desfecho real observado. O risco futuro é o `Score de Risco`.

**Por que tenho menos alunos no dashboard que no XLS bruto?**  
Porque a coorte usa alunos pareados entre anos e regras de consistência do pipeline.

**Em qual aba vejo prioridade de ação?**  
`Alertas` (ranking operacional) e detalhe do aluno (`/aluno/<ra>`).

**Quando usar "Métricas de Desempenho"?**  
Para validar se a política Top-K está capturando casos com ganho real (`Recall`, `Lift`).
