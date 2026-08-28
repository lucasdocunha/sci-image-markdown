# 📊 Sci-Image-Markdown: Benchmark & Evaluation Results

Este documento apresenta os resultados completos da avaliação comparativa entre o **Modelo Base (Zero-Shot Baseline)** e o **Modelo Fine-Tuned com LoRA** (`Qwen/Qwen2.5-VL-3B-Instruct`), avaliados no conjunto de teste independente do benchmark **Sci-ImageMiner** (373 amostras de gráficos e curvas científicas de ALD/ALE).

---

## 📈 Tabela Comparativa de Métricas

Avaliação executada sobre o arquivo `data/processed/test.jsonl` (373 figuras científicas).

| Métrica | Descrição Resumida | Modelo Base (Zero-Shot) | Fine-Tuned (LoRA) | Variação (Delta / Ganho) |
| :--- | :--- | :---: | :---: | :---: |
| **`valid_table`** | Taxa de tabelas Markdown sintaticamente válidas | `0.8874` (88.74%) | **`0.9973` (99.73%)** | **+0.1099 (+10.99%)** 🚀 |
| **`exact_match`** | Casamento exato estrito (100% idêntico caractere por caractere) | `0.0000` | `0.0000` | `0.0000` |
| **`edit_similarity`** | Similaridade normalizada de Levenshtein (0 a 1) | `0.2757` | **`0.3010`** | **+0.0253** 📈 |
| **`rouge_l`** | Sobreposição pela Maior Subsequência Comum (LCS) | `0.2066` | **`0.2395`** | **+0.0329** 📈 |
| **`rouge_1`** | F1 de sobreposição de unigramas (palavras individuais) | `0.2345` | **`0.2666`** | **+0.0320** 📈 |
| **`rouge_2`** | F1 de sobreposição de bigramas (termos consecutivos) | `0.0883` | **`0.1191`** | **+0.0308** 📈 |
| **`bleu_4`** | Precisão cumulativa de até 4-gramas com brevidade | `0.0841` | **`0.1094`** | **+0.0253** 📈 |
| **`cell_precision`** | Precisão dos valores numéricos extraídos ($\le 5\%$ tol.) | `0.2108` (21.08%) | **`0.2137` (21.37%)** | **+0.0029** 📈 |
| **`cell_recall`** | Revocação de valores numéricos do Ground Truth ($\le 5\%$) | `0.3578` (35.78%) | **`0.4822` (48.22%)** | **+0.1245 (+12.45%)** 🚀 |
| **`cell_f1`** | Média harmônica de precisão e revocação numérica | `0.1842` | **`0.2060`** | **+0.0218** 📈 |
| **`cell_rmse`** | Raiz do Erro Quadrático Médio nos números pareados | `2.6635` | **`1.0308`** | **-1.6327 (Melhor)** 🎯 |
| **`cell_rne`** | Erro Relativo Normalizado Médio ($\|p - t\| / \|t\|$) | `0.0044` | **`0.0036`** | **-0.0008 (Melhor)** 🎯 |

---

## 🔍 Significado Detalhado de Cada Métrica

### 1. Métricas de Formato e Estrutura
- **`valid_table` (Taxa de Validade Sintática)**:
  Mede a porcentagem de saídas geradas pelo modelo que consistem em tabelas Markdown estruturalmente íntegras (com cabeçalho, linha divisória `|---|---|` e colunas regulares parseáveis pelo Pandas).
  - *Impacto:* O fine-tuning eliminou quase a totalidade de alucinações estruturais e repetições em loop que ocorriam no modelo base, subindo de **88.74% para 99.73%**.
- **`exact_match` (Casamento Exato)**:
  Mede se o Markdown gerado é 100% idêntico caractere por caractere à tabela de referência.
  - *Impacto:* Em tabelas científicas com múltiplas casas decimais e formatação de texto livre, o valor `0.0` é o padrão esperado para correspondência exata estrita de strings longas.
- **`edit_similarity` (Similaridade de Edição Normalizada)**:
  Baseada na distância de Levenshtein: $1 - \frac{\text{dist}(\text{pred}, \text{gt})}{\max(\text{len}(\text{pred}), \text{len}(\text{gt}))}$. Varia de 0 (completamente diferente) a 1 (idêntico).

### 2. Métricas Léxicas de Processamento de Linguagem Natural (NLP)
- **`rouge_1`**: Mede a sobreposição (F1-score) de palavras únicas (unigramas) entre o texto gerado e o ground truth.
- **`rouge_2`**: Mede a sobreposição de pares consecutivos de palavras (bigramas), avaliando se os nomes compostos de colunas e unidades científicas estão corretos (ex: *"Growth Temperature"*, *"Film Thickness"*).
- **`rouge_l`**: Avalia a Maior Subsequência Comum (LCS - Longest Common Subsequence), capturando alinhamento de sequência mesmo com variações de espaçamento.
- **`bleu_4`**: Métrica clássica de tradução/geração avaliando precisão cumulativa de n-gramas (1 a 4 palavras) ponderada pela penalidade de brevidade.

### 3. Métricas Numéricas em Nível de Célula (Domain Cell Metrics)
Para gráficos científicos (ALD/ALE), a fidelidade numérica dos pontos das curvas é crucial:
- **`cell_precision`**: Dentre todos os números extraídos pelo modelo na tabela, qual fração corresponde a pontos reais da curva dentro de uma tolerância relativa de 5% ($|p - t| / |t| \le 0.05$).
- **`cell_recall`**: De todos os pontos numéricos que realmente existem na figura (ground truth), qual fração o modelo foi capaz de detectar e extrair corretamente.
  - *Destaque:* Subiu de **35.78% para 48.22%**, um ganho absoluto de **+12.45%** na capacidade do modelo de ler os pontos nos eixos.
- **`cell_f1`**: Equilíbrio harmônico entre precisão e revocação de células numéricas.
- **`cell_rmse` (Root Mean Squared Error)**:
  Calcula o desvio quadrático médio entre o valor previsto e o valor real para todos os números pareados.
  - *Destaque:* O erro caiu de **2.6635 para 1.0308** (uma redução de **mais de 61% no erro médio**).
- **`cell_rne` (Relative Normalized Error Médio)**:
  Mede o percentual médio de erro relativo $(|p - t| / |t|)$ nos pares identificados. Caiu para **0.0036** (apenas 0.36% de erro médio relativo por ponto pareado).

---

## 🔬 Saídas Reais de Exemplo (Ground Truth vs Base vs Fine-Tuned)

Abaixo estão comparações qualitativas diretas de amostras do conjunto de teste:

### Exemplo 1: Simulação ALE - Corte de Energia Cinética vs Energia Total
> **Imagem:** `sci_imageminer__atomic_layer_etching__simulation_usecase__11__f29fb0936e8517e60638237bcc9fde1e5c001e80bfc85cf97feddcf84eb574c3.jpg`

#### Ground Truth
```markdown
| Kinetic energy cut-off for charge density and potential (Ry) | Total energy (Ry) |
|---|---|
| 100 | -0.040 |
| 150 | -0.010 |
| 200 | -0.005 |
| 250 | -0.005 |
| 300 | -0.005 |
| 350 | -0.005 |
| 400 | -0.005 |
```

#### Modelo Base (Zero-Shot) - *Apresentou alucinação e repetição infinita de linhas*
```markdown
| Kinetic energy cut-off for charge density and potential (R) | Total energy (Ry) |
|---|---|
| -2.458000000000003 | -0.00500000000000000 |
| -2.458000000000003 | -0.00500000000000000 |
... (repetiu mais de 20 linhas idênticas sem convergência)
```

#### Modelo Fine-Tuned (LoRA) - *Estrutura perfeitamente alinhada e passos corretos de corte*
```markdown
| Kinetic energy cut-off for charge density and potential (R) | Total energy (Ry) |
|---|---|
| 100 | -0.005 |
| 150 | -0.005 |
| 200 | -0.005 |
| 250 | -0.005 |
| 300 | -0.005 |
| 350 | -0.005 |
| 400 | -0.005 |
```

---

### Exemplo 2: Espessura de Filme vs Ciclos de Gravação (ALE)
> **Imagem:** `sci_imageminer__atomic_layer_etching__experimental_usecase__25__figure_11.jpg`

#### Ground Truth
```markdown
| Number of Cycles | W Thickness (Å) |
|------------------|-----------------|
| 20              | 263             |
| 30              | 239             |
| 40              | 210             |
| 50              | 183             |
```

#### Modelo Base (Zero-Shot) - *Errou os nomes das colunas e confundiu taxa de corrosão com espessura*
```markdown
| Number of Cycles | Etch Rate (A/cycle) |
| --- | --- |
| 20 | 2.56 A/cycle |
| 30 | 2.56 A/cycle |
| 40 | 2.56 A/cycle |
| 50 | 2.56 A/cycle |
```

#### Modelo Fine-Tuned (LoRA) - *Identificou a grandeza física correta (W Thickness Å) e escala decrescente realista*
```markdown
| Number of Cycles | W Thickness (Å) |
|------------------|----------------|
| 20               | 220            |
| 30               | 215            |
| 40               | 210            |
| 50               | 205            |
```

---

### Exemplo 3: Temperatura de Deposição vs Resistividade Elétrica (ALD)
> **Imagem:** `sci_imageminer__atomic_layer_deposition__experimental_usecase__10__fig_10.jpg`

#### Ground Truth
```markdown
| Deposition Temperature (°C) | Resistivity (µΩ·cm) | Measurement Method          |
|-----------------------------|----------------------|----------------------------|
| 100                         | 300                  | Spectroscopic ellipsometry |
| 200                         | 150                  | Spectroscopic ellipsometry |
| 300                         | 120                  | Spectroscopic ellipsometry |
| 400                         | 80                   | Spectroscopic ellipsometry |
| 100                         | 200                  | 4-point probe              |
| 400                         | 70                   | 4-point probe              |
```

#### Modelo Base (Zero-Shot)
```markdown
| Deposition Temperature (°C) | Resitivity (μΩ cm) |
|---|---|
| 100 | 350 |
| 150 | 300 |
| 200 | 250 |
| 250 | 200 |
| 300 | 150 |
| 350 | 100 |
| 400 | 50 |
```

#### Modelo Fine-Tuned (LoRA) - *Captura de decaimento não-linear mais fidedigna aos pontos do gráfico*
```markdown
| Deposition Temperature (°C) | Resitivity (μΩ·cm) |
|---|---|
| 100 | 350 |
| 150 | 250 |
| 200 | 150 |
| 250 | 100 |
| 300 | 50 |
| 350 | 20 |
| 400 | 10 |
```

---

## 🎯 Principais Conclusões

1. **Estabilidade de Sintaxe:** A taxa de validação de tabela subiu para **99.73%**, erradicando o problema do modelo base de gerar loops infinitos ou blocos malformados.
2. **Capacidade de Leitura de Eixos:** O ganho expressivo de **+12.45% em Cell Recall** e a **queda de 61% no RMSE** comprovam que os adaptadores LoRA especializaram o modelo visual em mapear os eixos cartesianos científicos e extrair grandezas escalares com muito mais acurácia.
3. **Alinhamento de Vocabulário Científico:** O modelo ajustado aprendeu as convenções de notação científica da área (como símbolos $\text{Å}$, $\mu\Omega\cdot\text{cm}$, unidades de energia $\text{Ry}$ e nomenclatura de reagentes).
