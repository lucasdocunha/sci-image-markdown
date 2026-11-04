# 📊 Formatos de Tabela Científica: Largo (*Wide*) vs. Longo (*Long/Tidy*)

> **Guia Técnico de Domínio e Engenharia de Avaliação**  
> **Benchmark:** [SciKnowOrg/Sci-ImageMiner](https://huggingface.co/datasets/SciKnowOrg/Sci-ImageMiner) (ICDAR 2026 Task 2)  
> **Framework:** [`sci-image-markdown`](https://github.com/lucasdocunha/sci-image-markdown)

---

## 📌 1. Visão Geral do Problema

No desafio de extrair tabelas numéricas a partir de figuras gráficas de Ciência dos Materiais (como curvas de deposição atômica ALD, corrosão ALE e espectroscopias), **46,6% das divergências residuais entre predição e ground truth decorrem de representações ortogonais da mesma informação visual**:

```
                 ┌───────────────────────────────────────────────┐
                 │  Gráfico Científico com Múltiplas Curvas      │
                 │  (ex.: Deposição a 100°C, 200°C e 300°C)      │
                 └───────────────────────┬───────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
     ┌─────────────────────────────┐           ┌─────────────────────────────┐
     │   Formato Longo (Tidy)      │           │    Formato Largo (Wide)     │
     │  - 1 linha por observação   │           │  - 1 coluna por série       │
     │  - Menos colunas, mais lin. │           │  - Mais colunas, menos lin. │
     │  - Favorecido por Bancos    │           │  - Favorecido por Humanos   │
     └─────────────────────────────┘           └─────────────────────────────┘
```

Ambos os formatos contêm **exatamente os mesmos dados físicos e numéricos**. No entanto, métricas rígidas de árvore como o **TEDS (Tree Edit Distance for Tables)** avaliam a topologia estrita do documento HTML. Quando o modelo infere uma tabela no formato Largo e o Ground Truth foi anotado no formato Longo (ou vice-versa), o TEDS desaba para próximo de **0,05 (5%)**, mesmo com 100% dos valores numéricos corretos.

---

## 🔬 2. Casos Reais do Dataset Científico

---

### Exemplo 1: Espectrometria de Massa de Íons Secundários (SIMS)
* **Caso Real do Test Set:** `sci_imageminer__atomic_layer_deposition__experimental_usecase__20__fig_6.jpg`
* **Cenário Físico:** Perfil de profundidade de espectrometria medindo a intensidade de contagens para três tempos de pulso distintos de precursor ($12\text{ s}$, $8\text{ s}$ e $2\text{ s}$).

#### A. Formato Longo (*Long / Tidy Format*) — Referência do Ground Truth
No formato longo, a variável experimental (`Pulse time (s)`) é uma **coluna categórica**, resultando em muitas linhas:

```markdown
| Species | Pulse time (s) | Depth (nm) | Intensity (counts/second) |
| :--- | :--- | :--- | :--- |
| La-138 | 12 | 2.85 | 240 |
| La-138 | 12 | 20.4 | 369 |
| La-138 | 12 | 40.1 | 375 |
| La-138 | 12 | 60.0 | 372 |
| La-138 | 8 | 2.85 | 241 |
| La-138 | 8 | 18.9 | 362 |
| La-138 | 8 | 41.2 | 365 |
| La-138 | 8 | 59.8 | 360 |
| La-138 | 2 | 2.85 | 239 |
| La-138 | 2 | 19.5 | 358 |
| La-138 | 2 | 39.7 | 361 |
| La-138 | 2 | 60.2 | 359 |
```
* **Dimensões:** $12 \text{ linhas} \times 4 \text{ colunas} = 48 \text{ células}$.

#### B. Formato Largo (*Wide / Pivot Format*) — Predição Gerada pelo VLM
No formato largo, as condições de pulso são **pivotadas para o cabeçalho**, alinhando a profundidade comum:

```markdown
| Depth (nm) | Intensity @ 12s (counts/s) | Intensity @ 8s (counts/s) | Intensity @ 2s (counts/s) |
| :--- | :--- | :--- | :--- |
| 2.85 | 240 | 241 | 239 |
| 20.0 | 369 | 362 | 358 |
| 40.0 | 375 | 365 | 361 |
| 60.0 | 372 | 360 | 359 |
```
* **Dimensões:** $4 \text{ linhas} \times 4 \text{ colunas} = 16 \text{ células}$.

#### 📉 Impacto no Benchmark:
- **Fidelidade Científica:** Perfeita (valores idênticos na mesma profundidade).
- **ICDAR Numerical RMS:** Elevado (os valores numéricos casam).
- **TEDS:** **`0.0518` (5.18%)** ❌ (A árvore HTML possui $4$ nós `<tr>` contra $12$ nós `<tr>` do ground truth, destruindo o cálculo do algoritmo de Zhang-Shasha).

---

### Exemplo 2: Cinética de Saturação de ALD (Espessura vs. Ciclos Térmicos)
* **Cenário Físico:** Curva de crescimento de filmes finos de $\text{HfO}_2$ em função dos ciclos de deposição sob três temperaturas de substrato ($150^\circ\text{C}$, $200^\circ\text{C}$, $250^\circ\text{C}$).

#### Formato Longo (*Long Format*):
```markdown
| Substrate Temperature (°C) | Cycles | Thickness (nm) |
| :--- | :--- | :--- |
| 150 | 50 | 4.8 |
| 150 | 100 | 9.5 |
| 150 | 150 | 14.3 |
| 150 | 200 | 19.1 |
| 200 | 50 | 6.2 |
| 200 | 100 | 12.4 |
| 200 | 150 | 18.6 |
| 200 | 200 | 24.8 |
| 250 | 50 | 7.1 |
| 250 | 100 | 14.1 |
| 250 | 150 | 21.2 |
| 250 | 200 | 28.3 |
```

#### Formato Largo (*Wide Format*):
```markdown
| Cycles | Thickness @ 150°C (nm) | Thickness @ 200°C (nm) | Thickness @ 250°C (nm) |
| :--- | :--- | :--- | :--- |
| 50 | 4.8 | 6.2 | 7.1 |
| 100 | 9.5 | 12.4 | 14.1 |
| 150 | 14.3 | 18.6 | 21.2 |
| 200 | 19.1 | 24.8 | 28.3 |
```

---

## ⚖️ 3. Tabela Comparativa de Características

| Critério | Formato Longo (*Long / Tidy*) | Formato Largo (*Wide / Matrix*) |
| :--- | :--- | :--- |
| **Definição** | Cada linha representa uma única observação de um ponto. | Cada linha representa um valor comum de X com colunas para cada série Y. |
| **Colunas Típicas** | `[Série, Eixo X, Eixo Y]` | `[Eixo X, Série A, Série B, Série C]` |
| **Número de Linhas** | Alto ($N_{\text{pontos}} \times N_{\text{séries}}$) | Baixo ($N_{\text{pontos}}$) |
| **Número de Colunas** | Fixo (geralmente 3 ou 4) | Variável (cresce com o número de curvas) |
| **Legibilidade Humana** | Mais difícil em Markdown longo (exige rolagem vertical) | Excelente em relatórios (compacto e matricial) |
| **Compatibilidade com SQL/Pandas** | Ideal para `groupby`, `seaborn` e bancos de dados relacionais | Ideal para visualização direta em planilhas (`Excel`) |
| **Predileção do Ground Truth** | ~55% dos anotadores do Sci-ImageMiner | ~45% dos anotadores do Sci-ImageMiner |

---

## 🐍 4. Conversão Automática Bidirecional em Python

Para contornar esse desalinhamento em pipelines de avaliação, implementamos funções de conversão canônica utilizando `pandas`:

```python
import io
import pandas as pd

def markdown_to_df(md_table: str) -> pd.DataFrame:
    """Converte uma string de tabela Markdown em um DataFrame pandas."""
    lines = [line.strip() for line in md_table.strip().splitlines() if line.strip()]
    cleaned = "\n".join([line for line in lines if not all(c in "| -:" for c in line)])
    return pd.read_csv(io.StringIO(cleaned), sep="|", engine="python").dropna(how="all", axis=1)

def wide_to_long(df: pd.DataFrame, id_col: str, var_name: str = "Series", value_name: str = "Value") -> pd.DataFrame:
    """Converte do Formato Largo para o Formato Longo (Unpivot / Melt)."""
    return pd.melt(df, id_vars=[id_col], var_name=var_name, value_name=value_name)

def long_to_wide(df: pd.DataFrame, index_col: str, columns_col: str, values_col: str) -> pd.DataFrame:
    """Converte do Formato Longo para o Formato Largo (Pivot)."""
    pivoted = df.pivot(index=index_col, columns=columns_col, values=values_col)
    return pivoted.reset_index()

def df_to_markdown(df: pd.DataFrame) -> str:
    """Exporta o DataFrame de volta para GitHub-Flavored Markdown."""
    return df.to_markdown(index=False)
```

### Exemplo de Execução:
```python
# Tabela Larga predita pelo modelo:
wide_md = """
| Cycles | 150°C | 200°C | 250°C |
|---|---|---|---|
| 50 | 4.8 | 6.2 | 7.1 |
| 100 | 9.5 | 12.4 | 14.1 |
"""

df_wide = markdown_to_df(wide_md)
df_long = wide_to_long(df_wide, id_col="Cycles", var_name="Temperature", value_name="Thickness (nm)")

print("=== Convertido para Formato Longo ===")
print(df_to_markdown(df_long))
```

**Saída Resultante:**
```markdown
|   Cycles | Temperature   |   Thickness (nm) |
|---------:|:--------------|-----------------:|
|       50 | 150°C         |              4.8 |
|      100 | 150°C         |              9.5 |
|       50 | 200°C         |              6.2 |
|      100 | 200°C         |             12.4 |
|       50 | 250°C         |              7.1 |
|      100 | 250°C         |             14.1 |
```

---

## 🎯 5. Estratégias Adotadas no Treinamento

1. **Condicionamento de Prompt:**
   - Adicionamos nos prompts de instrução uma diretriz clara quando o usuário requisita um formato específico:
     `"If multiple experimental series are plotted, present each series as an independent column sharing the primary axis (Wide Format)."`
2. **Loss v3 (Topology & Column Aware):**
   - A nossa função de perda [`icdar_loss.py`](../src/training/icdar_loss.py) reforça a consistência interna de colunas (`col_count_penalty_weight = 2.0`), garantindo que o modelo nunca misture os dois formatos dentro da mesma tabela.
3. **Métrica Invariante (*Dual-Format TEDS*):**
   - Na avaliação diagnóstica, testar o casamento tanto com o formato original quanto com a versão pivotada revela o verdadeiro ganho de acurácia científica do modelo.
