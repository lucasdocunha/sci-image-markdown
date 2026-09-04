# 🔬 sci-image-markdown

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--3B-blueviolet.svg?logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![PEFT](https://img.shields.io/badge/Fine--Tuning-4--bit%20QLoRA%20(NF4)-success.svg)](https://github.com/huggingface/peft)
[![Benchmark](https://img.shields.io/badge/Benchmark-ICDAR%202026%20Task%202-orange.svg)](https://arxiv.org/abs/2607.26848)
[![Tests](https://img.shields.io/badge/Tests-13%2F13%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Framework de Pesquisa e Engenharia para Extração de Tabelas Numéricas a partir de Figuras Científicas em Markdown Estruturado.**  
*Avaliado no benchmark oficial ICDAR 2026 / Sci-ImageMiner (Task 2: Scientific Figure to Data Table Extraction).*

[📘 Detalhamento Técnico Completo](docs/detalhamento.md) • [📊 Resultados de Benchmark](docs/BENCHMARK_RESULTS.md) • [📓 Caderno Interativo (Jupyter)](detalhamento.ipynb) • [🌐 Modelo de Domínio](docs/CONTEXT.md)

</div>

---

## 📌 Sumário
- [1. Visão Geral e Contexto de Domínio](#1-visão-geral-e-contexto-de-domínio)
- [2. Arquitetura Multimodal (Qwen2.5-VL)](#2-arquitetura-multimodal-qwen25-vl)
- [3. Fundamentação Matemática: LoRA e QLoRA](#3-fundamentação-matemática-lora-e-qlora)
- [4. Mecânica da Função de Perda (Loss) & Otimização de VRAM](#4-mecânica-da-função-de-perda-loss--otimização-de-vram)
- [5. Censo Oficial do Dataset Científico (1.200 Figuras)](#5-censo-oficial-do-dataset-científico-1200-figuras)
- [6. Dicionário de Hiperparâmetros Chave](#6-dicionário-de-hiperparâmetros-chave)
- [7. Métricas de Avaliação Científica de Tabelas](#7-métricas-de-avaliação-científica-de-tabelas)
- [8. Resultados de Benchmark & Comparação com SOTA](#8-resultados-de-benchmark--comparação-com-sota)
- [9. Guia de Início Rápido](#9-guia-de-início-rápido)
- [10. Estrutura de Diretórios](#10-estrutura-de-diretórios)
- [11. Testes Automatizados](#11-testes-automatizados)

---

## 1. Visão Geral e Contexto de Domínio

Em artigos científicos de Ciência dos Materiais e Engenharia de Semicondutores, os dados experimentais não estão armazenados em arquivos brutos, mas incorporados visualmente em **figuras gráficas densas**:
- **ALD (Atomic Layer Deposition):** Curvas de saturação de precursores químicos e janelas térmicas de crescimento atômico ($\text{Å/ciclo}$).
- **ALE (Atomic Layer Etching):** Cinética de corrosão atômica de filmes finos metálicos (ex.: Tungstênio $\text{W}$, $\text{HfO}_2$, $\text{Al}_2\text{O}_3$).
- **Espectroscopias Ópticas e Químicas:** Elipsometria espectroscópica, espectros de absorção FTIR ($\text{cm}^{-1}$) e XPS.

O objetivo do framework **`sci-image-markdown`** é alimentar um modelo de visão e linguagem (**VLM**) exclusivamente com a imagem da figura científica e instruí-lo a gerar a tabela numérica subjacente em formato **GitHub-Flavored Markdown estruturado**.

---

## 2. Arquitetura Multimodal (Qwen2.5-VL)

O projeto é centralizado no modelo de fronteira **`Qwen/Qwen2.5-VL-3B-Instruct`**:

```
       [ Imagem Científica ]                         [ Prompt de Instrução ]
                 │                                            │
                 ▼                                            ▼
     ┌───────────────────────┐                    ┌───────────────────────┐
     │  Vision Encoder (ViT) │                    │   Tokenizer BPE       │
     │  - Resolução Dinâmica │                    │   (Vocab: 152.064)    │
     │  - 2D-RoPE Posicional │                    └───────────┬───────────┘
     └───────────┬───────────┘                                │
                 ▼                                            │
     ┌───────────────────────┐                                │
     │ Spatial Merge / MLP   │                                │
     │ (Compressão 2x2 -> 1) │                                │
     └───────────┬───────────┘                                │
                 │                                            │
                 └──────────────► [ Concatenação ] ◄──────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │      LLM Causal Backbone      │
                         │    - SwiGLU / RMSNorm         │
                         │    - Grouped Query Attention  │
                         │    - Adaptadores LoRA (q, v)  │
                         │    - Pesos Base em 4-bit NF4  │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │   Loss Causal / lm_head       │
                         │   - Fatiamento de Memória     │
                         │   - Cross-Entropy Mascarada   │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                            [ Tabela Markdown Gerada ]
```

### Inovações Multimodais Relevantes:
1. **ViT com Resolução Dinâmica (estilo NaViT):** Ao contrário de arquiteturas legadas que redimensionam forçadamente imagens para grades quadradas fixas (ex.: $336 \times 336$) distorcendo as escalas dos eixos, o Qwen2.5-VL fatia a imagem em patches adaptativos preservando a proporção (*aspect ratio*) nativa.
2. **2D-RoPE (Rotary Position Embeddings):** Codifica posições bidimensionais $(x, y)$, permitindo que o modelo ancore coordenadas físicas de pontos de dispersão aos valores numéricos dos eixos.
3. **Vocabulário BPE Expandido (152.064 tokens):** Representação otimizada para delimitadores de Markdown (`|`, `-`), expoentes, notações de potência ($10^{-3}$) e unidades químicas/físicas ($\text{Å}$, $\mu\Omega\cdot\text{cm}$, $\text{cm}^{-1}$).

---

## 3. Fundamentação Matemática: LoRA e QLoRA

### 3.1 Orçamento de Memória: Full Fine-Tuning vs. QLoRA
Para ajustar um modelo de 3 Bilhões de parâmetros por *Full Fine-Tuning*, o custo de memória por parâmetro é de **16 a 18 bytes** (pesos FP16 + gradientes FP16 + estados AdamW FP32 + cópia mestre):
$$\text{Memória} \approx 3 \times 10^9 \times 16 \text{ bytes} \approx 48 \text{ GB}$$
Isso inviabiliza o treino em GPUs comerciais de 8 GB a 16 GB.

### 3.2 Formulação do LoRA
O LoRA congela a matriz de pesos pré-treinada $W_0 \in \mathbb{R}^{d \times k}$ e injeta duas matrizes de posto reduzido $r \ll \min(d, k)$:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} \cdot B \cdot A$$
- $A \in \mathbb{R}^{r \times k} \sim \mathcal{N}(0, \sigma^2)$
- $B \in \mathbb{R}^{d \times r} = 0 \implies \Delta W = 0 \text{ no início do treinamento.}$
- Rank $r=16$, Alpha $\alpha=32$ ($\text{scaling} = \alpha/r = 2.0$).
- Projeções adaptadas: `["q_proj", "v_proj"]`.

### 3.3 QLoRA (4-bit NormalFloat & Paged AdamW)
1. **NF4 (NormalFloat 4-bit):** Quantização teórica ótima com quantis de áreas equivalentes sob uma distribuição normal $\mathcal{N}(0, \sigma^2)$.
2. **Double Quantization (DQ):** Quantiza as próprias constantes de escala de FP32 para 8 bits, economizando $0.37 \text{ bits/parâmetro}$.
3. **Paged AdamW (8-bit):** Migração dinâmica de estados do otimizador para RAM via CUDA Paging durante picos transitórios de memória, eliminando erros de *CUDA Out-Of-Memory*.
4. **Pegada de Memória:** O modelo roda com **menos de 7 GB de VRAM**, viabilizando treinamento e inferência em GPUs como RTX 3060, RTX 4060 ou T4.

---

## 4. Mecânica da Função de Perda (Loss) & Otimização de VRAM

### 4.1 Cross-Entropy Mascarada Autoregressiva
Dada a sequência multimodal $X = (x_1, \dots, x_M, x_{M+1}, \dots, x_N)$, onde $x_1 \dots x_M$ correspondem aos tokens de imagem e instrução, e $x_{M+1} \dots x_N$ aos tokens da tabela alvo:
$$\mathcal{L} = - \frac{1}{|T|} \sum_{t \in T} \log P(x_t \mid x_{<t})$$
- **Máscara de Contexto (`label = -100`):** Posições de imagem e prompt recebem o valor $-100$, sendo ignoradas nativamente pelo PyTorch (`ignore_index=-100`).
- **Shift Temporal:** Alinhamento dos logits previstos em $t$ com os rótulos alvos em $t+1$.

### 4.2 Otimização Crítica: Fatiamento de Vocabulário na `lm_head`
No Qwen2.5-VL, a camada linear de saída `lm_head` projeta o estado oculto ($D=2048$) para o vocabulário ($V=152.064$):
- **Alocação Padrão Ingênua (2048 tokens):**
  $$1 \times 2048 \times 152.064 \times 4 \text{ bytes (FP32)} \approx \mathbf{1.246 \text{ GB de VRAM}}$$
- **Nossa Implementação com Fatiamento (`_memory_efficient_qwen2_vl_forward`):**
  Filtramos os estados ocultos **antes** da camada `lm_head`, projetando estritamente os tokens onde `label != -100` (tipicamente ~180 tokens de tabela):
  $$1 \times 180 \times 152.064 \times 4 \text{ bytes (FP32)} \approx \mathbf{0.109 \text{ GB de VRAM}} \quad \implies \mathbf{91.2\% \text{ de Economia!}}$$

---

## 5. Censo Oficial do Dataset Científico (1.200 Figuras)

O dataset do benchmark oficial **Sci-ImageMiner (ICDAR 2026 Task 2)** foi baixado na íntegra e limpo para descartar metadados irrelevantes herdados de classificação (`classification`, `subdomain`, `caption`, `figure_number`):

| Split Oficial | Arquivo Local | Figuras com Tabela Numérica (Task 2) | Proporção |
| :--- | :--- | :---: | :---: |
| **Treino** | `data/processed/train.jsonl` | **723** | 60.25% |
| **Validação** | `data/processed/val.jsonl` | **104** | 8.67% |
| **Teste** | `data/processed/test.jsonl` | **373** | 31.08% |
| **Total Geral** | — | **1.200 figuras reais** | **100.0%** |

Cada linha é estritamente formatada no esquema canônico enxuto:
```json
{
  "id": "sci_imageminer__atomic_layer_deposition__experimental_usecase__12__fig_10",
  "image": "images/sci_imageminer__atomic_layer_deposition__experimental_usecase__12__fig_10.jpg",
  "table": "| AB Cycles | Tungsten Film Thickness (Å) |\n|---|---|\n| 0 | 0 |\n| 10 | 25 |\n| 20 | 50 |\n| 30 | 75 |\n| 40 | 100 |\n| 50 | 125 |\n| 60 | 150 |\n| 70 | 175 |\n| 80 | 200 |"
}
```

---

## 6. Dicionário de Hiperparâmetros Chave

| Parâmetro | Valor Padrão | Explicação Técnica |
| :--- | :---: | :--- |
| `name_or_path` | `Qwen/Qwen2.5-VL-3B-Instruct` | VLM base pré-treinado nativamente multimodal. |
| `load_in_4bit` | `true` | Quantização de pesos base congelados em 4-bit NormalFloat (NF4). |
| `r` / `lora_alpha` | `16` / `32` | Posto e fator multiplicativo de escala do adaptador LoRA ($\text{scale} = 2.0$). |
| `target_modules` | `["q_proj", "v_proj"]` | Módulos de atenção que recebem matrizes LoRA acopladas. |
| `learning_rate` | `2.0e-4` | Taxa de aprendizado inicial com decaimento cossenoidal (`cosine`). |
| `per_device_train_batch_size` | `1` | Mini-batch unitário para acomodar resoluções dinâmicas sem padding redundante. |
| `gradient_accumulation_steps`| `8` | Acumula gradientes em 8 iterações (batch size efetivo = 8). |
| `gradient_checkpointing` | `true` | Recalcula ativações no backward pass (economiza ~60% de memória de ativação). |
| `optim` | `paged_adamw_8bit` | Otimizador AdamW quantizado com paginação de memória para CPU. |
| `temperature` | `0.0` | Decodificação determinística gulosa (*greedy search*), eliminando alucinações. |

---

## 7. Métricas de Avaliação Científica de Tabelas

A biblioteca em [`src/metrics/`](src/metrics/) avalia a extração em duas frentes complementares:

### 1. Integridade Estrutural e Sintática
- **Taxa de Tabelas Válidas (VTR):** Mede se a saída gerada possui delimitadores válidos (`|`, `-`), cabeçalhos e linhas que podem ser parseadas em um `pandas.DataFrame`.
- **Similaridade de Edição Normalizada (TEDS / Edit Sim):** Distância de Levenshtein normalizada:
  $$\text{Edit Sim} = 1 - \frac{\text{Levenshtein}(S_{\text{pred}}, S_{\text{true}})}{\max(|S_{\text{pred}}|, |S_{\text{true}}|)}$$

### 2. Fidelidade Numérica por Casamento Bipartido
Os números extraídos da tabela predita e da tabela real são submetidos a um casamento bipartido guloso com tolerância relativa máxima de **$5\%$** ($\text{rel\_tol} = 0.05$):
$$\text{Match se: } \text{RNE}(p, t) = \frac{|p - t|}{\max(|t|, 10^{-7})} \le 0.05$$
- **Cell Precision:** $TP / |\text{Pred}|$
- **Cell Recall:** $TP / |\text{Target}|$
- **Cell F1:** $2 \cdot \frac{P \cdot R}{P + R}$
- **Cell RMSE:** Raiz do erro quadrático médio dos pontos casados: $\sqrt{\frac{1}{K}\sum_{i=1}^K (p_i - t_i)^2}$
- **Cell RNE Médio:** Desvio relativo percentual médio dos pontos casados.

---

## 8. Resultados de Benchmark & Comparação com SOTA

Avaliação executada sobre as **373 figuras de teste independentes** do benchmark ICDAR 2026 Task 2:

| Posição / Modelo | Parâmetros | VTR (Tabelas Válidas) | Cell Recall ($\le 5\%$ tol.) | Cell RMSE | Score Final ICDAR | Requisito de Hardware |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **TeleOCR-VL** | Multi-VLM | ~99.0% | ~49.5% | ~1.05 | **41.81** | Cluster Multi-GPU |
| 🥈 **VLMinators** | ~7B | ~98.8% | ~46.8% | ~1.15 | **40.80** | 16-24 GB VRAM |
| 🚀 **sci-image-markdown (Ours)** | **~3B** | **`99.73%`** | **`48.22%`** | **`1.0308`** | **`40.50`** 🎯 | **< 7 GB VRAM (GPU única)** |
| 🥉 **Ricoh_SRCB** | ~7B-14B | ~98.0% | ~44.1% | ~1.30 | **38.67** | Multi-GPU |
| 📌 **Baseline Oficial ICDAR** | ~8B | ~92.5% | ~40.2% | ~1.85 | **35.97** | ~16 GB VRAM |
| ⚠️ **Modelo Base (Zero-Shot)** | ~3B | 88.74% | 35.78% | 2.6635 | **31.08** | < 8 GB VRAM |

👉 Consulte o relatório completo com análise qualitativa e detalhes dos competidores em [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md).

---

## 9. Guia de Início Rápido

### 1. Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install matplotlib
```

### 2. Download do Dataset Completo (1.200 Imagens)

```bash
.venv/bin/python prepare_data.py --download-hf --raw-dir data/raw --processed-dir data/processed --num-workers 16
```

### 3. Treinamento QLoRA

```bash
# Execução direta interativa:
.venv/bin/python train.py --config configs/default.yaml

# Execução protegida em background com logs desvinculados:
bash run_train.sh
tail -f train.log
```

### 4. Avaliação Comparativa de Benchmark

```bash
bash run_eval_both.sh
```
Os relatórios estruturados são salvos em `outputs/eval_results/base_model/` e `outputs/eval_results/finetuned_model/`.

### 5. Inferência em Imagem Única (CLI)

```bash
.venv/bin/python predict.py \
    --image data/processed/images/sci_imageminer__atomic_layer_deposition__experimental_usecase__12__fig_10.jpg \
    --adapter-path outputs/checkpoints/final_adapter \
    --output-md tabela_extraida.md \
    --output-csv tabela_extraida.csv
```

### 6. Fusão e Exportação dos Pesos (*Merge & Unload*)

```bash
bash run_export.sh
```
Gera um modelo consolidado e desacoplado em `outputs/merged_model/` para servir com `AutoModelForImageTextToText`.

---

## 10. Estrutura de Diretórios

```
sci-image-markdown/
├── README.md                  # Apresentação geral do repositório (PT-BR)
├── detalhamento.ipynb         # Caderno interativo com plots reais (Base64) e simulações
├── configs/                   # Configurações modulares (default.yaml, qlora.yaml, etc.)
├── docs/                      # Central de documentação
│   ├── detalhamento.md        # Dissecação técnica profunda (loss, QLoRA, arquitetura)
│   ├── BENCHMARK_RESULTS.md   # Relatório oficial de benchmark e SOTA
│   ├── CONTEXT.md             # Glossário e contexto físico-químico ALD/ALE
│   ├── AGENTS.md              # Workflows de agentes autônomos
│   ├── adr/                   # Architecture Decision Records
│   └── agents/                # Especificações de labels e triage
├── src/                       # Código-fonte modular (data, models, metrics, training, inference)
├── prepare_data.py            # CLI: download e processamento de dados
├── train.py                   # CLI: treinamento QLoRA
├── evaluate.py                # CLI: avaliação de benchmark
├── predict.py                 # CLI: inferência em imagem única
├── export.py                  # CLI: fusão de pesos
├── run_train.sh               # Script bash de treinamento em background
├── run_eval_both.sh           # Script bash de avaliação comparativa
├── run_export.sh              # Script bash de fusão de pesos
└── tests/                     # 13 testes unitários e de integração
```

---

## 11. Testes Automatizados

Execute a suíte com o pytest:

```bash
.venv/bin/pytest -v
```

```text
============================= test session starts ==============================
tests/test_config.py::test_load_default_config PASSED                    [  7%]
tests/test_config.py::test_merge_configs PASSED                          [ 15%]
tests/test_dataset.py::test_synthetic_data_generation_and_loading PASSED [ 23%]
tests/test_metrics.py::test_normalized_edit_distance PASSED              [ 30%]
tests/test_metrics.py::test_extract_numerical_cells PASSED               [ 38%]
tests/test_metrics.py::test_compute_numerical_cell_metrics_perfect_match PASSED [ 46%]
tests/test_metrics.py::test_compute_numerical_cell_metrics_partial_match PASSED [ 53%]
tests/test_metrics.py::test_table_extraction_evaluator PASSED            [ 61%]
tests/test_table_parser.py::test_extract_markdown_table_block_raw PASSED [ 69%]
tests/test_table_parser.py::test_extract_markdown_table_block_fenced PASSED [ 76%]
tests/test_table_parser.py::test_parse_markdown_to_dataframe_valid PASSED [ 84%]
tests/test_table_parser.py::test_parse_markdown_to_dataframe_invalid PASSED [ 92%]
tests/test_table_parser.py::test_dataframe_to_markdown_roundtrip PASSED  [100%]
============================== 13 passed in 2.65s ==============================
```
