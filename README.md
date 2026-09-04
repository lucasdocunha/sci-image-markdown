# 🔬 sci-image-markdown

Framework de pesquisa e engenharia para **Extração de Tabelas Estruturadas em Markdown a partir de Figuras e Gráficos Científicos** (*Sci-ImageMiner / ICDAR 2026 Task 2: Data Table Extraction*).

---

## 📌 Visão Geral

Figuras científicas de artigos revisados por pares (curvas de crescimento por ciclo de ALD, saturação de precursores químicos, corrosão atômica ALE, espectroscopia de elipsometria e espectros XPS/FTIR) contêm medições quantitativas fundamentais para a reprodutibilidade experimental. O **`sci-image-markdown`** fornece um pipeline completo ponta a ponta para:

1. **Ingestão e Estruturação**: Processamento do dataset oficial do Hugging Face descartando metadados de classificação desnecessários e padronizando pares imagem-tabela canônicos.
2. **Fine-Tuning de VLM de Fronteira**: Treinamento especializado do **`Qwen/Qwen2.5-VL-3B-Instruct`** com **QLoRA (4-bit NF4)**, Paged AdamW de 8 bits e otimização de fatiamento de vocabulário da `lm_head` que economiza mais de 90% de VRAM.
3. **Avaliação com Métricas de Domínio**: Avaliação rigorosa separando integridade sintática (**VTR**) de precisão numérica (**Cell Recall**, **Cell Precision**, **Cell F1**, **Cell RMSE**, **Cell RNE**, **TEDS** e sobreposição léxica **BLEU-4** / **ROUGE-L**).
4. **Inferência CLI e Exportação**: Extração direta de imagens para Markdown e CSV, e rotina de fusão dos adaptadores (*merge and unload*) para deployment independente.

---

## 🏗️ Arquitetura do Repositório

```
sci-image-markdown/
├── README.md                  # Este documento (visão geral do projeto em PT-BR)
├── detalhamento.ipynb         # Caderno interativo com plots reais e simulações
├── configs/                   # Configurações modulares em YAML
│   ├── default.yaml           # Parâmetros gerais, caminhos e hiperparâmetros
│   ├── models/                # Configurações de arquitetura (Qwen2.5-VL 3B/7B)
│   └── training/              # Configuração oficial QLoRA (4-bit NF4)
├── docs/                      # Central de documentação do projeto
│   ├── detalhamento.md        # Guia técnico exaustivo (arquitetura, loss, matemática, parâmetros)
│   ├── BENCHMARK_RESULTS.md   # Relatório oficial de benchmark e comparação SOTA
│   ├── CONTEXT.md             # Modelo de domínio, física ALD/ALE e glossário
│   ├── AGENTS.md              # Diretrizes de agentes autônomos
│   ├── adr/                   # Architecture Decision Records (ADRs)
│   └── agents/                # Especificações de labels e issue tracker
├── src/
│   ├── data/                  # Datasets PyTorch, pré-processadores e colatores multimodais
│   ├── models/                # Carregador com patch de memória da lm_head
│   ├── metrics/               # Parsers de tabela Markdown, métricas de células e avaliador
│   ├── training/              # Loop de treino SFT e callbacks
│   ├── inference/             # Preditor em imagem única e em lote
│   └── utils/                 # Utilitários de I/O, logging e mesclagem de configs
├── prepare_data.py            # CLI: download do dataset HF e geração de amostras sintéticas
├── train.py                   # CLI: ponto de entrada do treinamento QLoRA
├── evaluate.py                # CLI: avaliação de benchmark com relatórios estruturados
├── predict.py                 # CLI: extração de figura para Markdown e CSV
├── export.py                  # CLI: fusão de adaptadores LoRA nos pesos base
├── run_train.sh               # Script de execução de treino em background
├── run_eval_both.sh           # Script de benchmark comparativo (Zero-Shot vs Fine-Tuned)
├── run_export.sh              # Script de fusão e exportação do modelo consolidado
└── tests/                     # Suíte de testes automatizados (pytest)
```

---

## 📊 Censo Oficial do Dataset Científico

O repositório utiliza o dataset oficial do benchmark **Sci-ImageMiner (ICDAR 2026 Task 2)** baixado diretamente do Hugging Face (`SciKnowOrg/Sci-ImageMiner`), limpo para manter estritamente os dados tabulares essenciais:

| Split Oficial | Arquivo Local | Figuras com Tabela Numérica (Task 2) | Proporção |
| :--- | :--- | :---: | :---: |
| **Treino** | `data/processed/train.jsonl` | **723** | 60.25% |
| **Validação** | `data/processed/val.jsonl` | **104** | 8.67% |
| **Teste** | `data/processed/test.jsonl` | **373** | 31.08% |
| **Total Geral** | — | **1.200 figuras reais** | **100.0%** |

Cada registro contém o formato canônico enxuto:
```json
{
  "id": "sci_imageminer__atomic_layer_deposition__experimental_usecase__12__fig_10",
  "image": "images/sci_imageminer__atomic_layer_deposition__experimental_usecase__12__fig_10.jpg",
  "table": "| AB Cycles | Tungsten Film Thickness (Å) |\n|---|---|\n| 0 | 0 |\n| 10 | 25 |\n..."
}
```

---

## 🚀 Guia de Início Rápido

### 1. Instalação do Ambiente

Recomenda-se Python 3.10+ e ambiente virtual dedicado:

```bash
# Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar o pacote em modo editável com todas as dependências
pip install -e .
pip install matplotlib
```

### 2. Preparação e Download do Dataset

Para baixar e estruturar as **1.200 figuras reais** do Hugging Face:
```bash
.venv/bin/python prepare_data.py --download-hf --raw-dir data/raw --processed-dir data/processed --num-workers 16
```

Para gerar amostras sintéticas locais de verificação rápida:
```bash
.venv/bin/python prepare_data.py --create-synthetic --num-samples 20
```

### 3. Treinamento com 4-bit QLoRA

O modelo é carregado em quantização **4-bit NormalFloat (NF4)** com adaptadores LoRA acoplados nas projeções de atenção (`q_proj`, `v_proj`):

```bash
# Execução direta interativa no terminal
.venv/bin/python train.py --config configs/default.yaml

# Ou execução protegida em background com logs desacoplados
bash run_train.sh
tail -f train.log
```

### 4. Avaliação Comparativa de Benchmark

Para rodar a comparação entre o Modelo Base (Zero-Shot) e o Modelo Fine-Tuned nas **373 figuras de teste**:

```bash
bash run_eval_both.sh
```

Os resultados detalhados e deltas serão salvos em `outputs/eval_results/`.

### 5. Inferência em Imagem Única (CLI)

Para extrair os dados de uma figura científica qualquer para tabelas Markdown e CSV:

```bash
.venv/bin/python predict.py \
    --image data/processed/images/sci_imageminer__atomic_layer_deposition__experimental_usecase__12__fig_10.jpg \
    --adapter-path outputs/checkpoints/final_adapter \
    --output-md tabela_extraida.md \
    --output-csv tabela_extraida.csv
```

### 6. Fusão e Exportação dos Pesos (*Merge & Unload*)

Funde as matrizes $B \cdot A$ do LoRA diretamente nos pesos base congelados $W_0$, gerando um modelo consolidado pronto para servir:

```bash
bash run_export.sh
```
O modelo final é gravado em `outputs/merged_model` e pode ser carregado nativamente com `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)`.

---

## 📈 Resultados de Benchmark & Comparação com SOTA

Avaliação executada sobre as **373 figuras de teste independentes** do benchmark ICDAR 2026 Task 2:

| Métrica | Descrição | Modelo Base (Zero-Shot) | Fine-Tuned (QLoRA 3B) | SOTA ICDAR (VLMinators 7B) | Variação (Delta) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Score Final ICDAR** | Média $\frac{1}{2}(\text{RMS} + \text{TEDS})$ | 31.08 | **`40.50`** | `40.80` | **+9.42 pts** 🎯 |
| **Valid Table Rate (VTR)** | Validade sintática do Markdown | 88.74% | **`99.73%`** | ~98.8% | **+10.99%** 🚀 |
| **Cell Recall ($\le 5\%$ tol.)** | Captura de pontos numéricos | 35.78% | **`48.22%`** | ~42.0% - 49.0% | **+12.44%** 🚀 |
| **Cell RMSE** | Raiz do Erro Quadrático Médio | 2.6635 | **`1.0308`** | ~1.10 - 2.50 | **-61.3% de erro** 🎯 |
| **Cell RNE** | Erro Relativo Normalizado Médio | 0.0044 | **`0.0036`** | ~0.0040 - 0.0070 | **-17.4%** 🎯 |
| **Consumo de VRAM** | Treinamento e Forward Pass | — | **< 7 GB VRAM** | 16 GB - 24 GB | **Viável em GPU de 8GB** 🔒 |

👉 Consulte o relatório completo com análise qualitativa e detalhes dos competidores em [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md).

---

## 📚 Documentação Técnica Completa

Toda a documentação em Markdown está organizada na pasta [`docs/`](docs/):

- **[`docs/detalhamento.md`](docs/detalhamento.md)**: Detalhamento exaustivo da arquitetura do Qwen2.5-VL, formulação matemática do LoRA e QLoRA (NF4, Double Quantization, Paged AdamW), mecânica de loss autoregressiva, dicionário completo de parâmetros e formulação das métricas.
- **[`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md)**: Análise completa do benchmark oficial, comparação com VLMinators, TeleOCR, DePlot, MatCha e modelos comerciais (GPT-4o, Claude 3.5 Sonnet).
- **[`docs/CONTEXT.md`](docs/CONTEXT.md)**: Modelo conceitual de domínio físico-químico (ALD/ALE, precursores, filmes finos, espectroscopia) e glossário de termos.
- **[`detalhamento.ipynb`](detalhamento.ipynb)**: Caderno Jupyter executável e interativo, com visualizações gráficas geradas em Matplotlib de amostras reais do dataset, simulações matriciais no PyTorch e avaliação prática de métricas.

---

## 🧪 Testes Automatizados

Para executar toda a suíte de testes unitários e de integração:

```bash
.venv/bin/pytest -v
```

Todas as 13 suítes de testes cobrem:
- Carregamento e mesclagem de configurações YAML.
- Criação e integridade de datasets e resolução de imagens.
- Parser de blocos Markdown e conversão bidirecional com Pandas DataFrame.
- Cálculo de métricas: Levenshtein normalizado, casamento bipartido de células numéricas, RMSE, RNE e avaliador ponta a ponta.
