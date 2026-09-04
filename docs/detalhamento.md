# 📘 Detalhamento Técnico: Framework `sci-image-markdown`

Este documento apresenta uma dissecação técnica profunda e exaustiva da arquitetura, formulação matemática, estratégia de quantização, cálculo de perda (*loss*), dicionário de parâmetros e rotinas operacionais implementadas no repositório `sci-image-markdown` para a tarefa de **Extração de Tabelas de Figuras Científicas em Markdown Estruturado** (*Sci-ImageMiner / ICDAR 2026 Task 2*).

---

## 📑 Sumário

1. [Arquitetura do Modelo (Qwen2.5-VL)](#1-arquitetura-do-modelo-qwen25-vl)
2. [Fundamentação Matemática: LoRA e QLoRA](#2-fundamentação-matemática-lora-e-qlora)
3. [Mecânica da Função de Perda (Loss) e Otimização de Memória](#3-mecânica-da-função-de-perda-loss-e-otimização-de-memória)
4. [Dicionário Completo de Argumentos e Parâmetros](#4-dicionário-completo-de-argumentos-e-parâmetros)
5. [Pipeline de Dados, Pré-Processamento e Collação](#5-pipeline-de-dados-pré-processamento-e-collação)
6. [Métricas de Avaliação Científica de Tabelas](#6-métricas-de-avaliação-científica-de-tabelas)
7. [Guia Operacional: Como Treinar, Avaliar e Exportar](#7-guia-operacional-como-treinar-avaliar-e-exportar)

---

## 1. Arquitetura do Modelo (Qwen2.5-VL)

O modelo base utilizado é o **`Qwen/Qwen2.5-VL-3B-Instruct`**, um Vision-Language Model (VLM) de fronteira desenhado para processamento multimodal nativo de alta resolução.

```
       [ Imagem Científica ]                         [ Texto do Prompt ]
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
     │ (Redução de Tokens)   │                                │
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

### 1.1 Componentes Principais

1. **Vision Encoder (ViT com Resolução Dinâmica - NaViT style):**
   - Ao contrário de VLMs legados (como LLaVA 1.5 ou CLIP) que redimensionam forçadamente a imagem para uma grade quadrada fixa (ex.: $336 \times 336$ ou $448 \times 448$) distorcendo a proporção dos eixos cartesianos, o Qwen2.5-VL divide a imagem em patches adaptativos de $14 \times 14$ mantendo o *aspect ratio* original.
   - Utiliza **2D Rotary Position Embeddings (2D-RoPE)** para codificar a posição espacial bidimensional $(x, y)$ de cada patch, essencial para figuras científicas onde a coordenada física de um ponto no gráfico mapeia diretamente para o valor da tabela.

2. **Vision-to-Language Merger (Adapter):**
   - Agrupa blocos espaciais de patches (geralmente $2 \times 2$) comprimindo 4 tokens visuais em 1 token multimodal, reduzindo a complexidade de atenção quadrática da sequência.

3. **Causal Language Model (LLM Backbone):**
   - Arquitetura baseada em Transformer decodificador autoregressivo com ativação **SwiGLU**, normalização **RMSNorm** e **Grouped-Query Attention (GQA)**.
   - Vocabulário amplo com **152.064 tokens**, otimizado para lidar nativamente com código, Markdown, números científicos e fórmulas químicas/matemáticas.

---

## 2. Fundamentação Matemática: LoRA e QLoRA

### 2.1 Por que não Full Fine-Tuning?

No treinamento tradicional (*Full Fine-Tuning*), para cada parâmetro treinável $W \in \mathbb{R}^{d \times k}$, é necessário armazenar:
- **Pesos originais:** 2 bytes (FP16/BF16)
- **Gradientes:** 2 bytes (FP16/BF16)
- **Momentum do AdamW ($m_t$):** 4 bytes (FP32)
- **Variância do AdamW ($v_t$):** 4 bytes (FP32)
- **Cópia mestre dos pesos:** 4 bytes (FP32)

Total: **16 a 18 bytes por parâmetro**. Para um modelo de 3 Bilhões de parâmetros:
$$\text{Memória do Otimizador e Pesos} \approx 3 \times 10^9 \times 16 \text{ bytes} \approx 48 \text{ GB}$$
Isso impossibilita o treinamento em GPUs comerciais de 8 GB a 16 GB sem clusters distribuídos.

---

### 2.2 Formulação do LoRA (Low-Rank Adaptation)

O LoRA (Hu et al., 2021) parte da premissa de que a mudança intrínseca de pesos $\Delta W$ durante a adaptação para uma tarefa específica reside em um subespaço de **baixo posto dimensional** (*low intrinsic dimension*).

Dada uma camada linear pré-treinada com pesos congelados $W_0 \in \mathbb{R}^{d \times k}$, a atualização $\Delta W$ é decomposta no produto de duas matrizes de posto reduzido $r \ll \min(d, k)$:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} \cdot B \cdot A$$

Onde:
- $W_0 \in \mathbb{R}^{d \times k}$ permanece **congelado** (sem gradientes).
- $A \in \mathbb{R}^{r \times k}$ é inicializada com distribuição gaussiana $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d \times r}$ é inicializada com **zeros** ($\Delta W = 0$ no início do treino, preservando o comportamento original).
- $r$ é o **Rank** (ex.: $r=16$).
- $\alpha$ é o **Fator de Escala** (*Scaling factor*, ex.: $\alpha=32$).

#### Para uma entrada $x$:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B (A x)$$

#### Análise dos Hiperparâmetros de LoRA:
- **Rank ($r$):** Controla a capacidade expressiva dos adaptadores. No nosso caso, $r=16$ equilibra perfeitamente capacidade e regularização para geração de tabelas.
- **Alpha ($\alpha$):** Constante que dimensiona a intensidade da atualização. A razão $\frac{\alpha}{r} = \frac{32}{16} = 2.0$ atua como uma taxa de aprendizado multiplicativa estática que estabiliza o gradiente quando $r$ é alterado.
- **Dropout ($lora\_dropout=0.05$):** Dropout probabilístico aplicado à ativação $A x$, prevenindo sobreajuste (*overfitting*) em termos textuais específicos.
- **Módulos Alvo (`target_modules: ["q_proj", "v_proj"]`):** Aplicar LoRA nas projeções de Query e Value da atenção multi-cabeça permite ao modelo alterar o padrão de ancoragem visual (*onde olhar*) e recuperação de conteúdo (*o que extrair*) sem instabilizar a projeção de saída ou o MLP.

---

### 2.3 Formulação do QLoRA (Quantized Low-Rank Adaptation)

O QLoRA (Dettmers et al., 2023) introduz três avanços seminais que tornam viável treinar o Qwen2.5-VL de 3B consumindo **menos de 7 GB de VRAM**:

#### 1. Quantização 4-bit NormalFloat (NF4)
Pesos de redes neurais pré-treinadas seguem aproximadamente uma distribuição normal $\mathcal{N}(0, \sigma^2)$. A quantização linear padrão (*int4*) cria intervalos uniformes que perdem muita informação nas caudas e no centro da curva de sino.
O tipo de dados **NF4** constrói 16 quantis ($2^4 = 16$) com áreas de probabilidade idênticas sob a distribuição normal padrão:
$$q_i = \frac{1}{2} \left( Q_X\left(\frac{i}{2^k}\right) + Q_X\left(\frac{i+1}{2^k}\right) \right)$$
Isso atinge uma perda de informação teoricamente ótima para pesos gaussianos, preservando a acurácia de FP16 mesmo comprimindo os pesos em 4 bits (redução de 4x de espaço de armazenamento).

#### 2. Double Quantization (DQ)
Os pesos quantizados em blocos precisam de constantes de escala (*quantization constants*) $c_1$. Em modelos grandes, essas constantes em FP32 ocupam cerca de 0.5 bits por parâmetro. O QLoRA quantiza as próprias constantes $c_1$ em inteiros de 8 bits com uma segunda escala $c_2$ a cada 256 blocos:
$$\text{Economia} = 0.5 \text{ bits/param} \longrightarrow 0.127 \text{ bits/param} \quad (\approx 0.37 \text{ bits/param poupados})$$

#### 3. Paged AdamW (8-bit)
Utiliza os recursos de memória unificada da NVIDIA (via `CUDA Paging`) para mover buffers do estado do otimizador da VRAM para a RAM da CPU automaticamente durante picos transitórios de alocação de memória (como o backward pass de sequências longas), prevenindo o erro fatal de Out-Of-Memory (`CUDA OOM`).

#### 4. Separação entre Storage Dtype e Compute Dtype:
- **Armazenamento de $W_0$:** Mantido em VRAM como **4-bit NF4**.
- **Cálculo da Ativação:** Quando o tensor de entrada $x$ (em FP16) chega, $W_0$ é dequantizado pontualmente em memória compartilhada rápida (*shared memory registers*) para **FP16**:
$$h = \text{dequant}(W_0) \cdot x + \frac{\alpha}{r} B (A x)$$
O resultado é acumulado em FP16 e os gradientes são calculados e propagados **apenas para as matrizes $A$ e $B$**, mantendo $W_0$ 100% estático.

---

## 3. Mecânica da Função de Perda (Loss) e Otimização de Memória

### 3.1 Loss Causal Autoregressiva (Cross-Entropy Mascarada)

A tarefa de geração de tabelas Markdown é modelada como modelagem de linguagem causal multimodal. Dada a sequência de tokens de entrada $X = (x_1, x_2, \dots, x_N)$, onde:
- Os tokens visuais e o prompt de sistema/usuário compõem o contexto: $X_{\text{prompt}} = (x_1, \dots, x_M)$.
- Os tokens da tabela alvo compõem a resposta esperada: $X_{\text{target}} = (x_{M+1}, \dots, x_N)$.

A probabilidade conjunta da sequência é:
$$P(X_{\text{target}} \mid X_{\text{prompt}}) = \prod_{t=M+1}^{N} P(x_t \mid x_1, \dots, x_{t-1})$$

A função de perda é a Entropia Cruzada (*Cross-Entropy Loss*):
$$\mathcal{L} = - \frac{1}{|T|} \sum_{t \in T} \log P(x_t \mid x_{<t})$$
Onde $T$ representa o conjunto de índices dos tokens pertencentes **estritamente à tabela Markdown alvo**.

#### Máscara de Label (`label = -100`):
Para evitar que o modelo gaste capacidade aprendendo a reproduzir os tokens do prompt ou a prever a própria imagem, todos os tokens de contexto recebem o valor sentinela $-100$:
$$\text{label}_t = \begin{cases} -100, & \text{se } t \le M \text{ (imagem e prompt)} \\ x_t, & \text{se } t > M \text{ (tabela alvo)} \end{cases}$$
O PyTorch ignora nativamente tokens com valor `-100` no cálculo do `F.cross_entropy(..., ignore_index=-100)`.

#### Alinhamento Temporal (Shift de Tokens):
Para previsão autoregressiva do próximo token, os rótulos devem ser deslocados de uma posição em relação às predições do decodificador:
$$\hat{y}_t \longleftrightarrow y_{t+1}$$
$$\text{Tokens de Entrada: } [x_1, x_2, \dots, x_{N-1}]$$
$$\text{Rótulos Alvo: } [x_2, x_3, \dots, x_N]$$

---

### 3.2 Otimização Crítica de Memória: Fatiamento de Vocabulário na `lm_head`

#### O Problema no Qwen2.5-VL:
O modelo possui uma camada de saída linear (`lm_head`) que projeta o estado oculto de dimensão $D=2048$ para o espaço do vocabulário $V=152.064$:
$$\text{logits} = \text{hidden\_states} \times W_{\text{lm\_head}}^T \quad \in \mathbb{R}^{B \times S \times 152.064}$$

Se o batch size $B=1$ e o comprimento da sequência multimodal $S=2048$ tokens:
$$\text{Tamanho do tensor de logits} = 1 \times 2048 \times 152.064 \times 4 \text{ bytes (FP32)} \approx 1.246 \text{ GB}$$
Alocar esse tensor gigantesco na VRAM durante o forward pass, junto com o backward pass da Cross-Entropy, gerava **CUDA Out-Of-Memory** mesmo com QLoRA de 4 bits.

#### A Solução Implementada (`_memory_efficient_qwen2_vl_forward` em `src/models/loader.py`):
Em vez de projetar os 2048 tokens da sequência inteira contra os 152.064 logits, filtramos os estados ocultos **antes** de chamar a `lm_head`:

```python
# 1. Aplica o padding e shift dos labels
shift_labels = nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
shift_labels_flat = shift_labels.view(-1)

# 2. Identifica quais posições realmente têm perda a ser calculada (ignora -100)
valid_mask = shift_labels_flat != -100

if valid_mask.any():
    # 3. Fatiamento: extrai APENAS os estados ocultos dos tokens da tabela
    shift_hidden = hidden_states.view(-1, hidden_states.shape[-1])
    valid_hidden = shift_hidden[valid_mask]  # Ex: de 2048 tokens, restam apenas ~180!
    valid_labels = shift_labels_flat[valid_mask].to(valid_hidden.device)

    # 4. Projeta para o vocabulário APENAS os tokens válidos
    valid_logits = self.lm_head(valid_hidden).float()  # Redução de ~90% de VRAM!

    # 5. Calcula Cross-Entropy apenas nas posições necessárias
    loss = nn.functional.cross_entropy(valid_logits, valid_labels, reduction="mean")
```

**Impacto Prático:** A projeção de vocabulário cai de 2048 tokens para tipicamente 100 a 250 tokens por imagem, reduzindo o consumo de VRAM da loss em **mais de 88%**, permitindo o treinamento fluido em placas como RTX 3060/4060 (8GB VRAM).

---

## 4. Dicionário Completo de Argumentos e Parâmetros

A configuração do framework é modularizada em YAMLs (`configs/default.yaml` e `configs/training/qlora.yaml`). Abaixo está o mapeamento detalhado de cada parâmetro:

### 4.1 Bloco Geral
| Parâmetro | Tipo | Padrão | Explicação Técnica |
| :--- | :---: | :---: | :--- |
| `project_name` | `str` | `"sci-image-markdown"` | Identificador do projeto em logs, artefatos e exportações. |
| `experiment_name` | `str` | `"qwen2_5_vl_3b_qlora"` | Nome da rodada de experimento atual, usado para nomear diretórios de checkpoints e métricas. |
| `seed` | `int` | `42` | Semente pseudoaleatória para reprodutibilidade de divisão de dados, inicialização de LoRA e shuffle. |

---

### 4.2 Bloco `data` (Processamento de Dados)
| Parâmetro | Tipo | Padrão | Explicação Técnica |
| :--- | :---: | :---: | :--- |
| `raw_dir` | `str` | `"data/raw"` | Diretório de download bruto do Hugging Face contendo imagens originais e metadados. |
| `processed_dir` | `str` | `"data/processed"` | Diretório com os arquivos finais `train.jsonl`, `val.jsonl`, `test.jsonl` já estruturados e filtrados. |
| `splits_dir` | `str` | `"data/splits"` | Diretório reservado para particionamento estratificado de treino/validação/teste. |
| `train_file` | `str` | `"data/processed/train.jsonl"` | Caminho para o JSONL de treino. Cada linha contém `{"id", "image", "table"}`. |
| `val_file` | `str` | `"data/processed/val.jsonl"` | Caminho para o JSONL de validação periódica ao final de cada época. |
| `test_file` | `str` | `"data/processed/test.jsonl"` | Caminho para o JSONL de teste cego avaliado pelo benchmark oficial. |
| `image_folder` | `str` | `"data/raw/images"` | Diretório base de resolução de caminhos relativos de imagens. |
| `max_image_resolution`| `int` | `280` | Dimensão máxima (largura/altura em pixels) para o `thumbnail` da imagem. 280px mantém legibilidade dos eixos e números com uso ultrabaixo de tokens visuais. |
| `min_image_resolution`| `int` | `128` | Dimensão mínima admitida para figuras sem degradação excessiva dos ticks. |
| `max_table_chars` | `int` | `1500` | Truncamento defensivo de caracteres da tabela Markdown alvo para evitar sequências que estourem o buffer de contexto. |
| `system_prompt` | `str` | `"You are an expert..."`| Instrução de sistema passada ao modelo antes da imagem, condicionando o estilo da extração. |

---

### 4.3 Bloco `model` (Carregamento da Arquitetura)
| Parâmetro | Tipo | Padrão | Explicação Técnica |
| :--- | :---: | :---: | :--- |
| `name_or_path` | `str` | `"Qwen/Qwen2.5-VL-3B-Instruct"` | Identificador oficial do Hugging Face Hub ou diretório local contendo os pesos base pré-treinados. |
| `model_type` | `str` | `"qwen2_5_vl"` | Flag de arquitetura que aciona a classe especializada `Qwen2_5_VLForConditionalGeneration`. |
| `trust_remote_code` | `bool` | `true` | Permite execução de scripts de modelagem customizados publicados no repositório do HF. |
| `torch_dtype` | `str` | `"float16"` | Precisão dos tensores intermediários em memória (`float16` para estabilidade ou `bfloat16` em GPUs Ampere+). |
| `attn_implementation`| `str` | `"sdpa"` | Implementação de atenção. `"sdpa"` ativa o Scaled Dot-Product Attention acelerado do PyTorch 2+. Se disponível GPU sm80+, pode usar `"flash_attention_2"`. |
| `use_cache` | `bool` | `false` | Desabilita o cache de chaves/valores de atenção (*KV Cache*) durante o treinamento, pois é incompatível com *Gradient Checkpointing*. |

---

### 4.4 Bloco `training` (Otimização e Hiperparâmetros)
| Parâmetro | Tipo | Padrão | Explicação Técnica |
| :--- | :---: | :---: | :--- |
| `method` | `str` | `"qlora"` | Método de ajuste fino. `"qlora"` congela o modelo em 4-bit NF4 e acopla adaptadores treináveis. |
| `optim` | `str` | `"paged_adamw_8bit"` | Otimizador AdamW quantizado em 8 bits com paginação de memória para CPU RAM em picos de alocação. |
| `output_dir` | `str` | `"outputs/checkpoints"` | Pasta de destino onde os adaptadores PEFT (`adapter_model.safetensors`, `adapter_config.json`) são gravados. |
| `num_train_epochs` | `int` | `3` | Número de passagens completas pelo conjunto de dados de treino. |
| `per_device_train_batch_size` | `int` | `1` | Tamanho do mini-lote por GPU. Como as imagens possuem tamanhos variados, `1` é o valor seguro para evitar padding redundante. |
| `per_device_eval_batch_size` | `int` | `1` | Tamanho do lote durante validação periódica. |
| `gradient_accumulation_steps` | `int` | `8` | Acumula gradientes ao longo de 8 passos antes de executar o `optimizer.step()`. Simula um batch size efetivo de $1 \times 8 = 8$. |
| `learning_rate` | `float` | `2.0e-4` | Taxa de aprendizado inicial aplicada aos adaptadores LoRA. 2e-4 é a faixa empírica recomendada para convergência rápida sem destruição de representações. |
| `lr_scheduler_type`| `str` | `"cosine"` | Decaimento suave da taxa de aprendizado seguindo uma curva cossenoidal após o warmup. |
| `warmup_steps` | `int` | `20` | Primeiros 20 passos nos quais o learning rate sobe linearmente de 0 até 2e-4, prevenindo gradientes explosivos na inicialização. |
| `weight_decay` | `float` | `0.01` | Regularização $L_2$ aplicada às matrizes $A$ e $B$ do LoRA. |
| `logging_steps` | `int` | `10` | Frequência de iterações para impressão de loss, learning rate e velocidade no console. |
| `eval_strategy` | `str` | `"epoch"` | Roda avaliação ao final de cada época completa de treinamento. |
| `save_strategy` | `str` | `"epoch"` | Salva um checkpoint do adaptador ao término de cada época. |
| `save_total_limit`| `int` | `2` | Mantém apenas os 2 checkpoints mais recentes, deletando os mais antigos para economizar disco. |
| `fp16` | `bool` | `true` | Ativa treinamento com precisão mista FP16 (Half Precision). |
| `bf16` | `bool` | `false` | Bfloat16 (desativado automaticamente se a placa de vídeo tiver arquitetura anterior à Ampere). |
| `dataloader_num_workers`| `int` | `2` | Número de processos assíncronos para leitura e decodificação de imagens do disco. |
| `gradient_checkpointing`| `bool`| `true` | **Crucial para VRAM**: não armazena ativações intermediárias no forward pass; recalcula-as sob demanda no backward pass. Economiza até 60% de VRAM ao custo de apenas ~20% a mais de tempo de processamento. |

---

### 4.5 Bloco `peft` (Configuração LoRA)
| Parâmetro | Tipo | Padrão | Explicação Técnica |
| :--- | :---: | :---: | :--- |
| `r` | `int` | `16` | Posto (*Rank*) da decomposição matricial de baixa dimensão. |
| `lora_alpha` | `int` | `32` | Fator multiplicativo de escala de atualização ($\frac{\alpha}{r} = 2.0$). |
| `lora_dropout` | `float` | `0.05` | Taxa de dropout estocástico aplicado aos adaptadores. |
| `bias` | `str` | `"none"` | Não treina parâmetros de viés (*bias*), mantendo todos congelados. |
| `task_type` | `str` | `"CAUSAL_LM"` | Informa ao HuggingFace PEFT que a arquitetura alvo é de modelagem causal autorregressiva. |
| `target_modules` | `list` | `["q_proj", "v_proj"]` | Módulos de atenção que recebem matrizes LoRA acopladas. |

---

### 4.6 Bloco `quantization` (Configuração BitsAndBytes)
| Parâmetro | Tipo | Padrão | Explicação Técnica |
| :--- | :---: | :---: | :--- |
| `load_in_4bit` | `bool` | `true` | Carrega os pesos da rede base congelada utilizando precisão inteira de 4 bits. |
| `bnb_4bit_quant_type` | `str` | `"nf4"` | Tipo de quantização ótima para distribuições normais (*NormalFloat 4*). |
| `bnb_4bit_compute_dtype` | `str` | `"float16"` | Precisão em que a multiplicação matricial é computada nos registradores da GPU. |
| `bnb_4bit_use_double_quant`| `bool` | `true` | Ativa a dupla quantização das escalas, economizando 0.37 bits por parâmetro adicionais. |

---

### 4.7 Bloco `evaluation` (Configuração de Inferência e Benchmarking)
| Parâmetro | Tipo | Padrão | Explicação Técnica |
| :--- | :---: | :---: | :--- |
| `batch_size` | `int` | `1` | Tamanho do lote de geração na avaliação. |
| `max_new_tokens` | `int` | `1024` | Número máximo de novos tokens gerados na tabela de saída antes da interrupção. |
| `temperature` | `float` | `0.0` | Amostragem determinística (*greedy search*). Temperatura zero garante que o modelo escolha sempre o token mais provável, eliminando alucinações estocásticas em números. |
| `do_sample` | `bool` | `false` | Desativa amostragem probabilística (Top-p / Top-k) em favor de decodificação determinística gulosa. |
| `numerical_relative_tolerance`| `float` | `0.05` | Tolerância relativa para validação de células numéricas ($\le 5\%$ de erro em relação ao Ground Truth). |

---

## 5. Pipeline de Dados, Pré-Processamento e Collação

### 5.1 Censo Oficial do Dataset Sci-ImageMiner (ICDAR 2026 Task 2)
O repositório original do Hugging Face (`SciKnowOrg/Sci-ImageMiner`) contém 1.951 painéis de figuras de artigos científicos de Ciência dos Materiais e Semicondutores. Deles, **exatamente 1.200 amostras** pertencem à Task 2 (Extração de Dados / Tabelas Numéricas):

| Split Oficial | Total de Imagens | Figuras com Tabela Numérica (Task 2) | Proporção |
| :--- | :---: | :---: | :---: |
| **Treino (`train.jsonl`)** | 1.170 | **723** | 60.25% |
| **Validação (`val.jsonl`)** | 201 | **104** | 8.67% |
| **Teste (`test.jsonl`)** | 580 | **373** | 31.08% |
| **Total Global** | **1.951** | **1.200** | **100.0%** |

Todas as **1.200 figuras reais** foram baixadas e integradas localmente em `data/raw/images/` e `data/processed/images/`.

### 5.2 Estrutura do Arquivo JSONL Limpo
Após a limpeza executada para remoção dos dados da antiga autora, cada linha dos arquivos `train.jsonl`, `val.jsonl` e `test.jsonl` contém estritamente o esquema canônico mínimo:

```json
{
  "id": "sci_imageminer__atomic_layer_deposition__experimental_usecase__12__fig_10",
  "image": "images/sci_imageminer__atomic_layer_deposition__experimental_usecase__12__fig_10.jpg",
  "table": "| AB Cycles | Tungsten Film Thickness (Å) |\n|---|---|\n| 0 | 0 |\n| 10 | 25 |\n| 20 | 50 |\n| 30 | 75 |\n| 40 | 100 |\n| 50 | 125 |\n| 60 | 150 |\n| 70 | 175 |\n| 80 | 200 |"
}
```

### 5.2 Formatação de Mensagens (Chat Template)
Em [`src/data/preprocessor.py`](file:///home/lucas/masters/sci-image-markdown/src/data/preprocessor.py), a função `format_qwen_vl_conversation` converte a imagem e a tabela no formato padrão de conversação do Qwen2.5-VL:

```python
messages = [
    {"role": "system", "content": "You are an expert scientific figure analyzer..."},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "Extract all numerical data points from this scientific figure panel into a Markdown table with clear column headers."}
        ]
    },
    {"role": "assistant", "content": target_table}
]
```

### 5.3 O Collatador Multimodal (`QwenVLDataCollator`)
Em [`src/data/collator.py`](file:///home/lucas/masters/sci-image-markdown/src/data/collator.py):
1. Aplica o `processor.apply_chat_template(...)` transformando as mensagens em texto com tags especiais (`<|im_start|>`, `<|im_end|>`, `<|vision_start|>`, `<|vision_end|>`).
2. Extrai e redimensiona as matrizes de pixels com `process_vision_info`.
3. Converte tudo em tensores com padding automático.
4. Gera a máscara de labels, definindo `-100` nas posições de padding.

---

## 6. Métricas de Avaliação Científica de Tabelas

A biblioteca de métricas implementada em [`src/metrics/`](file:///home/lucas/masters/sci-image-markdown/src/metrics/) separa rigorosamente a **integridade sintática** da **precisão numérica**:

```
                       Geração do Modelo (Markdown Text)
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │    Parser de Tabela (Regex)   │
                       └───────────────┬───────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
          [ Tabela Inválida ]                   [ Tabela Válida ]
          - VTR = 0                             - VTR = 1 (99.73%)
          - Loss de loop                        - Parse para pandas.DataFrame
                                                         │
                                                         ▼
                                       ┌───────────────────────────────┐
                                       │     Alinhamento Numérico      │
                                       │  Tolerância Relativa <= 5%    │
                                       └───────────────┬───────────────┘
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     ▼                                 ▼                                 ▼
          [ Cell Recall: 48.22% ]            [ Cell RMSE: 1.03 ]               [ Cell RNE: 0.36% ]
          Captura dos pontos reais           Erro absoluto nas curvas          Desvio percentual médio
```

1. **Taxa de Tabelas Válidas (`valid_table` / VTR):**
   Mede se o texto gerado possui cabeçalho, divisória (`|---|---|`) e linhas que podem ser convertidas com sucesso em um `pandas.DataFrame`.
   - *Baseline Zero-Shot:* 88.74% (11.26% de falhas por mode collapse / loops infinitos).
   - *Modelo Fine-Tuned com QLoRA:* **99.73%**.

2. **Cell Recall ($\le 5\%$ de tolerância):**
   Percentual de pontos reais plotados na curva do Ground Truth que foram devidamente recuperados pelo modelo dentro de 5% de tolerância:
   $$\text{Match se: } \frac{|y_{\text{pred}} - y_{\text{true}}|}{|y_{\text{true}}| + \epsilon} \le 0.05$$
   - *Baseline Zero-Shot:* 35.78%
   - *Modelo Fine-Tuned com QLoRA:* **48.22%** (+34.78% de ganho relativo).

3. **Cell RMSE (Root Mean Squared Error):**
   Raiz do erro quadrático médio entre os pontos preditos e os pontos reais correspondentes:
   $$\text{RMSE} = \sqrt{\frac{1}{K} \sum_{i=1}^{K} (y_{\text{pred}, i} - y_{\text{true}, i})^2}$$
   - *Baseline Zero-Shot:* 2.6635
   - *Modelo Fine-Tuned com QLoRA:* **1.0308** (redução de **61.3%** no erro).

4. **Normalized Edit Similarity:**
   Distância de edição normalizada de Levenshtein entre a string Markdown inteira gerada e a real:
   $$\text{Sim} = 1 - \frac{\text{Levenshtein}(S_{\text{pred}}, S_{\text{true}})}{\max(|S_{\text{pred}}|, |S_{\text{true}}|)}$$

---

## 7. Guia Operacional: Como Treinar, Avaliar e Exportar

### 7.1 Preparação do Dataset
Para gerar amostras sintéticas limpas de teste rápido:
```bash
.venv/bin/python prepare_data.py --create-synthetic --num-samples 20
```

Para processar o dataset científico real baixado do Hugging Face:
```bash
.venv/bin/python prepare_data.py --download-hf --raw-dir data/raw --processed-dir data/processed
```

---

### 7.2 Execução do Treinamento (QLoRA)

#### Opção A: Execução direta no terminal interativo
```bash
.venv/bin/python train.py --config configs/default.yaml
```

#### Opção B: Execução em background com log protegido
```bash
bash run_train.sh
# Acompanhe em tempo real com:
tail -f train.log
```

---

### 7.3 Avaliação do Benchmark (Base vs Fine-Tuned)
Para rodar a comparação completa das 373 figuras científicas de teste:
```bash
bash run_eval_both.sh
```
Os relatórios estruturados serão gravados em:
- `outputs/eval_results/base_model/report.json`
- `outputs/eval_results/finetuned_model/report.json`

---

### 7.4 Inferência em Imagem Única (CLI)
Para extrair os dados de uma figura científica arbitrária diretamente para Markdown e CSV:
```bash
.venv/bin/python predict.py \
    --image data/processed/images/sci_imageminer__atomic_layer_deposition__experimental_usecase__12__fig_10.jpg \
    --adapter-path outputs/checkpoints/final_adapter \
    --output-md tabela_extraida.md \
    --output-csv tabela_extraida.csv
```

---

### 7.5 Exportação e Fusão dos Pesos (*Merge & Unload*)
Para fundir os adaptadores LoRA diretamente nos pesos base do modelo e produzir um pacote independente de inferência (sem dependência de PEFT):
```bash
bash run_export.sh
```
O modelo consolidado final será salvo em `outputs/merged_model`.
