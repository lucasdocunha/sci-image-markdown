# Relatório — Augmentation (Degradação de Imagens) v2

Pipeline Albumentations 2.0.8 do script `augment_images.py`, branch `synthetic_dataset`.

---

## 1. Objetivo

Treinar o VLM (Qwen2.5-VL, extração figura → tabela Markdown) para ler figuras
científicas em condições reais de aquisição — **impressão, escaneamento e foto de
página** — e não apenas os JPEGs "perfeitos" do dataset processado. Cada cópia
degradada mantém a **tabela Markdown do original intacta**; só a imagem muda.

## 2. O que o pipeline aplica (acumulado, em ordem)

Cada cópia é uma passagem completa pelo pipeline abaixo com valores sorteados
nas faixas da severidade escolhida (determinístico por seed):

| # | Transform | O que simula | mild | moderate | strong |
|---|---|---|---|---|---|
| 1 | `Affine` (rotate + scale por eixo) | rotação + esticamento de print/repositório | ±5°, escala 0.90–1.05 | ±10°, 0.85–1.15 | ±15°, 0.80–1.25 |
| 2 | `Perspective` (keep_size, p=0/0.15/0.30) | foto da página em ângulo | — | ≤7% | ≤10% |
| 3 | `ToGray` (p=0/0.15/0.30) | impresso P&B / PDF B&W | — | 15% | 30% |
| 4 | `HueSaturationValue` (p=0.5) | cast de cor de tela/flash | ±5 h, ±10 s/v | ±10/±20 | ±15/±30 |
| 5 | `RandomGamma` (p=0.3/0.5/0.7) | exposição errada | 75–125 | 60–140 | 50–150 |
| 6 | `RandomBrightnessContrast` (p=0.7) | brilho/contraste | ±0.08 | ±0.15 | ±0.20 |
| 7 | `RandomShadow` (p=0/0.15/0.25) | sombra na foto do papel | — | 0.3–0.6 | 0.3–0.6 |
| 8 | `OneOf` — **exatamente uma**: | | | | |
| | `GaussianBlur` | scan desuniforme | k=3 | 3–5 | 3–7 |
| | `GaussNoise` | sensor de câmera | std 0.02–0.05 | 0.04–0.09 | 0.06–0.14 |
| | `ISONoise` | ruído de câmera celular | (0.01–0.03, 0.2–0.4) | (0.01–0.05, 0.3–0.7) | (0.02–0.08, 0.5–1.0) |
| | `Downscale` | baixa dpi + reampliação | 0.80–0.95 | 0.65–0.90 | 0.55–0.85 |
| 9 | `Sharpen` (p=0/0.15/0.25) | serrilhado de rescale de PDF | — | | |
| 10 | `ImageCompression` | artefatos JPEG | q 75–90 | q 45–75 | q 30–55 |

Regras de segurança integradas ao desenho:

- **Borda branca + `fit_output`** em todas as transformações geométricas: eixos,
  legendas e números de eixo nunca são cortados.
- **Downscale com piso 0.55**: protege picos finos de espectros (FTIR/XPS) —
  destruição de detalhe fino fica a cargo de ruído e JPEG (isométricos), não de
  reampliação que "apaga" picos.
- **ToGray never at 100%** em nenhuma severidade: evita treinar séries
  coloridas num regime impossível.
- **Todo o resto que não "corta conteúdo"**: flip/crop agressivo/CoarseDropout
  ficam de fora de propósito (invalidariam tabelas ou esconderiam eixos).

## 3. Dataset gerado (`data/augmented`, v2)

| Parâmetro | Valor |
|---|---|
| Comando | `python augment_images.py --variants 3 --output-dir data/augmented` |
| Severidade | `moderate` |
| Seed | 42 (reprodutível: seed por variante = `sha256(set:seed:split:id:variante)`) |

| Split | Originais | Cópias geradas |
|---|---|---|
| train | 723 | 2 169 |
| val | 104 | 312 |
| test | 373 | 1 119 |
| **total** | **1 200** | **3 600** |

Tamanho em disco: **1,3 GB** (3600 PNG). Mediana de largura das cópias: 687 px
vs 658 px dos originais (expansão de rotação/perspectiva compensa o
esticamento parcial).

### Frequência efetiva dos transforms (amostrada no run)

| Transform | Em quantas das 3600 cópias |
|---|---|
| Affine (rotate+stretch) | 3600 (100%) |
| Perspective | 500 (≈14%) |
| ImageCompression | 3600 (100%) |
| BrightnessContrast | 2496 (≈69%) |
| HueSaturationValue / RandomGamma | 1795 (≈50%) cada |
| Blur / GaussNoise / ISONoise / Downscale | 856 / 884 / 943 / 917 (uniforme) |
| ToGray / Shadow / Sharpen | 500 (≈14%) cada |

A lista exata de transforms aplicados **por cópia** fica em `manifest.json`
(`images[].transforms`), junto de `source_id`, `source_image` (ligação ao
original), split, seed de sorteio e o pipeline serializado por inteiro.

## 4. Esquema e rastreabilidade

- Saída contém **somente cópias** (`images/*.png` em PNG), `train/val/test.jsonl`
  no esquema canônico `id/image/table` e `manifest.json`.
- Os **originais permanecem intocados** em `data/processed` (garantido por
  testes de integridade byte a byte); ligação para treino combinado via
  `train_combined.jsonl` (README: originais por caminho absoluto + cópias).
- Nomes de arquivo **derivados da figura original**
  (`<figura_original>__aug_001.png`); stems duplicados entre pastas ganham
  sufixo determinístico `_d{hash6}`.
- **Pertença ao split original preservada** (ninguém migra entre
  treino/val/teste); o script rejeita entrada sintética/já-aumentada, saída
  não vazia e sobreposição source↔output.

## 5. Modo diagnóstico (`--diagnostic`)

Gera um conjunto por transformação **isolada** — para medir a queda de
desempenho causada por cada degradação isoladamente, sem misturar efeitos:

```bash
python augment_images.py --diagnostic --split test --severity strong \
  --output-dir data/diagnostic
```

13 conjuntos `baseline / rotate / stretch / grayscale / hue_gamma /
perspective / shadow / blur / gauss_noise / iso_noise / downscale / sharpen /
jpeg`, cada um em `<output>/<conjunto>/` com seu próprio JSONL e manifest.
Avaliar cada conjunto com `evaluate.py` e comparar ao `baseline` produz o
ranking de culpados pelas partes que o modelo erra.

## 6. Limitações e próximos passos

1. **A degradação não resolve erros de conteúdo** (juntar vários painéis,
   "linearizar" picos): esses dependem da distribuição de dados — tratados pelo
   gerador de sintéticos (`generate_synthetic_dataset.py`, 25 famílias,
   incluindo `growth_dual_panel` e espectros hard) e pelo treino combinado,
   não por este pipeline.
2. **Validação visual pendente**: inspecionar 5–10 cópias aleatórias (por
   severidade) antes de treinar, com prioridade para espectros com downscale
   agressivo e figuras multi-painel com grayscale.
3. **Análise de culpados**: gerar `data/diagnostic` e cruzar os
   `predictions.jsonl` de cada conjunto com o do `baseline` num script
   `scripts/analyze_degradation.py` ainda a ser construído.
4. Extensões futuras possíveis: oversampling dirigido (`--variants` maior para
   figuras multi-tabela/espectro) e pesos não uniformes no `OneOf`.
