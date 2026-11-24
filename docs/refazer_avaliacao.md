# Refazer a avaliação com a métrica oficial

Guia para repontuar os checkpoints no **teste (373 figuras)** com o script oficial do Sci-ImageMiner (Task 2), por painel. Foi escrito para rodar no servidor, onde estão os adapters e os dados.

## 1. Por que refazer

Os números do `docs/BENCHMARK_RESULTS.md` (52,32 do LightOnOCR, 52,11 do 7B, e assim por diante) vêm do `icdar_score` **local** (`src/metrics/table_metrics.py`). Ele usa a mesma fórmula da competição (½ TEDS + ½ RMS), mas com outro cálculo:

| | Local (`src/metrics`) | Oficial (`third_party/sci_imageminer_eval`) |
| :--- | :--- | :--- |
| Unidade | figura inteira, todas as tabelas juntas | **painel por painel**; as tabelas são mapeadas por ordem para os painéis |
| RMS | `recall × (1 − erro relativo médio)`, casamento guloso a 5% | `table_datapoints_f1` do DePlot |
| TEDS | implementação própria do repositório | `teds_pubtables` oficial |

Dois sinais de que as escalas diferem:

- O mesmo 3B+LoRA marcou **0,43 na local e 35,0 na oficial** (relatório da Larissa, `docs/analise_erros/analise_erros_dev.md` na branch `analise-de-erros`).
- A tabela do README mistura os dois: o RMS local dos nossos modelos fica em 58 a 62, e o RMS oficial das equipes do leaderboard fica em 12 a 17. A nota 41,81 do 1º lugar é oficial, no teste.

Por isso o 52,32 **não pode ser comparado** com o 41,81 até ser repontuado. Não se sabe quanto ele cai.

Só a avaliação precisa ser refeita. Nenhum treino.

## 2. O que é necessário

1. **O código da branch `analise-de-erros`.** Ela tem `analyze_errors.py`, `src/analysis/` e o script oficial em `third_party/sci_imageminer_eval/`. Ela também muda o `evaluate.py` para salvar `raw_output` e `id` nas predições. Ela está à frente da `main` e não está atrás dela, então tem todas as configs de treino atuais.
2. **Predições com `raw_output`.** Os `test_predictions.jsonl` gerados antes dessa mudança **não servem**: eles só têm `markdown_table`, que guarda apenas a primeira tabela de cada figura. Figuras com vários painéis seriam pontuadas só pelo primeiro painel. Por isso é preciso **rodar a inferência de novo** com o `evaluate.py` novo. Isso usa GPU, mas é só inferência.
3. **O metadata do teste**, `data/raw/test_metadata.jsonl`. Ele traz o gabarito por painel. O `prepare_data.py --download-hf` já baixa esse arquivo.

## 3. Preparação

```bash
cd ~/sci-image-markdown            # clone do servidor (ajuste o caminho)
git fetch origin
git worktree add --detach ../sci-image-markdown-oficial origin/analise-de-erros
cd ../sci-image-markdown-oficial
ln -s ../sci-image-markdown/data data   # reaproveita imagens e jsonl
```

Confira o metadata do teste:

```bash
ls -l data/raw/test_metadata.jsonl
```

Se o arquivo não existir, baixe só ele:

```bash
curl -L -o data/raw/test_metadata.jsonl \
  "https://huggingface.co/datasets/SciKnowOrg/Sci-ImageMiner/resolve/main/test/metadata.jsonl"
```

Use o mesmo venv do treino e instale as dependências do script oficial. O pin `numpy==1.26.4` pode rebaixar o numpy do venv. A Larissa rodou os testes com `torch` e CUDA nesse venv. Se houver conflito, use um venv só para pontuar, porque a pontuação não usa GPU.

```bash
source ~/sci-image-markdown/.venv/bin/activate   # ajuste
pip install -r third_party/sci_imageminer_eval/requirements.txt
pip install pandas click rich rapidfuzz jsonlines pytest
export PYTHONPATH=.
```

Valide a ferramenta. Este teste roda o script oficial e compara com a nota por painel:

```bash
python -m pytest tests/test_analysis_scoring.py tests/test_analyze_errors_cli.py -q
```

## 4. Checkpoints

Os comandos abaixo repetem os de `scripts/slurm_train_*.sh` e `scripts/run_eval_7b.sh` da `main`. Os caminhos de adapter variam entre os scripts (`/media/ssd2/datasets/...` e `/media/ssd2/lucas.ocunha/...`), então ajuste `DATA_OUT` ao que existe no servidor.

| Nome | `--config` | Adapter (em `$DATA_OUT/`) |
| :--- | :--- | :--- |
| `lightonocr_52_32` | `configs/training/train_lightonocr_3_4b.yaml` | `lightonocr_3_4b_icdar_loss/final_adapter` |
| `lightonocr_synth15` | `configs/training/train_lightonocr_3_4b_synth15.yaml` | `lightonocr_3_4b_synth15_icdar_loss/final_adapter` |
| `qwen7b` | `configs/training/train_7b_icdar_loss.yaml` | `qwen2_5_vl_7b_icdar_loss/final_adapter` |
| `qwen3_vl_8b` | `configs/training/train_qwen3_vl_8b.yaml` | `qwen3_vl_8b_icdar_loss/final_adapter` |
| `qwen3b` | `configs/default.yaml` (com `--training-config configs/evaluation/original_resolution.yaml` se foi avaliado em resolução original) | `outputs/checkpoints_icdar_loss/final_adapter` |

Antes de rodar, confirme **qual adapter gerou cada número**:

- **LightOnOCR (52,32):** o relatório do recorde está em `eval_results/lightonocr_3_4b_icdar_loss_record_05232/`. O `slurm_train_lightonocr_pipeline.sh` fala em "Loss v2", e o README fala em "Loss v3". Se o adapter foi sobrescrito depois do recorde, o 52,32 não é reproduzível, e vale registrar isso.
- **7B (52,11):** o `run_eval_7b.sh` usa `qwen2_5_vl_7b_icdar_loss`, mas o YAML atual grava em `qwen2_5_vl_7b_icdar_loss_v2`.
- **3B (49,49):** confirme se foi avaliado a 280 px ou em resolução original.

Use **o mesmo config, o mesmo prompt e a mesma resolução** do número que está sendo repontuado. Qualquer mudança aí muda o resultado.

## 5. Rodar (script em lote)

Salve como `rodar_avaliacao_oficial.sh` no worktree e envie com `sbatch` (precisa de GPU). Ajuste `DATA_OUT` e `OUT`.

```bash
#!/bin/bash
#SBATCH --job-name=aval-oficial
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=24:00:00
#SBATCH --output=logs/aval_oficial_%j.out
#SBATCH --error=logs/aval_oficial_%j.err

set -euo pipefail
cd "$HOME/sci-image-markdown-oficial"        # ajuste
source "$HOME/sci-image-markdown/.venv/bin/activate"   # ajuste
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export HF_HOME=/media/ssd2/lucas.ocunha/.cache/huggingface   # ajuste

DATA_OUT=/media/ssd2/datasets/sci-image-markdown-data/outputs   # ajuste
OUT=$DATA_OUT/eval_oficial
META=data/raw/test_metadata.jsonl
mkdir -p "$OUT" logs

roda() {  # roda NOME CONFIG ADAPTER [flags extras do evaluate.py]
  nome=$1; cfg=$2; adapter=$3; shift 3
  mkdir -p "$OUT/$nome"
  # 1) inferência no teste, salvando raw_output e id
  python -u evaluate.py --config "$cfg" "$@" \
    --adapter-path "$adapter" \
    --test-file data/processed/test.jsonl \
    --output-report "$OUT/$nome/test_report_local.json" \
    --save-predictions "$OUT/$nome/test_predictions.jsonl"
  # 2) nota oficial por painel
  python analyze_errors.py \
    --metadata "$META" \
    --repo-predictions "$OUT/$nome/test_predictions.jsonl" \
    --out-dir "$OUT/$nome"
}

roda lightonocr_52_32   configs/training/train_lightonocr_3_4b.yaml        "$DATA_OUT/lightonocr_3_4b_icdar_loss/final_adapter"
roda lightonocr_synth15 configs/training/train_lightonocr_3_4b_synth15.yaml "$DATA_OUT/lightonocr_3_4b_synth15_icdar_loss/final_adapter"
roda qwen7b             configs/training/train_7b_icdar_loss.yaml           "$DATA_OUT/qwen2_5_vl_7b_icdar_loss/final_adapter"
roda qwen3_vl_8b        configs/training/train_qwen3_vl_8b.yaml             "$DATA_OUT/qwen3_vl_8b_icdar_loss/final_adapter"
# 3B: acrescente "--training-config configs/evaluation/original_resolution.yaml" se foi avaliado em resolução original
# roda qwen3b configs/default.yaml outputs/checkpoints_icdar_loss/final_adapter --training-config configs/evaluation/original_resolution.yaml
```

Cada `roda` grava em `$OUT/<nome>/`:

- `test_predictions.jsonl`: a saída do modelo, com `raw_output` e `id`.
- `official_predictions.jsonl`: as tabelas mapeadas por painel, no formato oficial.
- `panels.csv`: uma linha por painel, com `rms`, `teds`, `weighted` e as categorias de erro.
- `summary.json`: as médias.

Para repontuar de novo sem GPU, quando as predições já existem, rode só o passo 2:

```bash
python analyze_errors.py --metadata data/raw/test_metadata.jsonl \
  --repo-predictions "$OUT/<nome>/test_predictions.jsonl" --out-dir "$OUT/<nome>"
```

## 6. Ler o resultado

A **nota oficial** é o campo `weighted` do `summary.json`. Para resumir todos de uma vez:

```bash
for d in "$OUT"/*/; do
  echo "$(basename "$d")  $(python -c "import json;s=json.load(open('${d}summary.json'));print({k:round(v,2) if isinstance(v,float) else v for k,v in s.items()})")"
done
```

## 7. Conferências antes de confiar nos números

1. **Sem aviso de `raw_output`.** O log do `analyze_errors.py` não pode mostrar `records have no raw_output`. Se mostrar, a predição é antiga e precisa ser regenerada.
2. **`n_panels`.** Deve ser o número de painéis com tabela no `test_metadata.jsonl`. Figuras sem predição entram com nota 0, e não somem da média.
3. **`extra_tables` no `summary.json`.** O mapeamento é por ordem: tabelas além do número de painéis são ignoradas, e painéis sem tabela contam como erro. Um `extra_tables` alto indica que o modelo está quebrando painéis em várias tabelas.
4. **RMS zerado por formatação.** O RMS oficial divide as células por `" | "`. Uma tabela como `|33|7.5|` (sem espaços ao redor de `|`) zera o RMS, e a ferramenta reproduz isso de propósito. Veja quantos painéis válidos têm RMS 0:

```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv("COLE_AQUI/panels.csv")   # ajuste para o panels.csv de um checkpoint
valid = df[df.pred_valid]
print("painéis válidos:", len(valid), "| com RMS zero:", int((valid.rms == 0).sum()))
EOF
```

   Se esse número for grande, o ganho possível está no formato de saída (espaço ao redor de `|`) e não no modelo.
5. **Os mesmos 373.** Confirme que o `data/processed/test.jsonl` e o `test_metadata.jsonl` vêm do mesmo split do Hugging Face.

## 8. Registrar

Preencha e atualize o `docs/BENCHMARK_RESULTS.md` com **duas colunas separadas**, local e oficial. Não coloque mais as duas na mesma coluna ao lado do leaderboard.

| Checkpoint | Local (antigo) | Oficial (`weighted`) | RMS oficial | TEDS oficial | `n_panels` | `extra_tables` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| lightonocr_52_32 | 52,32 | | | | | |
| lightonocr_synth15 | 51,72 | | | | | |
| qwen7b | 52,11 | | | | | |
| qwen3_vl_8b | 48,72 | | | | | |
| qwen3b | 49,49 | | | | | |

Referências oficiais no teste: 1º lugar 41,81 e baseline 35,97.

## 9. Observações

- **Teste só para relatar.** O relatório da Larissa mantém o teste fora das decisões para não viciar o ajuste. Para decidir mudanças (prompt, recorte, loss), use o dev: o `data/raw/validation_metadata.jsonl` com o `data/processed/val.jsonl`.
- **Recorte por painel.** A branch `recorte-por-painel` usa o bbox do metadata para recortar cada painel. Se o teste oficial não trouxer o bbox, essa inferência não roda como está.
- **Quem fez o quê.** `analyze_errors.py`, `src/analysis/` e o script oficial são da Larissa (branch `analise-de-erros`).
