#!/bin/bash
#SBATCH --job-name=pos-eval
#SBATCH --partition=dev
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/pos_avaliacao_%j.out
#SBATCH --error=logs/pos_avaliacao_%j.err

set -euo pipefail
cd /home/lucas.ocunha/sci-image-markdown
source /home/lucas.ocunha/sci-image-markdown/.venv/bin/activate
export PYTHONPATH=.

echo "================================================================="
echo "📊 Consolidando resultados da avaliacao oficial..."
echo "Data/Hora: $(date)"
echo "================================================================="

# 1) Consolidar resultados
python scripts/consolidar_resultados.py

# 2) Atualizar BENCHMARK_RESULTS.md
python - <<'EOF'
import json
from pathlib import Path

benchmark_file = Path("docs/BENCHMARK_RESULTS.md")
out_dir = Path("data/resultados_pro_slide")
checkpoints = [
    ("lightonocr_52_32", "LightOnOCR-3-4B", "52,32"),
    ("lightonocr_synth15", "LightOnOCR-3-4B + Synth15", "51,72"),
    ("qwen7b", "Qwen2.5-VL-7B-Instruct", "52,11"),
    ("qwen3_vl_8b", "Qwen3-VL-8B-Instruct", "48,72"),
    ("qwen3b", "Qwen2.5-VL-3B-Instruct (Orig Res)", "49,49"),
]

rows = []
for cid, name, local_old in checkpoints:
    s_path = out_dir / cid / "summary.json"
    if s_path.exists():
        with open(s_path) as f:
            s = json.load(f)
        weighted = f"{s.get('weighted', 0):.2f}"
        rms = f"{s.get('rms', 0):.2f}"
        teds = f"{s.get('teds', 0):.2f}"
        n_panels = s.get('n_panels', 0)
        extra = s.get('extra_tables', 0)
    else:
        weighted = rms = teds = n_panels = extra = "-"
    rows.append(f"| {name} | {local_old} | {weighted} | {rms} | {teds} | {n_panels} | {extra} |")

section = [
    "## Avaliação Oficial (Task 2 Sci-ImageMiner - Por Painel)",
    "",
    "Repontuação no split de teste (373 figuras, 605 painéis) com a métrica oficial.",
    "Referências: 1º Lugar do Leaderboard = **41,81**, Baseline Oficial = **35,97**.",
    "",
    "| Checkpoint | Local (antigo) | Oficial (`weighted`) | RMS oficial | TEDS oficial | `n_panels` | `extra_tables` |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
] + rows + [""]

content = benchmark_file.read_text()
if "## Avaliação Oficial" in content:
    idx = content.find("## Avaliação Oficial")
    content = content[:idx] + "\n".join(section)
else:
    content = content + "\n\n" + "\n".join(section)

benchmark_file.write_text(content)
print("docs/BENCHMARK_RESULTS.md atualizado com sucesso!")
EOF

# 3) Git commit & push
echo "Subindo resultados no git..."
git add data/resultados_pro_slide docs/BENCHMARK_RESULTS.md .gitignore scripts/
git status

if git diff --staged --quiet; then
  echo "Nada para commitar."
else
  git commit -m "feat(eval): adicionar resultados da avaliacao oficial para slides"
  git push origin main
  echo "✅ Push realizado com sucesso no branch main!"
fi

echo "================================================================="
echo "🎉 Pipeline completo finalizado com sucesso!"
echo "Data/Hora: $(date)"
echo "================================================================="
