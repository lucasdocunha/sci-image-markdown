#!/usr/bin/env python3
import json
import os
import subprocess
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/lucas.ocunha/sci-image-markdown")
OUT_DIR = BASE_DIR / "data" / "resultados_pro_slide"

CHECKPOINTS = [
    {
        "id": "lightonocr_52_32",
        "display_name": "LightOnOCR-3-4B (Recorde)",
        "local_target": "52,32",
    },
    {
        "id": "lightonocr_synth15",
        "display_name": "LightOnOCR-3-4B + Synth15",
        "local_target": "51,72",
    },
    {
        "id": "qwen7b",
        "display_name": "Qwen2.5-VL-7B-Instruct",
        "local_target": "52,11",
    },
    {
        "id": "qwen3_vl_8b",
        "display_name": "Qwen3-VL-8B-Instruct",
        "local_target": "48,72",
    },
    {
        "id": "qwen3b",
        "display_name": "Qwen2.5-VL-3B-Instruct (Orig Res)",
        "local_target": "49,49",
    },
]

def main():
    rows = []
    for cp in CHECKPOINTS:
        cid = cp["id"]
        cp_dir = OUT_DIR / cid
        summary_file = cp_dir / "summary.json"
        local_report_file = cp_dir / "test_report_local.json"

        local_icdar = None
        local_rms = None
        local_teds = None
        if local_report_file.exists():
            with open(local_report_file) as f:
                rep = json.load(f)
                local_icdar = rep.get("icdar_score")
                local_rms = rep.get("icdar_rms")
                local_teds = rep.get("teds")

        if summary_file.exists():
            with open(summary_file) as f:
                s = json.load(f)
            
            weighted = s.get("weighted", 0.0)
            rms = s.get("rms", 0.0)
            teds = s.get("teds", 0.0)
            n_panels = s.get("n_panels", 0)
            extra_tables = s.get("extra_tables", 0)
            
            rows.append({
                "Checkpoint": cp["display_name"],
                "ID": cid,
                "Local (antigo)": cp["local_target"],
                "Local (re-executado)": f"{local_icdar*100:.2f}" if local_icdar is not None else "-",
                "Oficial (weighted)": f"{weighted:.2f}",
                "RMS oficial": f"{rms:.2f}",
                "TEDS oficial": f"{teds:.2f}",
                "n_panels": n_panels,
                "extra_tables": extra_tables,
                "Status": "Concluído"
            })
        else:
            rows.append({
                "Checkpoint": cp["display_name"],
                "ID": cid,
                "Local (antigo)": cp["local_target"],
                "Local (re-executado)": f"{local_icdar*100:.2f}" if local_icdar is not None else "-",
                "Oficial (weighted)": "-",
                "RMS oficial": "-",
                "TEDS oficial": "-",
                "n_panels": "-",
                "extra_tables": "-",
                "Status": "Pendente"
            })

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "tabela_resultados_slides.csv"
    df.to_csv(csv_path, index=False)
    print(f"Salvo CSV em {csv_path}")

    # Gera Markdown para apresentações/slides
    md_lines = [
        "# Resultados da Avaliação Oficial (Task 2 - Sci-ImageMiner)",
        "",
        "Avaliação realizada no split de **teste oficial (373 figuras, 605 painéis)** utilizando a métrica oficial por painel (DePlot Datapoints F1 para RMS e PubTables-1M para TEDS).",
        "",
        "### Referências da Competição no Teste:",
        "- **1º Lugar do Leaderboard:** 41,81",
        "- **Baseline Oficial:** 35,97",
        "",
        "### Tabela Comparativa:",
        "",
        "| Checkpoint | Métrica Local (antigo) | Oficial (`weighted`) | RMS Oficial | TEDS Oficial | Painéis Avaliados | Tabelas Extras | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['Checkpoint']} | {r['Local (antigo)']} | {r['Oficial (weighted)']} | {r['RMS oficial']} | {r['TEDS oficial']} | {r['n_panels']} | {r['extra_tables']} | {r['Status']} |"
        )

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("*Gerado automaticamente pelo pipeline de avaliação oficial.*")

    readme_path = OUT_DIR / "README.md"
    readme_path.write_text("\n".join(md_lines))
    print(f"Salvo README em {readme_path}")
    print("\n" + "\n".join(md_lines[:18]))

if __name__ == "__main__":
    main()
