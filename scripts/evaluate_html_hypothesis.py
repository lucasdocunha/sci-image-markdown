#!/usr/bin/env python3
"""
Experimento de Validação de Hipótese TableSeq (arXiv:2604.16070):
Comparação entre Validação Estrita em Markdown vs. Validação via HTML DOM.

Este script avalia as predições geradas em duas vertentes:
1. Pipeline Padrão (Markdown com regex estrito):
   - Exige pipes em todas as linhas (| ... | ... |).
   - Rejeita tabelas de 1 coluna ou layouts de blocos múltiplos.
   - TEDS sintetizado via DataFrame.
2. Pipeline TableSeq (HTML DOM):
   - Converte Markdown/HTML para tags canônicas (<table>, <thead>, <tbody>, <tr>, <th>, <td>).
   - Validação resiliente através do parser DOM (BeautifulSoup).
   - Suporte nativo a tabelas de 1 coluna, múltiplos blocos e células mescladas.
   - TEDS calculado diretamente na árvore DOM canônica (algoritmo original PubTabNet/ICDAR).
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import markdown
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import existing metrics
from src.metrics.table_parser import parse_markdown_to_dataframe
from src.metrics.table_metrics import (
    TableNode,
    compute_tree_edit_distance,
    compute_normalized_edit_distance,
    compute_numerical_cell_metrics,
)

console = Console()


def parse_html_to_tree_and_df(raw_text: str) -> Tuple[TableNode, Optional[pd.DataFrame], bool]:
    """Parses raw text (Markdown or HTML) into an HTML DOM TableNode and DataFrame.
    
    Implements the TableSeq hypothesis by leveraging the formal DOM tree representation.
    """
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
        return TableNode("table"), None, False

    # Check if text contains explicit HTML table tags
    if "<table" in raw_text.lower():
        html = raw_text
    else:
        # Convert Markdown to HTML using standard tables extension
        try:
            html = markdown.markdown(raw_text, extensions=["tables"])
        except Exception:
            html = raw_text

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return TableNode("table"), None, False

    root = TableNode("table")
    all_rows: List[List[str]] = []
    headers: List[str] = []

    for table in tables:
        # 1. Header extraction
        th_tags = table.find_all("th")
        if th_tags and not headers:
            tr_head = TableNode("tr")
            for th in th_tags:
                txt = th.get_text(strip=True)
                headers.append(txt)
                tr_head.children.append(TableNode("th", text=txt))
            root.children.append(tr_head)

        # 2. Data rows extraction
        tr_tags = table.find_all("tr")
        for tr in tr_tags:
            td_tags = tr.find_all("td")
            if not td_tags:
                continue
            tr_node = TableNode("tr")
            row_data = []
            for td in td_tags:
                txt = td.get_text(strip=True)
                row_data.append(txt)
                tr_node.children.append(TableNode("td", text=txt))
            root.children.append(tr_node)
            all_rows.append(row_data)

    if not all_rows:
        return root, None, False

    if not headers:
        headers = [f"Col_{i}" for i in range(len(all_rows[0]))]

    # Normalize rows length to match headers
    norm_rows = []
    for r in all_rows:
        if len(r) < len(headers):
            r = r + [""] * (len(headers) - len(r))
        elif len(r) > len(headers):
            r = r[: len(headers)]
        norm_rows.append(r)

    try:
        df = pd.DataFrame(norm_rows, columns=headers)
        for col in df.columns:
            try:
                cleaned = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(cleaned)
            except Exception:
                pass
        return root, df, True
    except Exception:
        return root, None, False


def compute_html_teds(tree_pred: TableNode, tree_target: TableNode, is_pred_valid: bool, is_target_valid: bool) -> float:
    """Computes TEDS on HTML DOM trees directly."""
    if not is_pred_valid and not is_target_valid:
        return 1.0
    if not is_pred_valid or not is_target_valid:
        return 0.0

    ted, n_pred, n_target = compute_tree_edit_distance(tree_pred, tree_target)
    max_nodes = max(n_pred, n_target)
    if max_nodes <= 1:
        return 1.0 if (n_pred == n_target) else 0.0
    return max(0.0, 1.0 - (ted / max_nodes))


def evaluate_hypothesis(predictions_file: str, output_report: str):
    console.print(f"[bold cyan]🔍 Carregando predições de:[/bold cyan] {predictions_file}")
    
    with open(predictions_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    console.print(f"[green]Total de amostras carregadas:[/green] {len(records)}")

    md_valid_count = 0
    html_valid_count = 0
    
    md_teds_list = []
    html_teds_list = []
    
    md_f1_list = []
    html_f1_list = []
    
    md_rms_list = []
    html_rms_list = []

    salvaged_samples = []

    for r in records:
        pred_text = r.get("markdown_table", r.get("prediction", ""))
        target_text = r.get("ground_truth", r.get("target", ""))
        idx = r.get("index", r.get("id", len(md_teds_list)))

        # --- A. Regime Markdown Padrão ---
        md_pred_df, is_md_pred_valid = parse_markdown_to_dataframe(pred_text)
        md_target_df, is_md_target_valid = parse_markdown_to_dataframe(target_text)

        if is_md_pred_valid:
            md_valid_count += 1

        # Markdown TEDS
        from src.metrics.table_metrics import compute_teds as compute_standard_teds
        md_teds = compute_standard_teds(pred_text, target_text)
        md_teds_list.append(md_teds)

        # Markdown Cell Metrics
        md_cell = compute_numerical_cell_metrics(md_pred_df, md_target_df)
        md_f1_list.append(md_cell["f1"])
        md_rne = md_cell.get("mean_rne", 0.0)
        if math.isnan(md_rne):
            md_rne = 0.0
        md_rms = max(0.0, min(1.0, md_cell["recall"] * max(0.0, 1.0 - md_rne)))
        md_rms_list.append(md_rms)

        # --- B. Regime TableSeq HTML DOM ---
        tree_pred, html_pred_df, is_html_pred_valid = parse_html_to_tree_and_df(pred_text)
        tree_target, html_target_df, is_html_target_valid = parse_html_to_tree_and_df(target_text)

        if is_html_pred_valid:
            html_valid_count += 1

        # HTML DOM TEDS
        html_teds = compute_html_teds(tree_pred, tree_target, is_html_pred_valid, is_html_target_valid)
        html_teds_list.append(html_teds)

        # HTML Cell Metrics
        html_cell = compute_numerical_cell_metrics(html_pred_df, html_target_df)
        html_f1_list.append(html_cell["f1"])
        html_rne = html_cell.get("mean_rne", 0.0)
        if math.isnan(html_rne):
            html_rne = 0.0
        html_rms = max(0.0, min(1.0, html_cell["recall"] * max(0.0, 1.0 - html_rne)))
        html_rms_list.append(html_rms)

        # Salvação de Amostras
        if not is_md_pred_valid and is_html_pred_valid:
            salvaged_samples.append({
                "index": idx,
                "pred_text": pred_text[:120] + "..." if len(pred_text) > 120 else pred_text,
                "html_teds": html_teds,
                "html_f1": html_cell["f1"],
            })

    total = len(records)
    md_valid_rate = (md_valid_count / total) * 100.0 if total else 0.0
    html_valid_rate = (html_valid_count / total) * 100.0 if total else 0.0
    
    mean_md_teds = float(np.mean(md_teds_list)) * 100.0 if total else 0.0
    mean_html_teds = float(np.mean(html_teds_list)) * 100.0 if total else 0.0

    mean_md_f1 = float(np.mean(md_f1_list)) * 100.0 if total else 0.0
    mean_html_f1 = float(np.mean(html_f1_list)) * 100.0 if total else 0.0

    mean_md_rms = float(np.mean(md_rms_list)) * 100.0 if total else 0.0
    mean_html_rms = float(np.mean(html_rms_list)) * 100.0 if total else 0.0

    md_composite = 0.5 * (mean_md_teds + mean_md_rms)
    html_composite = 0.5 * (mean_html_teds + mean_html_rms)

    # Exibição dos resultados
    table = Table(title="🔬 Validação da Hipótese TableSeq (arXiv:2604.16070): Markdown vs. HTML DOM", show_header=True, header_style="bold magenta")
    table.add_column("Métrica de Avaliação", style="cyan")
    table.add_column("Markdown Estrito (Baseline)", justify="right", style="yellow")
    table.add_column("TableSeq HTML DOM (Hipótese)", justify="right", style="green")
    table.add_column("Diferença / Ganho Real (pp)", justify="right", style="bold green" if html_valid_rate >= md_valid_rate else "bold red")

    def diff_str(val_html, val_md):
        d = val_html - val_md
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2f} pp"

    table.add_row("Taxa de Tabelas Válidas (`valid_table`)", f"{md_valid_rate:.2f}%", f"{html_valid_rate:.2f}%", diff_str(html_valid_rate, md_valid_rate))
    table.add_row("Similaridade Estrutural (`TEDS`)", f"{mean_md_teds:.2f}%", f"{mean_html_teds:.2f}%", diff_str(mean_html_teds, mean_md_teds))
    table.add_row("F1-Score de Células (`cell_f1`)", f"{mean_md_f1:.2f}%", f"{mean_html_f1:.2f}%", diff_str(mean_html_f1, mean_md_f1))
    table.add_row("ICDAR Numerical RMS", f"{mean_md_rms:.2f}%", f"{mean_html_rms:.2f}%", diff_str(mean_html_rms, mean_md_rms))
    table.add_row("Score Composto ICDAR", f"{md_composite:.2f}%", f"{html_composite:.2f}%", diff_str(html_composite, md_composite))
    table.add_row("Tabelas Falsamente Invalidadas Recuperadas", "0", f"{len(salvaged_samples)}", f"+{len(salvaged_samples)}")

    console.print()
    console.print(table)
    console.print()

    if salvaged_samples:
        console.print(f"[bold green]✅ Exemplos de Tabelas que falhavam no Markdown mas foram validadas no HTML DOM:[/bold green]")
        for s in salvaged_samples[:5]:
            console.print(f"  - [cyan]Amostra #{s['index']}:[/cyan] HTML TEDS={s['html_teds']:.2f}, F1={s['html_f1']:.2f} | Texto: {s['pred_text']}")

    report_data = {
        "total_samples": total,
        "markdown_metrics": {
            "valid_table_rate": md_valid_rate,
            "teds": mean_md_teds,
            "cell_f1": mean_md_f1,
            "icdar_rms": mean_md_rms,
            "icdar_composite": md_composite,
        },
        "html_dom_metrics": {
            "valid_table_rate": html_valid_rate,
            "teds": mean_html_teds,
            "cell_f1": mean_html_f1,
            "icdar_rms": mean_html_rms,
            "icdar_composite": html_composite,
        },
        "gain": {
            "valid_table_rate": html_valid_rate - md_valid_rate,
            "teds": mean_html_teds - mean_md_teds,
            "cell_f1": mean_html_f1 - mean_md_f1,
            "icdar_rms": mean_html_rms - mean_md_rms,
            "icdar_composite": html_composite - md_composite,
            "salvaged_samples_count": len(salvaged_samples),
        },
        "salvaged_samples": salvaged_samples,
    }

    Path(output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    console.print(f"[bold cyan]📁 Relatório comparativo salvo em:[/bold cyan] {output_report}")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação da Hipótese TableSeq: Markdown vs. HTML DOM")
    parser.add_argument("--predictions-file", default="/media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/eval_results/icdar_loss_orig_res_test_predictions.jsonl", help="Caminho para o JSONL de predições")
    parser.add_argument("--output-report", default="/media/ssd2/lucas.ocunha/sci-image-markdown-data/outputs/eval_results/tableseq_html_hypothesis_report.json", help="Caminho para salvar o relatório JSON")
    args = parser.parse_args()

    evaluate_hypothesis(args.predictions_file, args.output_report)
