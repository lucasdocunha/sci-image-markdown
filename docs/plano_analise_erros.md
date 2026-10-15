# Análise de Erros (Task 2) — Plano de Implementação

> **Para agentes:** executar tarefa a tarefa (superpowers:executing-plans ou subagent-driven-development). Passos em checkbox (`- [ ]`). **Sem commits** até a Larissa pedir.

**Objetivo:** ferramenta agnóstica de modelo que recebe um arquivo de predições, dá a nota **oficial** do ICDAR por painel e mostra onde e por que o modelo erra, priorizado pelo quanto de nota cada erro custa.

**Arquitetura:** conversor (predições do repo → formato oficial, tabelas separadas por ordem) → nota por painel com o código oficial copiado sem alterações → colunas de recorte → `panels.csv` → notebook de revisão manual. As categorias de erro e o relatório entram por cima do `panels.csv` (Partes 2 e 3).

**Stack:** Python 3.10, pandas, código oficial do Sci-ImageMiner (DePlot RMS + TEDS PubTables), pytest, click (CLI), Jupyter (revisão).

**Spec:** a seção "Design aprovado" abaixo.

## Status

| Parte | Tarefas | Status |
|---|---|---|
| 1 — Fluxo e peças | T1–T7 | **concluída em 15/10**: 42/42 testes passando (`.venv` Python 3.11, torch 2.14 + CUDA). Cada teste novo foi visto falhando com a implementação sabotada. Desvios: o filtro de cercas do `split_tables` saiu (era redundante, porque cerca já encerra o bloco), e o teste de "uma fonte de predições" passou a checar a mensagem de erro (antes passava pelo motivo errado). O notebook (T7) ainda não foi aberto com dados reais |
| 2 — Categorias de erro | T8–T9 | **código concluído em 15/10**: 67/67 testes, cada teste novo visto falhando (25 sabotagens). Notebook executado de ponta a ponta com o `panels.csv` real (50 painéis, 50 imagens, sem erro). Revisão dos 50 painéis e calibração **concluídas em 15/10** (ver T9) |
| 3 — Priorização e relatório | T10–T11 | **parcial em 15/10**: `category_costs` e `slice_scores` prontos (70/70 testes) e relatório em `docs/analise_erros/analise_erros_dev.md` (+ 8 casos, no formato do `exemplos_analise_erros/`). Pendente: oráculo e `write_report` automático |

---

## Design aprovado

**Decisões (08/10/2026):**

- Entregável: ferramenta reutilizável + relatório do modelo atual.
- Agnóstica de modelo: a entrada é um arquivo de predições, não importa quem gerou.
- Análise que guia decisões: **dev/validação** (104 figuras com tabela, 167 painéis; 36 figuras multi-painel). O teste (373) fica fora.
- Régua: a **métrica oficial por painel** (`½·RMS + ½·TEDS`, ambas 0–100). As métricas locais (`src/metrics/`) servem só de diagnóstico auxiliar.
- Painéis: a saída do modelo é quebrada em tabelas e mapeada **por ordem** (1ª → 1º painel com gabarito não vazio, e assim por diante), como uma submissão real faria.
- Entrada canônica: **formato oficial** (`{"sample_id", "data_extraction": {painel: tabela}}`, JSONL), com conversor a partir do `predictions.jsonl` do repo.
- Recortes: tipo de figura (rótulo da Task 1), nº de painéis, tamanho do gabarito.
- Gabarito suspeito: sinalizado e reportado à parte, fora da conta de "o que corrigir no modelo".
- Forma: script CLI + notebook só de revisão.
- Prioridade: pontos perdidos por categoria (total e exclusivo) + oráculo nos erros consertáveis por código.
- Revisão manual: ~50 painéis (25 piores + 25 aleatórios, seed 42).
- 1º relatório: 3B LoRA do HF (`lucasoc/sci-image-models`) + base zero-shot, dev, `configs/evaluation/original_resolution.yaml`.

**Parte 1 — Fluxo:**

```
predictions.jsonl (repo) ─► convert ─► predições oficiais (JSONL)
metadata.jsonl (HF, split validation) ─► reference ──┤
                                                     ▼
                         scoring (código oficial, painel a painel)
                                                     ▼
                         panels.csv + summary.json  ─►  notebook de revisão ─► review.csv
```

**Parte 2 — Categorias** (um painel pode ter várias; limiares calibrados na revisão):

| Categoria | Regra automática |
|---|---|
| `painel_faltando` | painel com gabarito sem tabela prevista |
| `tabela_invalida` | reprovada em `is_valid_markdown_table_strict` |
| `colunas_divergentes` | nº de colunas ≠ gabarito |
| `amostragem_linhas` | `pred_rows / gt_rows` fora de [0,67; 1,5] |
| `cabecalho_desalinhado` | mesmo nº de colunas, similaridade média dos cabeçalhos (rapidfuzz `token_sort_ratio`) < 60 |
| `escala_magnitude` | mediana de `pred/gt` na 2ª coluna numérica com `|log10| ≥ 0,9` e a menos de 0,15 de um inteiro |
| `grade_regular_inventada` | 1ª coluna prevista com passo constante (desvio < 1% do passo médio) e a do gabarito não |
| `numero_nao_numerico` | células previstas com `~`, `10^`, `approx`, `≈` |

Gabarito suspeito (`gt_suspeito`): mais de 50 linhas, coluna majoritariamente numérica com texto, célula com `;` ou intervalo `\d+\s*-\s*\d+`, vírgula decimal (`21,78`) ou `~`/`≈`/`approx`/`10^` (os dois últimos vieram dos casos reais do Lucas e do dev).

**Parte 3 — Prioridade e relatório:** perda do painel = `100 − weighted`. Custo da categoria = Σ perda dos painéis dela ÷ nº total de painéis. Versão exclusiva = só painéis com aquela única categoria. Oráculos: (a) limpar `~`/`≈`/`approx` e converter `a×10^b`; (b) multiplicar pelo fator de escala detectado. Re-pontua e reporta o ganho. Recortes com *n* ao lado de cada %, sem conclusões com n < 5. Relatórios em `docs/analise_erros/<modelo>_dev.md`; saídas brutas em `outputs/analise_erros/` (gitignored).

## Global Constraints

- O código oficial é copiado **sem alterações** para `third_party/sci_imageminer_eval/` (CC BY 4.0 + Apache 2.0 do DePlot/pix2struct), com as licenças.
- A nota de um painel tem que ser idêntica à que o script oficial daria: `decode_unicode` nos dois lados, predição vazia ou inválida → `""` → 0, painel com gabarito vazio fica fora da média.
- RMS do painel = `table_datapoints_precision_recall([[flatten(gt)]], [flatten(pred)])["table_datapoints_f1"]`. TEDS = `teds_pubtables_official(gt, pred) * 100`.
- Novas dependências: `beautifulsoup4`, `rapidfuzz`, `editdistance`, `pycocoevalcap`, `jsonlines`.
- Não mudar o comportamento de treino nem de métricas locais. O único arquivo existente tocado é `evaluate.py` (dois campos a mais no JSONL).

## Review Focus

1. **Tabela prevista sem espaço ao redor de `|`** (`|33|7.5|`): o RMS oficial divide por `" | "` e zera. A ferramenta deve **reproduzir** isso, não consertar. Teste em T5.
2. **Mais tabelas previstas que painéis:** as extras são ignoradas na nota e contadas em `extra_tables`. Teste em T4.
3. **Figura do dev sem predição** (arquivo parcial ou erro de inferência): os painéis dela entram com nota 0, não somem da média. Teste em T6.
4. **Saída com cercas ```` ```markdown ```` e texto antes/depois:** `split_tables` devolve só os blocos de tabela. Teste em T4.
5. **`predictions.jsonl` antigo sem `raw_output`:** o conversor usa `markdown_table` (só a 1ª tabela, por causa do `extract_markdown_table_block`) e emite um aviso de que o multi-painel fica subestimado. Teste em T4.

---

## Parte 1

### T1: Código oficial copiado + dependências

**Files:**
- Create: `third_party/sci_imageminer_eval/` ← `sci-imageminer_evaluation.py`, `metrics.py`, `teds_pubtables.py`, `deplot_metric/metrics.py`, `pix2struct/metrics.py`, `license.txt`, `README.md` (de `github.com/sciknoworg/sci-imageminer/tree/main/evaluation_scripts`)
- Create: `third_party/sci_imageminer_eval/ORIGEM.md` (URL, data da cópia, "não editar")
- Modify: `requirements.txt`, `pyproject.toml` (dependências da Global Constraints)

- [x] Copiar os arquivos sem editar.
- [x] Verificar: `python -c "import sys; sys.path.insert(0,'third_party/sci_imageminer_eval'); from deplot_metric import metrics; from teds_pubtables import teds_pubtables_official"` → sem erro.

### T2: `evaluate.py` grava `id` e `raw_output`

**Files:** Modify `evaluate.py:108-114`; Test `tests/test_evaluate_saved_fields.py`

**Produces:** cada linha do `predictions.jsonl` passa a ter também `"id": rec.get("id")` e `"raw_output": res["raw_output"]` (`""` em caso de erro).

- [x] Teste `test_saved_predictions_have_id_and_raw_output`: `CliRunner` com `TablePredictor` monkeypatched (`predict_image` → `{"raw_output": "| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 2 |", "markdown_table": "| a |\n|---|\n| 1 |", "is_valid_table": True}`), `val.jsonl` de 1 registro em `tmp_path`; asserta `item["id"] == "fig_x"` e `"| b |" in item["raw_output"]`.
- [x] Rodar → FAIL (KeyError `id`). Implementar. Rodar → PASS.

### T3: Gabarito por painel

**Files:** Create `src/analysis/__init__.py`, `src/analysis/reference.py`; Test `tests/test_analysis_reference.py`; Fixture `tests/fixtures/analysis/metadata.jsonl` (3 figuras: 1 painel; 2 painéis com tabela; 2 painéis em que só `b` tem tabela)

**Produces:**
- `@dataclass PanelRef: sample_id: str, figure_id: str, image_file: str, panel_id: str, gt_table: str, figure_type: str, n_panels: int, bbox: Optional[dict]`. `n_panels` = nº de painéis da figura **com** tabela; `figure_id` = campo `id`; `image_file` = basename de `file_name`.
- `load_panels(metadata_path: str) -> List[PanelRef]`: só painéis com `data` não vazio, na ordem da lista `data_extraction`.
- `to_official_reference(panels: List[PanelRef]) -> List[dict]`: `[{"sample_id", "data_extraction": {panel_id: gt_table}}]`.

- [x] Testes: a 3ª figura gera 1 `PanelRef` com `panel_id == "b"` e `n_panels == 1`; `figure_type` vem de `classification[panel_id]`.
- [x] FAIL → implementar → PASS.

### T4: Conversor

**Files:** Create `src/analysis/convert.py`; Test `tests/test_analysis_convert.py`

**Consumes:** `PanelRef`, `load_panels` (T3).
**Produces:**
- `split_tables(text: str) -> List[str]`: remove todas as cercas ```` ``` ````, depois junta sequências máximas de linhas cujo `strip()` começa com `|`. Cada sequência com ≥ 2 linhas vira uma tabela (linhas com `strip`: o conversor faz o papel de quem monta a submissão, e o validador oficial reprova linha com espaço à esquerda).
- `to_official_predictions(pred_records: List[dict], panels: List[PanelRef]) -> Tuple[List[dict], List[dict]]`: devolve `(predicoes_oficiais, info)`. Casa o registro com a figura por `id` e, na falta dele, por basename de `image` == `image_file`. Usa `raw_output` e, na falta dele, `markdown_table` com `warnings.warn`. Tabelas mapeadas por ordem aos painéis da figura. `info` traz uma linha por figura: `{"sample_id", "n_pred_tables", "extra_tables"}`.

- [x] Testes: `test_split_tables_ignores_fences_and_text`; `test_maps_tables_by_order` (2 tabelas → `a`, `b`); `test_extra_tables_are_counted` (3 tabelas para 2 painéis → `extra_tables == 1`); `test_matches_by_basename_without_id`; `test_missing_raw_output_warns` (`pytest.warns(UserWarning)`).
- [x] FAIL → implementar → PASS.

### T5: Nota oficial por painel

**Files:** Create `src/analysis/scoring.py`; Test `tests/test_analysis_scoring.py`; Fixtures `tests/fixtures/analysis/reference.json`, `tests/fixtures/analysis/predictions.jsonl` (gerados de T3/T4 sobre o `metadata.jsonl` da fixture, com uma predição boa, uma ruim, uma sem espaços `|33|7.5|` e um painel sem predição)

**Consumes:** código de T1 (carregar `sci-imageminer_evaluation.py` via `importlib.util.spec_from_file_location`, porque o nome tem hífen).
**Produces:** `@dataclass PanelScore: pred_valid: bool, rms: float, teds: float, weighted: float` e `score_panel(gt_table: str, pred_table: Optional[str]) -> PanelScore`, seguindo a Global Constraints.

- [x] Teste `test_mean_matches_official_script`: roda o script oficial por `subprocess` (`--task data_extraction --reference-path ... --prediction-path ... --output-scores-path tmp`) e compara `rms`, `teds` e `weighted_score` com a média de `score_panel` nos mesmos painéis (`pytest.approx(abs=1e-9)`).
- [x] Teste `test_table_without_spaces_gets_zero_rms` (Review Focus 1): `score_panel(gt, "|x|y|\n|---|---|\n|33|7.5|").rms == 0`.
- [x] FAIL → implementar → PASS.

### T6: CLI `analyze_errors.py`

**Files:** Create `analyze_errors.py` (raiz, padrão click de `evaluate.py`); Test `tests/test_analyze_errors_cli.py`

**Consumes:** T3, T4, T5.
**Interface:** `python analyze_errors.py --metadata data/raw/validation_metadata.jsonl (--repo-predictions P.jsonl | --official-predictions O.jsonl) --out-dir outputs/analise_erros/<nome>`
**Produces:**
- `panels.csv`: `sample_id, panel_id, figure_type, n_panels, gt_rows, gt_cols, pred_rows, pred_cols, pred_valid, rms, teds, weighted, gt_table, pred_table`. `gt_*`/`pred_*` contam linhas de dados e colunas pela linha de cabeçalho, com 0 quando não há tabela.
- `summary.json`: `{"n_panels", "rms", "teds", "weighted"}` (médias sobre todos os painéis com gabarito), mais `extra_tables` (soma) quando a entrada é `--repo-predictions`.
- `official_predictions.jsonl` (resultado do conversor, reaproveitável no script oficial).

- [x] Teste `test_missing_figure_scores_zero` (Review Focus 3): fixture com uma figura ausente nas predições → os painéis dela aparecem no CSV com `weighted == 0` e `summary["n_panels"]` conta todos.
- [x] FAIL → implementar → PASS.

### T7: Notebook de revisão

**Files:** Create `notebooks/revisao_erros.ipynb`

**Consumes:** `panels.csv` (T6), imagens em `data/raw/images/`, `bbox` via `load_panels`.
- [x] Células: (1) parâmetros `PANELS_CSV`, `REVIEW_CSV`, `SEED = 42`; (2) amostra = 25 menores `weighted` + 25 aleatórios do resto; (3) para cada painel, crop pelo `bbox` + `gt_table` e `pred_table` lado a lado; (4) grava ou atualiza `review.csv` com `sample_id, panel_id, categorias, gt_suspeito, observacao`, preenchido à mão.
- [ ] Verificação manual: abrir com um `panels.csv` real e conferir 3 painéis (15/10: notebook executado headless sem erro e 1 painel conferido, o recorte do bbox está certo).

## Parte 2 — aguardando ok

### T8: Regras de categoria

**Files:** Create `src/analysis/taxonomy.py`; Test `tests/test_analysis_taxonomy.py`
**Produces:** `classify_panel(gt_table: str, pred_table: Optional[str]) -> List[str]`, com os nomes exatos da tabela da Parte 2, mais `CATEGORIES: Tuple[str, ...]`. O CLI ganha a coluna `categorias` (separada por `;`).
- [x] Um teste por regra, com um par mínimo gt/pred que dispara e outro que não dispara. FAIL → implementar → PASS. `is_valid_table` foi extraída para `scoring.py` (usada pelo `score_panel` e pela `tabela_invalida`); a comparação de cabeçalho usa `utils.default_process` (ignora maiúsculas e pontuação).

### T9: Gabarito suspeito + calibração

**Files:** Modify `src/analysis/taxonomy.py` (`flag_suspicious_gt(gt_table: str) -> bool`); Create `src/analysis/calibration.py` (`rule_precision(panels_csv: str, review_csv: str) -> pd.DataFrame`: precisão e recall de cada regra contra a marcação manual).
- [x] Testes de cada condição do `gt_suspeito`. FAIL → implementar → PASS. O `review.csv` do notebook ganhou `categorias_regra` (só leitura), e a calibração devolve zeros quando nada foi revisado ainda.
- [x] Revisar os 50 painéis, rodar a calibração, ajustar os limiares e registrar aqui o desvio. **Feito em 15/10.** Desvio: a revisão foi feita por 5 revisores Claude (10 painéis cada), sem ver a saída das regras e com um guia de definições comum. Conferi à mão 5 casos (27, 29, 30, 36 e 45), todos coerentes com a imagem. Precisão/recall por regra: `painel_faltando` 1,00/1,00 · `tabela_invalida` 1,00/1,00 · `colunas_divergentes` 0,93/0,81 · `amostragem_linhas` 0,50→**0,80**/0,89 (limite inferior de 0,67 para 0,55 e ignora gabarito de 1 linha; ajustado e medido na mesma amostra, então é otimista) · `grade_regular_inventada` 0,67/0,60 · `cabecalho_desalinhado` 0,50/0,33 e `escala_magnitude` 0/— (poucos casos, sem ajuste). `gt_suspeito`: o revisor marcou **45/50** (20 dos 25 aleatórios) com valor ou eixo errado; a regra pega só formato (precisão 0,88, recall 0,16). Achado: 34% das tabelas de treino têm X em grade regular, contra 70% nas predições do LoRA.

## Parte 3 — depende da Parte 2

### T10: Custo, oráculo e relatório

**Files:** Create `src/analysis/report.py` (`category_costs(df) -> pd.DataFrame` com `n`, `custo_total`, `custo_exclusivo`; `oracle_gains(df) -> pd.DataFrame`; `write_report(df, out_md: str, model_name: str)`); CLI ganha `--report`.
- [x] Teste: 2 painéis de 50 pontos em `escala_magnitude` num total de 4 → `custo_total == 25.0`. Desvio: em vez de `write_report`, o relatório foi escrito à mão a partir de `category_costs`/`slice_scores`, com diagnósticos conferidos nas imagens.
- [ ] Teste: o oráculo de `~0.5` → `0.5` aumenta o `rms` do painel de exemplo.

### T11: Primeiro relatório (precisa de GPU)

- [x] Baixar o dev: `python prepare_data.py --download-hf` (gera `data/processed/val.jsonl` e `data/raw/validation_metadata.jsonl`).
- [x] LoRA: `python evaluate.py --training-config configs/evaluation/original_resolution.yaml --test-file data/processed/val.jsonl --adapter-path lucasoc/sci-image-models --output-report outputs/analise_erros/lora3b/report_local.json --save-predictions outputs/analise_erros/lora3b/predictions.jsonl`. O prompt de usuário mudou na `main` em 14/10 (pede uma tabela por painel), então anotar o commit usado.
- [x] Base: o mesmo comando sem `--adapter-path`, com saída em `outputs/analise_erros/base3b/`.
- [x] Predições geradas em 15/10 (commit `713a02e`, adapter antigo + prompt da `main`) e nota oficial conferida contra o script oficial (idêntica): LoRA 3B **35,04** (RMS 15,56 · TEDS 54,53), base 3B **23,20** (RMS 8,79 · TEDS 37,61), 167 painéis do dev. O LoRA repete a mesma tabela em ~30 figuras (69 tabelas extras), provável efeito do prompt novo.
- [ ] `analyze_errors.py ... --report` nos dois, e salvar em `docs/analise_erros/lora3b_dev.md` e `base3b_dev.md`.
