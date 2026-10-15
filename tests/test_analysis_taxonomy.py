"""
Unit tests for the error category rules and the suspicious ground-truth flag.
"""

import pytest

from src.analysis.taxonomy import CATEGORIES, classify_panel, flag_suspicious_gt


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


GT = table(["Time (s)", "Mass (ng/cm2)"], [[0, 0], [7, 21.8], [15, 34.9], [42, 32.0], [46, 53.7]])


def test_categories_are_the_approved_ones():
    assert CATEGORIES == (
        "painel_faltando", "tabela_invalida", "colunas_divergentes", "amostragem_linhas",
        "cabecalho_desalinhado", "escala_magnitude", "grade_regular_inventada", "numero_nao_numerico",
    )


def test_exact_copy_has_no_category():
    assert classify_panel(GT, GT) == []


@pytest.mark.parametrize("pred", [None, ""])
def test_missing_panel(pred):
    assert classify_panel(GT, pred) == ["painel_faltando"]


def test_invalid_table():
    assert classify_panel(GT, "Time, Mass\n0, 0") == ["tabela_invalida"]


def test_column_mismatch():
    pred = table(["Time (s)", "Mass (ng/cm2)", "Note"], [[0, 0, "a"], [7, 21.8, "b"], [15, 34.9, "c"], [42, 32.0, "d"], [46, 53.7, "e"]])
    assert "colunas_divergentes" in classify_panel(GT, pred)


def test_row_sampling_too_few():
    pred = table(["Time (s)", "Mass (ng/cm2)"], [[0, 0], [46, 53.7]])
    assert "amostragem_linhas" in classify_panel(GT, pred)


def test_row_sampling_within_tolerance():
    pred = table(["Time (s)", "Mass (ng/cm2)"], [[0, 0], [7, 21.8], [15, 34.9], [46, 53.7]])
    assert "amostragem_linhas" not in classify_panel(GT, pred)


def test_misaligned_header():
    pred = table(["Temperature (K)", "Pressure (Torr)"], [[0, 0], [7, 21.8], [15, 34.9], [42, 32.0], [46, 53.7]])
    assert "cabecalho_desalinhado" in classify_panel(GT, pred)


def test_header_with_small_typo_and_case_is_aligned():
    pred = table(["TIME(S)", "MASS (NG/CM²)"], [[0, 0], [7, 21.8], [15, 34.9], [42, 32.0], [46, 53.7]])
    assert "cabecalho_desalinhado" not in classify_panel(GT, pred)


def test_scale_off_by_power_of_ten():
    pred = table(["Time (s)", "Mass (ng/cm2)"], [[0, 0], [7, 218], [15, 349], [42, 320], [46, 537]])
    assert "escala_magnitude" in classify_panel(GT, pred)


def test_scale_off_by_factor_two_is_not_magnitude():
    pred = table(["Time (s)", "Mass (ng/cm2)"], [[0, 0], [7, 43.6], [15, 69.8], [42, 64.0], [46, 107.4]])
    assert "escala_magnitude" not in classify_panel(GT, pred)


def test_invented_regular_grid():
    pred = table(["Time (s)", "Mass (ng/cm2)"], [[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]])
    assert "grade_regular_inventada" in classify_panel(GT, pred)


def test_regular_grid_matching_regular_gt_is_fine():
    gt = table(["Cycles", "Thickness"], [[0, 0], [10, 25], [20, 50], [30, 75]])
    assert "grade_regular_inventada" not in classify_panel(gt, gt)


def test_non_numeric_number():
    pred = table(["Time (s)", "Mass (ng/cm2)"], [[0, "~0"], [7, "~22"], [15, "≈35"], [42, 32.0], [46, "5x10^1"]])
    assert "numero_nao_numerico" in classify_panel(GT, pred)


def test_clean_gt_is_not_suspicious():
    assert flag_suspicious_gt(GT) is False


@pytest.mark.parametrize("gt", [
    table(["x", "y"], [[i, i] for i in range(51)]),                       # too many rows
    table(["x", "y"], [[1, 2], [2, 3], [3, "sharply rising"]]),           # text in numeric column
    table(["x", "y"], [[1, "8.4 ; 7.5"], [2, 3]]),                        # two values in one cell
    table(["Wavelength", "y"], [["360-400", 2], [780, 3]]),              # range as a value
    table(["Time (s)", "Mass"], [[0, "0,00"], [7, "21,78"]]),            # decimal comma
    table(["x", "y"], [[0, "~10"], [5, "~48"]]),                         # approximate values
])
def test_suspicious_gt(gt):
    assert flag_suspicious_gt(gt) is True
