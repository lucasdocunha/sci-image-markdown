"""Generate Sci-ImageMiner-like figures paired with Markdown data tables.

The generator mirrors the compact ``id``/``image``/``table`` schema and split
proportions of Sci-ImageMiner Task 2. It creates varied, publication-style
scientific charts while retaining the exact plotted values in each target.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, Sequence

import click
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image


DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_WEIGHTS = {"easy": 0.45, "medium": 0.40, "hard": 0.15}
SPLIT_PROPORTIONS = {"train": 723 / 1200, "val": 104 / 1200, "test": 373 / 1200}
FIGURE_SIZES = ((600, 460), (640, 480), (675, 520), (720, 500), (760, 570))
COLORS = ("#111111", "#d62728", "#1756b1", "#2ca02c", "#9467bd", "#e3771f")
MARKERS = ("o", "s", "^", "v", "D", "P")


@dataclass(frozen=True)
class Template:
    """A chart family and the Sci-ImageMiner identity fields it represents."""

    name: str
    difficulty: str
    subdomain: str
    study_type: str
    builder: Callable[[random.Random], tuple[Image.Image, str]]


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    """Return a compact table in the canonical dataset format."""
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _format_number(value: float, decimals: int = 2) -> str:
    rounded = round(value, decimals)
    return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")


def _render_chart(
    rng: random.Random,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float]]],
    x_label: str,
    y_label: str,
    *,
    errors: Sequence[Sequence[float]] | None = None,
    log_y: bool = False,
    panels: bool = False,
    annotation: str | None = None,
    plot_kind: str = "line",
    x_tick_labels: Sequence[str] | None = None,
) -> Image.Image:
    """Render a compact chart using conventions common in journal figures."""
    width, height = rng.choice(FIGURE_SIZES)
    if panels:
        width = max(width, 720)
        height = min(height, 500)
    dpi = 100
    figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2) if panels else [figure.subplots()]
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes.flat)
        except AttributeError:
            axes = [axes]

    if panels:
        midpoint = math.ceil(len(series) / 2)
        grouped_series = [
            [(i, *series[i]) for i in range(midpoint)],
            [(i, *series[i]) for i in range(midpoint, len(series))],
        ]
    else:
        grouped_series = [[(i, *item) for i, item in enumerate(series)]]

    for panel_index, (axis, panel_series) in enumerate(zip(axes, grouped_series)):
        for series_index, name, y_values in panel_series:
            color = COLORS[series_index % len(COLORS)]
            marker = MARKERS[series_index % len(MARKERS)]
            line_style = "-" if series_index % 3 else "--"
            if errors is not None:
                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=errors[series_index],
                    color=color,
                    marker=marker,
                    linestyle=line_style,
                    linewidth=1.35,
                    markersize=5,
                    capsize=2.5,
                    label=name,
                )
            elif plot_kind == "scatter":
                axis.scatter(
                    x_values,
                    y_values,
                    color=color,
                    marker=marker,
                    s=30,
                    facecolors="none" if series_index % 2 == 0 else color,
                    linewidths=1.2,
                    label=name,
                )
            elif plot_kind == "step":
                axis.step(
                    x_values,
                    y_values,
                    where="post",
                    color=color,
                    linewidth=1.4,
                    label=name,
                )
                axis.scatter(x_values, y_values, color=color, marker=marker, s=13)
            else:
                axis.plot(
                    x_values,
                    y_values,
                    color=color,
                    marker=marker,
                    linestyle=line_style,
                    linewidth=1.4,
                    markersize=5,
                    markerfacecolor=color if series_index % 2 else "white",
                    markeredgewidth=1.1,
                    label=name,
                )

        axis.set_xlabel(x_label, fontsize=11)
        axis.set_ylabel(y_label, fontsize=11)
        if x_tick_labels is not None:
            axis.set_xticks(x_values, x_tick_labels, rotation=18)
        axis.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
        axis.minorticks_on()
        for spine in axis.spines.values():
            spine.set_color("#202020")
            spine.set_linewidth(1.0)
        if rng.random() < 0.28:
            axis.grid(True, color="#d5d5d5", linewidth=0.55, alpha=0.8)
        if log_y:
            axis.set_yscale("log")
        if len(series) > 1:
            axis.legend(frameon=False, fontsize=8, loc="best", handlelength=2.2)
        if panels:
            axis.text(
                0.02,
                0.96,
                f"({chr(97 + panel_index)})",
                transform=axis.transAxes,
                va="top",
                fontsize=12,
                fontweight="bold",
            )
        if annotation and panel_index == 0:
            axis.text(
                0.05,
                0.84,
                annotation,
                transform=axis.transAxes,
                fontsize=9,
                fontweight="bold",
            )
        axis.margins(x=0.06, y=0.12)

    figure.subplots_adjust(
        left=0.14 if not panels else 0.09,
        right=0.97,
        bottom=0.16,
        top=0.96,
        wspace=0.34 if panels else 0.2,
    )
    return _figure_to_image(figure, dpi)


def _figure_to_image(figure: Figure, dpi: int = 100) -> Image.Image:
    """Serialize a Matplotlib figure to the RGB JPEG-like image used by builders."""
    buffer = BytesIO()
    figure.savefig(buffer, format="jpeg", dpi=dpi, facecolor="white", pil_kwargs={"quality": 92})
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        image = rendered.convert("RGB")
    figure.clear()
    buffer.close()
    return image


def _style_axis(axis, *, grid: bool = False) -> None:
    axis.tick_params(which="both", direction="in", top=True, right=True, labelsize=9)
    axis.minorticks_on()
    for spine in axis.spines.values():
        spine.set_color("#202020")
        spine.set_linewidth(1.0)
    if grid:
        axis.grid(True, color="#d5d5d5", linewidth=0.55, alpha=0.8)


def _render_bar_chart(
    rng: random.Random,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    x_label: str,
    y_label: str,
) -> Image.Image:
    width, height = rng.choice(FIGURE_SIZES)
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    positions = list(range(len(labels)))
    bar_width = 0.72 / len(series)
    for index, (name, values) in enumerate(series):
        offsets = [position + (index - (len(series) - 1) / 2) * bar_width for position in positions]
        axis.bar(
            offsets,
            values,
            width=bar_width,
            color=COLORS[(index + 2) % len(COLORS)],
            edgecolor="#202020",
            linewidth=0.6,
            label=name,
        )
    axis.set_xticks(positions, labels, rotation=25 if any(len(label) > 6 for label in labels) else 0)
    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel(y_label, fontsize=11)
    _style_axis(axis, grid=rng.random() < 0.2)
    if len(series) > 1:
        axis.legend(frameon=False, fontsize=8)
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.2, top=0.96)
    return _figure_to_image(figure)


def _render_grouped_scatter(
    rng: random.Random,
    groups: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    x_label: str,
    y_label: str,
    *,
    connect: bool = False,
) -> Image.Image:
    width, height = rng.choice(FIGURE_SIZES)
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for index, (name, xs, ys) in enumerate(groups):
        color = COLORS[index % len(COLORS)]
        if connect:
            axis.plot(xs, ys, color=color, linewidth=1.0, alpha=0.8)
        axis.scatter(
            xs,
            ys,
            color=color,
            marker=MARKERS[index % len(MARKERS)],
            s=31,
            facecolors="none" if index % 2 == 0 else color,
            linewidths=1.1,
            label=name,
        )
    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel(y_label, fontsize=11)
    _style_axis(axis, grid=rng.random() < 0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2 if len(groups) > 4 else 1)
    axis.margins(x=0.06, y=0.12)
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.16, top=0.96)
    return _figure_to_image(figure)


def _render_dual_axis(
    rng: random.Random,
    x_values: Sequence[float],
    left_series: Sequence[tuple[str, Sequence[float]]],
    right_series: Sequence[tuple[str, Sequence[float]]],
    x_label: str,
    left_label: str,
    right_label: str,
) -> Image.Image:
    width, height = rng.choice(FIGURE_SIZES)
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    left_axis = figure.subplots()
    right_axis = left_axis.twinx()
    handles = []
    for index, (name, values) in enumerate(left_series):
        handle = left_axis.plot(
            x_values,
            values,
            color=COLORS[index],
            marker=MARKERS[index],
            linewidth=1.3,
            label=name,
        )[0]
        handles.append(handle)
    for offset, (name, values) in enumerate(right_series, start=len(left_series)):
        handle = right_axis.plot(
            x_values,
            values,
            color=COLORS[offset],
            marker=MARKERS[offset],
            linestyle="--",
            linewidth=1.3,
            label=name,
        )[0]
        handles.append(handle)
    left_axis.set_xlabel(x_label, fontsize=11)
    left_axis.set_ylabel(left_label, fontsize=11)
    right_axis.set_ylabel(right_label, fontsize=11)
    _style_axis(left_axis, grid=rng.random() < 0.2)
    right_axis.tick_params(direction="in", labelsize=9)
    left_axis.legend(handles, [handle.get_label() for handle in handles], frameon=False, fontsize=8)
    figure.subplots_adjust(left=0.14, right=0.86, bottom=0.16, top=0.96)
    return _figure_to_image(figure)


def _render_distinct_panels(
    rng: random.Random,
    panels: Sequence[
        tuple[Sequence[float], Sequence[tuple[str, Sequence[float]]], str, str]
    ],
) -> Image.Image:
    vertical = rng.random() < 0.5
    width, height = ((700, 930) if vertical else (980, 480))
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1) if vertical else figure.subplots(1, 2)
    for panel_index, (axis, (xs, series, x_label, y_label)) in enumerate(zip(axes.flat, panels)):
        for series_index, (name, ys) in enumerate(series):
            axis.plot(
                xs,
                ys,
                color=COLORS[series_index],
                marker=MARKERS[series_index],
                linewidth=1.25,
                markersize=4,
                label=name,
            )
        axis.set_xlabel(x_label, fontsize=10)
        axis.set_ylabel(y_label, fontsize=10)
        axis.text(0.02, 0.96, f"({chr(97 + panel_index)})", transform=axis.transAxes,
                  va="top", fontsize=12, fontweight="bold")
        _style_axis(axis)
        if len(series) > 1:
            axis.legend(frameon=False, fontsize=7)
    figure.subplots_adjust(
        left=0.11 if vertical else 0.08,
        right=0.97,
        bottom=0.09 if vertical else 0.16,
        top=0.97,
        hspace=0.30,
        wspace=0.30,
    )
    return _figure_to_image(figure)


def _render_with_inset(
    rng: random.Random,
    main_x: Sequence[float],
    main_y: Sequence[float],
    inset_x: Sequence[float],
    inset_y: Sequence[float],
    x_label: str,
    y_label: str,
    inset_x_label: str,
) -> Image.Image:
    width, height = rng.choice(FIGURE_SIZES)
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(main_x, main_y, color=COLORS[2], marker="s", linewidth=1.3)
    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel(y_label, fontsize=11)
    _style_axis(axis)
    inset = axis.inset_axes((0.52, 0.48, 0.42, 0.42))
    inset.plot(inset_x, inset_y, color=COLORS[1], marker="o", linewidth=1.0, markersize=3)
    inset.set_xlabel(inset_x_label, fontsize=7)
    inset.set_ylabel(y_label, fontsize=7)
    inset.tick_params(direction="in", labelsize=6)
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.16, top=0.96)
    return _figure_to_image(figure)


def _easy_growth(rng: random.Random) -> tuple[Image.Image, str]:
    step = rng.choice((5, 10, 20))
    x_values = list(range(0, step * rng.randint(7, 11), step))
    growth = rng.uniform(0.07, 0.19)
    offset = rng.uniform(0.0, 0.3)
    y_values = [max(0.0, offset + growth * x + rng.uniform(-0.18, 0.18)) for x in x_values]
    y_values = [round(value, 2) for value in y_values]
    image = _render_chart(
        rng,
        x_values,
        [("Film thickness", y_values)],
        "Number of ALD Cycles",
        "Film Thickness (nm)",
        annotation=f"GPC = {growth:.3f} nm/cycle",
    )
    table = markdown_table(
        ["Number of ALD Cycles", "Film Thickness (nm)"],
        ((str(x), _format_number(y)) for x, y in zip(x_values, y_values)),
    )
    return image, table


def _easy_temperature(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = list(range(100, 351, 50))
    start = rng.uniform(-5, 15)
    slope = rng.uniform(-0.9, -0.45)
    y_values = [round(start + slope * (x - 100) + rng.uniform(-7, 7), 1) for x in x_values]
    image = _render_chart(
        rng,
        x_values,
        [("Mass change", y_values)],
        "Temperature (°C)",
        "Mass Change Per Cycle (ng/cm²)",
    )
    table = markdown_table(
        ["Temperature (°C)", "Mass Change Per Cycle (ng/cm²)"],
        ((str(x), _format_number(y, 1)) for x, y in zip(x_values, y_values)),
    )
    return image, table


def _easy_spectrum(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = list(range(500, 4001, 250))
    center = rng.uniform(1700, 2600)
    width = rng.uniform(260, 500)
    y_values = [
        round(0.05 + 0.72 * math.exp(-((x - center) / width) ** 2) + rng.uniform(-0.025, 0.025), 3)
        for x in x_values
    ]
    image = _render_chart(
        rng,
        x_values,
        [("Absorbance", y_values)],
        "Wavenumber (cm⁻¹)",
        "Absorbance (a.u.)",
    )
    table = markdown_table(
        ["Wavenumber (cm⁻¹)", "Absorbance (a.u.)"],
        ((str(x), f"{y:.3f}") for x, y in zip(x_values, y_values)),
    )
    return image, table


def _easy_publications_bar(rng: random.Random) -> tuple[Image.Image, str]:
    years = list(range(1992, 2013, 2))
    growth = rng.uniform(0.20, 0.28)
    values = [max(0, round(math.exp(growth * (year - 1992)) + rng.uniform(-2, 2))) for year in years]
    image = _render_bar_chart(
        rng,
        [str(year) for year in years],
        [("Publications", values)],
        "Publication Year",
        "Publications per Year",
    )
    table = markdown_table(
        ["Year", "Publications per Year"],
        ((str(year), str(value)) for year, value in zip(years, values)),
    )
    return image, table


def _easy_surface_coverage(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = list(range(0, rng.choice((61, 71, 81)), 10))
    slope = rng.uniform(0.38, 0.62)
    delay = rng.choice((0, 3, 5))
    y_values = [
        round(max(0, slope * (cycle - delay) + rng.uniform(-0.7, 0.7)), 1)
        for cycle in x_values
    ]
    image = _render_chart(
        rng,
        x_values,
        [("Ti coverage", y_values)],
        "Number of Cycles",
        "Ti Atom Surface Coverage (×10¹⁵/cm²)",
        plot_kind="scatter",
    )
    table = markdown_table(
        ["Number of Cycles", "Ti Atom Surface Coverage (×10¹⁵/cm²)"],
        ((str(x), f"{y:.1f}") for x, y in zip(x_values, y_values)),
    )
    return image, table


def _easy_resistivity(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = [0, 1, 2, 5, 10, 20, 30]
    floor = rng.uniform(65, 130)
    start = rng.uniform(6000, 14000)
    decay = rng.uniform(0.35, 0.65)
    y_values = [round(floor + start * math.exp(-decay * x) * rng.uniform(0.94, 1.06), 1) for x in x_values]
    image = _render_chart(
        rng,
        x_values,
        [("4-point probe", y_values)],
        "H₂ Plasma Exposure Time (s)",
        "Resistivity (µΩ·cm)",
        log_y=True,
        plot_kind="scatter",
    )
    table = markdown_table(
        ["H₂ Plasma Exposure Time (s)", "Resistivity (µΩ·cm)"],
        ((str(x), f"{y:.1f}") for x, y in zip(x_values, y_values)),
    )
    return image, table


def _easy_saturation(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    plateau = rng.uniform(24, 38)
    rate = rng.uniform(1.0, 2.8)
    y_values = [round(plateau * (1 - math.exp(-rate * x)) + rng.uniform(-0.5, 0.5), 2) for x in x_values]
    y_values[0] = max(0.0, y_values[0])
    image = _render_chart(
        rng,
        x_values,
        [("TMA exposure", y_values)],
        "Exposure Time (s)",
        "Mass Gain per Cycle (ng/cm²)",
        plot_kind="scatter",
    )
    table = markdown_table(
        ["Exposure Time (s)", "Mass Gain per Cycle (ng/cm²)"],
        ((f"{x:.1f}", f"{y:.2f}") for x, y in zip(x_values, y_values)),
    )
    return image, table


def make_easy(rng: random.Random) -> tuple[Image.Image, str]:
    builders = (
        _easy_growth,
        _easy_temperature,
        _easy_spectrum,
        _easy_publications_bar,
        _easy_surface_coverage,
        _easy_resistivity,
        _easy_saturation,
    )
    return rng.choice(builders)(rng)


def _medium_temperature_series(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = list(range(0, 21, 2))
    temperatures = ("128°C", "160°C", "196°C")
    starts = [rng.uniform(125, 145) for _ in temperatures]
    slopes = [rng.uniform(0.35, 1.8) for _ in temperatures]
    values = [
        [round(start - slope * x + rng.uniform(-0.8, 0.8), 1) for x in x_values]
        for start, slope in zip(starts, slopes)
    ]
    image = _render_chart(
        rng,
        x_values,
        list(zip(temperatures, values)),
        "Number of Cycles",
        "Film Thickness (Å)",
    )
    table = markdown_table(
        ["Number of Cycles", *temperatures],
        ((str(x), *(f"{column[row]:.1f}" for column in values)) for row, x in enumerate(x_values)),
    )
    return image, table


def _medium_decay(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = [0, 5, 10, 20, 35, 50, 75]
    low_power = [round(9.8 * math.exp(-x / 52) + rng.uniform(-0.12, 0.12), 3) for x in x_values]
    high_power = [round(9.8 * math.exp(-x / 25) + rng.uniform(-0.12, 0.12), 3) for x in x_values]
    image = _render_chart(
        rng,
        x_values,
        [("100 W", low_power), ("500 W", high_power)],
        "O₂ Plasma Exposure Time (s)",
        "Effective Carrier Lifetime (ms)",
    )
    table = markdown_table(
        ["O₂ Plasma Exposure Time (s)", "100 W (ms)", "500 W (ms)"],
        ((str(x), f"{low_power[i]:.3f}", f"{high_power[i]:.3f}") for i, x in enumerate(x_values)),
    )
    return image, table


def _medium_pressure(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = [round(i * 0.005, 3) for i in range(10)]
    pressures = ("0 torr", "0.05 torr", "0.1 torr")
    values = []
    for index in range(len(pressures)):
        amplitude = rng.uniform(0.8e-5, 2.8e-5) * (index + 1)
        values.append(
            [
                max(
                    1e-8,
                    round(amplitude * (1 - math.exp(-x / 0.009)) + rng.uniform(-8e-7, 8e-7), 8),
                )
                for x in x_values
            ]
        )
    image = _render_chart(
        rng,
        x_values,
        list(zip(pressures, values)),
        "Time (s)",
        "Concentration (mol/m³)",
    )
    table = markdown_table(
        ["Time (s)", *(f"TMA – {label} (mol/m³)" for label in pressures)],
        ((f"{x:.3f}", *(f"{column[row]:.2e}" for column in values)) for row, x in enumerate(x_values)),
    )
    return image, table


def _medium_reactant_saturation(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    water_plateau = rng.uniform(30, 37)
    tma_plateau = rng.uniform(26, 32)
    water = [round(water_plateau * (1 - math.exp(-1.2 * x)) + rng.uniform(-0.45, 0.45), 2) for x in x_values]
    tma = [round(tma_plateau * (1 - math.exp(-3.0 * x)) + rng.uniform(-0.45, 0.45), 2) for x in x_values]
    image = _render_grouped_scatter(
        rng,
        [("Water (1,30,x,30)", x_values, water), ("TMA (y,30,1,30)", x_values, tma)],
        "Exposure Time (s)",
        "Mass Gain per Cycle (ng/cm²)",
        connect=True,
    )
    rows = []
    for reagent, values in (("Water (1,30,x,30)", water), ("TMA (y,30,1,30)", tma)):
        rows.extend((f"{x:.1f}", f"{y:.2f}", reagent) for x, y in zip(x_values, values))
    return image, markdown_table(
        ["Exposure Time (s)", "Mass Gain per Cycle (ng/cm²)", "Reagent (cycle conditions)"],
        rows,
    )


def _medium_measurement_methods(rng: random.Random) -> tuple[Image.Image, str]:
    temperatures = [33, 58, 80, 102, 125, 177]
    methods = (
        "Ellipsometry + QCM",
        "Profilometry + QCM",
        "AFM + QCM",
        "XRR",
        "RBS/FReS + Ellipsometry",
        "Lorentz–Lorenz",
    )
    groups = []
    rows = []
    for index, method in enumerate(methods):
        values = [
            round(2.35 + 0.0037 * temperature + index * 0.025 + rng.uniform(-0.12, 0.12), 2)
            for temperature in temperatures
        ]
        groups.append((method, temperatures, values))
        rows.extend((str(x), f"{y:.2f}", method) for x, y in zip(temperatures, values))
    image = _render_grouped_scatter(
        rng,
        groups,
        "Growth Temperature (°C)",
        "Density (g/cm³)",
    )
    return image, markdown_table(
        ["Growth Temperature (°C)", "Density (g/cm³)", "Measurement"],
        rows,
    )


def _medium_dual_axis(rng: random.Random) -> tuple[Image.Image, str]:
    temperatures = [33, 58, 80, 102, 125, 177]
    dielectric = [round(rng.uniform(7.2, 8.4), 2) for _ in temperatures]
    field = [round(3.55 + 0.005 * temperature + rng.uniform(-0.18, 0.18), 2) for temperature in temperatures]
    image = _render_dual_axis(
        rng,
        temperatures,
        [("Dielectric constant", dielectric)],
        [("Breakdown field", field)],
        "Growth Temperature (°C)",
        "Dielectric Constant, k",
        "Breakdown Field (MV/cm)",
    )
    table = markdown_table(
        ["Growth Temperature (°C)", "Dielectric Constant, k", "Breakdown Field (MV/cm)"],
        ((str(x), f"{dielectric[i]:.2f}", f"{field[i]:.2f}") for i, x in enumerate(temperatures)),
    )
    return image, table


def _medium_emission_spectra(rng: random.Random) -> tuple[Image.Image, str]:
    wavelengths = list(range(100, 801, 50))
    gases = ("O₂", "H₂", "N₂")
    centers = [rng.uniform(240, 340), rng.uniform(300, 430), rng.uniform(370, 520)]
    values = []
    for gas_index, center in enumerate(centers):
        secondary = center + rng.uniform(160, 250)
        values.append(
            [
                round(
                    0.03
                    + (0.8 + 0.15 * gas_index) * math.exp(-((x - center) / 42) ** 2)
                    + 0.45 * math.exp(-((x - secondary) / 65) ** 2)
                    + rng.uniform(0, 0.035),
                    3,
                )
                for x in wavelengths
            ]
        )
    image = _render_chart(
        rng,
        wavelengths,
        list(zip(gases, values)),
        "Wavelength (nm)",
        "Emission Intensity (a.u.)",
    )
    table = markdown_table(
        ["Wavelength (nm)", *(f"{gas} Intensity (a.u.)" for gas in gases)],
        ((str(x), *(f"{column[row]:.3f}" for column in values)) for row, x in enumerate(wavelengths)),
    )
    return image, table


def _medium_barrier_methods(rng: random.Random) -> tuple[Image.Image, str]:
    thicknesses = [0, 5, 10, 15]
    labels = ("PE-ALD TaN", "PVD TaN", "PE-ALD Ta", "PVD Ta")
    values = []
    for index in range(len(labels)):
        start = rng.uniform(560, 625) + index * 5
        values.append([round(start + rng.uniform(15, 20) * thickness) for thickness in thicknesses])
    image = _render_chart(
        rng,
        thicknesses,
        list(zip(labels, values)),
        "Film Thickness (nm)",
        "Barrier Failure Temperature (°C)",
    )
    table = markdown_table(
        ["Film Thickness (nm)", *labels],
        ((str(x), *(str(column[row]) for column in values)) for row, x in enumerate(thicknesses)),
    )
    return image, table


def make_medium(rng: random.Random) -> tuple[Image.Image, str]:
    builders = (
        _medium_temperature_series,
        _medium_decay,
        _medium_pressure,
        _medium_reactant_saturation,
        _medium_measurement_methods,
        _medium_dual_axis,
        _medium_emission_spectra,
        _medium_barrier_methods,
    )
    return rng.choice(builders)(rng)


def _hard_error_bars(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = [5, 10, 15, 20, 30, 45, 60, 80]
    labels = ("850 mT", "1600 mT", "2500 mT", "4000 mT")
    values = []
    errors = []
    for index in range(len(labels)):
        floor = 0.048 + index * 0.002
        values.append(
            [
                round(
                    floor + 0.105 * math.exp(-x / (8.5 + index)) + rng.uniform(-0.002, 0.002),
                    3,
                )
                for x in x_values
            ]
        )
        errors.append([round(rng.uniform(0.001, 0.004), 3) for _ in x_values])
    image = _render_chart(
        rng,
        x_values,
        list(zip(labels, values)),
        "Plasma Pulse (s)",
        "GPC (nm/cycle)",
        errors=errors,
    )
    table = markdown_table(
        [
            "Plasma Pulse (s)",
            *(
                heading
                for label in labels
                for heading in (f"{label} GPC (nm/cycle)", f"{label} SD")
            ),
        ],
        (
            (
                str(x),
                *(
                    cell
                    for column in range(len(values))
                    for cell in (f"{values[column][row]:.3f}", f"{errors[column][row]:.3f}")
                ),
            )
            for row, x in enumerate(x_values)
        ),
    )
    return image, table


def _hard_dual_panel(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = list(range(1, 14))
    labels = ("Present study A", "Reference A", "Present study B", "Reference B")
    values = []
    for index in range(len(labels)):
        baseline = 0.42 + 0.30 * (index // 2)
        values.append(
            [
                round(
                    baseline
                    + 0.035 * x
                    + 0.13 * math.sin(x * (0.65 + index * 0.05))
                    + rng.uniform(-0.04, 0.04),
                    2,
                )
                for x in x_values
            ]
        )
    image = _render_chart(
        rng,
        x_values,
        list(zip(labels, values)),
        "Cycle",
        "Growth per cycle (Å)",
        panels=True,
    )
    table = markdown_table(
        ["Cycle", *labels],
        ((str(x), *(f"{column[row]:.2f}" for column in values)) for row, x in enumerate(x_values)),
    )
    return image, table


def _hard_log_spectrum(rng: random.Random) -> tuple[Image.Image, str]:
    x_values = list(range(1000, 10001, 500))
    labels = ("TiN", "50c ALE", "100c ALE", "200c ALE")
    values = []
    for index in range(len(labels)):
        scale = 10 ** (rng.uniform(9.8, 10.3) - index * rng.uniform(0.95, 1.25))
        decay_length = rng.uniform(1400, 2050) + index * rng.uniform(80, 180)
        phase = rng.uniform(-0.7, 0.7)
        values.append(
            [
                round(
                    max(
                        1.0,
                        scale
                        * math.exp(-(x - 1000) / decay_length)
                        * (1 + rng.uniform(0.08, 0.19) * math.sin(x / rng.uniform(460, 610) + phase)),
                    ),
                    2,
                )
                for x in x_values
            ]
        )
    image = _render_chart(
        rng,
        x_values,
        list(zip(labels, values)),
        "ω–2θ (arcsec)",
        "X-Ray Intensity",
        log_y=True,
    )
    table = markdown_table(
        ["ω–2θ (arcsec)", *(f"{label} (Intensity)" for label in labels)],
        ((str(x), *(f"{column[row]:.2e}" for column in values)) for row, x in enumerate(x_values)),
    )
    return image, table


def _hard_dense_qcm(rng: random.Random) -> tuple[Image.Image, str]:
    points_per_cycle = 4
    cycles = rng.randint(15, 24)
    times = []
    masses = []
    time = rng.uniform(0, 3)
    mass = 0.0
    for _ in range(cycles):
        for phase, duration, gain in (
            ("TMA", rng.uniform(2, 5), rng.uniform(13, 19)),
            ("purge", rng.uniform(20, 32), rng.uniform(-4, -1)),
            ("H₂O", rng.uniform(2, 5), rng.uniform(8, 14)),
            ("purge", rng.uniform(20, 32), rng.uniform(-2, 0)),
        ):
            del phase
            time += duration
            mass += gain + rng.uniform(-0.8, 0.8)
            times.append(round(time, 1))
            masses.append(round(mass, 2))
    assert len(times) == cycles * points_per_cycle
    image = _render_chart(
        rng,
        times,
        [("QCM mass", masses)],
        "Time (s)",
        "Mass (ng/cm²)",
        plot_kind="step",
    )
    table = markdown_table(
        ["Time (s)", "Mass (ng/cm²)"],
        ((f"{x:.1f}", f"{y:.2f}") for x, y in zip(times, masses)),
    )
    return image, table


def _hard_stacked_xrr(rng: random.Random) -> tuple[Image.Image, str]:
    angles = [round(0.04 + index * 0.058, 3) for index in range(24)]
    temperatures = (33, 58, 102, 177)
    plot_series = []
    rows = []
    for temperature_index, temperature in enumerate(temperatures):
        scale = 10 ** (4.2 + temperature_index * 1.7 + rng.uniform(-0.2, 0.2))
        experimental = []
        fitted = []
        frequency = rng.uniform(18, 25)
        for angle in angles:
            envelope = scale * math.exp(-angle * rng.uniform(5.4, 6.4))
            fringe = (0.23 + math.sin(frequency * angle + temperature_index * 0.5) ** 2) ** 2
            fit_value = max(1.0, envelope * fringe)
            experimental_value = max(1.0, fit_value * rng.uniform(0.78, 1.24))
            fitted.append(round(fit_value, 3))
            experimental.append(round(experimental_value, 3))
        plot_series.extend(
            (
                (f"{temperature}°C exp.", experimental),
                (f"{temperature}°C fit", fitted),
            )
        )
        rows.extend(
            (str(temperature), "Experimental", f"{angle:.3f}", f"{value:.4g}")
            for angle, value in zip(angles, experimental)
        )
        rows.extend(
            (str(temperature), "Fit", f"{angle:.3f}", f"{value:.4g}")
            for angle, value in zip(angles, fitted)
        )
    image = _render_chart(
        rng,
        angles,
        plot_series,
        "θ (degrees)",
        "Intensity (a.u.)",
        log_y=True,
    )
    table = markdown_table(
        ["Temperature (°C)", "Source", "θ (degrees)", "Intensity (a.u.)"],
        rows,
    )
    return image, table


def _hard_three_properties(rng: random.Random) -> tuple[Image.Image, str]:
    temperatures = [33, 58, 80, 102, 125, 177]
    density = [round(2.32 + 0.004 * x + rng.uniform(-0.08, 0.08), 2) for x in temperatures]
    refractive = [round(1.49 + 0.00075 * x + rng.uniform(-0.012, 0.012), 3) for x in temperatures]
    growth = [
        round(
            1.08 + 0.27 * math.exp(-((x - 112) / 70) ** 2) + rng.uniform(-0.025, 0.025),
            3,
        )
        for x in temperatures
    ]
    image = _render_dual_axis(
        rng,
        temperatures,
        [("Density", density), ("Refractive index", refractive)],
        [("Growth rate", growth)],
        "Growth Temperature (°C)",
        "Density / Refractive Index",
        "Growth Rate (Å/cycle)",
    )
    table = markdown_table(
        ["Growth Temperature (°C)", "Density (g/cm³)", "Refractive Index n", "Growth Rate (Å/cycle)"],
        (
            (str(x), f"{density[i]:.2f}", f"{refractive[i]:.3f}", f"{growth[i]:.3f}")
            for i, x in enumerate(temperatures)
        ),
    )
    return image, table


def _hard_distinct_panels(rng: random.Random) -> tuple[Image.Image, str]:
    trench = [0, 20, 40, 60, 80, 100]
    profiles = []
    profile_labels = (
        "Reaction-limited",
        "Diffusion-limited",
        "Recombination-limited A",
        "Recombination-limited B",
    )
    for index, label in enumerate(profile_labels):
        drop = rng.uniform(0.04, 0.16) * (index + 1)
        values = [round(100 - drop * depth + rng.uniform(-1.2, 1.2), 1) for depth in trench]
        profiles.append((label, values))
    probabilities = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    dose_10 = [round(1 + 650 * value**2 + rng.uniform(-1, 1), 1) for value in probabilities]
    dose_30 = [round(1 + 1100 * value**1.65 + rng.uniform(-2, 2), 1) for value in probabilities]
    image = _render_distinct_panels(
        rng,
        [
            (trench, profiles, "Distance into Trench (%)", "Equivalent Thickness (%)"),
            (
                probabilities,
                [("Aspect ratio 10", dose_10), ("Aspect ratio 30", dose_30)],
                "Surface Recombination Probability",
                "Normalized Dose for Saturation",
            ),
        ],
    )
    first_table = markdown_table(
        ["Distance into Trench (%)", *profile_labels],
        ((str(x), *(f"{series[1][row]:.1f}" for series in profiles)) for row, x in enumerate(trench)),
    )
    second_table = markdown_table(
        [
            "Surface Recombination Probability (r)",
            "Dose for Saturation (Aspect Ratio = 10)",
            "Dose for Saturation (Aspect Ratio = 30)",
        ],
        (
            (f"{x:.2f}", f"{dose_10[i]:.1f}", f"{dose_30[i]:.1f}")
            for i, x in enumerate(probabilities)
        ),
    )
    return image, first_table + "\n\n" + second_table


def _hard_inset_wvtr(rng: random.Random) -> tuple[Image.Image, str]:
    temperatures = list(range(20, 121, 10))
    optimum = rng.uniform(55, 80)
    minimum = rng.uniform(0.004, 0.012)
    temperature_wvtr = [
        round(minimum + rng.uniform(1.2e-6, 3.5e-6) * (temperature - optimum) ** 2, 4)
        for temperature in temperatures
    ]
    thicknesses = [0, 5, 10, 20, 30, 40, 50]
    thickness_wvtr = [
        round(0.38 * math.exp(-thickness / rng.uniform(5.5, 9.0)) + minimum, 4)
        for thickness in thicknesses
    ]
    image = _render_with_inset(
        rng,
        temperatures,
        temperature_wvtr,
        thicknesses,
        thickness_wvtr,
        "Deposition Temperature (°C)",
        "WVTR (g m⁻² day⁻¹)",
        "Al₂O₃ Thickness (nm)",
    )
    main_table = markdown_table(
        ["Deposition Temperature (°C)", "WVTR (g m⁻² day⁻¹)"],
        ((str(x), f"{y:.4f}") for x, y in zip(temperatures, temperature_wvtr)),
    )
    inset_table = markdown_table(
        ["Al₂O₃ Thickness (nm)", "WVTR (g m⁻² day⁻¹)"],
        ((str(x), f"{y:.4f}") for x, y in zip(thicknesses, thickness_wvtr)),
    )
    return image, main_table + "\n\n" + inset_table


def _hard_process_window(rng: random.Random) -> tuple[Image.Image, str]:
    processes = (
        ("ALD", 1, 10, 1e9, 1e12),
        ("Plasma Chemistry", 1, 1e3, 1e9, 1e15),
        ("Film Deposition", 1, 1e4, 1e9, 1e18),
        ("Etching", 10, 1e4, 1e11, 1e17),
        ("Implantation", 1e2, 1e5, 1e14, 1e17),
    )
    jittered = []
    for name, x0, x1, y0, y1 in processes:
        factor = rng.uniform(0.85, 1.15)
        jittered.append((name, x0 * factor, x1 * factor, y0, y1))
    width, height = rng.choice(FIGURE_SIZES)
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.set_xscale("log")
    axis.set_yscale("log")
    for index, (name, x0, x1, y0, y1) in enumerate(jittered):
        patch = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=COLORS[index],
            edgecolor=COLORS[index],
            alpha=0.18,
            linewidth=1.2,
        )
        axis.add_patch(patch)
        axis.text(math.sqrt(x0 * x1), math.sqrt(y0 * y1), name, ha="center", va="center", fontsize=8)
    axis.set_xlim(0.7, 2e5)
    axis.set_ylim(5e8, 3e18)
    axis.set_xlabel("Ion Energy (eV)", fontsize=11)
    axis.set_ylabel("Ion Flux Density (cm⁻²s⁻¹)", fontsize=11)
    _style_axis(axis, grid=True)
    figure.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.96)
    image = _figure_to_image(figure)
    table = markdown_table(
        ["Process", "Ion Energy Range (eV)", "Ion Flux Density Range (cm⁻²s⁻¹)"],
        (
            (name, f"{x0:.2g} – {x1:.2g}", f"{y0:.0e} – {y1:.0e}")
            for name, x0, x1, y0, y1 in jittered
        ),
    )
    return image, table


def _hard_energy_pathways(rng: random.Random) -> tuple[Image.Image, str]:
    steps = ("Reactant", "Complex", "TS", "Product A", "Product B")
    x_values = list(range(len(steps)))
    pathways = []
    for index, label in enumerate(("Path A", "Path B", "Path C")):
        barrier = rng.uniform(28, 48) + index * rng.uniform(2, 6)
        product = rng.uniform(-22, 8) + index * rng.uniform(3, 9)
        values = [
            0.0,
            round(rng.uniform(-18, -6), 1),
            round(barrier, 1),
            round(product, 1),
            round(product + rng.uniform(4, 15), 1),
        ]
        pathways.append((label, values))
    image = _render_chart(
        rng,
        x_values,
        pathways,
        "Reaction Pathway",
        "ΔE (kcal/mol)",
        x_tick_labels=steps,
    )
    table = markdown_table(
        ["Reaction Step", *(name for name, _ in pathways)],
        ((step, *(f"{values[row]:.1f}" for _, values in pathways)) for row, step in enumerate(steps)),
    )
    return image, table


def make_hard(rng: random.Random) -> tuple[Image.Image, str]:
    builders = (
        _hard_error_bars,
        _hard_dual_panel,
        _hard_log_spectrum,
        _hard_dense_qcm,
        _hard_stacked_xrr,
        _hard_three_properties,
        _hard_distinct_panels,
        _hard_inset_wvtr,
        _hard_process_window,
        _hard_energy_pathways,
    )
    return rng.choice(builders)(rng)


BUILDERS: dict[str, Callable[[random.Random], tuple[Image.Image, str]]] = {
    "easy": make_easy,
    "medium": make_medium,
    "hard": make_hard,
}


TEMPLATES = (
    Template("ald_growth", "easy", "atomic_layer_deposition", "experimental", _easy_growth),
    Template("ale_mass_change", "easy", "atomic_layer_etching", "experimental", _easy_temperature),
    Template("infrared_spectrum", "easy", "atomic_layer_deposition", "experimental", _easy_spectrum),
    Template("publication_bar", "easy", "atomic_layer_deposition", "experimental", _easy_publications_bar),
    Template("surface_coverage", "easy", "atomic_layer_deposition", "experimental", _easy_surface_coverage),
    Template("plasma_resistivity", "easy", "atomic_layer_deposition", "experimental", _easy_resistivity),
    Template("single_saturation", "easy", "atomic_layer_deposition", "experimental", _easy_saturation),
    Template("temperature_series", "medium", "atomic_layer_etching", "experimental", _medium_temperature_series),
    Template("carrier_lifetime", "medium", "atomic_layer_deposition", "experimental", _medium_decay),
    Template("precursor_transport", "medium", "atomic_layer_deposition", "simulation", _medium_pressure),
    Template("reactant_saturation", "medium", "atomic_layer_deposition", "experimental", _medium_reactant_saturation),
    Template("measurement_methods", "medium", "atomic_layer_deposition", "experimental", _medium_measurement_methods),
    Template("electrical_dual_axis", "medium", "atomic_layer_deposition", "experimental", _medium_dual_axis),
    Template("emission_spectra", "medium", "atomic_layer_deposition", "experimental", _medium_emission_spectra),
    Template("barrier_methods", "medium", "atomic_layer_deposition", "experimental", _medium_barrier_methods),
    Template("gpc_error_bars", "hard", "atomic_layer_deposition", "experimental", _hard_error_bars),
    Template("growth_dual_panel", "hard", "atomic_layer_deposition", "simulation", _hard_dual_panel),
    Template("ale_log_spectrum", "hard", "atomic_layer_etching", "experimental", _hard_log_spectrum),
    Template("dense_qcm", "hard", "atomic_layer_deposition", "experimental", _hard_dense_qcm),
    Template("stacked_xrr", "hard", "atomic_layer_deposition", "experimental", _hard_stacked_xrr),
    Template("three_properties", "hard", "atomic_layer_deposition", "experimental", _hard_three_properties),
    Template("recombination_panels", "hard", "atomic_layer_deposition", "simulation", _hard_distinct_panels),
    Template("wvtr_inset", "hard", "atomic_layer_deposition", "experimental", _hard_inset_wvtr),
    Template("plasma_process_window", "hard", "atomic_layer_deposition", "simulation", _hard_process_window),
    Template("reaction_pathways", "hard", "atomic_layer_deposition", "simulation", _hard_energy_pathways),
)


def _allocate_counts(total: int, proportions: dict[str, float]) -> dict[str, int]:
    exact = {name: total * proportion for name, proportion in proportions.items()}
    sizes = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(sizes.values())
    order = sorted(exact, key=lambda name: exact[name] - sizes[name], reverse=True)
    for name in order[:remainder]:
        sizes[name] += 1
    return sizes


def _schedule_templates(
    total: int,
    difficulty: str,
    rng: random.Random,
) -> list[Template]:
    if difficulty == "all":
        difficulty_counts = _allocate_counts(total, DIFFICULTY_WEIGHTS)
        levels = [level for level in DIFFICULTIES for _ in range(difficulty_counts[level])]
        rng.shuffle(levels)
    else:
        levels = [difficulty] * total

    templates_by_level = {
        level: [template for template in TEMPLATES if template.difficulty == level]
        for level in DIFFICULTIES
    }
    queues: dict[str, list[Template]] = {level: [] for level in DIFFICULTIES}
    scheduled = []
    for level in levels:
        if not queues[level]:
            queues[level] = templates_by_level[level].copy()
            rng.shuffle(queues[level])
        scheduled.append(queues[level].pop())
    return scheduled


def _split_sizes(total: int) -> dict[str, int]:
    """Allocate records with the official 723/104/373 split proportions."""
    return _allocate_counts(total, SPLIT_PROPORTIONS)


def split_records(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Split records using the official Sci-ImageMiner Task 2 proportions."""
    sizes = _split_sizes(len(records))
    result: dict[str, list[dict[str, str]]] = {}
    cursor = 0
    for name in ("train", "val", "test"):
        result[name] = records[cursor : cursor + sizes[name]]
        cursor += sizes[name]
    return result


def generate_dataset(
    output_dir: str | Path,
    num_samples: int,
    difficulty: str = "all",
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, int]:
    """Create JPEG figures and canonical JSONL splits."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if difficulty != "all" and difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of: all, {', '.join(DIFFICULTIES)}")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{destination} is not empty; choose an empty directory or set overwrite=True")
        shutil.rmtree(destination)
    images_dir = destination / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    generated: list[tuple[dict[str, str], str, str]] = []
    seen_tables: set[str] = set()
    duplicate_retries = 0
    for index, template in enumerate(_schedule_templates(num_samples, difficulty, rng)):
        for attempt in range(25):
            image, table = template.builder(rng)
            if table not in seen_tables:
                break
            duplicate_retries += 1
        else:
            raise RuntimeError(f"could not create a unique sample from template '{template.name}'")
        seen_tables.add(table)
        figure_name = ("figure_1", "fig_1")[index % 2]
        sample_id = (
            f"synthetic__{template.subdomain}__{template.study_type}_usecase__"
            f"{index + 1}__{figure_name}"
        )
        image_name = f"{sample_id}.jpg"
        image.save(images_dir / image_name, format="JPEG", quality=92, subsampling=0)
        record = {"id": sample_id, "image": f"images/{image_name}", "table": table}
        generated.append((record, template.difficulty, template.name))
    rng.shuffle(generated)

    records = [record for record, _, _ in generated]
    splits = split_records(records)
    for split_name, split_records_ in splits.items():
        with (destination / f"{split_name}.jsonl").open("w", encoding="utf-8") as file:
            for record in split_records_:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "num_samples": num_samples,
        "seed": seed,
        "difficulty_counts": dict(Counter(level for _, level, _ in generated)),
        "template_counts": dict(Counter(template for _, _, template in generated)),
        "unique_tables": len(seen_tables),
        "duplicate_retries": duplicate_retries,
        "splits": {name: len(items) for name, items in splits.items()},
        "split_proportions": SPLIT_PROPORTIONS,
        "image_format": "JPEG",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest["splits"]


@click.command()
@click.option("--output-dir", default="data/synthetic", show_default=True, help="Dataset output directory.")
@click.option(
    "--num-samples",
    default=90,
    show_default=True,
    type=click.IntRange(1, None),
    help="Total samples across all splits.",
)
@click.option(
    "--difficulty",
    default="all",
    show_default=True,
    type=click.Choice(["all", *DIFFICULTIES]),
    help="Difficulty level, or a processed-data-like weighted mix.",
)
@click.option("--seed", default=42, show_default=True, type=int, help="Random seed for reproducible data.")
@click.option("--overwrite", is_flag=True, help="Replace a non-empty output directory.")
def main(output_dir: str, num_samples: int, difficulty: str, seed: int, overwrite: bool) -> None:
    """Generate a synthetic scientific-image-to-Markdown dataset."""
    try:
        splits = generate_dataset(output_dir, num_samples, difficulty, seed, overwrite)
    except (FileExistsError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    except OSError as error:
        raise click.ClickException(
            f"Cannot create or replace '{output_dir}': {error.strerror or 'filesystem error'}. "
            "Choose a writable path, for example 'data/synthetic' relative to this project."
        ) from error
    click.echo(
        f"Generated {num_samples} samples in {output_dir} "
        f"(train={splits['train']}, val={splits['val']}, test={splits['test']})."
    )


if __name__ == "__main__":
    main()
