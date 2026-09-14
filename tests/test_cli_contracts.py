"""
Static contract tests between the CLI entrypoints and the classes they construct.

These exist because train.py called SciImageTableDataset(max_image_resolution=...)
after that parameter had been removed from the constructor, which made `python
train.py` fail with a TypeError on the first line of work. The unit tests did not
catch it: they instantiate the dataset directly with their own arguments.

The checks are AST-based on purpose. Importing train.py pulls in torch,
transformers and bitsandbytes, so an import-based test could not run on a machine
without CUDA -- exactly where this bug would be introduced and reviewed.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# CLI module -> (constructed class, module defining it)
CONSTRUCTOR_CONTRACTS = [
    ("train.py", "SciImageTableDataset", "src/data/dataset.py"),
    ("evaluate.py", "TablePredictor", "src/inference/predictor.py"),
    ("predict.py", "TablePredictor", "src/inference/predictor.py"),
]


def _parse(rel_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def _init_signature(rel_path: str, class_name: str):
    """Returns (accepted_param_names, accepts_kwargs) for a class __init__."""
    for node in ast.walk(_parse(rel_path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = item.args
                    names = {a.arg for a in args.args} | {a.arg for a in args.kwonlyargs}
                    return names - {"self"}, args.kwarg is not None
    raise AssertionError(f"{class_name}.__init__ not found in {rel_path}")


def _call_keywords(rel_path: str, callee_name: str):
    """Every keyword argument used to call `callee_name` in a module."""
    keywords = []
    for node in ast.walk(_parse(rel_path)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != callee_name:
            continue
        keywords.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return keywords


@pytest.mark.parametrize("cli_module,class_name,class_module", CONSTRUCTOR_CONTRACTS)
def test_cli_constructs_class_with_accepted_arguments(cli_module, class_name, class_module):
    accepted, accepts_kwargs = _init_signature(class_module, class_name)
    if accepts_kwargs:
        pytest.skip(f"{class_name} accepts **kwargs; no static contract to enforce")

    call_sites = _call_keywords(cli_module, class_name)
    assert call_sites, f"{cli_module} never constructs {class_name}"

    for passed in call_sites:
        unknown = passed - accepted
        assert not unknown, (
            f"{cli_module} passes {sorted(unknown)} to {class_name}(), which accepts "
            f"only {sorted(accepted)}. Update the constructor or the call site."
        )


def test_resolution_is_controlled_in_exactly_one_place():
    """Training and inference must share one resolution mechanism.

    Previously the only resize lived in predictor.py, so the model trained at
    near-native resolution and was evaluated on 280px thumbnails. The budget now
    lives on the processor, which both paths go through.
    """
    loader_src = (REPO_ROOT / "src/models/loader.py").read_text(encoding="utf-8")
    assert "min_pixels" in loader_src and "max_pixels" in loader_src

    resize_free = [
        "src/inference/predictor.py",
        "src/data/dataset.py",
        "src/data/collator.py",
        "predict.py",
        "evaluate.py",
        "train.py",
    ]
    for rel_path in resize_free:
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert ".thumbnail(" not in src, (
            f"{rel_path} resizes images itself, which desynchronises training and "
            "inference. Resolution belongs to the processor budget in loader.py."
        )
