"""
Pipeline do experimento B: M1 (esqueleto) e M2 (valores) sobre a imagem, depois M3 (fusão, só texto).

M1 e M2 são independentes entre si: rodam em sequência neste processo, mas podem ser
submetidos como jobs separados (um por adapter) e combinados depois.
"""

from typing import Any, Dict, Optional

import torch
from PIL import Image

from ..data.decomposed import (
    FUSION_SYSTEM_PROMPT,
    build_fusion_user_text,
    clean_model_output,
    fill_skeleton,
    is_table_line,
)
from ..models.loader import load_model_and_processor
from ..models.qwen_vl import QwenVLTableExtractor
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def _prepare_image(image: Image.Image, max_res: Optional[int]) -> Image.Image:
    image = image.convert("RGB")
    if max_res:
        image.thumbnail((max_res, max_res))
    return image


class _ImageExtractor:
    """M1 ou M2: imagem -> texto (saída crua, limpa depois)."""

    def __init__(self, cfg: Dict[str, Any], adapter_path: Optional[str]):
        eval_cfg = cfg.get("evaluation", {})
        data_cfg = cfg.get("data", {})
        self.max_res = data_cfg.get("max_image_resolution")
        self.model, self.processor = load_model_and_processor(cfg, is_training=False, adapter_path=adapter_path)
        self.model.eval()
        self.extractor = QwenVLTableExtractor(
            model=self.model,
            processor=self.processor,
            system_prompt=data_cfg["system_prompt"],
            max_new_tokens=eval_cfg.get("max_new_tokens", 1024),
            temperature=eval_cfg.get("temperature", 0.0),
            min_pixels=data_cfg.get("min_pixels"),
            max_pixels=data_cfg.get("max_pixels"),
        )

    def predict(self, image: Image.Image) -> str:
        return self.extractor.predict(_prepare_image(image, self.max_res))


class _TextFuser:
    """M3: texto (esqueleto + valores) -> tabela final. Sem imagem."""

    def __init__(self, cfg: Dict[str, Any], adapter_path: Optional[str]):
        eval_cfg = cfg.get("evaluation", {})
        data_cfg = cfg.get("data", {})
        self.system_prompt = data_cfg.get("system_prompt") or FUSION_SYSTEM_PROMPT
        self.max_new_tokens = eval_cfg.get("max_new_tokens", 1024)
        self.model, self.processor = load_model_and_processor(cfg, is_training=False, adapter_path=adapter_path)
        self.model.eval()

    @torch.no_grad()
    def fuse(self, skeleton: str, values: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [{"type": "text", "text": build_fusion_user_text(skeleton, values)}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class DecomposedTablePipeline:
    """Encadeia M1 + M2 -> M3. Mantém a fusão determinística como referência/fallback."""

    def __init__(self, cfg_skeleton: Dict[str, Any], cfg_values: Dict[str, Any], cfg_fusion: Dict[str, Any],
                 adapters: Dict[str, Optional[str]]):
        self.skeleton = _ImageExtractor(cfg_skeleton, adapters.get("skeleton"))
        self.values = _ImageExtractor(cfg_values, adapters.get("values"))
        self.fuser = _TextFuser(cfg_fusion, adapters.get("fusion"))

    def predict_image(self, image: Image.Image) -> Dict[str, Any]:
        skeleton = clean_model_output(self.skeleton.predict(image))
        values = clean_model_output(self.values.predict(image))
        deterministic = fill_skeleton(skeleton, values)

        fused = clean_model_output(self.fuser.fuse(skeleton, values))
        fallback_used = not any(is_table_line(ln) for ln in fused.split("\n"))
        if fallback_used:
            logger.warning("M3 produced no table; falling back to deterministic fill.")

        return {
            "skeleton": skeleton,
            "values": values,
            "deterministic_table": deterministic,
            "markdown_table": deterministic if fallback_used else fused,
            "fused_raw": fused,
            "fallback_used": fallback_used,
        }
