"""Reusable on-the-fly degradation pipeline (Albumentations) for Sci-Image figures."""

import os

# Avoid a network version check when this module is imported in training/eval.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2

SEVERITY_LEVELS = ("mild", "moderate", "strong")

# Each factor gets harder from mild -> strong. Rotation comes from the CLI flag.
_SEVERITY_SETTINGS = {
    "mild": {
        "stretch": (0.9, 1.05),
        "gray_p": 0.0,
        "hsv_limit": (5, 10, 10),
        "gamma": (75, 125),
        "gamma_p": 0.3,
        "brightness": 0.08,
        "perspective": None,
        "shadow_p": 0.0,
        "blur": (3, 3),
        "gauss_std": (0.02, 0.05),
        "iso": ((0.01, 0.03), (0.2, 0.4)),
        "downscale": (0.8, 0.95),
        "sharpen_p": 0.0,
        "quality": (75, 90),
    },
    "moderate": {
        "stretch": (0.85, 1.15),
        "gray_p": 0.15,
        "hsv_limit": (10, 20, 20),
        "gamma": (60, 140),
        "gamma_p": 0.5,
        "brightness": 0.15,
        "perspective": (0.03, 0.07),
        "perspective_p": 0.15,
        "shadow_p": 0.15,
        "blur": (3, 5),
        "gauss_std": (0.04, 0.09),
        "iso": ((0.01, 0.05), (0.3, 0.7)),
        "downscale": (0.65, 0.9),
        "sharpen_p": 0.15,
        "quality": (45, 75),
    },
    "strong": {
        "stretch": (0.8, 1.25),
        "gray_p": 0.3,
        "hsv_limit": (15, 30, 30),
        "gamma": (50, 150),
        "gamma_p": 0.7,
        "brightness": 0.2,
        "perspective": (0.05, 0.1),
        "perspective_p": 0.3,
        "shadow_p": 0.25,
        "blur": (3, 7),
        "gauss_std": (0.06, 0.14),
        "iso": ((0.02, 0.08), (0.5, 1.0)),
        "downscale": (0.55, 0.85),
        "sharpen_p": 0.25,
        "quality": (30, 55),
    },
}

_WHITE = (255, 255, 255)


def _affine(settings: dict, rotation_limit: float) -> A.Affine:
    """Rotation plus independent per-axis stretch, always with a white border fit."""
    stretch = settings["stretch"]
    return A.Affine(
        rotate=(-rotation_limit, rotation_limit),
        scale={"x": stretch, "y": stretch},
        fit_output=True,
        border_mode=cv2.BORDER_CONSTANT,
        fill=_WHITE,
        p=1.0,
    )


def build_pipeline(rotation_limit: float, severity: str) -> A.Compose:
    """Simulate a realistic photo/scan of a printed figure, accumulated in one pass."""
    s = _SEVERITY_SETTINGS[severity]
    transforms = [_affine(s, rotation_limit)]
    if s["perspective"]:
        transforms.append(
            A.Perspective(
                scale=s["perspective"],
                keep_size=True,
                fit_output=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=_WHITE,
                p=s["perspective_p"],
            )
        )
    if s["gray_p"] > 0:
        transforms.append(A.ToGray(p=s["gray_p"]))
    transforms.append(A.HueSaturationValue(*s["hsv_limit"], p=0.5))
    transforms.append(A.RandomGamma(*s["gamma"], p=s["gamma_p"]))
    transforms.append(
        A.RandomBrightnessContrast(
            brightness_limit=s["brightness"], contrast_limit=s["brightness"], p=0.7
        )
    )
    if s["shadow_p"] > 0:
        transforms.append(
            A.RandomShadow(
                shadow_roi=(0.0, 0.0, 1.0, 1.0),
                num_shadows_limit=(1, 2),
                shadow_intensity_range=(0.3, 0.6),
                p=s["shadow_p"],
            )
        )
    transforms.append(
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=s["blur"], p=1.0),
                A.GaussNoise(std_range=s["gauss_std"], per_channel=False, p=1.0),
                A.ISONoise(color_shift=s["iso"][0], intensity=s["iso"][1], p=1.0),
                A.Downscale(
                    scale_range=s["downscale"],
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=1.0,
                ),
            ],
            # Every copy receives at least one local degradation.
            p=1.0,
        )
    )
    if s["sharpen_p"] > 0:
        transforms.append(A.Sharpen(p=s["sharpen_p"]))
    transforms.append(A.ImageCompression(quality_range=s["quality"], p=1.0))
    return A.Compose(transforms, strict=True, save_applied_params=True)
