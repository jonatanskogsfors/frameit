"""The e-ink image pipeline: from a colour image to the panel's palette."""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numba
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import uniform_filter

type Rgb = tuple[int, int, int]
type Resample = Literal["lanczos", "nearest"]
type OutputFormat = Literal["bmp", "png"]

PALETTE_DIR = Path(__file__).parent / "palettes"
PRESET_DIR = Path(__file__).parent / "presets"
SETTINGS_VERSION = 1
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

LINEAR_SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
LMS_TO_LINEAR_SRGB = np.linalg.inv(LINEAR_SRGB_TO_LMS)
OKLAB_TO_LMS = np.linalg.inv(LMS_TO_OKLAB)

SRGB_LINEAR_SEGMENT_END = 0.04045
LINEAR_LINEAR_SEGMENT_END = 0.0031308

NEUTRAL_CHROMA = 0.08
SKIN_TONE_HUES = (40, 80)
SKIN_TONE_MAX_CHROMA = 0.12
LIGHT_SKY_HUES = (200, 260)
LIGHT_SKY_MIN_LIGHTNESS = 0.7
FULLY_SNAPPED = 0.99


def srgb_to_linear(srgb: NDArray) -> NDArray:
    return np.where(
        srgb <= SRGB_LINEAR_SEGMENT_END, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
    )


def srgb_to_oklab(srgb: NDArray) -> NDArray:
    return np.cbrt(srgb_to_linear(srgb) @ LINEAR_SRGB_TO_LMS.T) @ LMS_TO_OKLAB.T


def oklab_to_linear(oklab: NDArray) -> NDArray:
    return np.clip(((oklab @ OKLAB_TO_LMS.T) ** 3) @ LMS_TO_LINEAR_SRGB.T, 0, 1)


def oklab_to_srgb(oklab: NDArray) -> NDArray:
    linear = oklab_to_linear(oklab)
    return np.where(
        linear <= LINEAR_LINEAR_SEGMENT_END,
        12.92 * linear,
        1.055 * linear ** (1 / 2.4) - 0.055,
    )


def oklab_to_image(oklab: NDArray) -> Image.Image:
    return Image.fromarray((oklab_to_srgb(oklab) * 255 + 0.5).astype(np.uint8))


@dataclass(frozen=True)
class Palette:
    """The first two colours are black and white, the rest are chromatic pigments.

    measured_rgb is what the panel actually shows and drives dithering and preview;
    ideal_rgb is what the frame's firmware expects in the image file.
    """

    id: str
    name: str
    size: tuple[int, int]
    color_names: tuple[str, ...]
    measured_rgb: tuple[Rgb, ...]
    ideal_rgb: tuple[Rgb, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Palette:
        return cls(
            id=data["id"],
            name=data["name"],
            size=tuple(data["size"]),
            color_names=tuple(data["color_names"]),
            measured_rgb=tuple(tuple(color) for color in data["measured_rgb"]),
            ideal_rgb=tuple(tuple(color) for color in data["ideal_rgb"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "hash": self.content_hash, **asdict(self)}

    @cached_property
    def content_hash(self) -> str:
        # Names are left out so that renaming a palette doesn't make it a different one
        content = json.dumps([self.measured_rgb, self.ideal_rgb, list(self.size)])
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    @cached_property
    def measured_srgb(self) -> NDArray:
        return np.array(self.measured_rgb) / 255.0

    @cached_property
    def measured_linear(self) -> NDArray:
        return srgb_to_linear(self.measured_srgb)

    @cached_property
    def oklab(self) -> NDArray:
        return srgb_to_oklab(self.measured_srgb)

    @cached_property
    def _pigments_by_hue(self) -> tuple[NDArray, NDArray, NDArray]:
        pigments = self.oklab[2:]
        hues = np.arctan2(pigments[:, 2], pigments[:, 1])
        order = np.argsort(hues)
        chromas = np.hypot(pigments[:, 1], pigments[:, 2])
        return hues[order], chromas[order], pigments[:, 0][order]

    @property
    def pigment_hues(self) -> NDArray:
        return self._pigments_by_hue[0]

    @property
    def pigment_chromas(self) -> NDArray:
        return self._pigments_by_hue[1]

    @property
    def pigment_lightnesses(self) -> NDArray:
        return self._pigments_by_hue[2]


def load_palettes() -> dict[str, Palette]:
    return {
        path.stem: Palette.from_dict({"id": path.stem, **json.loads(path.read_text())})
        for path in sorted(PALETTE_DIR.glob("*.json"))
    }


@dataclass(frozen=True)
class Tuning:
    gamma: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    flatten_threshold: float = 0.008
    flatten_window: int = 5
    gamut_strength: float = 1.0
    hue_pull_strength: float = 0.15
    hue_pull_range: float = 20
    snap_radius: float = 0.0
    dither_strength: float = 0.8
    chroma_weight: float = 3.0


@dataclass(frozen=True)
class Preset:
    """A starting point for the tuning, suited to a kind of material."""

    id: str
    name: str
    description: str
    resample: Resample
    tuning: Tuning

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Preset:
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            resample=data.get("resample", "lanczos"),
            tuning=Tuning(**data.get("tuning", {})),
        )


def load_presets() -> dict[str, Preset]:
    return {
        path.stem: Preset.from_dict({"id": path.stem, **json.loads(path.read_text())})
        for path in sorted(PRESET_DIR.glob("*.json"))
    }


def fit_to_panel(image: Image.Image, palette: Palette, resample: Resample) -> Image.Image:
    method = (
        Image.Resampling.LANCZOS if resample == "lanczos" else Image.Resampling.NEAREST
    )
    return ImageOps.fit(image.convert("RGB"), palette.size, method=method)


def adjust_tone(
    image: Image.Image, gamma: float, contrast: float, saturation: float
) -> Image.Image:
    image = image.point(lambda value: round(255 * (value / 255) ** gamma))
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return ImageEnhance.Color(image).enhance(saturation)


def flatten(image: Image.Image, threshold: float, window: int) -> NDArray:
    oklab = srgb_to_oklab(np.asarray(image, dtype=np.float64) / 255.0)
    if threshold > 0:
        size = (window, window, 1)
        mean = uniform_filter(oklab, size=size)
        variance = uniform_filter(oklab**2, size=size) - mean**2
        deviation = np.sqrt(np.clip(variance.sum(axis=-1), 0, None))
        oklab = np.where((deviation < threshold)[..., None], mean, oklab)
    return oklab


def map_gamut(oklab: NDArray, palette: Palette, strength: float) -> NDArray:
    lightness, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
    chroma, hue = np.hypot(a, b), np.arctan2(b, a)
    black_lightness, white_lightness = palette.oklab[0, 0], palette.oklab[1, 0]
    target_lightness = black_lightness + lightness * (white_lightness - black_lightness)

    # The reachable chroma is interpolated between the pigments' hues and shrinks
    # towards black and white
    hues = palette.pigment_hues
    wrapped_hues = np.concatenate([hues - 2 * np.pi, hues, hues + 2 * np.pi])
    max_chroma = np.interp(hue, wrapped_hues, np.tile(palette.pigment_chromas, 3))
    pigment_lightness = np.clip(
        np.interp(hue, wrapped_hues, np.tile(palette.pigment_lightnesses, 3)),
        black_lightness + 0.03,
        white_lightness - 0.03,
    )
    max_chroma = max_chroma * np.clip(
        np.minimum(
            (target_lightness - black_lightness) / (pigment_lightness - black_lightness),
            (white_lightness - target_lightness) / (white_lightness - pigment_lightness),
        ),
        0,
        1,
    )
    knee = 0.8 * max_chroma
    compressed_chroma = np.where(
        chroma > knee,
        knee
        + (max_chroma - knee)
        * np.tanh((chroma - knee) / np.maximum(max_chroma - knee, 1e-6)),
        chroma,
    )

    lightness = lightness + strength * (target_lightness - lightness)
    chroma = chroma + strength * (compressed_chroma - chroma)
    # The panel's black and white aren't neutral; greys follow them
    lightness_fraction = np.clip(
        (lightness - black_lightness) / (white_lightness - black_lightness), 0, 1
    )[..., None]
    white_point_shift = strength * (
        palette.oklab[0, 1:] * (1 - lightness_fraction)
        + palette.oklab[1, 1:] * lightness_fraction
    )
    return np.stack(
        [
            lightness,
            chroma * np.cos(hue) + white_point_shift[..., 0],
            chroma * np.sin(hue) + white_point_shift[..., 1],
        ],
        axis=-1,
    )


def pull_hues_and_snap(
    oklab: NDArray,
    palette: Palette,
    pull_strength: float,
    pull_range: float,
    snap_radius: float,
) -> tuple[NDArray, float]:
    """Returns the adjusted colours and the share of pixels fully snapped."""
    lightness, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
    chroma, hue = np.hypot(a, b), np.arctan2(b, a)
    offsets = (palette.pigment_hues - hue[..., None] + np.pi) % (2 * np.pi) - np.pi
    nearest_offset = np.take_along_axis(
        offsets, np.argmin(np.abs(offsets), axis=-1)[..., None], axis=-1
    )[..., 0]
    pull = (
        pull_strength
        * np.clip(1 - np.abs(nearest_offset) / np.radians(pull_range), 0, 1)
        * np.clip(chroma / NEUTRAL_CHROMA, 0, 1)
    )
    degrees = np.degrees(hue) % 360
    skin_tone = (
        (degrees > SKIN_TONE_HUES[0])
        & (degrees < SKIN_TONE_HUES[1])
        & (chroma < SKIN_TONE_MAX_CHROMA)
    )
    light_sky = (
        (degrees > LIGHT_SKY_HUES[0])
        & (degrees < LIGHT_SKY_HUES[1])
        & (lightness > LIGHT_SKY_MIN_LIGHTNESS)
    )
    hue = hue + np.where(skin_tone | light_sky, 0.0, pull) * nearest_offset
    oklab = np.stack([lightness, chroma * np.cos(hue), chroma * np.sin(hue)], axis=-1)

    distances = np.sqrt(((oklab[..., None, :] - palette.oklab) ** 2).sum(-1))
    nearest, nearest_distance = np.argmin(distances, axis=-1), distances.min(axis=-1)
    radius = max(snap_radius, 1e-6)
    snap_weight = np.clip((radius - nearest_distance) / (0.5 * radius), 0, 1)
    snapped = oklab + snap_weight[..., None] * (palette.oklab[nearest] - oklab)
    return snapped, float((snap_weight >= FULLY_SNAPPED).mean())


@numba.njit
def _linear_to_oklab(red, green, blue, to_lms, to_oklab):
    long = (to_lms[0, 0] * red + to_lms[0, 1] * green + to_lms[0, 2] * blue) ** (1 / 3)
    medium = (to_lms[1, 0] * red + to_lms[1, 1] * green + to_lms[1, 2] * blue) ** (1 / 3)
    short = (to_lms[2, 0] * red + to_lms[2, 1] * green + to_lms[2, 2] * blue) ** (1 / 3)
    return (
        to_oklab[0, 0] * long + to_oklab[0, 1] * medium + to_oklab[0, 2] * short,
        to_oklab[1, 0] * long + to_oklab[1, 1] * medium + to_oklab[1, 2] * short,
        to_oklab[2, 0] * long + to_oklab[2, 1] * medium + to_oklab[2, 2] * short,
    )


@numba.njit
def _floyd_steinberg(
    linear_image, palette_linear, palette_oklab, to_lms, to_oklab, strength, chroma_weight
):
    """Error is spread in linear light; the nearest colour is picked in Oklab."""
    height, width, _ = linear_image.shape
    image = linear_image.copy()
    palette_indices = np.zeros((height, width), np.uint8)
    for y in range(height):
        direction = 1 if y % 2 == 0 else -1
        for step in range(width):
            x = step if direction == 1 else width - 1 - step
            red = min(max(image[y, x, 0], 0.0), 1.0)
            green = min(max(image[y, x, 1], 0.0), 1.0)
            blue = min(max(image[y, x, 2], 0.0), 1.0)
            lightness, a, b = _linear_to_oklab(red, green, blue, to_lms, to_oklab)
            best, best_distance = 0, 1e9
            for index in range(palette_oklab.shape[0]):
                distance = (lightness - palette_oklab[index, 0]) ** 2 + chroma_weight * (
                    (a - palette_oklab[index, 1]) ** 2
                    + (b - palette_oklab[index, 2]) ** 2
                )
                if distance < best_distance:
                    best, best_distance = index, distance
            palette_indices[y, x] = best
            for channel in range(3):
                clamped = min(max(image[y, x, channel], 0.0), 1.0)
                error = (clamped - palette_linear[best, channel]) * strength
                if 0 <= x + direction < width:
                    image[y, x + direction, channel] += error * 7 / 16
                if y + 1 < height:
                    if 0 <= x - direction < width:
                        image[y + 1, x - direction, channel] += error * 3 / 16
                    image[y + 1, x, channel] += error * 5 / 16
                    if 0 <= x + direction < width:
                        image[y + 1, x + direction, channel] += error * 1 / 16
    return palette_indices


def dither(
    oklab: NDArray, palette: Palette, strength: float, chroma_weight: float
) -> NDArray[np.uint8]:
    return _floyd_steinberg(
        np.ascontiguousarray(oklab_to_linear(oklab)),
        np.ascontiguousarray(palette.measured_linear),
        np.ascontiguousarray(palette.oklab),
        LINEAR_SRGB_TO_LMS,
        LMS_TO_OKLAB,
        float(strength),
        float(chroma_weight),
    )


def to_preview_image(palette_indices: NDArray[np.uint8], palette: Palette) -> Image.Image:
    return Image.fromarray(
        (palette.measured_srgb[palette_indices] * 255 + 0.5).astype(np.uint8)
    )


def to_panel_image(palette_indices: NDArray[np.uint8], palette: Palette) -> Image.Image:
    image = Image.fromarray(palette_indices, mode="P")
    image.putpalette([channel for color in palette.ideal_rgb for channel in color])
    return image


def process(image: Image.Image, palette: Palette, tuning: Tuning) -> NDArray[np.uint8]:
    """The whole pipeline for an image already fitted to the panel."""
    toned = adjust_tone(image, tuning.gamma, tuning.contrast, tuning.saturation)
    oklab = flatten(toned, tuning.flatten_threshold, tuning.flatten_window)
    oklab = map_gamut(oklab, palette, tuning.gamut_strength)
    oklab, _ = pull_hues_and_snap(
        oklab,
        palette,
        tuning.hue_pull_strength,
        tuning.hue_pull_range,
        tuning.snap_radius,
    )
    return dither(oklab, palette, tuning.dither_strength, tuning.chroma_weight)


def save_panel_image(
    palette_indices: NDArray[np.uint8],
    palette: Palette,
    directory: Path,
    stem: str,
    output_format: OutputFormat,
) -> str:
    image = to_panel_image(palette_indices, palette)
    file_name = f"{stem}.{output_format}"
    # The PhotoPainter firmware only reads 24-bit BMP
    (image.convert("RGB") if output_format == "bmp" else image).save(
        directory / file_name
    )
    return file_name


def write_file_list(directory: Path, file_names: list[str]) -> None:
    """The PhotoPainter firmware's mode 2 shows the images in this order."""
    with open(directory / "fileList.txt", "w", newline="") as file_list:
        file_list.write("".join(f"pic/{name}\r\n" for name in file_names))


@dataclass(frozen=True)
class Settings:
    """The palette is stored in full, so a settings file keeps giving the same result
    even if the palette file changes later."""

    palette: Palette
    resample: Resample = "lanczos"
    output_format: OutputFormat = "bmp"
    tuning: Tuning = field(default_factory=Tuning)
    preset: str | None = None

    def save(self, path: Path) -> None:
        data = {
            "version": SETTINGS_VERSION,
            "preset": self.preset,
            "palette": self.palette.to_dict(),
            "resample": self.resample,
            "output_format": self.output_format,
            "tuning": asdict(self.tuning),
        }
        path.write_text(_to_json_with_compact_lists(data) + "\n")

    @classmethod
    def load(cls, path: Path) -> Settings:
        data = json.loads(path.read_text())
        return cls(
            palette=Palette.from_dict(data["palette"]),
            resample=data.get("resample", "lanczos"),
            output_format=data.get("output_format", "bmp"),
            tuning=Tuning(**data.get("tuning", {})),
            preset=data.get("preset"),
        )


def _to_json_with_compact_lists(data: dict[str, Any]) -> str:
    """Lists without nested lists go on one line, so each RGB colour gets its own line."""
    text = json.dumps(data, indent=2, ensure_ascii=False)
    return re.sub(
        r"\[\s+([^\[\]{}]*?)\s+\]",
        lambda match: "[" + re.sub(r",\s+", ", ", match.group(1)) + "]",
        text,
    )


@dataclass(frozen=True)
class ExportResult:
    file_names: list[str]
    errors: list[str]


def find_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def export_images(
    files: list[Path],
    settings: Settings,
    output_folder: Path,
    on_progress: Callable[[], object] = lambda: None,
) -> ExportResult:
    """BMP output gets the same layout as the PhotoPainter's SD card: images in pic/
    and fileList.txt in the root. The settings are saved next to the images."""
    is_bmp = settings.output_format == "bmp"
    image_folder = output_folder / "pic" if is_bmp else output_folder
    image_folder.mkdir(parents=True, exist_ok=True)

    file_names, errors = [], []
    for path in files:
        try:
            image = fit_to_panel(Image.open(path), settings.palette, settings.resample)
            palette_indices = process(image, settings.palette, settings.tuning)
            file_names.append(
                save_panel_image(
                    palette_indices,
                    settings.palette,
                    image_folder,
                    path.stem,
                    settings.output_format,
                )
            )
        except OSError as error:
            errors.append(f"{path.name}: {error}")
        on_progress()

    settings.save(output_folder / "frameit.json")
    if is_bmp:
        write_file_list(output_folder, file_names)
    return ExportResult(file_names, errors)
