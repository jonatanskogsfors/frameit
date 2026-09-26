"""Frame extraction from films with ffmpeg, with explicit colour conversion."""

import itertools
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from PIL import Image

type ColorMatrix = Literal["bt709", "bt601"]
type ColorRange = Literal["tv", "pc"]

JPEG_QUALITY = 95
SD_MAX_HEIGHT = 576
HEAD_SECONDS = 5
TAIL_SECONDS = 60
# Container timestamps are often rounded to whole milliseconds
TIMING_TOLERANCE_SECONDS = 0.002
SEEK_MARGIN_SECONDS = 0.005

_COLOR_MATRICES: dict[str, ColorMatrix] = {
    "bt709": "bt709",
    "smpte170m": "bt601",
    "bt470bg": "bt601",
}


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int
    sample_aspect_ratio: Fraction
    color_matrix: ColorMatrix | None
    color_range: ColorRange | None
    # Seconds from the start for every frame, only when frames aren't evenly spaced
    frame_times: tuple[float, ...] | None = None

    @property
    def display_width(self) -> int:
        """Anamorphic sources (DVD) store non-square pixels; this is the width with
        square pixels, rounded to an even number."""
        return round(self.width * self.sample_aspect_ratio / 2) * 2

    @property
    def guessed_color_matrix(self) -> ColorMatrix:
        return "bt601" if self.height <= SD_MAX_HEIGHT else "bt709"

    def seek_seconds(self, frame_number: int) -> float:
        """A little before the frame, so that rounded container timestamps still land
        on it."""
        if self.frame_times is None:
            return max(0.0, (frame_number - 0.5) / self.fps)
        return max(0.0, self.frame_times[frame_number] - SEEK_MARGIN_SECONDS)


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not found; install with brew install ffmpeg"
        )


def probe_video(path: Path) -> VideoInfo:
    """Evenly spaced frames are counted from the last timestamp, which only needs the
    end of the file. Duration times fps would be wrong when the container's duration
    follows the audio. Unevenly spaced frames (pulldown, variable frame rate) need
    every timestamp, which reads the whole file."""
    data = _ffprobe(
        path,
        "-show_entries",
        "format=duration:stream=r_frame_rate,start_time,width,height,"
        "sample_aspect_ratio,color_space,color_range",
    )
    stream = data["streams"][0]
    nominal_fps = Fraction(stream["r_frame_rate"])
    start_time = float(stream.get("start_time", 0))
    duration = float(data["format"]["duration"])

    frame_times = None
    if _evenly_spaced(_packet_times(path, f"%+{HEAD_SECONDS}"), nominal_fps):
        tail = _packet_times(path, f"{max(0.0, duration - TAIL_SECONDS)}%")
        frame_count = round((tail[-1] - start_time) * nominal_fps) + 1
        fps = float(nominal_fps)
    else:
        frame_times = tuple(time - start_time for time in _packet_times(path))
        frame_count = len(frame_times)
        fps = (frame_count - 1) / (frame_times[-1] - frame_times[0])

    sample_aspect_ratio = stream.get("sample_aspect_ratio", "")
    color_range = stream.get("color_range")
    return VideoInfo(
        frame_count=frame_count,
        fps=fps,
        width=stream["width"],
        height=stream["height"],
        sample_aspect_ratio=(
            Fraction(sample_aspect_ratio.replace(":", "/")) or Fraction(1)
            if ":" in sample_aspect_ratio
            else Fraction(1)
        ),
        color_matrix=_COLOR_MATRICES.get(stream.get("color_space", "")),
        color_range=color_range if color_range in ("tv", "pc") else None,
        frame_times=frame_times,
    )


def _packet_times(path: Path, interval: str | None = None) -> list[float]:
    arguments = ["-show_entries", "packet=pts_time"]
    if interval is not None:
        arguments += ["-read_intervals", interval]
    return sorted(
        float(packet["pts_time"])
        for packet in _ffprobe(path, *arguments).get("packets", [])
        if packet.get("pts_time") not in (None, "N/A")
    )


def _evenly_spaced(times: list[float], fps: Fraction) -> bool:
    frame_duration = 1 / float(fps)
    intervals = [later - earlier for earlier, later in itertools.pairwise(times)]
    return bool(intervals) and all(
        abs(interval - frame_duration) <= TIMING_TOLERANCE_SECONDS
        for interval in intervals
    )


def _ffprobe(path: Path, *arguments: str) -> dict[str, Any]:
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            *arguments,
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(output)


def select_frames(
    total_frames: int, fps: float, count: int, offset_seconds: int, limit_seconds: int
) -> list[int]:
    """Evenly spread frames between the offset from the start and the limit from
    the end; a single frame is taken from the middle."""
    offset_frames = round(offset_seconds * fps)
    limit_frames = round(limit_seconds * fps)
    frames_in_selection = total_frames - offset_frames - limit_frames
    if count <= 1:
        return [frames_in_selection // 2 + offset_frames]
    frame_step = max(1, frames_in_selection - 1) // (count - 1)
    return [index * frame_step + offset_frames for index in range(count - 1)] + [
        total_frames - 1 - limit_frames
    ]


@dataclass(frozen=True)
class ExtractResult:
    saved: list[Path]
    errors: list[str]


def extract_frames(
    path: Path,
    info: VideoInfo,
    frame_numbers: list[int],
    output_folder: Path,
    color_matrix: ColorMatrix,
    color_range: ColorRange,
    on_progress: Callable[[], object] = lambda: None,
) -> ExtractResult:
    """Seeks to each frame with its own ffmpeg process, several in parallel, and saves
    the frames as JPEG named by frame number (1-based)."""
    output_folder.mkdir(parents=True, exist_ok=True)
    saved, errors = [], []
    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) // 2)) as pool:
        futures = {
            pool.submit(
                _extract_frame,
                path,
                info,
                number,
                output_folder,
                color_matrix,
                color_range,
            ): number
            for number in sorted(set(frame_numbers))
        }
        for future in as_completed(futures):
            try:
                saved.append(future.result())
            except RuntimeError as error:
                errors.append(f"frame {futures[future]}: {error}")
            on_progress()
    return ExtractResult(sorted(saved), errors)


def _extract_frame(
    path: Path,
    info: VideoInfo,
    number: int,
    output_folder: Path,
    color_matrix: ColorMatrix,
    color_range: ColorRange,
) -> Path:
    width, height = info.display_width, info.height
    seconds = info.seek_seconds(number)
    scale = (
        f"scale=w={width}:h={height}:in_color_matrix={color_matrix}:"
        f"in_range={color_range}:out_range=pc:"
        "flags=lanczos+accurate_rnd+full_chroma_int"
    )
    ffmpeg = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{seconds:.6f}",
            "-i",
            str(path),
            "-an",
            "-sn",
            "-frames:v",
            "1",
            "-vf",
            f"{scale},format=rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if ffmpeg.returncode != 0 or len(ffmpeg.stdout) != width * height * 3:
        raise RuntimeError(ffmpeg.stderr.decode().strip() or "no frame at this position")
    file_path = output_folder / f"frame_{number + 1:06}.jpg"
    Image.frombytes("RGB", (width, height), ffmpeg.stdout).save(
        file_path, quality=JPEG_QUALITY, subsampling="4:2:0"
    )
    return file_path
