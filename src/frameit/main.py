import sys
from dataclasses import replace
from pathlib import Path

import click

from frameit import convert, extract

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600


@click.group()
def cli() -> None:
    pass


@cli.command("extract")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("frames", type=int, default=1)
@click.option("-o", "--output-dir", type=Path, default=Path())
@click.option("-s", "--seconds-offset", type=int, default=0)
@click.option("-m", "--minutes-offset", type=int, default=0)
@click.option("-z", "--seconds-limit", type=int, default=0)
@click.option("-n", "--minutes-limit", type=int, default=0)
@click.option(
    "--color-matrix",
    type=click.Choice(["bt709", "bt601"]),
    help="Overrides the source's colour matrix.",
)
@click.option(
    "--range",
    "color_range",
    type=click.Choice(["tv", "pc"]),
    help="Overrides the source's level range (tv = limited, pc = full).",
)
def extract_frames(
    path: Path,
    frames: int,
    output_dir: Path,
    seconds_offset: int,
    minutes_offset: int,
    seconds_limit: int,
    minutes_limit: int,
    color_matrix: extract.ColorMatrix | None,
    color_range: extract.ColorRange | None,
) -> None:
    """Extract evenly spread frames from a film as JPEG."""
    try:
        extract.require_ffmpeg()
    except RuntimeError as error:
        sys.exit(str(error))
    print("Reading video information...")
    info = extract.probe_video(path)
    color_matrix = color_matrix or info.color_matrix or info.guessed_color_matrix
    color_range = color_range or info.color_range or "tv"
    print(
        f"Total length: {info.frame_count / info.fps / SECONDS_PER_MINUTE:.0f} min; "
        f"FPS: {info.fps:.4f}; "
        f"Total frames: {info.frame_count}; "
        f"{info.width}x{info.height} → {info.display_width}x{info.height}; "
        f"colour: {color_matrix} {color_range}"
        + ("" if info.color_matrix else " (guessed)")
        + (
            "; uneven frame timing (e.g. pulldown), every timestamp read"
            if info.frame_times
            else ""
        )
    )

    offset_seconds = seconds_offset + minutes_offset * SECONDS_PER_MINUTE
    limit_seconds = seconds_limit + minutes_limit * SECONDS_PER_MINUTE
    frame_numbers = extract.select_frames(
        info.frame_count, info.fps, frames, offset_seconds, limit_seconds
    )
    offset_frames = round(offset_seconds * info.fps)
    frames_in_selection = (
        info.frame_count - offset_frames - round(limit_seconds * info.fps)
    )
    if frames_in_selection < info.frame_count:
        print(f"Starting at frame {offset_frames}")
        hours, minutes, seconds = hours_minutes_seconds(frames_in_selection / info.fps)
        print(f"Selection length: {hours}:{minutes:02}:{seconds:.1f}")
    frame_step = max(1, frames_in_selection - 1) // max(1, frames - 1)
    print(
        f"{frames} of {frames_in_selection} frames (every {frame_step / info.fps:.1f} s.)"
    )

    with click.progressbar(
        length=len(set(frame_numbers)), label="Extracting"
    ) as progress:
        result = extract.extract_frames(
            path,
            info,
            frame_numbers,
            output_dir,
            color_matrix,
            color_range,
            on_progress=lambda: progress.update(1),
        )
    print(f"Saved {len(result.saved)} frames in {output_dir}")
    if result.errors:
        print("\n".join(result.errors), file=sys.stderr)
        sys.exit(1)


@cli.command("convert")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-s",
    "--settings",
    "settings_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Settings file (frameit.json) saved from the notebook.",
)
@click.option(
    "-o",
    "--output-dir",
    type=Path,
    help="Defaults to <source>_frame next to the source.",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["bmp", "png"]),
    help="Overrides the format in the settings file.",
)
def convert_images(
    source: Path,
    settings_path: Path,
    output_dir: Path | None,
    output_format: convert.OutputFormat | None,
) -> None:
    """Convert a folder of images, or a single image, for the e-ink frame."""
    settings = convert.Settings.load(settings_path)
    if output_format:
        settings = replace(settings, output_format=output_format)
    files = [source] if source.is_file() else convert.find_images(source)
    if not files:
        sys.exit(f"No images in {source}")
    output_folder = output_dir or source.parent / f"{source.stem}_frame"

    palette = settings.palette
    matching_id = next(
        (
            palette_id
            for palette_id, file_palette in convert.load_palettes().items()
            if file_palette.content_hash == palette.content_hash
        ),
        None,
    )
    match_note = f"{matching_id}.json" if matching_id else "matches no palette file"
    print(f"Palette: {palette.id} ({palette.content_hash}), {match_note}")

    with click.progressbar(length=len(files), label="Converting") as progress:
        result = convert.export_images(
            files, settings, output_folder, on_progress=lambda: progress.update(1)
        )

    print(f"Saved {len(result.file_names)} of {len(files)} images in {output_folder}")
    if result.errors:
        print("\n".join(result.errors), file=sys.stderr)
        sys.exit(1)


def hours_minutes_seconds(total_seconds: float) -> tuple[int, int, float]:
    hours, remainder = divmod(total_seconds, SECONDS_PER_HOUR)
    minutes, seconds = divmod(remainder, SECONDS_PER_MINUTE)
    return int(hours), int(minutes), seconds


if __name__ == "__main__":
    cli()
