import sys
from pathlib import Path

import click
import cv2
from PIL import Image


@click.group()
def cli():
    pass


@cli.command()
@click.argument("path", type=Path)
@click.argument("frames", type=int, default=1)
@click.option("-w", "--width", type=int, default=800)
@click.option("-h", "--height", type=int, default=480)
@click.option("-o", "--output-dir", type=Path, default=Path())
@click.option("-s", "--seconds-offset", type=int, default=0)
@click.option("-m", "--minutes-offset", type=int, default=0)
@click.option("-z", "--seconds-limit", type=int, default=0)
@click.option("-n", "--minutes-limit", type=int, default=0)
def movie(
    path: Path,
    frames: int,
    width: int,
    height: int,
    output_dir: Path,
    seconds_offset: int,
    minutes_offset: int,
    seconds_limit: int,
    minutes_limit: int,
):
    output_dir.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    print(
        f"Total length: {total_frames / fps / 60:.0f} min; "
        f"FPS: {fps:.4f}; "
        f"Total frames: {total_frames}"
    )

    offset_frames = round((seconds_offset + minutes_offset * 60) * fps)
    limit_frames = round((seconds_limit + minutes_limit * 60) * fps)

    frames_in_selection = total_frames - offset_frames - limit_frames
    selection_in_seconds = frames_in_selection / fps

    if frames_in_selection < total_frames:
        print(f"Starting at frame {offset_frames}")
        selection_h, selection_m, selection_s = hours_minutes_seconds(
            selection_in_seconds
        )
        print(f"Selection length: {selection_h}:{selection_m:02}:{selection_s:.1f}")

    batch = max(1, frames_in_selection - 1) // max(1, frames - 1)
    print(f"{frames} of {frames_in_selection} frames (every {batch/fps:.1f} s.)")
    selected_frames = (
        [n * batch + offset_frames for n in range(frames - 1)]
        + [total_frames - 1 - limit_frames]
        if frames > 1
        else [frames_in_selection // 2 + offset_frames]
    )

    for frame_number in selected_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        flag, frame = capture.retrieve()
        if not flag:
            sys.exit("OOPS")

        image = resize_and_crop(frame, width, height)
        dithered_image = dither_frame(image)
        dithered_image.save(output_dir / f"frame_{frame_number + 1:06}_d.bmp")
        # cv2.imwrite(output_dir / f"frame_{frame_number + 1:06}.jpg", image)


@cli.command()
@click.argument("path", type=Path)
@click.option("-w", "--width", type=int, default=800)
@click.option("-h", "--height", type=int, default=480)
@click.option("-o", "--output-dir", type=Path, default=Path())
def image(
    path: Path,
    width: int,
    height: int,
    output_dir: Path,
):
    output_dir.mkdir(exist_ok=True)
    image = cv2.imread(path)
    image = resize_and_crop(image, width, height)
    dithered_image = dither_frame(image)
    dithered_image.save(output_dir / f"{path.stem}_d.bmp")


def hours_minutes_seconds(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return int(hours), int(minutes), seconds


def resize_and_crop(image_array, width: int, height: int):
    frame_height, frame_width = image_array.shape[:2]
    frame_ratio = frame_width / frame_height
    target_ratio = width / height
    if frame_ratio <= target_ratio:
        image_array = cv2.resize(image_array, (width, round(width / frame_ratio)))
    else:
        image_array = cv2.resize(image_array, (round(height * frame_ratio), height))

    frame_height, frame_width = image_array.shape[:2]
    mid_x, mid_y = int(frame_width / 2), int(frame_height / 2)
    half_width, half_height = int(width / 2), int(height / 2)
    return image_array[
        mid_y - half_height : mid_y + half_height, mid_x - half_width : mid_x + half_width
    ]


def dither_frame(frame):
    pil_image = Image.fromarray(frame[:, :, ::-1])
    palette = Image.new("P", (1, 1))
    palette.putpalette(
        (
            (0, 0, 0)
            + (255, 255, 255)
            + (0, 255, 0)
            + (0, 0, 255)
            + (255, 0, 0)
            + (255, 255, 0)
            + (255, 128, 0)
            + (0, 0, 0) * 249
        )
    )

    return pil_image.quantize(
        dither=Image.Dither.FLOYDSTEINBERG,
        palette=palette,
    ).convert("RGB")


if __name__ == "__main__":
    cli()
