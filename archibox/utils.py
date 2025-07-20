import subprocess

import numpy as np
from PIL import Image


def save_frames_to_mp4(frames: list[np.ndarray], filepath, fps: int = 20):
    assert frames[0].dtype == np.uint8
    assert frames[0].shape[-1] == 3
    height, width, _ = frames[0].shape

    # fmt: off
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",  # overwrite output file if it exists
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",  # input from stdin
        "-an",  # no audio
        "-c:v", "libx264",
        str(filepath)
    ]
    # fmt: on
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            ffmpeg.stdin.write(frame.tobytes())
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()


def save_frames_to_gif(frames: list[np.ndarray], filepath, fps: int = 20):
    imgs = [Image.fromarray(frame) for frame in frames]
    imgs[0].save(
        filepath,
        save_all=True,
        append_images=imgs[1:],
        duration=1000 // fps,
        loop=0,
    )
