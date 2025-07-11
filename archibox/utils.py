import argparse
import functools
import os
import subprocess
from ast import literal_eval
from typing import Callable, get_type_hints

import numpy as np
from PIL import Image


def parse_config(main: Callable):
    @functools.wraps(main)
    def wrapped_main():
        main_hints = get_type_hints(main)
        main_hints.pop("return", None)
        if len(main_hints) == 0:
            raise ValueError("cfg argument in main() must be type annotated")
        if len(main_hints) > 1:
            raise ValueError("expected main() function to take only one argument")
        try:
            cfg = list(main_hints.values())[0]()
        except Exception as e:
            raise ValueError("failed to initialize default config") from e

        parser = argparse.ArgumentParser()
        parser.add_argument("overrides", nargs="*")
        args = parser.parse_args()

        casted_overrides = []
        for override in args.overrides:
            key, value_str = override.split("=", 1)
            cur = cfg
            *prefix, final_key = key.split(".")
            for k in prefix:
                if not hasattr(cur, k):
                    raise AttributeError(f"{type(cur)} has no config option {k!r}")
                cur = getattr(cfg, k)
            hints = get_type_hints(cur)
            if final_key not in hints:
                raise AttributeError(f"{type(cur)} has no config option {final_key!r}")
            target_type = hints[final_key]

            if target_type is bool:
                if value_str.lower() in ("true", "1", "yes", "y", "on"):
                    parsed_value = True
                elif value_str.lower() in ("false", "0", "no", "n", "off"):
                    parsed_value = False
                else:
                    raise TypeError(f"failed to interpret {value_str} as bool")
            else:
                try:
                    parsed_value = literal_eval(value_str)
                except Exception:
                    parsed_value = value_str
            try:
                casted_value = target_type(parsed_value)
            except Exception as e:
                raise TypeError(
                    f"failed to cast {value_str} to type {target_type}"
                ) from e
            setattr(cur, final_key, casted_value)

            casted_overrides.append(f"{key}={casted_value}")

        if len(args.overrides) > 0 and int(os.environ.get("RANK", "0")) == 0:
            print(f"running with overrides: {' '.join(casted_overrides)}")

        return main(cfg)

    return wrapped_main


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
