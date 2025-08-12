from pathlib import Path

import numpy as np

from picture_sim_app.image_utils import get_gif_dimensions, resize_gif

# Define paths (static)
INPUT_AIRFOIL_IMAGES = [
    "input_naca_002.png",
    "input_naca_015.png",
    "input_naca_030.png",
]

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_RUNS_DIR = SCRIPT_DIR / "output" / "batch_runs"

# Define path to output folder
OUTPUT_FOLDER = SCRIPT_DIR / "output" / "batch_runs"


def main():
    gif_files = list(BATCH_RUNS_DIR.glob("*.gif"))

    print(f"Check {len(gif_files)} GIF files exist and can be loaded:")

    problematic_files = []

    target_size = (800, 600)  # Target output size for gif images (upscale/downscale)

    for gif_path in gif_files:

        try:
            dimensions = get_gif_dimensions(gif_path)
            print(f"{gif_path.name}: {dimensions[0]}x{dimensions[1]}")
            print(f"\nProcessing GIFs to consistent size: {target_size}")
            resize_gif(
                input_path=gif_path,
                output_path=gif_path,
                target_size=target_size,
                maintain_aspect=False,  # Set to True to maintain aspect ratio (may add black bars)
                # Set to False to stretch to exact target size
            )

        except Exception as e:
            print(f"{gif_path.name}: ERROR - {e}")
            problematic_files.append(gif_path)

    if len(problematic_files) > 0:
        print(f"\n {len(problematic_files)} problematic GIF files found")


if __name__ == "__main__":
    main()
