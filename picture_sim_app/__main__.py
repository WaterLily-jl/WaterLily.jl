import sys
from pathlib import Path

import subprocess

from image_utils import capture_image, resize_gif, display_gif_fullscreen
from picture_sim_app.detect_aoa import calculate_aoa_from_markers, plot_processed_aoa_markers

SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths to input and output folders
INPUT_FOLDER = SCRIPT_DIR / "input"
OUTPUT_FOLDER = SCRIPT_DIR / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def main() -> None:
    capture_image(input_folder=INPUT_FOLDER)

    # Paths
    input_image_name = "input.png"
    input_path = INPUT_FOLDER / input_image_name
    output_gif_name = "output.gif"
    output_gif = OUTPUT_FOLDER / output_gif_name

    # Set grayscale threshold to identify color
    grayscale_threshold = 0.65
    max_image_res=1080
    t_sim=20.
    delta_t=0.05
    verbose="true"

    calculate_aoa = False
    if calculate_aoa:
        angle_of_attack, _, image_with_markers = calculate_aoa_from_markers(
            image_path=str(input_path),
            marker_color_rgb=(16,52,110),
            tolerance=50,
        )

        print(f"Calculated Angle of Attack: {angle_of_attack:.2f} degrees")

        plot_markers = False
        if plot_markers:
            plot_processed_aoa_markers(image_with_markers, angle_of_attack)

    julia_script = SCRIPT_DIR.parent / "test" / "TestPixelCamSim.jl"

    # Verify Julia script path
    if not julia_script.is_file():
        print(f"Error: Julia script not found at {julia_script}")
        sys.exit(1)

    cmd = [
        "julia",
        str(julia_script),
        str(input_path),
        str(output_gif),
        str(grayscale_threshold),
        str(max_image_res),
        str(t_sim),
        str(delta_t),
        verbose,
    ]
    print(f"Starting Julia: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"\nJulia process exited with code {result.returncode}")

    resize_gif(input_path=output_gif, output_path=output_gif)

    display_gif_fullscreen(gif_path=output_gif, monitor_index=1)


if __name__ == "__main__":
    main()
