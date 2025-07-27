import sys
from pathlib import Path

import subprocess

from picture_sim_app.image_utils import capture_image, resize_gif, display_gif_fullscreen, display_two_gifs_side_by_side
from picture_sim_app.detect_aoa import calculate_aoa_from_markers, plot_processed_aoa_markers

SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths to input and output folders
INPUT_FOLDER = SCRIPT_DIR / "input"
OUTPUT_FOLDER = SCRIPT_DIR / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # capture_image(input_folder=INPUT_FOLDER)

    # File I/O settings
    input_image_name = "input.png"
    # input_image_name = "input_red_hierarchy.png"
    # input_image_name = "input_red.png"
    # input_image_name = "input_naca_002.png"
    # input_image_name = "input_naca_015.png"
    # input_image_name = "input_naca_030.png"
    input_path = INPUT_FOLDER / input_image_name
    # output_animation_name = "output.gif"
    output_animation_name = "particleplot.gif"
    output_path = OUTPUT_FOLDER / output_animation_name

    # Image recognition settings (rule of thumb for both: Increase if too much background noise, decrease if solid not showing)
    threshold = 0.5 # First play with this until solid shows up, then adjust diff_threshold
    diff_threshold=0.2 # Increase if too much noise, decrease if solid not showing (only if using colors)
    solid_color="red" # Options are grey, red, green or blue
    manual_mode = False  # Set to True to use provided threshold values, otherwise smart detection is used
    force_invert_mask = False  # Set to True to force mask inversion if smart logic fails (will be seen as an incorrect
                               # ghost cell padding at the edges of the fluid boundary)

    # Image resolution cap (spatial resolution)
    max_image_res=800
    # Simulation duration and temporal resolution
    t_sim = 20.
    delta_t = 0.05

    # Flow settings
    Re = 200.
    epsilon = 1. # BDIM kernel width

    # Other settings
    verbose="true"
    sim_type="particles"
    mem="Array"

    # Estimate AoA from markers
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


    # Run Julia script 'TestPixelCamSim.jl'
    julia_script = SCRIPT_DIR.parent / "test" / "TestPixelCamSim.jl"

    # Verify Julia script path
    if not julia_script.is_file():
        print(f"Error: Julia script not found at {julia_script}")
        sys.exit(1)

    cmd = [
        "julia",
        str(julia_script),
        # File I/O settings
        str(input_path),
        str(output_path),
        # Image recognition settings
        str(threshold),
        str(diff_threshold),
        str(solid_color),
        str(manual_mode).lower(),
        str(force_invert_mask).lower(),
        # Image resolution cap (spatial resolution)
        str(max_image_res),
        # Simulation duration and temporal resolution
        str(t_sim),
        str(delta_t),
        # Flow settings
        str(Re),
        str(epsilon),
        # Other settings
        verbose,
        sim_type,
        mem,
    ]
    print(f"Starting Julia: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        raise Exception(f"\nJulia process exited with code {result.returncode}")

    output_path_gif_right = OUTPUT_FOLDER / "output.gif"

    resize_gif(input_path=output_path, output_path=output_path)
    resize_gif(input_path=output_path_gif_right, output_path=output_path_gif_right)

    # display_gif_fullscreen(gif_path=output_path, monitor_index=1)

    display_two_gifs_side_by_side(gif_path_left=output_path, gif_path_right=output_path_gif_right)


if __name__ == "__main__":
    main()
