import sys
from pathlib import Path
import subprocess

import numpy as np
import yaml

from picture_sim_app.characteristic_length_and_aoa_estimation import characteristic_length_and_aoa_pca
from picture_sim_app.create_visualizations import create_gifs
from picture_sim_app.detect_airfoil_type import detect_airfoil_type
# from julia.api import Julia

from picture_sim_app.image_utils import (
    capture_image, 
    resize_gif, 
    crop_gif,
    make_gifs_consistent_size,
    get_gif_dimensions,
    display_gif_fullscreen, 
    display_two_gifs_side_by_side
)
from picture_sim_app.live_simulation import run_julia_simulation_script
from picture_sim_app.pixel_body_python import PixelBodyMask

# Initialize Julia with your custom sysimage
# sysimage_path = str(Path(__file__).resolve().parent / "julia_sysimage_pixelbody.so")
# jl = Julia(sysimage=sysimage_path, compiled_modules=False)

# Define absolute path to the script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths to input and output folders
INPUT_FOLDER = SCRIPT_DIR / "input"
OUTPUT_FOLDER = SCRIPT_DIR / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def main() -> None:

    with open(SCRIPT_DIR / "configs/settings.yaml", "r") as f:
        settings = yaml.safe_load(f)

    # Capture image with interactive click-and-drag selection box
    # The selection box maintains aspect ratio while you drag
    # - Click and drag to define the selection area
    # - The box automatically maintains your target aspect ratio
    # capture_image(
    #     input_folder=INPUT_FOLDER,
    #     fixed_aspect_ratio=(4, 3),  # 4:3 aspect ratio for consistency
    #     selection_box_mode=True,    # Click-and-drag selection box
    #     # fixed_size=(800, 600),    # Alternative: exact pixel dimensions
    # )

    # File I/O paths
    io_settings = settings["io_settings"]
    input_path = INPUT_FOLDER / io_settings["input_image_name"]
    output_path_particle_plot = OUTPUT_FOLDER / io_settings["particle_plot_name"]
    output_path_heatmap_plot = OUTPUT_FOLDER / io_settings["heatmap_plot_name"]
    # output_path_data = OUTPUT_FOLDER / data_file_name

    #Unpack simulation settings:
    simulation_settings = settings["simulation_settings"]
    image_recognition_debug_mode = simulation_settings["image_recognition_debug_mode"]

    # Use image recognition to create a fluid-solid mask (1=Fluid, 0=Solid)
    pixel_body = PixelBodyMask(
        image_path=str(input_path),
        threshold=simulation_settings["threshold"],
        diff_threshold=simulation_settings["diff_threshold"],
        max_image_res=simulation_settings["max_image_res"],
        body_color=simulation_settings["solid_color"],
        manual_mode=simulation_settings["manual_mode"],
        force_invert_mask=simulation_settings["force_invert_mask"],
    )
    domain_mask = pixel_body.get_mask()

    # Estimate characteristic length and angle of attack using PCA
    l_c, aoa, thickness = characteristic_length_and_aoa_pca(
        mask=domain_mask,
        plot_method=image_recognition_debug_mode,
        show_components=False,
    )

    # Test rotating angle of attack
    domain_mask = pixel_body.rotate_mask(current_angle=aoa, target_angle=45)

    plot_mask = image_recognition_debug_mode
    if plot_mask:
        pixel_body.plot_mask()

    # Estimate characteristic length and angle of attack using PCA (after rotation)
    l_c, aoa, thickness = characteristic_length_and_aoa_pca(
        mask=domain_mask,
        plot_method=image_recognition_debug_mode,
        show_components=False,
        )

    # Estimate airfoil type based on thickness and characteristic length
    airfoil_type = detect_airfoil_type(thickness_to_cord_ratio=thickness/l_c)

    use_precomputed_results = False
    if not use_precomputed_results:

        run_julia_simulation_script(
            domain_mask=domain_mask,
            l_c=l_c,
            simulation_settings=simulation_settings,
            output_path_particle_plot=output_path_particle_plot,
            output_path_heatmap_plot=output_path_heatmap_plot,
            output_folder=OUTPUT_FOLDER,
            script_dir=SCRIPT_DIR,
        )

    else:
        print("\nUsing precomputed simulation results...")

    # # Create visualizations from the exported simulation data
    # print("\nCreating gifs from simulation data...")
    # create_gifs(
    #     data_path=output_path_data,
    #     particle_output=output_path_particle_plot,
    #     heatmap_output=output_path_heatmap_plot,
    # )

    # Define paths for both GIFs
    gif_paths = [output_path_particle_plot, output_path_heatmap_plot]

    # Get original dimensions before processing
    print("Original GIF dimensions:")
    for gif_path in gif_paths:
        if gif_path.exists():
            dims = get_gif_dimensions(gif_path)
            print(f"  {gif_path.name}: {dims[0]}x{dims[1]}")

    # Option 1: Crop specific regions if you know there's unwanted padding
    # Define crop boxes to remove margins/padding: (left, top, right, bottom)
    # crop_boxes = [
    #     (50, 50, 750, 550),  # Crop particleplot.gif
    #     (50, 50, 750, 550),  # Crop output.gif
    # ]

    # Option 2: Resize without cropping but with consistent dimensions
    target_size = (800, 600)  # Set your desired consistent size

    print(f"\nProcessing GIFs to consistent size: {target_size}")

    # Method A: Process each GIF individually with aspect ratio preservation
    for gif_path in gif_paths:
        if gif_path.exists():
            print(f"Processing {gif_path.name}...")
            resize_gif(
                input_path=gif_path,
                output_path=gif_path,
                target_size=target_size,
                maintain_aspect=False,  # Set to True to maintain aspect ratio (may add black bars)
                                        # Set to False to stretch to exact target size
            )

    # Method B: Use the batch processing function with optional cropping
    # make_gifs_consistent_size(
    #     gif_paths=gif_paths,
    #     target_size=target_size,
    #     maintain_aspect=True,
    #     crop_boxes=crop_boxes  # Optional: specify crop regions
    # )

    # Verify final dimensions
    print("\nFinal GIF dimensions:")
    for gif_path in gif_paths:
        if gif_path.exists():
            dims = get_gif_dimensions(gif_path)
            print(f"  {gif_path.name}: {dims[0]}x{dims[1]}")

    # # Monitor selection for display
    # print("\nAvailable monitors:")
    # from picture_sim_app.image_utils import list_monitors
    # monitors = list_monitors()
    #
    # # Default to secondary monitor (index 1) if available, otherwise primary (index 0)
    # default_monitor = 1 if len(monitors) > 1 else 0
    #
    # # Display the consistent-sized GIFs side by side on selected monitor
    # print(f"\nDisplaying GIFs on monitor {default_monitor}...")
    # display_two_gifs_side_by_side(
    #     gif_path_left=output_path,
    #     gif_path_right=output_path_gif_right,
    #     monitor_index=default_monitor
    # )


if __name__ == "__main__":
    main()
