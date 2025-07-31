import sys
from pathlib import Path
import subprocess

import numpy as np

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
    # Capture image with interactive click-and-drag selection box
    # The selection box maintains aspect ratio while you drag
    # - Click and drag to define the selection area
    # - The box automatically maintains your target aspect ratio
    capture_image(
        input_folder=INPUT_FOLDER,
        fixed_aspect_ratio=(4, 3),  # 4:3 aspect ratio for consistency
        selection_box_mode=True,    # Click-and-drag selection box
        # fixed_size=(800, 600),    # Alternative: exact pixel dimensions
    )

    # File I/O settings
    # input_image_name = "input.png"
    # input_image_name = "input_red_hierarchy.png"
    # input_image_name = "input_red.png"
    # input_image_name = "input_naca_002.png"
    input_image_name = "input_naca_015.png"
    # input_image_name = "input_naca_030.png"
    input_path = INPUT_FOLDER / input_image_name
    # output_animation_name = "output.gif"
    particle_plot_name = "particleplot.gif"
    heatmap_plot_name = "heatmap_plot.gif"
    data_file_name = "simulation_data.npz"
    output_path_particle_plot = OUTPUT_FOLDER / particle_plot_name
    output_path_heatmap_plot = OUTPUT_FOLDER / heatmap_plot_name
    output_path_data = OUTPUT_FOLDER / data_file_name

    # Image recognition settings (rule of thumb for both: Increase if too much background noise, decrease if solid not showing)
    threshold = 0.5 # First play with this until solid shows up, then adjust diff_threshold
    diff_threshold=0.2 # Increase if too much noise, decrease if solid not showing (only if not using gray)
    solid_color="gray" # Options are gray, red, green or blue
    manual_mode = False  # Set to True to use provided threshold values, otherwise smart detection is used
    force_invert_mask = False  # Set to True to force mask inversion if smart logic fails (will be seen as an incorrect
                               # ghost cell padding at the edges of the fluid boundary)
    image_recognition_debug_mode = True # Set to true to plot the image recognition mask and PCA components

    # Image resolution cap (spatial resolution)
    max_image_res=800
    # Simulation duration and temporal resolution
    t_sim = 2.
    delta_t = 0.05

    # Flow settings
    Re = 200.
    epsilon = 1. # BDIM kernel width

    # Other settings
    verbose="true"
    sim_type="sim_with_particle_and_heatmap_gif"  # Run the simulation. Options are:
                                  # "sim_with_vorticity_heatmap_gif" (the OG)
                                  # "particles" (uses PathLines library to create a particle plot)
                                  # "sim_only" (runs the sim without any plotting)
                                  # "sim_with_particle_and_heatmap_gif" (creates both the particle and heatmap gif)
    mem="Array"

    # Use image recognition to create a fluid-solid mask (1=Fluid, 0=Solid)
    pixel_body = PixelBodyMask(
        image_path=str(input_path),
        threshold=threshold,
        diff_threshold=diff_threshold,
        max_image_res=max_image_res,
        body_color=solid_color,
        manual_mode=manual_mode,
        force_invert_mask=force_invert_mask,
    )
    domain_mask = pixel_body.get_mask()

    plot_mask = image_recognition_debug_mode
    if plot_mask:
        pixel_body.plot_mask()

    # Estimate characteristic length and angle of attack using PCA
    l_c, aoa, thickness = characteristic_length_and_aoa_pca(
        mask=domain_mask,
        plot_method=image_recognition_debug_mode,
        show_components=False,
    )

    # Estimate airfoil type based on thickness and characteristic length
    airfoil_type = detect_airfoil_type(thickness_to_cord_ratio=thickness/l_c)

    # Save the boolean mask to a temporary file for Julia to read
    mask_file = OUTPUT_FOLDER / "temp_mask.npy"
    np.save(mask_file, domain_mask)

    # Run Julia script 'TestPixelCamSim.jl'
    julia_script = SCRIPT_DIR.parent / "test" / "TestPixelCamSim.jl"

    # Verify Julia script path
    if not julia_script.is_file():
        print(f"Error: Julia script not found at {julia_script}")
        sys.exit(1)

    cmd = [
        "julia",
        str(julia_script),
        # "--sysimage", "julia_sysimage_pixelbody.so",  # Add custom Julia sysimage for faster startup and precompiled
        # package loading (precompiles Julia packages and code)
        # File I/O settings - now pass mask file instead of image
        str(mask_file),  # Pass mask file instead of input image
        str(output_path_particle_plot),  # Pass full particle gif path for dual_gifs mode
        # Simulation parameters
        str(l_c),  # Pass characteristic length from Python
        str(Re),
        str(epsilon),
        str(t_sim),
        str(delta_t),
        # Other settings
        verbose,
        sim_type,
        mem,
        str(output_path_heatmap_plot),  # Pass full heatmap gif path for dual_gifs mode
    ]
    print(f"Starting Julia: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)

    # Clean up temporary mask file
    if mask_file.exists():
        mask_file.unlink()

    if result.returncode != 0:
        raise Exception(f"\nJulia process exited with code {result.returncode}")

    # Create visualizations from the exported simulation data
    print("\nCreating gifs from simulation data...")
    create_gifs(
        data_path=output_path_data,
        particle_output=output_path_particle_plot,
        heatmap_output=output_path_heatmap_plot,
    )

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
