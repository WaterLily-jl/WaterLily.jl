import subprocess
import sys
from pathlib import Path
import logging
from tqdm import tqdm

import numpy as np
import yaml

from picture_sim_app.characteristic_length_and_aoa_estimation import characteristic_length_and_aoa_pca
from picture_sim_app.create_visualizations import create_gifs
from picture_sim_app.image_utils import get_gif_dimensions, resize_gif
from picture_sim_app.pixel_body_python import PixelBodyMask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths (static)
INPUT_AIRFOIL_IMAGES = [
"input_naca_002.png",
"input_naca_015.png"
"input_naca_030.png",
]

# Define absolute path to the script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths to input and output folders
INPUT_FOLDER = SCRIPT_DIR / "input"
OUTPUT_FOLDER = SCRIPT_DIR / "output" / "batch_runs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Simulation-specific paths
JULIA_SCRIPT_PATH = SCRIPT_DIR.parent / "test" / "TestPixelCamSim.jl"
TEMP_MASK_PATH = OUTPUT_FOLDER / "temp_mask.npy"


def run_sim(
        domain_mask: np.ndarray,
        output_paths: dict,
        simulation_settings) -> None:

    l_c, _aoa, _thickness = characteristic_length_and_aoa_pca(
        mask=domain_mask,
        plot_method=False,
        show_components=False,
    )

    # Save the boolean mask to a temporary file for Julia to read
    mask_file = OUTPUT_FOLDER / "temp_mask.npy"
    np.save(mask_file, domain_mask)

    # Verify Julia script path
    if not JULIA_SCRIPT_PATH.is_file():
        print(f"Error: Julia script not found at {JULIA_SCRIPT_PATH}")
        sys.exit(1)

    # Unpack output paths
    output_path_particle_plot = output_paths["output_path_particle_plot"]
    output_path_heatmap_plot = output_paths["output_path_heatmap_plot"]
    output_path_data = output_paths["output_path_data"]

    cmd = [
        "julia",
        str(JULIA_SCRIPT_PATH),
        # "--sysimage", "julia_sysimage_pixelbody.so",  # Add custom Julia sysimage for faster startup and precompiled
        # package loading (precompiles Julia packages and code)
        # File I/O settings - now pass mask file instead of image
        str(mask_file),  # Pass mask file instead of input image
        str(output_path_particle_plot),  # Pass full particle gif path for dual_gifs mode
        # Simulation parameters
        str(l_c),  # Pass characteristic length from Python
        str(simulation_settings["Re"]),
        str(simulation_settings["epsilon"]),
        str(simulation_settings["t_sim"]),
        str(simulation_settings["delta_t"]),
        # Other settings
        str(simulation_settings["verbose"]).lower(),
        simulation_settings["sim_type"],
        simulation_settings["mem"],
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
    create_gifs(
        data_path=output_path_data,
        particle_output=output_path_particle_plot,
        heatmap_output=output_path_heatmap_plot,
    )

    # Define paths for both GIFs
    gif_paths = [output_path_particle_plot, output_path_heatmap_plot]

    # Get original dimensions before processing
    for gif_path in gif_paths:
        if gif_path.exists():
            dims = get_gif_dimensions(gif_path)

    target_size = tuple(simulation_settings["target_size"])

    print(f"\nProcessing GIFs to consistent size: {target_size}")

    # Process each GIF individually with aspect ratio preservation
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

    # Verify final dimensions
    print("\nFinal GIF dimensions:")
    for gif_path in gif_paths:
        if gif_path.exists():
            dims = get_gif_dimensions(gif_path)
            print(f"  {gif_path.name}: {dims[0]}x{dims[1]}")

def main() -> None:
    # Load simulation settings (same for all, so only need to load once)
    with open(SCRIPT_DIR / "sim_inputs_batch_run.yaml", "r") as f:
        batch_run_settings = yaml.safe_load(f)

    force_run = batch_run_settings.get("force_run", False)
    sim_settings = batch_run_settings["simulation_settings"]
    image_recognition_settings = sim_settings["image_recognition"]
    simulation_settings = sim_settings["simulation"]

    for base_input_image_name in tqdm(INPUT_AIRFOIL_IMAGES, desc="Airfoil:"):
        for sim_aoa in tqdm(range(-180, 181), desc="AoA", leave=False):

            # define output file names per simulation (naca type + angle of attack)
            input_path = INPUT_FOLDER / base_input_image_name

            naca_type = base_input_image_name.split('.')[0].replace("input_", "")
            file_postfix = f"{naca_type}_{sim_aoa}"

            particle_plot_name = f"particleplot_{file_postfix}.gif"
            heatmap_plot_name = f"heatmap_plot_{file_postfix}.gif"
            data_file_name = f"simulation_data_{file_postfix}.npz"

            if not force_run:
                # Check if output gifs already exist (skip if they do)
                particle_plot_output_path = OUTPUT_FOLDER / particle_plot_name
                heatmap_plot_output_path = OUTPUT_FOLDER / heatmap_plot_name
                if particle_plot_output_path.exists() and heatmap_plot_output_path.exists():
                    logger.warning(f"Gif output for {naca_type}, aoa={sim_aoa} already exists. Skipping simulation.")
                    continue


            output_path_particle_plot = OUTPUT_FOLDER / particle_plot_name
            output_path_heatmap_plot = OUTPUT_FOLDER / heatmap_plot_name
            output_path_data = OUTPUT_FOLDER / data_file_name

            # Pack output paths into a dictionary (to prevent confusing files when unpacking later in the pipeline)
            output_paths ={
                "output_path_particle_plot": output_path_particle_plot,
                "output_path_heatmap_plot": output_path_heatmap_plot,
                "output_path_data": output_path_data,
            }

            # Instantiate fluid-solid mask from base image
            pixel_body = PixelBodyMask(
                image_path=str(input_path),
                threshold=image_recognition_settings["threshold"],
                diff_threshold=image_recognition_settings["diff_threshold"],
                max_image_res=image_recognition_settings["max_image_res"],
                body_color=image_recognition_settings["solid_color"],
                manual_mode=image_recognition_settings["manual_mode"],
                force_invert_mask=image_recognition_settings["force_invert_mask"],
                verbose=False,
            )

            domain_mask = pixel_body.get_mask()
            _l_c, aoa, _thickness = characteristic_length_and_aoa_pca(
                mask=domain_mask,
                plot_method=False,
                show_components=False,
            )

            rotated_domain_mask = pixel_body.rotate_mask(current_angle=aoa, target_angle=sim_aoa)

            # Run sim for selected airfoil and angle of attack, and create plots
            try:
                run_sim(
                    rotated_domain_mask,
                    output_paths,
                    simulation_settings)
            except Exception as e:
                logger.warning(f"Simulation failed for {naca_type} at aoa={sim_aoa}: {e}")
                continue


if __name__ == "__main__":
    main()