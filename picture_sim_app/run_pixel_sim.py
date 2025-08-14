from pathlib import Path

import yaml

from picture_sim_app.characteristic_length_and_aoa_estimation import characteristic_length_and_aoa_pca
from picture_sim_app.detect_airfoil_type import detect_airfoil_type

from picture_sim_app.image_utils import (
    capture_image,
)
from picture_sim_app.live_simulation import run_julia_simulation_script
from picture_sim_app.pixel_body_python import PixelBodyMask

# Define absolute path to the script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths to input and output folders
INPUT_FOLDER = SCRIPT_DIR / "input"
OUTPUT_FOLDER = SCRIPT_DIR / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def run_simulation(settings):
    """Run one complete simulation cycle."""
    # Capture image (skips bounding box selection and uses cached option)
    capture_image(
        input_folder=INPUT_FOLDER,
        fixed_aspect_ratio=(16, 9),  # Aspect ratio of the selection box (width:height)
        selection_box_mode=True,  # Click-and-drag selection box
        # fixed_size=(800, 600),    # Alternative: exact pixel dimensions
        use_cached_box=True,  # Fixed typo: was use_chached_box
    )

    # File I/O paths
    io_settings = settings["io_settings"]
    input_path = INPUT_FOLDER / io_settings["input_image_name"]
    output_path_particle_plot = OUTPUT_FOLDER / io_settings["particle_plot_name"]
    output_path_heatmap_plot = OUTPUT_FOLDER / io_settings["heatmap_plot_name"]
    # output_path_data = OUTPUT_FOLDER / data_file_name

    # Unpack simulation settings:
    simulation_settings = settings["simulation_settings"]
    image_recognition_debug_mode = simulation_settings["image_recognition_debug_mode"]
    show_components_pca = simulation_settings["show_components_pca"]

    if show_components_pca and not image_recognition_debug_mode:
        raise Exception("'show_components_pca' is set to True, but 'image_recognition_debug_mode' is False."
                        "\n set 'image_recognition_debug_mode' to True to see PCA components.")

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

    plot_mask = image_recognition_debug_mode
    if plot_mask:
        pixel_body.plot_mask()

    if not simulation_settings["use_precomputed_results"]:
        # Estimate characteristic length and angle of attack using PCA
        l_c, aoa, thickness = characteristic_length_and_aoa_pca(
            mask=domain_mask,
            plot_method=image_recognition_debug_mode,
            show_components=show_components_pca,
        )

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
        # If using precomputed results, try to find the best matching GIF plots and use those instead of running the
        # simulation

        # Estimate characteristic length and angle of attack using PCA and airfoil type detection
        l_c, aoa, thickness = characteristic_length_and_aoa_pca(
            mask=domain_mask,
            plot_method=image_recognition_debug_mode,
            show_components=show_components_pca,
            object_is_airfoil=True,
        )

        # Estimate airfoil type based on thickness and characteristic length
        airfoil_type = detect_airfoil_type(thickness_to_cord_ratio=thickness / l_c)

        # Round angle of attack to nearest multiple of 3
        rounded_aoa = round(aoa / 3) * 3

        # Find gifs corresponding to the airfoil type and angle of attack
        particle_plot_name = f"particleplot_{airfoil_type}_{rounded_aoa}.gif"
        heatmap_plot_name = f"heatmap_plot_{airfoil_type}_{rounded_aoa}.gif"

        # Check if the plots exists
        output_path_particle_plot = OUTPUT_FOLDER / "batch_runs" / particle_plot_name
        output_path_heatmap_plot = OUTPUT_FOLDER / "batch_runs" / heatmap_plot_name

        if not output_path_particle_plot.exists():
            raise FileNotFoundError(f"Could not find {output_path_particle_plot}")

        if not output_path_heatmap_plot.exists():
            raise FileNotFoundError(f"Could not find {output_path_heatmap_plot}")

        # Overwrite the output paths to the found files (use symlink instead of copying)
        symlink_particle = OUTPUT_FOLDER / "particleplot.gif"
        symlink_heatmap = OUTPUT_FOLDER / "heatmap_plot.gif"

        # Remove existing symlinks/files if they exist
        if symlink_particle.exists() or symlink_particle.is_symlink():
            symlink_particle.unlink()
        if symlink_heatmap.exists() or symlink_heatmap.is_symlink():
            symlink_heatmap.unlink()

        # Create new symlinks pointing to the batch_runs files
        symlink_particle.symlink_to(output_path_particle_plot)
        symlink_heatmap.symlink_to(output_path_heatmap_plot)


def main() -> None:
    # Load settings once at start
    with open(SCRIPT_DIR / "configs/settings.yaml", "r") as f:
        settings = yaml.safe_load(f)

    # First run with interactive selection (don't use cached box)
    capture_image(
        input_folder=INPUT_FOLDER,
        fixed_aspect_ratio=(16, 9),
        selection_box_mode=True,
        use_cached_box=False,  # Interactive selection for first run
    )

    # Run first simulation
    run_simulation(settings)
    
    # Idle loop waiting for spacebar
    while True:
        user_input = input("\nPress ENTER to run again, 'r' to reselect box, or 'q' to quit: ").strip().lower()
        if user_input == 'q':
            print("Exiting...")
            break
        elif user_input == '':
            print("Running simulation again...")
            # Reload settings in case they changed
            with open(SCRIPT_DIR / "configs/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            run_simulation(settings)
        elif user_input == 'r':
            print("Reselecting bounding box...")
            capture_image(
                input_folder=INPUT_FOLDER,
                fixed_aspect_ratio=(16, 9),
                selection_box_mode=True,
                use_cached_box=False,  # Force interactive selection
            )
            # Reload settings and run simulation with new box
            with open(SCRIPT_DIR / "configs/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            run_simulation(settings)
        else:
            print("Invalid input. Press ENTER, 'r', or 'q'.")


if __name__ == "__main__":
    main()