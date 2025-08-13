from pathlib import Path

import numpy as np

from picture_sim_app.image_utils import get_gif_dimensions


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

def create_list_of_gif_files() -> list:
    list_gif_files = []
    aoa_range = np.arange(-180, 180 + 1, 3)

    for base_input_image_name in INPUT_AIRFOIL_IMAGES:
        for sim_aoa in aoa_range:

            naca_type = base_input_image_name.split('.')[0].replace("input_", "")
            file_postfix = f"{naca_type}_{sim_aoa}"

            particle_plot_name = f"particleplot_{file_postfix}.gif"
            heatmap_plot_name = f"heatmap_plot_{file_postfix}.gif"

            output_path_particle_plot = OUTPUT_FOLDER / particle_plot_name
            output_path_heatmap_plot = OUTPUT_FOLDER / heatmap_plot_name

            list_gif_files.append(output_path_particle_plot)
            list_gif_files.append(output_path_heatmap_plot)

    return list_gif_files


def main():

    # gif_files = list(BATCH_RUNS_DIR.glob("*.gif"))

    # print(f"Check {len(gif_files)} GIF files exist and can be loaded:")

    gif_files =  create_list_of_gif_files()

    problematic_files = []
    
    for gif_path in gif_files:

        if not gif_path.exists():
            print(f"{gif_path.name}: ERROR - File does not exist")
            problematic_files.append(gif_path)
            continue

        try:
            dimensions = get_gif_dimensions(gif_path)
            # print(f"{gif_path.name}: {dimensions[0]}x{dimensions[1]}")
        except Exception as e:
            print(f"{gif_path.name}: ERROR - {e}")
            problematic_files.append(gif_path)

    if len(problematic_files) > 0:
        print(f"\n {len(problematic_files)} problematic GIF files found")

if __name__ == "__main__":
    main()
