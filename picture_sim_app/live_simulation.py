import pathlib
import subprocess
import sys

import numpy as np


def run_julia_simulation_script(
        domain_mask: np.ndarray,
        l_c: float,
        simulation_settings: dict,
        output_path_particle_plot: str,
        output_path_heatmap_plot: str,
        output_folder: pathlib.Path,
        script_dir: pathlib.Path,
) -> None:

    # Save the boolean mask to a temporary file for Julia to read
    mask_file = output_folder / "temp_mask.npy"
    np.save(mask_file, domain_mask)

    # Run Julia script 'TestPixelCamSim.jl'
    julia_script = script_dir.parent / "test" / "TestPixelCamSim.jl"

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