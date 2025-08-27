from pathlib import Path
import json
import os, shutil
import time
import subprocess
import psutil

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

CAMERA_INDEX = 1


def find_display_processes():
    """Find running display processes by looking for our display scripts."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('persistent_gif_display' in str(arg) for arg in cmdline):
                processes.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def stop_display_processes():
    """Stop any running display processes."""
    pids = find_display_processes()
    stopped = []
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
            stopped.append(pid)
            print(f"[display] Stopped display process {pid}")
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                proc.kill()  # Force kill if terminate didn't work
                stopped.append(pid)
                print(f"[display] Force killed display process {pid}")
            except psutil.NoSuchProcess:
                pass
        except Exception as e:
            print(f"[display] Failed to stop process {pid}: {e}")
    return stopped


def start_display_process(monitor_index=1, use_qt_version=True, flip_90_degrees=False):
    """Start the display process."""
    try:
        if use_qt_version:
            if flip_90_degrees:
                script_path = SCRIPT_DIR / "persistent_gif_display_90_deg.py"
            else:
                script_path = SCRIPT_DIR / "persistent_gif_display_2.py"
        else:
            script_path = SCRIPT_DIR / "persistent_gif_display.py"

        # Start the process in the background
        proc = subprocess.Popen([
            "python", str(script_path), str(monitor_index)
        ], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)

        print(f"[display] Started display process {proc.pid} on monitor {monitor_index}")
        time.sleep(2)  # Give it time to start
        return proc.pid
    except Exception as e:
        print(f"[display] Failed to start display process: {e}")
        return None


def restart_display_process(monitor_index=1, use_qt_version=True, flip_90_degrees=False):
    """Stop existing display processes and start a new one."""
    print("[display] Restarting display process...")
    stopped_pids = stop_display_processes()
    time.sleep(1)  # Brief pause between stop and start
    new_pid = start_display_process(monitor_index, use_qt_version, flip_90_degrees)
    return new_pid


def safe_unlink_windows(path: Path, max_retries: int = 3, delay: float = 0.5):
    """
    Simplified unlink - display process will be restarted so no need for complex retry logic.
    """
    for attempt in range(max_retries):
        try:
            if path.exists() or path.is_symlink():
                path.unlink()
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"[unlink] File locked, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"[unlink] Failed to unlink {path} after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            print(f"[unlink] Unexpected error unlinking {path}: {e}")
            raise
    return False


def ensure_link(src: Path, dst: Path):
    """
    Create a symlink to src at dst. Simplified version since display will be restarted.
    """
    # Simple unlink since we're restarting the display process
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    try:
        dst.symlink_to(src)
        print(f"[link] symlink created: {dst} -> {src}")
        return
    except OSError as e:
        if getattr(e, "winerror", None) == 1314:
            print(f"[link] Symlink privilege missing (1314). Falling back to hardlink/copy.")
        else:
            print(f"[link] Symlink failed ({e}). Trying hardlink/copy.")

    # Hardlink attempt
    try:
        os.link(src, dst)
        print(f"[link] hardlink created: {dst} -> {src}")
        return
    except OSError as e:
        print(f"[link] Hardlink failed ({e}). Copying file.")

    # Copy fallback
    try:
        shutil.copy2(src, dst)
        print(f"[link] file copied: {dst} (from {src})")
    except Exception as e:
        raise RuntimeError(f"Failed to create link or copy from {src} to {dst}: {e}")


def run_simulation(settings):
    """Run one complete simulation cycle."""
    # Capture image (skips bounding box selection and uses cached option)
    capture_image(
        input_folder=INPUT_FOLDER,
        fixed_aspect_ratio=tuple(settings["capture_image_aspect_ratio"]),  # Aspect ratio of the selection box (width:height)
        selection_box_mode=True,  # Click-and-drag selection box
        # fixed_size=(800, 600),    # Alternative: exact pixel dimensions
        use_cached_box=True,  # Fixed typo: was use_chached_box
        camera_index=CAMERA_INDEX,
    )

    # File I/O paths
    io_settings = settings["io_settings"]
    input_path = INPUT_FOLDER / io_settings["input_image_name"]
    output_path_particle_plot = OUTPUT_FOLDER / io_settings["particle_plot_name"]
    output_path_heatmap_vorticity= OUTPUT_FOLDER / io_settings["heatmap_vorticity_name"]
    output_path_heatmap_pressure = OUTPUT_FOLDER / io_settings["heatmap_pressure_name"]
    # output_path_data = OUTPUT_FOLDER / data_file_name
    flip_90_degrees = io_settings["flip_90_degrees"]

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

    object_is_airfoil = True

    if not simulation_settings["use_precomputed_results"]:
        # Estimate characteristic length and angle of attack using PCA
        l_c, aoa, thickness = characteristic_length_and_aoa_pca(
            mask=domain_mask,
            plot_method=image_recognition_debug_mode,
            show_components=show_components_pca,
            object_is_airfoil=object_is_airfoil,
        )

        if object_is_airfoil:
            # Estimate airfoil type based on thickness and characteristic length
            airfoil_type = detect_airfoil_type(thickness_to_cord_ratio=thickness / l_c)

        run_julia_simulation_script(
            domain_mask=domain_mask,
            l_c=l_c,
            simulation_settings=simulation_settings,
            output_path_particle_plot=output_path_particle_plot,
            output_path_heatmap_vorticity=output_path_heatmap_vorticity,
            output_path_heatmap_pressure=output_path_heatmap_pressure,
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
            object_is_airfoil=object_is_airfoil,
        )
        if flip_90_degrees:
            # Rotate the detected angle 90 degrees counterclockwise to match angle if flow is coming from the top
            aoa -= 90

        if object_is_airfoil:
            # Estimate airfoil type based on thickness and characteristic length
            airfoil_type = detect_airfoil_type(thickness_to_cord_ratio=thickness / l_c)

        # Round angle of attack to nearest multiple of 3
        rounded_aoa = round(aoa / 3) * 3

        # Find gifs corresponding to the airfoil type and angle of attack
        particle_plot_name = f"particleplot_{airfoil_type}_{rounded_aoa}.gif"
        heatmap_vorticity_name = f"heatmap_vorticity_{airfoil_type}_{rounded_aoa}.gif"
        heatmap_pressure_name = f"heatmap_pressure_{airfoil_type}_{rounded_aoa}.gif"

        # Check if the plots exists
        output_path_particle_plot = OUTPUT_FOLDER / "batch_runs" / particle_plot_name
        output_path_heatmap_vorticity = OUTPUT_FOLDER / "batch_runs" / heatmap_vorticity_name
        output_path_heatmap_pressure = OUTPUT_FOLDER / "batch_runs" / heatmap_pressure_name

        if not output_path_particle_plot.exists():
            raise FileNotFoundError(f"Could not find {output_path_particle_plot}")

        if not output_path_heatmap_vorticity.exists():
            raise FileNotFoundError(f"Could not find {output_path_heatmap_vorticity}")

        if not output_path_heatmap_pressure.exists():
            raise FileNotFoundError(f"Could not find {output_path_heatmap_pressure}")

        # Overwrite the output paths to the found files (use symlink instead of copying)
        symlink_particle = OUTPUT_FOLDER / "particleplot.gif"
        symlink_heatmap_vorticity = OUTPUT_FOLDER / "heatmap_vorticity.gif"
        symlink_heatmap_pressure = OUTPUT_FOLDER / "heatmap_pressure.gif"

        # Stop display processes to release files
        print("Stopping display processes to update symlinks...")
        stop_display_processes()
        time.sleep(1)  # Give processes time to fully stop

        # Remove existing symlinks/files if they exist (should work now)
        safe_unlink_windows(symlink_particle)
        safe_unlink_windows(symlink_heatmap_vorticity)
        safe_unlink_windows(symlink_heatmap_pressure)

        # Create new symlinks pointing to the batch_runs files
        ensure_link(output_path_particle_plot, symlink_particle)
        ensure_link(output_path_heatmap_vorticity, symlink_heatmap_vorticity)
        ensure_link(output_path_heatmap_pressure, symlink_heatmap_pressure)

    # Restart display process
    print("Restarting display process...")
    restart_display_process(monitor_index=1, use_qt_version=True, flip_90_degrees=flip_90_degrees)

    # Save airfoil data to JSON (use actual AoA, not rounded)
    airfoil_data = {
        "airfoil_type": airfoil_type if object_is_airfoil else "unknown",
        "aoa": round(aoa, 1),
        "thickness": round(thickness, 3),
        "characteristic_length": round(l_c, 3)
    }

    airfoil_data_path = OUTPUT_FOLDER / "airfoil_data.json"
    with open(airfoil_data_path, "w") as f:
        json.dump(airfoil_data, f, indent=2)


def main() -> None:
    # Load settings once at start
    with open(SCRIPT_DIR / "configs/settings.yaml", "r") as f:
        settings = yaml.safe_load(f)

    # First run with interactive selection (don't use cached box)
    capture_image(
        input_folder=INPUT_FOLDER,
        fixed_aspect_ratio=tuple(settings["capture_image_aspect_ratio"]),
        selection_box_mode=True,
        use_cached_box=False,  # Interactive selection for first run
        camera_index=CAMERA_INDEX,
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
                fixed_aspect_ratio=tuple(settings["capture_image_aspect_ratio"]),
                selection_box_mode=True,
                use_cached_box=False,  # Force interactive selection
                camera_index=CAMERA_INDEX,
            )
            # Reload settings and run simulation with new box
            with open(SCRIPT_DIR / "configs/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            run_simulation(settings)
        else:
            print("Invalid input. Press ENTER, 'r', or 'q'.")


if __name__ == "__main__":
    main()