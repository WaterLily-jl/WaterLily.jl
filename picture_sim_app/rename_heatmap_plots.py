import os
from pathlib import Path

def rename_heatmap_plots():
    batch_dir = Path(__file__).resolve().parent.parent / "picture_sim_app" / "output" / "batch_runs"
    if not batch_dir.exists():
        print(f"Directory not found: {batch_dir}")
        return
    for file in batch_dir.iterdir():
        if file.is_file() and file.name.startswith("heatmap_plot"):
            new_name = "heatmap_vorticity" + file.name[len("heatmap_plot"):]
            new_path = batch_dir / new_name
            print(f"Renaming {file.name} -> {new_name}")
            file.rename(new_path)

if __name__ == "__main__":
    rename_heatmap_plots()
