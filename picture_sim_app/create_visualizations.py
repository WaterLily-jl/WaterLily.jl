"""
Create visualizations from Julia simulation data.
This script creates both particle and heatmap GIFs from simulation data saved by Julia.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_particle_gif(data_path, output_path, particle_size=0.3, particle_alpha=0.9, figsize=(8, 6)):
    """Create particle animation GIF from simulation data matching Julia style."""
    data = np.load(data_path)
    
    # Debug: print what keys are actually in the data
    print(f"Available keys in simulation data: {list(data.keys())}")
    
    # Check if the expected keys exist
    if 'particle_x' not in data or 'particle_y' not in data:
        print("Error: 'particle_x' or 'particle_y' not found in simulation data")
        print("Available data structure:")
        for key in data.keys():
            print(f"  {key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")
        return
    
    # Reconstruct particle positions from separate x,y arrays
    particle_x = data['particle_x']  # [frame, particle]
    particle_y = data['particle_y']  # [frame, particle]
    particle_positions = np.stack([particle_x, particle_y], axis=-1)  # [frame, particle, x/y]
    body_mask = data['body_mask']
    time_points = data['time_points']
    
    n_frames, n_particles, _ = particle_positions.shape
    
    # Setup figure with gray30 background like Julia (:gray30 ≈ #4d4d4d)
    fig, ax = plt.subplots(figsize=figsize, facecolor='#4d4d4d')
    ax.set_facecolor('#4d4d4d')  # Dark gray background matching Julia's :gray30
    ax.set_aspect('equal')
    
    # Remove all axes decorations like Julia
    ax.axis('off')
    
    # Plot body (solid region) with red color matching Julia's #990000
    body_solid = body_mask < 0.5  # True for solid pixels
    # Use red color with transparency matching Julia's overlay
    ax.imshow(body_solid.T, origin='lower', cmap='Reds', alpha=0.8, vmin=0, vmax=1,
              extent=[0, body_mask.shape[0], 0, body_mask.shape[1]])
    
    # Initialize scatter plot for particles - small white dots like Julia
    scat = ax.scatter([], [], c='white', s=particle_size, alpha=particle_alpha, 
                     marker='o', linewidths=0, edgecolors='none')
    
    ax.set_xlim(0, body_mask.shape[0])
    ax.set_ylim(0, body_mask.shape[1])
    
    def animate(frame):
        # Get particle positions for this frame
        pos = particle_positions[frame]
        
        # Filter out NaN positions (inactive particles)
        valid_mask = ~np.isnan(pos).any(axis=1)
        valid_pos = pos[valid_mask]
        
        if len(valid_pos) > 0:
            scat.set_offsets(valid_pos)
            # Use consistent small particle size
            sizes = np.ones(len(valid_pos)) * particle_size
            scat.set_sizes(sizes)
        else:
            scat.set_offsets(np.empty((0, 2)))
        
        # Add time display in white text like Julia
        ax.set_title(f'tU/L = {time_points[frame]:.3f}', color='white', fontsize=12, pad=10)
        return scat,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    
    # Save as GIF
    print(f"Saving particle animation to {output_path}")
    anim.save(output_path, writer='pillow', fps=20, dpi=100)
    plt.close()


def create_heatmap_gif(data_path, output_path, figsize=(8, 6)):
    """Create vorticity heatmap animation GIF from simulation data."""
    data = np.load(data_path)
    
    vorticity_fields = data['vorticity']  # [frame, i, j]
    body_mask = data['body_mask']
    time_points = data['time_points']
    
    n_frames = vorticity_fields.shape[0]
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_aspect('equal')
    
    # Determine colormap limits
    vmin, vmax = np.nanpercentile(vorticity_fields, [5, 95])
    vmax = max(abs(vmin), abs(vmax))  # Make symmetric
    vmin = -vmax
    
    # Initialize heatmap
    im = ax.imshow(np.zeros_like(vorticity_fields[0]).T, origin='lower', 
                   cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                   extent=[0, vorticity_fields.shape[1], 0, vorticity_fields.shape[2]])
    
    # Add body contour
    body_solid = body_mask < 0.5
    ax.contour(body_solid.T, levels=[0.5], colors='black', linewidths=2, 
               extent=[0, body_mask.shape[0], 0, body_mask.shape[1]])
    
    ax.set_xlim(0, vorticity_fields.shape[1])
    ax.set_ylim(0, vorticity_fields.shape[2])
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Vorticity', fontsize=10)
    
    def animate(frame):
        # Update heatmap data
        im.set_data(vorticity_fields[frame].T)
        ax.set_title(f't = {time_points[frame]:.2f}', fontsize=12)
        return im,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    
    # Save as GIF
    print(f"Saving heatmap animation to {output_path}")
    anim.save(output_path, writer='pillow', fps=20, dpi=100)
    plt.close()


def create_gifs(data_path, particle_output, heatmap_output, particle_size=0.3, particle_alpha=0.9):
    """Create both particle and heatmap GIFs from simulation data."""
    print(f"Creating visualizations from {data_path}")

    # Create particle GIF with Julia-style appearance
    create_particle_gif(data_path, particle_output, particle_size=particle_size, particle_alpha=particle_alpha)

    # Create heatmap GIF
    create_heatmap_gif(data_path, heatmap_output)

    print("✓ Both visualizations created successfully")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python create_visualizations.py data_path particle_output heatmap_output")
        sys.exit(1)
    
    data_path, particle_output, heatmap_output = sys.argv[1:4]
    create_gifs(data_path, particle_output, heatmap_output)
