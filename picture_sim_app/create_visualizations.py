"""
Create visualizations from Julia simulation data.
This script creates both particle and heatmap GIFs from simulation data saved by Julia.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_particle_gif(data_path, output_path, scale=5.0, figsize=(8, 6)):
    """Create particle animation GIF from simulation data."""
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
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='#4d4d4d')
    ax.set_facecolor('#4d4d4d')
    ax.set_aspect('equal')
    
    # Plot body (solid region)
    body_solid = body_mask < 0.5  # True for solid pixels
    ax.imshow(body_solid.T, origin='lower', cmap='Reds', alpha=0.8, extent=[0, body_mask.shape[0], 0, body_mask.shape[1]])
    
    # Initialize scatter plot for particles
    scat = ax.scatter([], [], c='white', s=1, alpha=0.8)
    
    ax.set_xlim(0, body_mask.shape[0])
    ax.set_ylim(0, body_mask.shape[1])
    ax.axis('off')
    
    def animate(frame):
        # Get particle positions for this frame
        pos = particle_positions[frame]
        
        # Filter out NaN positions (inactive particles)
        valid_mask = ~np.isnan(pos).any(axis=1)
        valid_pos = pos[valid_mask]
        
        if len(valid_pos) > 0:
            scat.set_offsets(valid_pos)
            # Vary particle size based on some criteria (optional)
            sizes = np.ones(len(valid_pos)) * scale
            scat.set_sizes(sizes)
        
        ax.set_title(f't = {time_points[frame]:.2f}', color='white', fontsize=12)
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


def create_gifs(data_path, particle_output, heatmap_output):
    """Create both particle and heatmap GIFs from simulation data."""
    print(f"Creating visualizations from {data_path}")

    # Create particle GIF
    create_particle_gif(data_path, particle_output)

    # Create heatmap GIF
    create_heatmap_gif(data_path, heatmap_output)

    print("✓ Both visualizations created successfully")
