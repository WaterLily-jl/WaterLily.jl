# WaterLily Live Simulation

Interactive real-time simulation workflow for WaterLily.jl fluid dynamics.

## Features

- **Interactive Setup**: Configure your camera field of view once using a visual selection tool
- **Live Simulation**: Press SPACE to capture image, run simulation, and display results
- **High Performance**: Uses PyJulia with precompiled sysimage for optimal speed
- **Multi-Monitor**: Display results on secondary monitor by default
- **Real-time Controls**: Adjust parameters and field of view without restarting

## Quick Start

### 1. Install Dependencies

**Python:**
```bash
pip install -r requirements_live.txt
```

**Julia Packages:**
```bash
julia -e 'using Pkg; Pkg.add(["WaterLily", "StaticArrays", "ReadVTK", "WriteVTK", "PlutoUI", "Plots", "PackageCompiler"])'
```

### 2. Build Precompiled Sysimage (Recommended)

For optimal performance, create a precompiled Julia sysimage:

```bash
julia build_sysimage.jl
```

This creates `julia_sysimage_pixelbody.so` (~100MB) that dramatically speeds up simulation startup.

### 3. Run Live Simulation

```bash
python live_simulation.py
```

## Workflow

### 1. Initial Setup & First Simulation
1. **Position your object** in the camera view
2. **Click and drag** to define the selection area  
3. **Use WASD keys** to fine-tune position
4. **Press SPACE** when satisfied with the selection
5. **First simulation runs automatically**
6. **Results are displayed** on your selected monitor

### 2. Continuous Live Mode
After the first simulation completes:
- **SPACE** - Capture new image and run next simulation
- **R** - Reconfigure field of view (if you need to adjust selection box)
- **ESC** - Exit live mode

### 3. Performance Benefits
- **No restart needed** - Run unlimited simulations in sequence
- **Julia stays loaded** - No compilation overhead after first run
- **Same selection box** - Consistent field of view between runs
- **Adjustable positioning** - Move objects between simulations

## Performance Notes

### With Sysimage (Recommended)
- First simulation: ~5-10 seconds
- Subsequent simulations: ~2-5 seconds
- Julia startup: <1 second

### Without Sysimage (Fallback)
- Each simulation: ~15-30 seconds  
- Julia startup: 5-15 seconds
- Uses subprocess mode

## Configuration

### Simulation Parameters

Edit the `simulation_params` dictionary in `LiveSimulation.__init__()`:

```python
self.simulation_params = {
    # Image recognition
    'threshold': 0.7,           # Solid detection threshold
    'diff_threshold': 0.2,      # Background noise threshold
    'solid_color': 'gray',      # 'gray', 'red', 'green', 'blue'
    'manual_mode': False,       # Use smart detection vs fixed thresholds
    'force_invert_mask': False, # Force mask inversion if needed
    
    # Resolution and timing
    'max_image_res': 800,       # Max image resolution
    't_sim': 2.0,              # Simulation time (seconds)
    'delta_t': 0.05,           # Time step
    
    # Flow parameters
    'Re': 200.0,               # Reynolds number
    'epsilon': 1.0,            # BDIM kernel width
    
    # Output processing
    'target_size': (800, 600), # GIF output size
    'maintain_aspect': False,   # Maintain aspect ratio vs stretch
}
```

### Field of View

The system remembers your field of view selection between simulation runs. To reconfigure:
- Press **R** during live mode, or
- Delete `input/field_of_view_setup.png` and restart

## Troubleshooting

### PyJulia Issues
```bash
# Install PyJulia
pip install julia

# Configure PyJulia (one-time setup)
python -c "import julia; julia.install()"
```

### Sysimage Build Fails
- Ensure all Julia packages install without errors
- Try running `TestPixelCamSim.jl` manually first
- Check Julia version compatibility

### Camera Issues
- Check camera permissions
- Try different camera index in `cv2.VideoCapture(0)`
- Ensure no other applications are using the camera

### Performance Issues
- Use the precompiled sysimage (major performance boost)
- Reduce `max_image_res` for faster processing
- Use `mem="Array"` instead of `"CuArray"` if CUDA issues

## File Structure

```
picture_sim_app/
├── live_simulation.py          # Main interactive script
├── build_sysimage.jl          # Creates precompiled Julia image
├── requirements_live.txt       # Python dependencies
├── input/
│   ├── field_of_view_setup.png # Your configured field of view
│   └── input.png               # Latest captured image
└── output/
    ├── particleplot.gif        # Particle visualization
    └── output.gif              # Flow field visualization
```

## Advanced Usage

### Custom Precompile Statements

Create `precompile_statements.jl` with common operations:

```julia
# Add to build_sysimage.jl
using WaterLily
sim = PixelSimAirfoil("test_image.png"; Re=200, ϵ=1)
# ... other common operations
```

### Batch Processing

For processing multiple images automatically, modify the live loop to read from a directory instead of camera input.

### Remote Display

Use X11 forwarding or VNC to display results on a remote machine:

```bash
ssh -X user@host python live_simulation.py
```
