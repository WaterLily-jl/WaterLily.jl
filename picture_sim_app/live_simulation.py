#!/usr/bin/env python3
"""
Interactive Live Simulation Mode

This script provides an interactive workflow where you:
1. Set up your field of view once using the selection tool
2. Hit space to capture image, run simulation, and display results
3. Repeat step 2 as many times as you want without restarting

Features:
- Uses PyJulia with precompiled sysimage for optimal performance
- Live camera preview with selection box overlay
- One-button capture and simulate workflow
- Configurable simulation parameters
- Multi-monitor support for GIF display
"""

from pathlib import Path
import subprocess

# Import Julia interface
try:
    from julia.api import Julia
    JULIA_AVAILABLE = True
except ImportError:
    print("Warning: PyJulia not available. Will use subprocess fallback.")
    JULIA_AVAILABLE = False

from picture_sim_app.image_utils import (
    capture_image, 
    resize_gif, 
    display_two_gifs_side_by_side,
    list_monitors
)

# Define absolute path to the script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths to input and output folders
INPUT_FOLDER = SCRIPT_DIR / "input"
OUTPUT_FOLDER = SCRIPT_DIR / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


class LiveSimulation:
    """Interactive live simulation controller."""
    
    def __init__(self):
        self.julia_initialized = False
        self.jl = None
        self.selection_box = None
        self.simulation_params = {}
        self.monitor_index = 1  # Default to secondary monitor
        self.setup_complete = False
        
        # File paths
        self.input_path = INPUT_FOLDER / "input.png"
        self.output_path_left = OUTPUT_FOLDER / "particleplot.gif"
        self.output_path_right = OUTPUT_FOLDER / "output.gif"
        
        # Initialize Julia if available
        self.initialize_julia()
        
        # Setup simulation parameters
        self.setup_simulation_parameters()
        
        # Setup monitor selection
        self.setup_monitor_selection()
    
    def initialize_julia(self):
        """Initialize Julia with precompiled sysimage."""
        if not JULIA_AVAILABLE:
            print("PyJulia not available. Using subprocess mode.")
            return
            
        sysimage_path = SCRIPT_DIR / "julia_sysimage_pixelbody.so"
        sysimage_no_cuda_path = SCRIPT_DIR / "julia_sysimage_pixelbody_no_cuda.so"
        julia_script = SCRIPT_DIR.parent / "test" / "TestPixelCamSim.jl"
        
        # Try CUDA-free sysimage first if available, then regular sysimage, then standard Julia
        sysimage_options = [
            (sysimage_no_cuda_path, "CUDA-free sysimage"),
            (sysimage_path, "standard sysimage")
        ]
        
        for sysimage_file, description in sysimage_options:
            if sysimage_file.exists():
                print(f"Attempting to load Julia with {description}: {sysimage_file}")
                try:
                    self.jl = Julia(sysimage=str(sysimage_file), compiled_modules=False)
                    print(f"✓ Julia initialized with {description}")
                    
                    # Load the Julia script
                    if julia_script.exists():
                        print(f"Loading Julia script: {julia_script}")
                        self.jl.include(str(julia_script))
                        self.julia_initialized = True
                        print("✓ Julia script loaded successfully")
                        return
                    else:
                        print(f"❌ Julia script not found: {julia_script}")
                        
                except Exception as e:
                    print(f"❌ Failed to initialize Julia with {description}: {e}")
                    print("🔄 Trying next option...")
                    # Clean up failed attempt
                    self.jl = None
        
        # Fallback to standard Julia without sysimage
        try:
            print("Loading Julia in standard mode...")
            self.jl = Julia(compiled_modules=False)
            print("✓ Julia initialized (standard mode)")
            print("💡 Tip: Create a CUDA-free sysimage for better performance:")
            print("   julia build_sysimage_no_cuda.jl")
            
            # Load the Julia script
            if julia_script.exists():
                print(f"Loading Julia script: {julia_script}")
                self.jl.include(str(julia_script))
                self.julia_initialized = True
                print("✓ Julia script loaded successfully")
            else:
                print(f"❌ Julia script not found: {julia_script}")
                
        except Exception as e:
            print(f"❌ Failed to initialize Julia in standard mode: {e}")
            print("🔄 Falling back to subprocess mode...")
            self.jl = None
            self.julia_initialized = False
    
    def setup_simulation_parameters(self):
        """Setup simulation parameters."""
        self.simulation_params = {
            # Image recognition settings
            'threshold': 0.7,
            'diff_threshold': 0.2,
            'solid_color': 'gray',
            'manual_mode': False,
            'force_invert_mask': False,
            
            # Image resolution cap
            'max_image_res': 800,
            
            # Simulation duration and temporal resolution
            't_sim': 2.0,
            'delta_t': 0.05,
            
            # Flow settings
            'Re': 200.0,
            'epsilon': 1.0,
            
            # Other settings
            'verbose': 'true',
            'sim_type': 'particles',
            'mem': 'Array',
            
            # Processing settings
            'target_size': (800, 600),
            'maintain_aspect': False,
        }
        
        print("Simulation Parameters:")
        print("=" * 40)
        for key, value in self.simulation_params.items():
            print(f"  {key}: {value}")
        print("=" * 40)
    
    def setup_monitor_selection(self):
        """Setup monitor for GIF display."""
        print("\nMonitor Setup:")
        monitors = list_monitors()
        
        # Default to secondary monitor if available
        self.monitor_index = 1 if len(monitors) > 1 else 0
        print(f"Display monitor: {self.monitor_index}")
    
    def setup_field_of_view(self):
        """Interactive field of view setup using selection box."""
        print("\n" + "="*60)
        print("FIELD OF VIEW SETUP")
        print("="*60)
        print("Set up your camera field of view:")
        print("1. Position your object in the camera view")
        print("2. Click and drag to define the selection area")
        print("3. Use WASD keys to fine-tune position")
        print("4. Press SPACE when satisfied with the selection")
        print("5. Press ESC to quit setup")
        print("-" * 60)
        
        # Capture initial image with selection box - this also sets up the selection area
        try:
            capture_image(
                input_folder=INPUT_FOLDER,
                image_name="input.png",  # Save directly as input.png
                fixed_aspect_ratio=(4, 3),
                selection_box_mode=True
            )
            
            # Check if image was captured
            if self.input_path.exists():
                print("✓ Field of view setup complete!")
                self.setup_complete = True
                return True
            else:
                print("❌ Field of view setup cancelled")
                return False
                
        except Exception as e:
            print(f"❌ Error during field of view setup: {e}")
            return False
    
    def live_simulation_loop(self):
        """Main live simulation loop - capture, simulate, display, repeat."""
        print("\n" + "="*60)
        print("LIVE SIMULATION MODE")
        print("="*60)
        print("Instructions:")
        print("  - Each cycle: Capture image → Run simulation → Display results")
        print("  - After displaying results, you can:")
        print("    * Press SPACE to run another simulation")
        print("    * Press R to reconfigure field of view")
        print("    * Press ESC to exit")
        print("  - You can adjust the object position between runs")
        print("-" * 60)
        
        simulation_count = 1  # We already ran the first simulation
        
        while True:
            print(f"\n🎬 Simulation #{simulation_count} completed!")
            print("📺 Displaying results...")
            
            # Step 1: Display results
            self.display_results()
            
            # Step 2: Wait for user input for next action
            print(f"\n🔄 Ready for simulation #{simulation_count + 1}")
            print("Controls:")
            print("  SPACE - Capture new image and run next simulation")
            print("  R     - Reconfigure field of view")  
            print("  ESC   - Exit live simulation")
            
            # Simple input loop
            while True:
                try:
                    action = input("\nPress [SPACE], [R], or [ESC]: ").strip().lower()
                    
                    if action in ['space', ' ', 's', '']:
                        # Run next simulation
                        break
                    elif action in ['r', 'reconfigure']:
                        # Reconfigure field of view
                        print("\n🔧 Reconfiguring field of view...")
                        if self.setup_field_of_view():
                            print("✓ Field of view reconfigured!")
                            break
                        else:
                            print("❌ Reconfiguration cancelled. Exiting.")
                            return
                    elif action in ['esc', 'escape', 'exit', 'q', 'quit']:
                        print("👋 Exiting live simulation mode...")
                        return
                    else:
                        print("Invalid input. Please press SPACE, R, or ESC.")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Exiting live simulation mode...")
                    return
            
            # Step 3: Capture new image using the same selection settings
            print(f"\n🚀 Starting simulation #{simulation_count + 1}...")
            print("📸 Capture new image using your established field of view...")
            
            try:
                # Capture new image with the same selection box settings
                capture_image(
                    input_folder=INPUT_FOLDER,
                    image_name="input.png",
                    fixed_aspect_ratio=(4, 3),
                    selection_box_mode=True  # Allow user to adjust if needed
                )
                
                if not self.input_path.exists():
                    print("❌ Image capture cancelled. Exiting live simulation.")
                    return
                    
            except Exception as e:
                print(f"❌ Error capturing image: {e}")
                continue
            
            # Step 4: Run simulation
            print("🧮 Running simulation...")
            if self.julia_initialized and self.jl:
                success = self.run_julia_simulation()
            else:
                success = self.run_subprocess_simulation()
            
            if not success:
                print("❌ Simulation failed!")
                continue
            
            # Step 5: Process GIFs
            print("🎬 Processing GIFs...")
            self.process_gifs()
            
            simulation_count += 1
    
    def run_julia_simulation(self):
        """Run simulation using PyJulia."""
        try:
            # Call Julia function directly
            params = self.simulation_params
            result = self.jl.run_simulation(
                str(self.input_path),
                str(self.output_path_left),
                params['threshold'],
                params['diff_threshold'],
                params['solid_color'],
                params['manual_mode'],
                params['force_invert_mask'],
                params['max_image_res'],
                params['t_sim'],
                params['delta_t'],
                params['Re'],
                params['epsilon'],
                params['verbose'],
                params['sim_type'],
                params['mem']
            )
            return result == 0
        except Exception as e:
            print(f"Julia simulation failed: {e}")
            return False
    
    def run_subprocess_simulation(self):
        """Run simulation using subprocess (fallback)."""
        try:
            julia_script = SCRIPT_DIR.parent / "test" / "TestPixelCamSim.jl"
            params = self.simulation_params
            
            cmd = [
                "julia",
                str(julia_script),
                str(self.input_path),
                str(self.output_path_left),
                str(params['threshold']),
                str(params['diff_threshold']),
                str(params['solid_color']),
                str(params['manual_mode']).lower(),
                str(params['force_invert_mask']).lower(),
                str(params['max_image_res']),
                str(params['t_sim']),
                str(params['delta_t']),
                str(params['Re']),
                str(params['epsilon']),
                params['verbose'],
                params['sim_type'],
                params['mem'],
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            print(f"Subprocess simulation failed: {e}")
            return False
    
    def process_gifs(self):
        """Process GIFs to consistent size."""
        gif_paths = [self.output_path_left, self.output_path_right]
        target_size = self.simulation_params['target_size']
        
        for gif_path in gif_paths:
            if gif_path.exists():
                resize_gif(
                    input_path=gif_path,
                    output_path=gif_path,
                    target_size=target_size,
                    maintain_aspect=self.simulation_params['maintain_aspect']
                )
    
    def display_results(self):
        """Display the resulting GIFs."""
        if self.output_path_left.exists() and self.output_path_right.exists():
            display_two_gifs_side_by_side(
                gif_path_left=self.output_path_left,
                gif_path_right=self.output_path_right,
                monitor_index=self.monitor_index
            )
        else:
            print("Output GIFs not found")
    
    def run(self):
        """Main execution loop."""
        print("🌊 WaterLily Live Simulation")
        print("=" * 50)
        
        # Step 1: Setup field of view and capture first image
        print("Setting up field of view and capturing first image...")
        if not self.setup_field_of_view():
            print("Setup cancelled. Exiting.")
            return
            
        # Step 2: Run first simulation
        print("\n🚀 Running first simulation...")
        print("🧮 Processing simulation...")
        
        if self.julia_initialized and self.jl:
            success = self.run_julia_simulation()
        else:
            success = self.run_subprocess_simulation()
            
        if not success:
            print("❌ First simulation failed! Check your setup.")
            return
            
        # Step 3: Process first results
        print("🎬 Processing GIFs...")
        self.process_gifs()
        
        # Step 4: Enter continuous simulation loop  
        self.live_simulation_loop()
        
        print("\n👋 Live simulation ended. Goodbye!")


def main():
    """Main function."""
    try:
        sim = LiveSimulation()
        sim.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
