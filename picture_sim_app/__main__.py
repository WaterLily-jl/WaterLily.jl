import os
import sys
from pathlib import Path

import cv2
import subprocess
from PIL import Image, ImageSequence
import pygame


SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths to input and output folders
INPUT_FOLDER = SCRIPT_DIR / "input"
OUTPUT_FOLDER = SCRIPT_DIR / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)



def capture_image(image_name: str = "input.png") -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found or cannot be opened.")

    print("Press [space] to capture image, or [ESC] to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == 32:  # Spacebar
            # Use the global INPUT_FOLDER path
            path = INPUT_FOLDER / image_name
            cv2.imwrite(str(path), frame) # cv2.imwrite often prefers string paths
            print(f"Image saved to {path}")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Read the saved image for cropping
    img = cv2.imread(str(path)) # cv2.imread often prefers string paths
    if img is None:
        raise RuntimeError(f"Failed to load image {path} for cropping.")

    # Let the user select ROI with mouse
    print("Select ROI (drag mouse to crop). Press ENTER or SPACE to confirm, or 'c' to cancel.")
    roi = cv2.selectROI("Crop Image", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w > 0 and h > 0:
        # Crop and overwrite the image
        cropped_img = img[int(y):int(y+h), int(x):int(x+w)]
        cv2.imwrite(str(path), cropped_img) # cv2.imwrite often prefers string paths
        print(f"Cropped image saved to {path}")
    else:
        print("No crop selected, original image kept.")

def list_monitors():
    """List available monitors and their properties."""
    pygame.init()
    
    # Get number of displays
    num_displays = pygame.display.get_num_displays()
    print(f"Number of displays detected: {num_displays}")
    
    monitors = []
    for i in range(num_displays):
        # Get display bounds
        display_info = pygame.display.get_desktop_sizes()[i] if hasattr(pygame.display, 'get_desktop_sizes') else (1920, 1080)
        monitors.append({
            'index': i,
            'width': display_info[0],
            'height': display_info[1]
        })
        print(f"  Monitor {i}: {display_info[0]}x{display_info[1]}")
    
    pygame.quit()
    return monitors

def display_gif_fullscreen(gif_path, monitor_index=0, force_windowed=False):
    """
    Display a GIF in fullscreen mode on a specific monitor using pygame.
    
    Args:
        gif_path (str): Absolute path to the GIF file
        monitor_index (int): Index of the monitor to use (0 for primary, 1 for secondary, etc.)
        force_windowed (bool): If True, starts in windowed mode instead of fullscreen
    """
    try:
        # Initialize pygame
        pygame.init()
        
        # Get monitor information
        num_displays = pygame.display.get_num_displays()
        print(f"Available displays: {num_displays}")
        
        if monitor_index >= num_displays:
            print(f"Monitor {monitor_index} not found. Using monitor 0.")
            monitor_index = 0
        
        # Get all desktop sizes
        desktop_sizes = pygame.display.get_desktop_sizes() if hasattr(pygame.display, 'get_desktop_sizes') else [(1920, 1080)] * num_displays
        print(f"Desktop sizes: {desktop_sizes}")
        
        # Calculate position for the target monitor
        x_offset = 0
        for i in range(monitor_index):
            if i < len(desktop_sizes):
                x_offset += desktop_sizes[i][0]
        
        y_offset = 0  # Assuming monitors are horizontally aligned
        
        # Get target monitor dimensions
        if monitor_index < len(desktop_sizes):
            screen_width, screen_height = desktop_sizes[monitor_index]
        else:
            screen_width, screen_height = 1920, 1080
        
        print(f"Target monitor {monitor_index}: {screen_width}x{screen_height} at offset ({x_offset}, {y_offset})")
        
        # Set window position BEFORE creating the display
        os.environ['SDL_VIDEO_WINDOW_POS'] = f'{x_offset},{y_offset}'
        
        # Start in windowed mode first, then go fullscreen
        # This helps with proper monitor detection
        if force_windowed:
            screen = pygame.display.set_mode((screen_width, screen_height))
            fullscreen = False
        else:
            # Create windowed first, then switch to fullscreen
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.time.wait(100)  # Brief pause
            screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
            fullscreen = True
        pygame.display.set_caption("GIF Player")
        
        # Load and process GIF
        print("Loading GIF...")
        gif = Image.open(gif_path)
        frames = []
        durations = []
        
        # Extract all frames
        frame_count = 0
        while True:
            try:
                # Convert PIL image to pygame surface
                frame = gif.convert('RGBA')
                pygame_image = pygame.image.fromstring(frame.tobytes(), frame.size, 'RGBA')
                
                # Scale frame to fit screen while maintaining aspect ratio
                frame_rect = pygame_image.get_rect()
                screen_rect = screen.get_rect()
                
                # Calculate scaling factor
                scale_x = screen_rect.width / frame_rect.width
                scale_y = screen_rect.height / frame_rect.height
                scale = min(scale_x, scale_y)
                
                new_width = int(frame_rect.width * scale)
                new_height = int(frame_rect.height * scale)
                
                # Scale the image
                scaled_image = pygame.transform.scale(pygame_image, (new_width, new_height))
                
                # Center the image
                centered_rect = scaled_image.get_rect(center=screen_rect.center)
                
                frames.append((scaled_image, centered_rect))
                
                # Get frame duration
                duration = gif.info.get('duration', 100)  # Default 100ms
                durations.append(duration)
                
                frame_count += 1
                gif.seek(frame_count)
                
            except EOFError:
                break
        
        if not frames:
            print("No frames found in the GIF")
            return
        
        print(f"Loaded {len(frames)} frames")
        print("Controls: ESC/Q=quit, SPACE=pause/resume, F=toggle fullscreen, M=move to next monitor")
        
        # Animation loop
        clock = pygame.time.Clock()
        frame_index = 0
        running = True
        paused = False
        last_frame_time = pygame.time.get_ticks()
        current_monitor = monitor_index
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        if paused:
                            print("Paused")
                        else:
                            print("Resumed")
                            last_frame_time = current_time
                    elif event.key == pygame.K_f:
                        # Toggle fullscreen
                        fullscreen = not fullscreen
                        if fullscreen:
                            screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
                            print("Switched to fullscreen")
                        else:
                            screen = pygame.display.set_mode((screen_width, screen_height))
                            print("Switched to windowed")
                    elif event.key == pygame.K_m:
                        # Move to next monitor
                        current_monitor = (current_monitor + 1) % num_displays
                        
                        # Calculate new position
                        new_x_offset = 0
                        for i in range(current_monitor):
                            if i < len(desktop_sizes):
                                new_x_offset += desktop_sizes[i][0]
                        
                        # Get new monitor dimensions
                        if current_monitor < len(desktop_sizes):
                            screen_width, screen_height = desktop_sizes[current_monitor]
                        
                        print(f"Moving to monitor {current_monitor}: {screen_width}x{screen_height}")
                        
                        # Recreate window on new monitor
                        os.environ['SDL_VIDEO_WINDOW_POS'] = f'{new_x_offset},0'
                        if fullscreen:
                            screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
                        else:
                            screen = pygame.display.set_mode((screen_width, screen_height))
            
            # Update animation if not paused
            if not paused and current_time - last_frame_time >= durations[frame_index]:
                frame_index = (frame_index + 1) % len(frames)
                last_frame_time = current_time
            
            # Draw current frame
            screen.fill((0, 0, 0))  # Black background
            frame_surface, frame_rect = frames[frame_index]
            screen.blit(frame_surface, frame_rect)
            
            pygame.display.flip()
            clock.tick(60)  # Limit to 60 FPS for smooth playback
        
    except FileNotFoundError:
        print(f"Error: GIF file not found at {gif_path}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()


def resize_gif(input_path, output_path, target_size=(1920, 1080), fill_color=(0, 0, 0, 0)):
    """
    Resize each frame of a GIF while preserving aspect ratio, then pad to target size.

    Args:
        input_path (str): Path to the input GIF.
        output_path (str): Path to save the resized and padded GIF.
        target_size (tuple): Desired (width, height), e.g., (1920, 1080).
        fill_color (tuple): RGBA fill color for padding, default transparent/black.
    """
    with Image.open(input_path) as img:
        frames = []
        durations = []

        for frame in ImageSequence.Iterator(img):
            frame = frame.convert("RGBA")
            orig_w, orig_h = frame.size
            target_w, target_h = target_size

            # Compute scale to preserve aspect ratio
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            # Resize with aspect ratio
            resized = frame.resize((new_w, new_h), Image.BICUBIC)

            # Create new canvas and paste resized frame in center
            canvas = Image.new("RGBA", target_size, fill_color)
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            canvas.paste(resized, (offset_x, offset_y))

            frames.append(canvas)
            durations.append(frame.info.get('duration', 100))

        # Save as animated GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2
        )


def main() -> None:
    capture_image()

    # Paths
    input_image_name = "input.png"
    input_path = INPUT_FOLDER / input_image_name
    output_gif_name = "output.gif"
    output_gif = OUTPUT_FOLDER / output_gif_name

    julia_script = SCRIPT_DIR.parent / "test" / "TestPixelCamSim.jl"

    # Verify Julia script path
    if not julia_script.is_file():
        print(f"Error: Julia script not found at {julia_script}")
        sys.exit(1)

    cmd = ["julia", str(julia_script), str(input_path), str(output_gif)]
    print(f"Starting Julia: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"\nJulia process exited with code {result.returncode}")

    resize_gif(input_path=output_gif, output_path=output_gif)

    display_gif_fullscreen(gif_path=output_gif, monitor_index=1)


if __name__ == "__main__":
    main()
