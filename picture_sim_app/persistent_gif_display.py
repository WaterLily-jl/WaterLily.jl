#!/usr/bin/env python3
"""
Persistent GIF Display Monitor

A standalone script that continuously displays two GIFs side by side on a secondary monitor.
Automatically reloads GIFs when files are modified, with manual reload option.

Usage:
    python persistent_gif_display.py

Controls:
    - ESC/Q: Quit
    - R: Manual reload GIFs
    - M: Switch monitor
    - F: Toggle fullscreen
    - P: Pause/resume file monitoring
    - SPACE: Pause/resume animation

Features:
    - Automatic file monitoring with debounced reloading
    - Manual reload capability
    - Monitor switching
    - Graceful error handling for missing files
    - Memory efficient GIF loading
"""

import time
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pygame
from PIL import Image, ImageSequence
import os
from typing import Tuple, List

class GifFileHandler(FileSystemEventHandler):
    """Handles file system events for GIF files."""
    
    def __init__(self, callback, gif_paths: List[Path], debounce_seconds: float = 1.0):
        self.callback = callback
        self.gif_paths = [Path(p) for p in gif_paths]
        self.gif_names = {p.name for p in self.gif_paths}
        self.debounce_seconds = debounce_seconds
        self.last_modified = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.name in self.gif_names:
            current_time = time.time()
            
            # Debounce: only trigger if enough time has passed since last modification
            if file_path in self.last_modified:
                if current_time - self.last_modified[file_path] < self.debounce_seconds:
                    return
                    
            self.last_modified[file_path] = current_time
            
            # Small delay to ensure file write is complete
            threading.Timer(0.5, lambda: self.callback(file_path)).start()

class PersistentGifDisplay:
    """Main display class for persistent GIF viewing."""
    
    def __init__(self, gif_path_left: Path, gif_path_right: Path, monitor_index: int = 1):
        self.gif_path_left = Path(gif_path_left)
        self.gif_path_right = Path(gif_path_right)
        self.monitor_index = monitor_index
        
        # Display state
        self.screen = None
        self.running = True
        self.animation_paused = False
        self.monitoring_paused = False
        self.fullscreen = False
        
        # GIF data
        self.frames_left = []
        self.frames_right = []
        self.durations_left = []
        self.durations_right = []
        self.frame_indices = [0, 0]
        self.last_frame_times = [0, 0]
        
        # File monitoring
        self.observer = None
        self.reload_pending = False
        self.reload_lock = threading.Lock()
        
        # Initialize pygame
        pygame.init()
        self.setup_display()
        self.setup_file_monitoring()
        
    def setup_display(self):
        """Initialize the display on the specified monitor."""
        num_displays = pygame.display.get_num_displays()
        desktop_sizes = pygame.display.get_desktop_sizes() if hasattr(pygame.display, "get_desktop_sizes") else [(1920, 1080)] * num_displays
        
        print(f"Available displays: {num_displays}")
        for i, size in enumerate(desktop_sizes):
            status = " (CURRENT)" if i == self.monitor_index else ""
            print(f"  Monitor {i}: {size[0]}x{size[1]}{status}")
        
        if self.monitor_index >= num_displays:
            print(f"Monitor {self.monitor_index} not available. Using monitor 0.")
            self.monitor_index = 0
        
        # Calculate monitor position
        x_offset = sum(desktop_sizes[i][0] for i in range(self.monitor_index))
        self.screen_width, self.screen_height = desktop_sizes[self.monitor_index]
        
        # Set window position and create display
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x_offset},0"
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags)
        pygame.display.set_caption(f"Persistent GIF Display - Monitor {self.monitor_index}")
        
    def setup_file_monitoring(self):
        """Setup file system monitoring for automatic reloads."""
        # Monitor the directories containing the GIF files
        directories_to_watch = set()
        for gif_path in [self.gif_path_left, self.gif_path_right]:
            if gif_path.exists():
                directories_to_watch.add(gif_path.parent)
        
        if directories_to_watch:
            self.observer = Observer()
            handler = GifFileHandler(
                callback=self.on_file_changed,
                gif_paths=[self.gif_path_left, self.gif_path_right],
                debounce_seconds=1.0
            )
            
            for directory in directories_to_watch:
                self.observer.schedule(handler, str(directory), recursive=False)
                print(f"Monitoring directory: {directory}")
            
            self.observer.start()
            
    def on_file_changed(self, file_path: Path):
        """Callback for when a monitored file changes."""
        if not self.monitoring_paused:
            print(f"File changed: {file_path.name} - Reloading GIFs...")
            with self.reload_lock:
                self.reload_pending = True
    
    def load_gif_frames(self, gif_path: Path, target_height: int) -> Tuple[List, List, int]:
        """Load and scale GIF frames to fit the target height."""
        if not gif_path.exists():
            print(f"Warning: GIF file not found: {gif_path}")
            # Return empty placeholder frames
            placeholder = pygame.Surface((100, target_height))
            placeholder.fill((50, 50, 50))  # Dark gray
            return [placeholder], [1000], 100  # 1 second duration, 100px width
        
        try:
            with Image.open(gif_path) as gif:
                frames, durations = [], []
                width = 0
                
                for frame in ImageSequence.Iterator(gif):
                    frame = frame.convert('RGBA')
                    orig_w, orig_h = frame.size
                    
                    # Scale to target height while maintaining aspect ratio
                    scale = target_height / orig_h
                    new_w = int(orig_w * scale)
                    width = new_w  # Store width of last frame (should be consistent)
                    
                    frame_resized = frame.resize((new_w, target_height), Image.LANCZOS)
                    surf = pygame.image.fromstring(frame_resized.tobytes(), frame_resized.size, 'RGBA')
                    frames.append(surf)
                    durations.append(frame.info.get('duration', 100))
                
                return frames, durations, width
                
        except Exception as e:
            print(f"Error loading GIF {gif_path}: {e}")
            # Return error placeholder
            placeholder = pygame.Surface((100, target_height))
            placeholder.fill((100, 0, 0))  # Dark red for error
            return [placeholder], [1000], 100
    
    def reload_gifs(self):
        """Reload both GIF files and rescale for current display."""
        print("Reloading GIFs...")
        
        # Load frames at full screen height first
        frames_left, durations_left, width_left = self.load_gif_frames(self.gif_path_left, self.screen_height)
        frames_right, durations_right, width_right = self.load_gif_frames(self.gif_path_right, self.screen_height)
        
        # Check if combined width exceeds screen width
        total_width = width_left + width_right
        if total_width > self.screen_width:
            # Scale down both to fit
            scale = self.screen_width / total_width
            new_height = int(self.screen_height * scale)
            
            frames_left, durations_left, width_left = self.load_gif_frames(self.gif_path_left, new_height)
            frames_right, durations_right, width_right = self.load_gif_frames(self.gif_path_right, new_height)
            
            self.y_offset = (self.screen_height - new_height) // 2
        else:
            self.y_offset = 0
        
        # Update instance variables
        self.frames_left = frames_left
        self.frames_right = frames_right
        self.durations_left = durations_left
        self.durations_right = durations_right
        self.width_left = width_left
        self.frame_indices = [0, 0]
        self.last_frame_times = [pygame.time.get_ticks(), pygame.time.get_ticks()]
        
        print(f"Loaded: {len(frames_left)} frames (left), {len(frames_right)} frames (right)")
        
    def switch_monitor(self):
        """Switch to the next available monitor."""
        num_displays = pygame.display.get_num_displays()
        self.monitor_index = (self.monitor_index + 1) % num_displays
        print(f"Switching to monitor {self.monitor_index}")
        self.setup_display()
        
        # Reload GIFs for new screen dimensions
        if self.frames_left or self.frames_right:
            self.reload_gifs()
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        self.fullscreen = not self.fullscreen
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags)
        print(f"Fullscreen: {'ON' if self.fullscreen else 'OFF'}")
    
    def update_animation(self):
        """Update animation frame indices based on timing."""
        if self.animation_paused or not self.frames_left or not self.frames_right:
            return
            
        current_time = pygame.time.get_ticks()
        
        # Update left GIF
        if current_time - self.last_frame_times[0] >= self.durations_left[self.frame_indices[0]]:
            self.frame_indices[0] = (self.frame_indices[0] + 1) % len(self.frames_left)
            self.last_frame_times[0] = current_time
            
        # Update right GIF
        if current_time - self.last_frame_times[1] >= self.durations_right[self.frame_indices[1]]:
            self.frame_indices[1] = (self.frame_indices[1] + 1) % len(self.frames_right)
            self.last_frame_times[1] = current_time
    
    def draw(self):
        """Draw the current frame of both GIFs."""
        self.screen.fill((0, 0, 0))  # Black background
        
        if self.frames_left and self.frames_right:
            # Draw left GIF
            self.screen.blit(self.frames_left[self.frame_indices[0]], (0, self.y_offset))
            
            # Draw right GIF
            self.screen.blit(self.frames_right[self.frame_indices[1]], (self.width_left, self.y_offset))
        
        # Draw status indicators
        self.draw_status()
        
        pygame.display.flip()
    
    def draw_status(self):
        """Draw status information overlay."""
        font = pygame.font.Font(None, 36)
        y_pos = 10
        
        # Status messages
        status_lines = []
        
        if self.monitoring_paused:
            status_lines.append("FILE MONITORING: PAUSED")
        
        if self.animation_paused:
            status_lines.append("ANIMATION: PAUSED")
            
        if self.reload_pending:
            status_lines.append("RELOAD PENDING...")
        
        # Draw status text
        for line in status_lines:
            text_surface = font.render(line, True, (255, 255, 0))  # Yellow text
            text_rect = text_surface.get_rect()
            text_rect.topleft = (10, y_pos)
            
            # Draw background rectangle
            bg_rect = text_rect.inflate(10, 5)
            pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
            
            self.screen.blit(text_surface, text_rect)
            y_pos += 40
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    self.running = False
                    
                elif event.key == pygame.K_r:
                    # Manual reload
                    with self.reload_lock:
                        self.reload_pending = True
                    print("Manual reload requested")
                    
                elif event.key == pygame.K_m:
                    self.switch_monitor()
                    
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()
                    
                elif event.key == pygame.K_p:
                    # Toggle file monitoring
                    self.monitoring_paused = not self.monitoring_paused
                    status = "PAUSED" if self.monitoring_paused else "ACTIVE"
                    print(f"File monitoring: {status}")
                    
                elif event.key == pygame.K_SPACE:
                    # Toggle animation
                    self.animation_paused = not self.animation_paused
                    status = "PAUSED" if self.animation_paused else "PLAYING"
                    print(f"Animation: {status}")
    
    def run(self):
        """Main run loop."""
        clock = pygame.time.Clock()
        
        # Initial load
        self.reload_gifs()
        
        print("\n" + "="*60)
        print("PERSISTENT GIF DISPLAY - RUNNING")
        print("="*60)
        print("Controls:")
        print("  ESC/Q: Quit")
        print("  R: Manual reload GIFs")
        print("  M: Switch monitor")
        print("  F: Toggle fullscreen")
        print("  P: Pause/resume file monitoring")
        print("  SPACE: Pause/resume animation")
        print("\nAutomatic file monitoring is ACTIVE")
        print("="*60)
        
        try:
            while self.running:
                # Handle pending reloads
                with self.reload_lock:
                    if self.reload_pending:
                        self.reload_gifs()
                        self.reload_pending = False
                
                # Handle events
                self.handle_events()
                
                # Update animation
                self.update_animation()
                
                # Draw everything
                self.draw()
                
                # Limit FPS
                clock.tick(60)
                
        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            
        pygame.quit()
        print("Shutdown complete")

def main():
    """Main entry point."""
    # Define paths relative to script location
    script_dir = Path(__file__).resolve().parent
    output_folder = script_dir / "output"
    
    gif_left = output_folder / "particleplot.gif"
    gif_right = output_folder / "output.gif"
    
    # Default to secondary monitor if available
    import pygame
    pygame.init()
    num_displays = pygame.display.get_num_displays()
    default_monitor = 1 if num_displays > 1 else 0
    pygame.quit()
    
    print("Starting persistent GIF display...")
    print(f"Left GIF: {gif_left}")
    print(f"Right GIF: {gif_right}")
    print(f"Target monitor: {default_monitor}")
    
    # Create and run display
    display = PersistentGifDisplay(
        gif_path_left=gif_left,
        gif_path_right=gif_right,
        monitor_index=default_monitor
    )
    
    display.run()

if __name__ == "__main__":
    main()
