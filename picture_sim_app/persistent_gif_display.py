import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAVE_WATCHDOG = True
except ImportError:
    HAVE_WATCHDOG = False

import pygame
from PIL import Image, ImageSequence

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
GIF_LEFT = OUTPUT_DIR / "particleplot.gif"
GIF_RIGHT = OUTPUT_DIR / "heatmap_plot.gif"

# -------------------- GIF Loading -------------------- #

class LoadedGif:
    def __init__(self, frames: List[pygame.Surface], durations: List[int], width: int, height: int):
        self.frames = frames
        self.durations = durations
        self.width = width
        self.height = height

def load_gif(path: Path) -> Optional[LoadedGif]:
    if not path.exists():
        print(f"Warning: missing {path.name}")
        return None
    try:
        with Image.open(path) as im:
            frames = []
            durations = []
            for frame in ImageSequence.Iterator(im):
                frm = frame.convert("RGBA")
                surf = pygame.image.fromstring(frm.tobytes(), frm.size, "RGBA")
                frames.append(surf)
                durations.append(frame.info.get("duration", 100))
            if not frames:
                print(f"Warning: no frames in {path.name}")
                return None
            w = frames[0].get_width()
            h = frames[0].get_height()
            return LoadedGif(frames, durations, w, h)
    except Exception as e:
        print(f"Warning: failed to load {path.name}: {e}")
        return None

def scale_gif(g: LoadedGif, target_height: int) -> LoadedGif:
    if g is None:
        return None
    if g.height == target_height:
        return g
    ratio = target_height / g.height
    new_w = max(1, int(g.width * ratio))
    frames_scaled = [pygame.transform.smoothscale(f, (new_w, target_height)) for f in g.frames]
    return LoadedGif(frames_scaled, g.durations, new_w, target_height)

# -------------------- File Change Detection -------------------- #

def file_signature(path: Path):
    try:
        st_link = path.lstat()  # metadata of symlink itself (if symlink)
        st_target = path.stat()  # target metadata (follow)
        return (st_link.st_mtime_ns, st_link.st_ino, st_target.st_mtime_ns, st_target.st_ino, st_target.st_size)
    except FileNotFoundError:
        return None

class ChangeWatcher:
    def __init__(self, paths: List[Path], callback, poll_interval=0.5):
        self.paths = paths
        self.callback = callback
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._signatures = {p: file_signature(p) for p in paths}

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)

    def _loop(self):
        while not self._stop.is_set():
            for p in self.paths:
                sig = file_signature(p)
                if sig != self._signatures.get(p):
                    self._signatures[p] = sig
                    self.callback(p)
            time.sleep(self.poll_interval)

if HAVE_WATCHDOG:
    class DirHandler(FileSystemEventHandler):
        def __init__(self, watch_files: List[str], callback):
            self.watch_files = set(watch_files)
            self.callback = callback
        def on_any_event(self, event):
            try:
                name = Path(event.src_path).name
            except Exception:
                return
            if name in self.watch_files:
                self.callback(Path(event.src_path))

# -------------------- Display Manager -------------------- #

class DualGifDisplay:
    def __init__(self, monitor_index: int = 1):
        pygame.init()
        self.monitor_index = monitor_index
        self.fullscreen = False
        self._init_window()
        # state
        self.left: Optional[LoadedGif] = None
        self.right: Optional[LoadedGif] = None
        self.single_mode = False
        self.idx_left = 0
        self.idx_right = 0
        now = pygame.time.get_ticks()
        self.next_left = now
        self.next_right = now
        self.reload_lock = threading.Lock()
        self.reload_requested = False

    def _init_window(self):
        num_displays = pygame.display.get_num_displays()
        desktop_sizes = (pygame.display.get_desktop_sizes()
                         if hasattr(pygame.display, "get_desktop_sizes")
                         else [(1920,1080)] * num_displays)
        if self.monitor_index >= num_displays:
            self.monitor_index = 0
        self.desktops = desktop_sizes
        x_offset = sum(desktop_sizes[i][0] for i in range(self.monitor_index))
        self.screen_w, self.screen_h = desktop_sizes[self.monitor_index]
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x_offset},0"
        flags = 0 if not self.fullscreen else pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), flags)
        pygame.display.set_caption("Dual GIF Display")

    def cycle_monitor(self):
        self.monitor_index = (self.monitor_index + 1) % pygame.display.get_num_displays()
        self._init_window()
        # Force re-scale
        self._rescale()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self._init_window()
        self._rescale()

    def request_reload(self):
        with self.reload_lock:
            self.reload_requested = True

    def _consume_reload_flag(self):
        with self.reload_lock:
            flag = self.reload_requested
            self.reload_requested = False
        return flag

    def load_and_pack(self):
        base_left = load_gif(GIF_LEFT)
        base_right = load_gif(GIF_RIGHT)
        if base_left and base_right:
            # Try full height first
            target_h = self.screen_h
            left_scaled = scale_gif(base_left, target_h)
            right_scaled = scale_gif(base_right, target_h)
            total_w = left_scaled.width + right_scaled.width
            if total_w > self.screen_w:
                scale_factor = self.screen_w / total_w
                target_h = max(1, int(self.screen_h * scale_factor))
                left_scaled = scale_gif(base_left, target_h)
                right_scaled = scale_gif(base_right, target_h)
            self.left = left_scaled
            self.right = right_scaled
            self.single_mode = False
        elif base_left or base_right:
            # Single mode
            chosen = base_left if base_left else base_right
            scaled = scale_gif(chosen, self.screen_h)
            self.left = scaled
            self.right = None
            self.single_mode = True
            if base_left and not base_right:
                print("Warning: only particleplot.gif available; displaying single GIF.")
            elif base_right and not base_left:
                print("Warning: only heatmap_plot.gif available; displaying single GIF.")
        else:
            self.left = None
            self.right = None
            self.single_mode = True
            print("Warning: neither GIF could be loaded.")

        self.idx_left = self.idx_right = 0
        now = pygame.time.get_ticks()
        self.next_left = now
        self.next_right = now

    def _rescale(self):
        # Rescale current raw frames (reload fresh to avoid quality loss)
        self.load_and_pack()

    def update(self):
        # Animation stepping
        now = pygame.time.get_ticks()
        if self.left and now >= self.next_left:
            dur = self.left.durations[self.idx_left]
            self.idx_left = (self.idx_left + 1) % len(self.left.frames)
            self.next_left = now + (dur if dur > 0 else 100)
        if self.right and not self.single_mode and now >= self.next_right:
            dur = self.right.durations[self.idx_right]
            self.idx_right = (self.idx_right + 1) % len(self.right.frames)
            self.next_right = now + (dur if dur > 0 else 100)

    def draw(self):
        self.screen.fill((0,0,0))
        if not self.left:
            pygame.display.flip()
            return
        if self.single_mode:
            # center horizontally
            surf = self.left.frames[self.idx_left]
            x = (self.screen_w - surf.get_width()) // 2
            y = (self.screen_h - surf.get_height()) // 2
            self.screen.blit(surf, (x,y))
        else:
            left_surf = self.left.frames[self.idx_left]
            right_surf = self.right.frames[self.idx_right]
            # left at x=0
            self.screen.blit(left_surf, (0, (self.screen_h - left_surf.get_height()) // 2))
            # right immediately next
            self.screen.blit(right_surf, (self.left.width, (self.screen_h - right_surf.get_height()) // 2))
        pygame.display.flip()

# -------------------- Main Loop -------------------- #

def run(monitor_index: int = 1):
    display = DualGifDisplay(monitor_index=monitor_index)
    display.load_and_pack()

    # Setup watchers
    def changed(_):
        display.request_reload()

    if HAVE_WATCHDOG:
        observer = Observer()
        handler = DirHandler([GIF_LEFT.name, GIF_RIGHT.name], changed)
        observer.schedule(handler, str(OUTPUT_DIR), recursive=False)
        observer.start()
        poller = None
    else:
        observer = None
        poller = ChangeWatcher([GIF_LEFT, GIF_RIGHT], changed)
        poller.start()

    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            # Process reloads
            if display._consume_reload_flag():
                # Small debounce to allow both symlinks to settle
                time.sleep(0.15)
                display.load_and_pack()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE,):
                        running = False
                    elif event.key == pygame.K_f:
                        display.toggle_fullscreen()
                    elif event.key == pygame.K_m:
                        display.cycle_monitor()

            display.update()
            display.draw()
            clock.tick(60)
    finally:
        if observer:
            observer.stop()
            observer.join(timeout=1)
        if poller:
            poller.stop()
        pygame.quit()

if __name__ == "__main__":
    # Optional monitor index from argv
    mon = 1
    if len(sys.argv) > 1:
        try:
            mon = int(sys.argv[1])
        except ValueError:
            pass
    run(mon)
