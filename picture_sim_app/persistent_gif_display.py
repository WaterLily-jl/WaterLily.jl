import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional

import yaml  # <-- added

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
CONFIG_PATH = SCRIPT_DIR / "configs" / "settings.yaml"  # <-- added
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
    def __init__(self, monitor_index: int = 1, start_mode: str = "both"):
        pygame.init()
        self.monitor_index = monitor_index
        self.fullscreen = False
        self._init_window()
        # mode state
        self.mode = start_mode  # "both" | "particle_plot" | "heatmap"
        # gif containers
        self.left: Optional[LoadedGif] = None
        self.right: Optional[LoadedGif] = None
        self.single_mode = False
        # animation state
        self.idx_left = 0
        self.idx_right = 0
        now = pygame.time.get_ticks()
        self.next_left = now
        self.next_right = now
        # reload flag
        self.reload_lock = threading.Lock()
        self.reload_requested = False
        # watcher handles
        self.observer = None
        self.poller = None
        # placeholders
        self.placeholder = self._make_placeholder_surface()
        # initial load
        self.load_and_pack()

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

    def _make_placeholder_surface(self):
        surf = pygame.Surface((400, 300))
        surf.fill((0, 0, 0))
        font = pygame.font.Font(None, 32)
        txt = font.render("GIF missing", True, (180, 0, 0))
        r = txt.get_rect(center=surf.get_rect().center)
        surf.blit(txt, r)
        return surf

    def set_mode(self, mode: str):
        if mode not in ("both", "particle_plot", "heatmap"):
            print(f"Warning: invalid mode '{mode}' ignored.")
            return
        if mode == self.mode:
            return
        print(f"Switching mode to {mode}")
        self.mode = mode
        self._stop_watchers()
        self.load_and_pack()
        self._start_watchers()

    def _active_paths(self):
        if self.mode == "both":
            return [GIF_LEFT, GIF_RIGHT]
        if self.mode == "particle_plot":
            return [GIF_LEFT]
        return [GIF_RIGHT]

    def _start_watchers(self):
        paths = self._active_paths()
        names = [p.name for p in paths]
        def changed(_):
            self.request_reload()
        if HAVE_WATCHDOG:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            class Handler(FileSystemEventHandler):
                def on_any_event(self_inner, event):
                    try:
                        nm = Path(event.src_path).name
                        if nm in names:
                            changed(nm)
                    except Exception:
                        pass
            self.observer = Observer()
            self.observer.schedule(Handler(), str(OUTPUT_DIR), recursive=False)
            self.observer.start()
        else:
            self.poller = ChangeWatcher(paths, lambda _: changed(None))

    def _stop_watchers(self):
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=1)
            self.observer = None
        if self.poller:
            self.poller.stop()
            self.poller = None

    def shutdown(self):
        self._stop_watchers()

    def load_and_pack(self):
        """Load according to current mode."""
        base_left = None
        base_right = None
        if self.mode == "both":
            base_left = load_gif(GIF_LEFT)
            base_right = load_gif(GIF_RIGHT)
        elif self.mode == "particle_plot":
            base_left = load_gif(GIF_LEFT)
        elif self.mode == "heatmap":
            base_left = load_gif(GIF_RIGHT)  # treat selected gif as 'left' for centering

        if self.mode == "both" and base_left and base_right:
            target_h = self.screen_h
            l_scaled = scale_gif(base_left, target_h)
            r_scaled = scale_gif(base_right, target_h)
            total_w = l_scaled.width + r_scaled.width
            if total_w > self.screen_w:
                scale_factor = self.screen_w / total_w
                target_h = max(1, int(self.screen_h * scale_factor))
                l_scaled = scale_gif(base_left, target_h)
                r_scaled = scale_gif(base_right, target_h)
            self.left = l_scaled
            self.right = r_scaled
            self.single_mode = False
        else:
            # single mode (either explicit single selection or fallback)
            chosen = base_left
            if chosen:
                scaled = scale_gif(chosen, self.screen_h)
                self.left = scaled
            else:
                self.left = None
            self.right = None
            self.single_mode = True
            if not chosen:
                print(f"Warning: required GIF for mode '{self.mode}' missing.")

        self.idx_left = self.idx_right = 0
        now = pygame.time.get_ticks()
        self.next_left = now
        self.next_right = now

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
        self.screen.fill((0, 0, 0))
        if not self.left:
            # draw placeholder for single mode missing asset
            if self.single_mode:
                ps = self.placeholder
                self.screen.blit(ps, ps.get_rect(center=self.screen.get_rect().center))
            pygame.display.flip()
            return
        if self.single_mode:
            surf = self.left.frames[self.idx_left]
            x = (self.screen_w - surf.get_width()) // 2
            y = (self.screen_h - surf.get_height()) // 2
            self.screen.blit(surf, (x, y))
        else:
            left_surf = self.left.frames[self.idx_left]
            right_surf = self.right.frames[self.idx_right] if self.right else self.placeholder
            y_l = (self.screen_h - left_surf.get_height()) // 2
            y_r = (self.screen_h - right_surf.get_height()) // 2
            self.screen.blit(left_surf, (0, y_l))
            self.screen.blit(right_surf, (self.left.width, y_r))
        pygame.display.flip()

# -------------------- Main Loop -------------------- #

def run(monitor_index: int = 1, start_mode: str = "both"):
    display = DualGifDisplay(monitor_index=monitor_index, start_mode=start_mode)
    display._start_watchers()

    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            if display._consume_reload_flag():
                time.sleep(0.1)
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
                    elif event.key == pygame.K_r:
                        print("Manual reload requested")
                        display.load_and_pack()
                    elif event.key == pygame.K_1:
                        display.set_mode("both")
                    elif event.key == pygame.K_2:
                        display.set_mode("particle_plot")
                    elif event.key == pygame.K_3:
                        display.set_mode("heatmap")
            display.update()
            display.draw()
            clock.tick(60)
    finally:
        display.shutdown()
        pygame.quit()

if __name__ == "__main__":
    mon = 1
    if len(sys.argv) > 1:
        try:
            mon = int(sys.argv[1])
        except ValueError:
            pass
    # Load settings.yaml for initial mode
    start_mode = "both"
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
            gm = cfg.get("simulation_settings", {}).get("gif_display", "both")
            if gm in ("both", "particle_plot", "heatmap"):
                start_mode = gm
            else:
                print(f"Warning: gif_display '{gm}' invalid, defaulting to 'both'")
    except Exception as e:
        print(f"Warning: could not read settings.yaml ({e}), defaulting to 'both'")
    run(mon, start_mode=start_mode)
