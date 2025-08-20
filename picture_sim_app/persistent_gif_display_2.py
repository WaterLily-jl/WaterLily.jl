import sys
import os
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout, QDesktopWidget
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QTimer, Qt

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
CONFIG_PATH = SCRIPT_DIR / "configs" / "settings.yaml"  # <-- added
GIF_LEFT = OUTPUT_DIR / "particleplot.gif"
GIF_RIGHT = OUTPUT_DIR / "heatmap_pressure.gif"


class GifLabel(QLabel):
    def __init__(self, path):
        super().__init__()
        self.symlink_path = path
        self.real_target = None
        self.setAlignment(Qt.AlignCenter)  # center the GIF
        self.setScaledContents(False)  # Don't scale contents, we'll handle it manually
        self.load_movie()

    def load_movie(self):
        new_target = os.path.realpath(self.symlink_path)  # resolve symlink
        if new_target != self.real_target:
            self.real_target = new_target
            movie = QMovie(self.symlink_path)
            if movie.isValid():
                self.setMovie(movie)
                movie.start()
                self.movie = movie
                self._scale_movie()  # Scale with aspect ratio preservation
                print(f"Loaded GIF: {self.symlink_path} -> {self.real_target}")
            else:
                self.setText(f"Invalid GIF: {self.symlink_path}")
                print(f"Failed to load: {self.symlink_path}")

    def _scale_movie(self):
        """Scale movie preserving aspect ratio"""
        if hasattr(self, 'movie') and self.movie:
            # Get original size
            original_size = self.movie.frameRect().size()
            if original_size.isEmpty():
                return

            # Get available space (leave some margin for safety)
            available_size = self.size()
            margin = 10
            available_width = max(1, available_size.width() - margin)
            available_height = max(1, available_size.height() - margin)

            # Calculate scaling factor to fit while preserving aspect ratio
            scale_x = available_width / original_size.width()
            scale_y = available_height / original_size.height()
            scale_factor = min(scale_x, scale_y)  # Use smaller scale to fit entirely

            # Calculate new size
            new_width = int(original_size.width() * scale_factor)
            new_height = int(original_size.height() * scale_factor)

            from PyQt5.QtCore import QSize
            self.movie.setScaledSize(QSize(new_width, new_height))

    def resizeEvent(self, event):
        """Handle resize events to rescale the movie"""
        super().resizeEvent(event)
        self._scale_movie()

    def check_update(self):
        """Check if symlink target changed, reload if needed"""
        self.load_movie()

class GifDisplay(QWidget):
    def __init__(self, paths, monitor_index=1):
        super().__init__()
        self.monitor_index = monitor_index
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Small margins to prevent edge cropping
        layout.setSpacing(5)  # Small spacing between GIFs
        self.labels = []
        for p in paths:
            label = GifLabel(p)
            # Set minimum size to ensure visibility
            label.setMinimumSize(200, 150)
            layout.addWidget(label, 1)  # Equal stretch for both labels
            self.labels.append(label)

        # poll symlinks every 2s
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_symlinks)
        self.timer.start(2000)

    def poll_symlinks(self):
        for label in self.labels:
            label.check_update()

    def show_on_monitor(self):
        """Show the window on the specified monitor"""
        desktop = QDesktopWidget()
        screen_count = desktop.screenCount()
        
        if self.monitor_index >= screen_count:
            print(f"Monitor {self.monitor_index} not available, using primary monitor")
            self.monitor_index = 0
        
        screen_geometry = desktop.screenGeometry(self.monitor_index)
        print(f"Displaying on monitor {self.monitor_index} ({screen_geometry.width()}x{screen_geometry.height()})")
        
        # Move window to the specified monitor
        self.move(screen_geometry.x(), screen_geometry.y())
        self.resize(screen_geometry.width(), screen_geometry.height())
        self.showFullScreen()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Check command line args for monitor selection
    monitor_index = 1  # Default to secondary monitor
    if len(sys.argv) > 1:
        try:
            monitor_index = int(sys.argv[1])
        except ValueError:
            print("Invalid monitor index, using default (1)")

    # 🔥 replace with your symlink paths
    symlinks = [GIF_LEFT.__str__(), GIF_RIGHT.__str__()]

    w = GifDisplay(symlinks, monitor_index)
    w.setWindowTitle("Two GIFs side by side")
    w.show_on_monitor()

    sys.exit(app.exec_())
