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
        self.setScaledContents(True)  # Scale content to fit label size
        self.load_movie()

    def load_movie(self):
        new_target = os.path.realpath(self.symlink_path)  # resolve symlink
        if new_target != self.real_target:
            self.real_target = new_target
            movie = QMovie(self.symlink_path)
            if movie.isValid():
                movie.setScaledSize(self.size())  # Scale to label size
                self.setMovie(movie)
                movie.start()
                self.movie = movie
                print(f"Loaded GIF: {self.symlink_path} -> {self.real_target}")
            else:
                self.setText(f"Invalid GIF: {self.symlink_path}")
                print(f"Failed to load: {self.symlink_path}")

    def resizeEvent(self, event):
        """Handle resize events to rescale the movie"""
        super().resizeEvent(event)
        if hasattr(self, 'movie') and self.movie:
            self.movie.setScaledSize(self.size())

    def check_update(self):
        """Check if symlink target changed, reload if needed"""
        self.load_movie()

class GifDisplay(QWidget):
    def __init__(self, paths, monitor_index=1):
        super().__init__()
        self.monitor_index = monitor_index
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        layout.setSpacing(0)  # Remove spacing between GIFs
        self.labels = []
        for p in paths:
            label = GifLabel(p)
            layout.addWidget(label)
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
