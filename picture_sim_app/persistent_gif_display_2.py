import sys
import os
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout
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
                print(f"Loaded GIF: {self.symlink_path} -> {self.real_target}")
            else:
                self.setText(f"Invalid GIF: {self.symlink_path}")
                print(f"Failed to load: {self.symlink_path}")

    def check_update(self):
        """Check if symlink target changed, reload if needed"""
        self.load_movie()

class GifDisplay(QWidget):
    def __init__(self, paths):
        super().__init__()
        layout = QHBoxLayout(self)
        self.labels = []
        for p in paths:
            label = GifLabel(p)
            label.setFixedSize(400, 400)  # 👈 make sure it’s visible
            layout.addWidget(label)
            self.labels.append(label)

        # poll symlinks every 2s
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_symlinks)
        self.timer.start(2000)

    def poll_symlinks(self):
        for label in self.labels:
            label.check_update()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 🔥 replace with your symlink paths
    symlinks = [GIF_LEFT.__str__(), GIF_RIGHT.__str__()]

    w = GifDisplay(symlinks)
    w.setWindowTitle("Two GIFs side by side")
    w.resize(900, 450)
    w.show()

    sys.exit(app.exec_())
