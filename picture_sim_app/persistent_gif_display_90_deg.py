import sys
import os
import json
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QDesktopWidget
from PyQt5.QtGui import QMovie, QFont, QTransform
from PyQt5.QtCore import QTimer, Qt

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
CONFIG_PATH = SCRIPT_DIR / "configs" / "settings.yaml"
AIRFOIL_DATA_PATH = OUTPUT_DIR / "airfoil_data.json"
GIF_LEFT = OUTPUT_DIR / "particleplot.gif"
GIF_RIGHT = OUTPUT_DIR / "heatmap_pressure.gif"

AIRFOIL_NAME_MAPPING = {
    "naca_002": "NACA 0002",
    "naca_015": "NACA 0015",
    "naca_030": "NACA 0030",
}


class GifLabel(QLabel):
    def __init__(self, path):
        super().__init__()
        self.symlink_path = path
        self.real_target = None
        self.setAlignment(Qt.AlignCenter)  # center the GIF
        self.setScaledContents(False)  # Don't scale contents, we'll handle it manually
        self.rotation_angle = 90  # 90 degrees clockwise rotation
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
        """Scale movie preserving aspect ratio with rotation"""
        if hasattr(self, 'movie') and self.movie:
            # Get original size
            original_size = self.movie.frameRect().size()
            if original_size.isEmpty():
                return

            # For 90-degree rotation, swap width and height for scaling calculation
            original_width = original_size.width()
            original_height = original_size.height()

            # After 90-degree rotation, width becomes height and vice versa
            rotated_width = original_height
            rotated_height = original_width

            # Get available space (leave some margin for safety)
            available_size = self.size()
            margin = 10
            available_width = max(1, available_size.width() - margin)
            available_height = max(1, available_size.height() - margin)

            # Calculate scaling factor based on rotated dimensions
            scale_x = available_width / rotated_width
            scale_y = available_height / rotated_height
            scale_factor = min(scale_x, scale_y)  # Use smaller scale to fit entirely

            # Calculate new size (apply to original dimensions, rotation happens during display)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            from PyQt5.QtCore import QSize
            self.movie.setScaledSize(QSize(new_width, new_height))

    def paintEvent(self, event):
        """Override paint event to apply rotation"""
        if hasattr(self, 'movie') and self.movie and self.movie.currentPixmap():
            from PyQt5.QtGui import QPainter
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)

            # Get the current frame
            pixmap = self.movie.currentPixmap()

            # Calculate center point for rotation
            center_x = self.width() // 2
            center_y = self.height() // 2

            # Set up transformation
            transform = QTransform()
            transform.translate(center_x, center_y)
            transform.rotate(self.rotation_angle)
            transform.translate(-pixmap.width() // 2, -pixmap.height() // 2)

            painter.setTransform(transform)
            painter.drawPixmap(0, 0, pixmap)
        else:
            # Fallback to default painting if no movie
            super().paintEvent(event)

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

        # Main horizontal layout for GIFs (no title space taken)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        self.labels = []
        for p in paths:
            label = GifLabel(p)
            # Set minimum size to ensure visibility
            label.setMinimumSize(200, 150)
            main_layout.addWidget(label, 1)  # Equal stretch for both labels
            self.labels.append(label)

        # Create title label as overlay (no layout, positioned manually)
        self.title_label = QLabel("Loading...", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: black; padding: 5px;")  # Black text, no background
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents)  # Allow mouse events to pass through

        # Update title with airfoil data
        self.update_title()

        # poll symlinks every 2s
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_symlinks)
        self.timer.start(2000)

    def load_airfoil_data(self):
        """Load airfoil data from JSON file"""
        try:
            if AIRFOIL_DATA_PATH.exists():
                with open(AIRFOIL_DATA_PATH, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading airfoil data: {e}")
        return None

    def update_title(self):
        """Update title with current airfoil data"""
        airfoil_data = self.load_airfoil_data()
        if airfoil_data:
            airfoil_type = airfoil_data.get('airfoil_type', 'unknown')
            aoa = airfoil_data.get('aoa', 0)
            airfoil_name = AIRFOIL_NAME_MAPPING[airfoil_type]
            title_text = f"Airfoil={airfoil_name}, \t Angle of attack={aoa}°"
        else:
            title_text = "No airfoil data available"

        self.title_label.setText(title_text)

    def resizeEvent(self, event):
        """Position the title overlay when window is resized"""
        super().resizeEvent(event)
        # Position title at top center with wider width to prevent cropping
        title_width = min(800, self.width() - 20)  # Use most of screen width, max 800px
        title_height = 40  # Fixed height for title
        x = (self.width() - title_width) // 2
        y = 10  # 10px from top
        self.title_label.setGeometry(x, y, title_width, title_height)

    def poll_symlinks(self):
        for label in self.labels:
            label.check_update()
        # Also update title in case airfoil data changed
        self.update_title()

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

    # replace with your symlink paths
    symlinks = [GIF_LEFT.__str__(), GIF_RIGHT.__str__()]

    w = GifDisplay(symlinks, monitor_index)
    w.setWindowTitle("Two GIFs side by side")
    w.show_on_monitor()

    sys.exit(app.exec_())
