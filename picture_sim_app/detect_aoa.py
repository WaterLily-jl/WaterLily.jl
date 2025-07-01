import logging

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def calculate_aoa_from_markers(
        image_path: str,
        marker_color_rgb: tuple=(255, 0, 0),
        tolerance: int=20,
        show_processed_image: bool=True,
) ->tuple[float, list]:
    """
    Detects two marker dots of a specified color in an image, calculates their
    mean positions, and then computes the angle of the line connecting them.

    Args:
        image_path (str): Path to the input image.
        marker_color_rgb (tuple): The RGB color of the markers (e.g., (255, 0, 0) for red).
        tolerance (int): Tolerance for color detection (how much deviation from
                         the marker_color_rgb is allowed).
        show_processed_image(bool): If True, shows the estimated marker centroids and line between them.

    Returns:
        tuple: (angle_deg, marker_coords)
               - angle_deg (float): The calculated angle in degrees. None if not found.
               - marker_coords (list): A list of (x, y) tuples for the detected markers.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert RGB marker color to BGR for OpenCV
    marker_color_bgr = (marker_color_rgb[2], marker_color_rgb[1], marker_color_rgb[0])

    # Define color range (in BGR)
    # Convert the marker_color_bgr to a NumPy array for easier subtraction/addition
    color_np = np.array(marker_color_bgr)

    lower_bound = np.maximum(0, color_np - tolerance)
    upper_bound = np.minimum(255, color_np + tolerance)

    logging.info(f"Detecting color in BGR range: Lower {lower_bound}, Upper {upper_bound}")

    # Create mask for the specified color
    mask = cv2.inRange(img, lower_bound, upper_bound)

    # Find contours in the mask
    # RETR_EXTERNAL for outer contours, CHAIN_APPROX_SIMPLE for compressed points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marker_coords = []
    processed_img = img.copy()

    if len(contours) < 2:
        raise ValueError(f"Only {len(contours)} markers found. Need at least 2.")

    # Sort contours by area in descending order and take the largest two
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Find the centroid of each marker
    for i, contour in enumerate(contours):
        # Calculate the centroid (mean x, y) of each detected marker
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            marker_coords.append((cX, cY))
            # Draw the centroid on the image
            cv2.circle(processed_img, (cX, cY), 3, marker_color_rgb, -1)
        else:
            raise ValueError(f"Moment for contour {i} is zero (area of marker registered as zero).")

    if len(marker_coords) == 2:
        p1 = marker_coords[0]
        p2 = marker_coords[1]

        # Ensure consistent order for angle calculation (e.g., left-most as p1)
        if p1[0] > p2[0]: # If p1 is to the right of p2, swap them
            p1, p2 = p2, p1

        # Calculate dx and dy (distance between marker centroids)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Calculate angle between marker centroids
        aoa_rad = math.atan2(dy, dx)
        angle_of_attack = math.degrees(aoa_rad)

        # Draw the line connecting the markers
        cv2.line(processed_img, p1, p2, color=marker_color_rgb, thickness=2) # Green line

        if show_processed_image:
            plot_processed_aoa_markers(processed_img, angle_of_attack)

        return angle_of_attack, marker_coords

    else:
        raise ValueError("Could not find two distinct marker points.")


def plot_processed_aoa_markers(processed_marker_image: np.ndarray, angle_of_attack: float = None) -> None:
    """Displays the processed markers."""
    fig, ax = plt.subplots()
    ax.imshow(processed_marker_image)
    if angle_of_attack is not None:
        ax.set_title(f"Angle of attack {angle_of_attack:.2f} deg")
    else:
        ax.set_title(f"Processed markers (AoA estimation failed)")
    ax.axis('off')

    # plt.tight_layout()
    plt.show()


def main() -> None:

    image_file = "input/airfoil_30_deg_markers.png"
    marker_color_to_detect = (255, 0, 0)
    color_detection_tolerance = 30 # Adjust this if marker color varies

    angle, marker_coords = calculate_aoa_from_markers(
        image_file,
        marker_color_rgb=marker_color_to_detect,
        tolerance=color_detection_tolerance,
        show_processed_image=True,
    )

    print(f"Calculated Angle of Attack: {angle:.2f} degrees")

if __name__ == "__main__":
    main()
