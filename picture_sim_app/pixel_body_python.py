import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class PixelBodyMask:
    """
    Creates boolean masks that distinguish solid from fluid using image recognition logic.
    Supports both grayscale and colored images.
    Mask logic: 1=Fluid, 0=Solid
    """
    
    def __init__(self, image_path, threshold=0.5, diff_threshold=None, max_image_res=None, 
                 body_color="gray", manual_mode=False, force_invert_mask=False, verbose=True):
        """
        Create a PixelBody mask from an image.
        
        Args:
            image_path (str): Path to the input image
            threshold (float): Minimum intensity for solid detection (0.0-1.0)
            diff_threshold (float): Color channel difference threshold
            max_image_res (int): Maximum resolution limit for image resizing
            body_color (str): Color of the solid body ("gray", "red", "green", "blue")
            manual_mode (bool): Use manual thresholds instead of smart detection
            force_invert_mask (bool): Force inversion of the mask logic
        """
        self.image_path = image_path
        self.threshold = threshold
        self.diff_threshold = diff_threshold
        self.max_image_res = max_image_res
        self.body_color = body_color
        self.manual_mode = manual_mode
        self.force_invert_mask = force_invert_mask

        # Verbose mode (to disable some print statements)
        self.verbose = verbose
        
        # Load and process the image
        self.img = self._load_image()
        self.mask = self._create_mask()
    
    def _load_image(self):
        """Load image and apply resolution limit if specified."""
        img = Image.open(self.image_path)
        if self.verbose:
            print(f"Original image size: {img.size}")
        
        if self.max_image_res is not None:
            img = self._limit_resolution(img, self.max_image_res)
            if self.verbose:
                print(f"Image resized to: {img.size}")
        
        return img
    
    def _limit_resolution(self, img, max_grid):
        """Resize image if dimensions exceed the specified limit."""
        w, h = img.size
        max_dim = max(h, w)
        
        if max_dim <= max_grid:
            return img
        
        scale = max_grid / max_dim
        new_w = round(w * scale)
        new_h = round(h * scale)

        if self.verbose:
            print(f"Resizing from {w}x{h} → {new_w}x{new_h}")
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    def _create_mask(self):
        """Create fluid-solid mask using image recognition logic."""
        valid_colors = ["gray", "red", "green", "blue"]
        if self.body_color not in valid_colors:
            raise ValueError(f"Unsupported solid color: {self.body_color}. "
                           f"Supported colors are: {', '.join(valid_colors)}")
        
        if self.body_color == "gray":
            mask = self._process_grayscale()
        else:
            mask = self._process_color()

        # Invert logic to match the following convention: 1=Fluid, 0=Solid
        mask = ~mask

        return mask
    
    def _process_grayscale(self):
        """Process grayscale images."""
        # Convert to grayscale and normalize to [0,1]
        gray_img = self.img.convert('L')
        gray_array = np.array(gray_img, dtype=np.float64) / 255.0
        
        # Binary mask: assumes solid is dark (low values) and fluid is bright (high values)
        # Higher threshold = more restrictive = less solid detected
        mask = gray_array < (1.0 - self.threshold)
        
        # Transpose to align matrix indices with physical x-y
        mask = np.flipud(mask).T
        return mask
    
    def _process_color(self):
        """Process color images with smart body detection."""
        print("Colored figure selected. Smart body detection will be used.")
        
        # Convert to RGB and extract channels
        img_rgb = self.img.convert('RGB')
        img_array = np.array(img_rgb, dtype=np.float64) / 255.0
        R = img_array[:, :, 0]
        G = img_array[:, :, 1]
        B = img_array[:, :, 2]
        
        # Smart body detection uses channel hierarchy
        R_mean = np.mean(R)
        G_mean = np.mean(G)
        B_mean = np.mean(B)
        
        print(f"Channel means: R={R_mean:.3f}, G={G_mean:.3f}, B={B_mean:.3f}")
        print(f"Manual mode: {self.manual_mode}, Force invert mask: {self.force_invert_mask}")
        
        # Analyze channel hierarchy
        R_vs_G_diff = R_mean - G_mean
        R_vs_B_diff = R_mean - B_mean
        total_color_range = (np.max([np.max(R), np.max(G), np.max(B)]) - 
                           np.min([np.min(R), np.min(G), np.min(B)]))
        
        needs_inversion = False
        threshold = self.threshold
        diff_threshold = self.diff_threshold
        
        if self.manual_mode:
            # Manual mode: use provided thresholds but smart inversion logic
            print("MANUAL MODE: Using provided threshold values")
            if R_mean > G_mean and R_mean > B_mean:
                print("RED DOMINANT camera detected (mask inversion applied)")
                needs_inversion = True
            elif G_mean > R_mean and B_mean > R_mean:
                print("RED SUPPRESSED camera detected (mask inversion not applied)")
                needs_inversion = False
            else:
                print("BALANCED channels detected (mask inversion not applied)")
                needs_inversion = False
        else:
            # Smart mode: auto-adjust thresholds and determine inversion
            if R_mean > G_mean and R_mean > B_mean:
                print("RED DOMINANT camera detected")
                red_dominance = min(R_vs_G_diff, R_vs_B_diff) / total_color_range
                print(f"Red dominance factor: {red_dominance:.3f}")
                
                if red_dominance > 0.05:
                    threshold = 0.5 + red_dominance
                    diff_threshold = 0.05 + red_dominance * 0.5
                else:
                    threshold = 0.45
                    diff_threshold = 0.15
                needs_inversion = True
                
            elif G_mean > R_mean and B_mean > R_mean:
                print("RED SUPPRESSED camera detected")
                red_suppression = max(G_mean - R_mean, B_mean - R_mean) / total_color_range
                print(f"Red suppression factor: {red_suppression:.3f}")
                
                threshold = 0.35 + red_suppression * 0.2
                diff_threshold = 0.15 + red_suppression * 0.3
                needs_inversion = False
                
            else:
                print("BALANCED channels detected")
                if total_color_range > 0.5:
                    threshold = 0.4
                    diff_threshold = 0.2
                else:
                    threshold = 0.3
                    diff_threshold = 0.1
                needs_inversion = False
            
            # Ensure thresholds are within reasonable bounds
            threshold = np.clip(threshold, 0.2, 0.7)
            diff_threshold = np.clip(diff_threshold, 0.05, 0.4)
        
        # Force invert override
        if self.force_invert_mask:
            print("FORCE INVERT: Overriding smart inversion logic")
            needs_inversion = not needs_inversion
        
        print("FINAL THRESHOLDS:")
        print(f"   threshold = {threshold:.3f}")
        print(f"   diff_threshold = {diff_threshold:.3f}")
        print(f"   needs_inversion = {needs_inversion}")
        print(f"   manual_mode = {self.manual_mode}")
        print(f"   force_invert_mask = {self.force_invert_mask}")
        print("=" * 50)
        
        # Apply color detection logic
        if self.body_color == "red":
            red_detected = ((R > threshold) & 
                          ((R - G) > diff_threshold) & 
                          ((R - B) > diff_threshold))
            
            if needs_inversion:
                mask = red_detected
                print("Applied mask = red_detected (no inversion)")
            else:
                mask = ~red_detected
                print("Applied mask = ~red_detected (inverted)")
                
        elif self.body_color == "green":
            green_detected = ((G > threshold) & 
                            ((G - R) > diff_threshold) & 
                            ((G - B) > diff_threshold))
            mask = ~green_detected
            
        elif self.body_color == "blue":
            blue_detected = ((B > threshold) & 
                           ((B - G) > diff_threshold) & 
                           ((B - R) > diff_threshold))
            mask = ~blue_detected
        
        # Transpose to align matrix indices with physical x-y
        mask = np.flipud(mask).T

        return mask


    def rotate_mask(
        self, current_angle: float,
        target_angle: float = 0.0,
        replace_mask: bool = True,
    ) -> np.ndarray:
        """
        Rotate solid in domain mask to achieve 0 degrees angle of attack.

        Args:
            current_angle (float): Current angle of attack in degrees
            target_angle (float): Target angle of attack in degrees (angle after rotation)
            replace_mask (bool): If True, replace the current mask with the rotated one in the PixelBodyMask instance

        Returns:
            numpy.ndarray: New boolean mask with rotated solid (True=fluid, False=solid)
        """
        mask = self.mask

        # Don't rotate if already at target angle
        if current_angle == target_angle:
            return self.mask

        # Calculate necessary rotation to achieve target angle
        rotation_angle = target_angle - current_angle

        # Rotate the solid in the mask by the calculated angle
        rotated_solid_mask = self._rotate_solid_by_angle(mask, rotation_angle)

        # Only replace the original mask in the instance if requested
        if replace_mask:
            self.mask = rotated_solid_mask

        return rotated_solid_mask

    @staticmethod
    def _rotate_solid_by_angle(mask:np.ndarray[bool], rotation_angle: float) -> np.ndarray:
        """
        Internal function to rotate solid in boolean mask by a specific angle.

        Args:
            mask (numpy.ndarray): Boolean array where True=fluid, False=solid
            rotation_angle (float): Angle to rotate by in degrees (positive = counterclockwise)

        Returns:
            numpy.ndarray: New boolean mask with rotated solid (true=fluid, false=solid)
        """
        # Convert to numpy array
        mask = np.array(mask, dtype=bool)

        # Find solid pixels (where mask is False)
        solid_coords = np.where(~mask)

        if len(solid_coords[0]) == 0:
            return mask  # No solid pixels found, return original mask

        # Extract coordinates: solid_coords gives (row, col) = (y, x) in image terms
        # Convert to standard x, y coordinates
        x_coords = solid_coords[1].astype(np.float64)  # column indices = x
        y_coords = solid_coords[0].astype(np.float64)  # row indices = y

        # Calculate centroid of solid object
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)

        # Convert angle to radians
        angle_rad = np.radians(rotation_angle)

        # Create rotation matrix
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        # Translate coordinates to origin (centroid becomes 0,0)
        x_centered = x_coords - centroid_x
        y_centered = y_coords - centroid_y

        # Apply rotation matrix
        x_rotated = cos_theta * x_centered - sin_theta * y_centered
        y_rotated = sin_theta * x_centered + cos_theta * y_centered

        # Translate back to original position
        x_new = x_rotated + centroid_x
        y_new = y_rotated + centroid_y

        # Round to nearest integer pixel coordinates
        x_new = np.round(x_new).astype(int)
        y_new = np.round(y_new).astype(int)

        # Create new mask filled with fluid (True)
        new_mask = np.ones_like(mask, dtype=bool)

        # Get mask dimensions
        height, width = mask.shape

        # Filter coordinates to stay within bounds
        valid_indices = ((x_new >= 0) & (x_new < width) &
                         (y_new >= 0) & (y_new < height))

        x_valid = x_new[valid_indices]
        y_valid = y_new[valid_indices]

        # Set rotated solid pixels to False
        new_mask[y_valid, x_valid] = False

        return new_mask
    
    def get_mask(self):
        """
        Return the boolean mask.
        """
        return self.mask

    def plot_mask(self) -> None:
        """
        Plots the boolean mask.
        """
        transposed_mask = np.transpose(self.mask) # Need to transpose because mask indexing is structured as:
                                                  # row index = x axis index and column index = y axis index
        plt.imshow(transposed_mask, cmap='gray', origin='lower')
        plt.title("PixelBody Mask")
        plt.show()

        return
    
    def save_mask(self, output_path):
        """Save the mask as a binary image."""
        # Convert boolean mask to 0-255 image (True=255=white, False=0=black)
        mask_img = Image.fromarray((self.mask.T * 255).astype(np.uint8), mode='L')
        mask_img = np.flipud(mask_img)  # Flip back for saving
        mask_img = Image.fromarray(mask_img)
        mask_img.save(output_path)
        print(f"Mask saved to: {output_path}")


def create_pixel_body_mask(image_path, **kwargs):
    """
    Convenience function to create a PixelBody mask from an image.
    
    Args:
        image_path (str): Path to the input image
        **kwargs: Additional arguments passed to PixelBodyMask constructor
    
    Returns:
        numpy.ndarray: Boolean mask (True=solid, False=fluid)
    """
    pixel_body = PixelBodyMask(image_path, **kwargs)
    return pixel_body.get_mask()
