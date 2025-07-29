import numpy as np
import matplotlib.pyplot as plt


def characteristic_length_and_aoa_pca(mask, plot_method=False, show_components=True) -> tuple[float, float]:
    """
    Estimate the characteristic length and angle of attack of a PixelBody using Principal Component Analysis.
    
    Finds the direction perpendicular to the principal component of the pixel distribution
    (the direction in which points are most spread) and estimates the angle of attack
    of the object based on the principal axis direction.
    
    Args:
        mask (numpy.ndarray): Boolean mask where False=solid, True=fluid
        plot_method (bool): Whether to display visualization plots
        show_components (bool): Whether to show legend and PCA components in plot
    
    Returns:
        tuple: (characteristic_length, angle_degrees)
            - characteristic_length (float): Estimated characteristic length
            - angle_degrees (float): Signed angle of attack in degrees [-180, 180]
                                   Positive for clockwise, negative for counterclockwise
    """
    # Convert mask to numpy array if needed
    mask = np.array(mask)
    
    # Find solid pixels (where mask is False, value 0)
    solid_coords = np.where(~mask)
    
    if len(solid_coords[0]) == 0:
        raise ValueError("No solid detected when attempting to calculate characteristic length")
    
    # Extract x, y coordinates (note: solid_coords gives (row, col) = (y, x) in image terms)
    xs = solid_coords[0].astype(np.float64)  # row indices = x coordinates
    ys = solid_coords[1].astype(np.float64)  # column indices = y coordinates
    
    # Create points matrix (2 x N)
    pts = np.vstack([xs, ys])
    
    # Calculate mean (centroid)
    mu = np.mean(pts, axis=1, keepdims=True)
    
    # Center the data
    X = pts - mu
    
    # Perform SVD (equivalent to PCA)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # First principal component
    p1 = U[:, 0]
    
    # Project onto first principal axis
    projections = np.dot(p1, X)
    half_span = np.max(np.abs(projections))
    
    characteristic_length = half_span * 2
    
    # Estimate signed angle of attack in degrees from x-axis
    angle_degrees = -np.rad2deg(np.arctan2(p1[1], p1[0]))
    
    if plot_method:
        # Compute line endpoints for visualization
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        p_start = mu.flatten() + min_proj * p1
        p_end = mu.flatten() + max_proj * p1
        
        plt.figure(figsize=(10, 8))
        plt.scatter(xs, ys, s=2, c='black', alpha=0.6, 
                   label='Solid Pixels' if show_components else None)
        
        if show_components:
            plt.scatter(mu[0], mu[1], marker='x', s=100, c='red', label='Centroid')
            
            plt.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                    linewidth=2, color='orange', label='Principal Axis')
            
            plt.scatter([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                       s=60, c='orange', label='Extent')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'PCA Analysis\nCharacteristic Length: {characteristic_length:.2f}, Angle: {angle_degrees:.1f}°')
        
        if show_components:
            plt.legend()
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return float(characteristic_length), float(angle_degrees)


def characteristic_length_bbox(mask, plot_method=False):
    """
    Estimate the characteristic length using bounding box method.
    
    Identifies a bounding box around the object and assumes the characteristic
    length is the longest diagonal in the box.
    
    Args:
        mask (numpy.ndarray): Boolean mask where False=solid, True=fluid
        plot_method (bool): Whether to display visualization plots
    
    Returns:
        float: Characteristic length (diagonal of bounding box)
    """
    # Convert mask to numpy array if needed
    mask = np.array(mask)
    
    # Find solid pixels (where mask is False, value 0)
    solid_coords = np.where(~mask)
    
    if len(solid_coords[0]) == 0:
        raise ValueError("No solid detected when attempting to calculate characteristic length")
    
    # Extract coordinates
    xs = solid_coords[0]
    ys = solid_coords[1]
    
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    
    dx = xmax - xmin
    dy = ymax - ymin
    characteristic_length = np.sqrt(dx**2 + dy**2)
    
    if plot_method:
        plt.figure(figsize=(10, 8))
        
        # Plot mask background
        plt.imshow(mask.T, cmap='gray', origin='lower', alpha=0.3, extent=[0, mask.shape[0], 0, mask.shape[1]])
        
        # Bounding box corners
        xcorners = [xmin, xmax, xmax, xmin, xmin]
        ycorners = [ymin, ymin, ymax, ymax, ymin]
        plt.plot(xcorners, ycorners, color='red', linewidth=2, label='Bounding Box')
        
        # Diagonal line
        plt.plot([xmin, xmax], [ymin, ymax], color='orange', linewidth=2, 
                linestyle='--', label='Diagonal')
        
        plt.scatter(xs, ys, s=2, c='black', alpha=0.4, label='Solid Pixels')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Bounding Box Analysis\nCharacteristic Length: {characteristic_length:.2f}')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return characteristic_length
