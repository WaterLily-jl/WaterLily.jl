import numpy as np
import matplotlib.pyplot as plt


def characteristic_length_and_aoa_pca(
        mask,
        plot_method=False,
        show_components=True,
        flow_xy=(1., 0.),
        debug_mode=False,
) -> tuple[float, float, float]:
    """
    Estimate the characteristic length, angle of attack, and maximum thickness of a PixelBody using Principal Component Analysis.
    
    Finds the direction perpendicular to the principal component of the pixel distribution
    (the direction in which points are most spread) and estimates the angle of attack
    and maximum thickness of the object based on the principal axis direction.
    
    Args:
        mask (numpy.ndarray): Boolean mask where False=solid, True=fluid
        plot_method (bool): Whether to display visualization plots
        show_components (bool): Whether to show legend and PCA components in plot
        flow_xy (tuple): Tuple of (x, y) coordinates representing the flow direction. If not provided, will assume
                        flow direciton is in the direction of x, that is, (1, 0).
        debug_mode (bool): If True, prints debug information to console.

    Returns:
        tuple: (characteristic_length, angle_degrees, max_thickness)
            - characteristic_length (float): Estimated characteristic length
            - angle_degrees (float): Signed angle of attack in degrees [-180, 180]
                                   Positive for clockwise, negative for counterclockwise
            - max_thickness (float): Maximum thickness perpendicular to principal axis
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
    
    # Calculate maximum thickness using cross-sectional analysis FIRST
    # Method: Sweep along the principal axis and find max thickness perpendicular to it
    p2 = U[:, 1]  # Second principal component (perpendicular to principal axis)
    proj_perp_all = np.dot(p2, X)  # Project all points onto perpendicular axis
    
    # Create bins along the principal axis for cross-sectional analysis
    n_bins = 50  # Number of cross-sections to analyze
    proj_along_axis = projections
    
    # Find thickness at each cross-section
    bin_edges = np.linspace(np.min(proj_along_axis), np.max(proj_along_axis), n_bins + 1)
    max_thickness = 0.0
    max_thickness_location = 0.0
    
    for i in range(n_bins):
        # Points in this cross-section bin
        in_bin = (proj_along_axis >= bin_edges[i]) & (proj_along_axis < bin_edges[i + 1])
        
        if np.sum(in_bin) > 1:  # Need at least 2 points to measure thickness
            # Find the span of points in the perpendicular direction for this cross-section
            perp_coords_in_bin = proj_perp_all[in_bin]
            thickness = np.max(perp_coords_in_bin) - np.min(perp_coords_in_bin)
            
            if thickness > max_thickness:
                max_thickness = thickness
                max_thickness_location = (bin_edges[i] + bin_edges[i + 1]) / 2

    # Determine airfoil orientation using thickness position method
    # For airfoils, max thickness is typically closer to leading edge
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    
    # Distance from max thickness to each end
    dist_to_min = abs(max_thickness_location - min_proj)
    dist_to_max = abs(max_thickness_location - max_proj)
    
    # The end closer to max thickness is likely the leading edge
    if dist_to_min < dist_to_max:
        # Min projection end is leading edge
        leading_proj = min_proj
        trailing_proj = max_proj
    else:
        # Max projection end is leading edge
        leading_proj = max_proj
        trailing_proj = min_proj
    
    # Calculate actual positions of leading and trailing edges
    p_center = mu.flatten()
    p_leading = p_center + leading_proj * p1    # Leading edge position
    p_trailing = p_center + trailing_proj * p1  # Trailing edge position
    
    # Store coordinates for easy access
    x_trailing, y_trailing = p_trailing[0], p_trailing[1]
    x_leading, y_leading = p_leading[0], p_leading[1]

    # Calculate Angle of attack in degrees
    angle_degrees = calculate_angle_of_attack(
        leading_edge_xy=(x_leading, y_leading),
        trailing_edge_xy=(x_trailing, y_trailing),
        flow_xy=flow_xy,
    )

    if debug_mode:
        print("Debug info:")
        print(f"  Distance from max thickness to min proj: {dist_to_min:.3f}")
        print(f"  Distance from max thickness to max proj: {dist_to_max:.3f}")
        print(f"  Leading edge detected at: {'min' if dist_to_min < dist_to_max else 'max'} projection end")
        print(f"  Trailing edge coords: ({x_trailing:.1f}, {y_trailing:.1f})")
        print(f"  Leading edge coords: ({x_leading:.1f}, {y_leading:.1f})")
        print(f"  Maximum thickness: {max_thickness:.2f}")
        print(f"  Max thickness location: {max_thickness_location:.2f} along principal axis")
        print(f" Angle of attack (degrees): {angle_degrees:.2f}")

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
            
            # Plot full principal axis
            plt.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                    linewidth=2, color='orange', label='Principal Axis')
            
            # Plot leading and trailing edges with different markers
            plt.scatter([x_trailing], [y_trailing], s=80, c='blue', marker='s', 
                       label='Trailing Edge', edgecolor='darkblue', linewidth=2)
            plt.scatter([x_leading], [y_leading], s=80, c='green', marker='^', 
                       label='Leading Edge', edgecolor='darkgreen', linewidth=2)
            
            # Plot maximum thickness location
            # Calculate the position where max thickness occurs
            max_thick_pos = mu.flatten() + max_thickness_location * p1
            # Calculate the perpendicular line endpoints at max thickness location
            thick_half_span = max_thickness / 2
            thick_start = max_thick_pos + thick_half_span * p2
            thick_end = max_thick_pos - thick_half_span * p2
            
            plt.plot([thick_start[0], thick_end[0]], [thick_start[1], thick_end[1]], 
                    linewidth=3, color='magenta', label=f'Max Thickness ({max_thickness:.2f})')
            plt.scatter([max_thick_pos[0]], [max_thick_pos[1]], s=60, c='magenta', marker='o', 
                       label='Max Thickness Center', edgecolor='darkmagenta', linewidth=2)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'PCA Analysis\nLength: {characteristic_length:.2f}, Angle: {angle_degrees:.1f}°, Max Thickness: {max_thickness:.2f}')
        
        if show_components:
            plt.legend()
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return float(characteristic_length), float(angle_degrees), float(max_thickness)


def calculate_angle_of_attack(
        leading_edge_xy: tuple[float, float], trailing_edge_xy: tuple[float, float], flow_xy: tuple[float, float]
) -> float:
    """
    Calculate the angle of attack of an air based on leading and trailing edge coordinates, and provided flow direction.
    """

    # Unpack coordinates
    flow_x, flow_y = flow_xy
    x_leading, y_leading = leading_edge_xy
    x_trailing, y_trailing = trailing_edge_xy

    # Build chord vector (direction is from leading to trailing edge)
    chord_vec = np.array([x_trailing - x_leading, y_trailing - y_leading])

    # Build flow vector (from provded coordinates)
    flow_vec = np.array([flow_x, flow_y])

    # Normalize vectors (for angle calculation we only care about direction)
    chord_vec = chord_vec / np.linalg.norm(chord_vec)
    flow_vec = flow_vec / np.linalg.norm(flow_vec)

    # Calculate signed angle from flow to chord (in degrees)
    angle_rad = np.arctan2(
        chord_vec[0]*flow_vec[1] - chord_vec[1]*flow_vec[0],  # cross product (z-component)
        chord_vec[0]*flow_vec[0] + chord_vec[1]*flow_vec[1]   # dot product
    )
    angle_deg = np.degrees(angle_rad)

    return angle_deg