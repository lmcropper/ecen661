"""
Optical Ray Tracing using ABCD Matrices

This module provides tools for tracing rays through optical systems using ABCD matrices
and visualizing the results. Supports free space propagation and thin lenses.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings


class OpticalSystem:
    """
    A class to represent an optical system using ABCD matrices.
    """
    
    def __init__(self):
        """Initialize an empty optical system."""
        self.elements = []  # List of (z_position, element_type, parameter)
        self.total_distance = 0
        
    def add_free_space(self, distance):
        """Add a free space propagation element."""
        self.elements.append({
            'type': 'free_space',
            'distance': distance,
            'position': self.total_distance
        })
        self.total_distance += distance
        
    def add_lens(self, focal_length):
        """Add a thin lens element."""
        self.elements.append({
            'type': 'lens',
            'focal_length': focal_length,
            'position': self.total_distance
        })
        
    def get_abcd_matrix(self, start_idx=0, end_idx=None):
        """
        Calculate the combined ABCD matrix for a segment of the optical system.
        
        Parameters:
        -----------
        start_idx : int
            Starting element index (default: 0)
        end_idx : int or None
            Ending element index (default: all elements)
            
        Returns:
        --------
        M : ndarray
            2x2 ABCD matrix
        """
        if end_idx is None:
            end_idx = len(self.elements)
            
        M = np.eye(2)  # Start with identity matrix
        
        for i in range(start_idx, end_idx):
            element = self.elements[i]
            
            if element['type'] == 'free_space':
                # Free space matrix
                M_element = np.array([
                    [1.0, element['distance']],
                    [0.0, 1.0]
                ])
            elif element['type'] == 'lens':
                # Thin lens matrix
                f = element['focal_length']
                M_element = np.array([
                    [1.0, 0.0],
                    [-1.0/f, 1.0]
                ])
            
            M = M_element @ M
            
        return M
    
    def trace_ray(self, initial_height, initial_angle):
        """
        Trace a single ray through the entire optical system.
        
        Parameters:
        -----------
        initial_height : float
            Initial ray height (y position) in mm
        initial_angle : float
            Initial ray angle in radians
            
        Returns:
        --------
        positions : ndarray
            Shape (2, n_points) with z positions and y positions along the ray path
        """
        ray_state = np.array([[initial_height], [initial_angle]])
        
        positions = np.array([[0.0], [initial_height]])  # Start position
        current_z = 0.0
        
        for element in self.elements:
            if element['type'] == 'free_space':
                # Propagate through free space
                d = element['distance']
                M = np.array([
                    [1.0, d],
                    [0.0, 1.0]
                ])
                ray_state = M @ ray_state
                current_z += d
                
                # Record endpoint
                new_pos = np.array([[current_z], [ray_state[0, 0]]])
                positions = np.hstack((positions, new_pos))
                
            elif element['type'] == 'lens':
                # Apply lens refraction (no distance change, just angle change)
                f = element['focal_length']
                M = np.array([
                    [1.0, 0.0],
                    [-1.0/f, 1.0]
                ])
                ray_state = M @ ray_state
                # Don't record a new z position for the lens
        
        return positions, ray_state
    
    def get_element_positions(self):
        """Return list of (position, type) for all optical elements."""
        return [(e['position'], e['type']) for e in self.elements]


def create_ray_types(system, object_height, object_distance, focal_length):
    """
    Create the three standard ray types for a given optical system.
    
    Parameters:
    -----------
    system : OpticalSystem
        The optical system
    object_height : float
        Height of the object in mm
    object_distance : float
        Distance from object to first lens in mm
    focal_length : float
        Focal length of the lens in mm
        
    Returns:
    --------
    rays : dict
        Dictionary with 'parallel', 'chief', 'focal' ray initial conditions
    """
    # All rays start at object height
    
    # 1. Parallel ray: parallel to optical axis (angle = 0)
    parallel_ray = (object_height, 0.0)
    
    # 2. Chief ray: goes through center of lens (angle makes it pass through center)
    # The lens is at distance object_distance from the object
    # To pass through lens center (height = 0), angle needed is:
    chief_angle = -object_height / object_distance
    chief_ray = (object_height, chief_angle)
    
    # 3. Focal ray: goes through the front focal point of the lens
    # The front focal point is at distance focal_length BEFORE the lens
    # Object is at distance object_distance from lens
    # The ray must pass through front focal point at height 0
    # Front focal point position: object_distance - focal_length
    # Angle needed: -object_height / (object_distance - focal_length)
    focal_angle = -object_height / (object_distance - focal_length)
    focal_ray = (object_height, focal_angle)
    
    return {
        'parallel': parallel_ray,
        'chief': chief_ray,
        'focal': focal_ray
    }


def plot_optical_system(system, object_height, object_distance, focal_length, 
                       title="Optical System Ray Trace"):
    """
    Plot the optical system with all three ray types.
    
    Parameters:
    -----------
    system : OpticalSystem
        The optical system with free space and lens elements
    object_height : float
        Height of the object in mm
    object_distance : float
        Distance from object to first lens in mm
    focal_length : float
        Focal length of the first lens in mm
    title : str
        Title for the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get ray types based on first lens position
    rays = create_ray_types(system, object_height, object_distance, focal_length)
    
    # Ray colors
    colors = {
        'parallel': '#1f77b4',  # blue
        'chief': '#ff7f0e',      # orange
        'focal': '#2ca02c'       # green
    }
    
    # Track all y values to set proper bounds
    all_y_values = [object_height, 0]  # Start with object height and optical axis
    ray_end_states = {}  # Store final states for back-projection
    
    # Trace and plot each ray type
    for ray_name, (init_height, init_angle) in rays.items():
        positions, final_state = system.trace_ray(init_height, init_angle)
        z = positions[0, :]
        y = positions[1, :]
        
        # Track all y values for bounds calculation
        all_y_values.extend(y)
        ray_end_states[ray_name] = (z[-1], y[-1], final_state[0, 0], final_state[1, 0])
        
        ax.plot(z, y, color=colors[ray_name], linewidth=2, 
               label=f'{ray_name.capitalize()} Ray', marker='o', markersize=4)
    
    # Handle virtual images by back-projecting diverging rays
    # Virtual images only exist when rays are diverging (positive angles) at the end
    z_end = system.total_distance
    
    # Check if rays are diverging (all have positive angles at the end)
    ray_names = list(ray_end_states.keys())
    end_angles = [ray_end_states[name][3] for name in ray_names]
    rays_diverging = all(angle > 1e-10 for angle in end_angles)  # All angles positive (diverging)
    
    # Only look for virtual image if rays are diverging
    if rays_diverging:
        # For each pair of rays, find where they would intersect backward
        for i in range(len(ray_names)):
            for j in range(i + 1, len(ray_names)):
                z1, y1, height1, angle1 = ray_end_states[ray_names[i]]
                z2, y2, height2, angle2 = ray_end_states[ray_names[j]]
                
                # Check if rays will intersect
                if abs(angle1 - angle2) > 1e-10:  # Rays not parallel
                    # Solve for intersection: y1 + angle1 * dz = y2 + angle2 * dz
                    dz = (y2 - y1) / (angle1 - angle2)
                    z_intersect = z1 + dz
                    y_intersect = y1 + angle1 * dz
                    
                    # Virtual image must be behind the system (z_intersect < z_end)
                    # and we must go backward to reach it (dz < 0)
                    if z_intersect < z_end and dz < 0:
                        # Extend rays backward to show virtual image
                        ax.plot([z_end, z_intersect], [y1, y_intersect], 
                               color=colors[ray_names[i]], linewidth=2, linestyle='--', alpha=0.6)
                        ax.plot([z_end, z_intersect], [y2, y_intersect], 
                               color=colors[ray_names[j]], linewidth=2, linestyle='--', alpha=0.6)
                        
                        # Mark virtual image
                        ax.plot(z_intersect, y_intersect, 'mo', markersize=10)
                        ax.axvline(x=z_intersect, color='purple', linestyle='--', linewidth=2, alpha=0.5)
                        ax.text(z_intersect + 3, y_intersect, 'Virtual Image', fontsize=9, 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                        
                        all_y_values.append(y_intersect)
                        break
    
    # Draw optical elements
    # Object plane with arrow
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.annotate('', xy=(0, object_height), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(-3, object_height/2, 'Object', fontsize=10, ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.plot(0, object_height, 'ko', markersize=8)
    
    # Mark all lenses in the system
    current_z = 0
    lens_count = 0
    for element in system.elements:
        if element['type'] == 'free_space':
            current_z += element['distance']
        elif element['type'] == 'lens':
            lens_count += 1
            f = element['focal_length']
            # Mark lens
            ax.axvline(x=current_z, color='red', linestyle='-', linewidth=3, alpha=0.7)
            
            # Mark focal points for first lens only (for clarity)
            if lens_count == 1:
                front_focal = current_z - f
                back_focal = current_z + f
                
                # Front focal point
                ax.plot(front_focal, 0, 'o', color='red', markersize=11, 
                       markeredgewidth=2.5, markeredgecolor='darkred', zorder=10)
                ax.text(front_focal, 0.3, 'F', fontsize=11, ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                edgecolor='red', linewidth=2, alpha=0.95))
                
                # Back focal point
                ax.plot(back_focal, 0, 'o', color='red', markersize=11, 
                       markeredgewidth=2.5, markeredgecolor='darkred', zorder=10)
                ax.text(back_focal, 0.3, "F'", fontsize=11, ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                edgecolor='red', linewidth=2, alpha=0.95))
    
    # Add legend entry for first lens only
    first_lens_z = object_distance
    ax.plot([], [], color='red', linestyle='-', linewidth=3, alpha=0.7, 
           label=f'Lens (f={focal_length}mm)')
    
    # Calculate and mark image position (for first lens)
    u = object_distance
    f = focal_length
    v = (u * f) / (u - f)
    image_position = object_distance + v
    image_height = object_height * (-v / u)  # magnification
    
    # Only mark first lens image if it's a real image (v > 0) and within reasonable bounds
    if v > 0:
        all_y_values.append(image_height)
        # Image plane with arrow
        ax.axvline(x=image_position, color='purple', linestyle='--', linewidth=2, alpha=0.5)
        ax.annotate('', xy=(image_position, image_height), xytext=(image_position, 0),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax.text(image_position + 3, image_height/2, 'Image', fontsize=10, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.plot(image_position, image_height, 'mo', markersize=8)
    
    # Add optical axis
    z_max = system.total_distance + 10
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    # Calculate dynamic y-axis bounds based on all ray positions
    y_min = min(all_y_values)
    y_max = max(all_y_values)
    y_range = y_max - y_min
    padding = max(0.3, y_range * 0.2)  # 20% padding or at least 0.3mm
    
    # Set labels and formatting
    ax.set_xlabel('Distance from Object (mm)', fontsize=12)
    ax.set_ylabel('Height from Optical Axis (mm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', color='red')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, z_max)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Set equal aspect ratio for better visualization
    ax.set_aspect('auto')
    
    return fig, ax


def analyze_system(system, object_height, object_distance, focal_length):
    """
    Analyze the optical system and print key information.
    
    Parameters:
    -----------
    system : OpticalSystem
        The optical system
    object_height : float
        Height of the object in mm
    object_distance : float
        Distance from object to lens in mm
    focal_length : float
        Focal length of the lens in mm
    """
    # Calculate image position using thin lens equation
    # 1/f = 1/object_distance + 1/image_distance
    u = object_distance  # object distance
    f = focal_length
    v = (u * f) / (u - f)  # image distance
    
    # Magnification
    magnification = -v / u
    image_height = object_height * magnification
    
    print("=" * 60)
    print("Optical System Analysis")
    print("=" * 60)
    print(f"Object Height: {object_height} mm")
    print(f"Object Distance: {object_distance} mm")
    print(f"Lens Focal Length: {focal_length} mm")
    print(f"Image Distance: {v:.2f} mm")
    print(f"Image Height: {image_height:.2f} mm")
    print(f"Magnification: {magnification:.2f}x")
    
    if v > 0:
        print("Image Type: Real (inverted)")
    else:
        print("Image Type: Virtual (upright)")
    
    print("=" * 60)
    
    return {
        'image_distance': v,
        'image_height': image_height,
        'magnification': magnification
    }


def create_optical_system_auto(object_distance, focal_length):
    """
    Create an optical system and automatically extend it to include the image position.
    
    Parameters:
    -----------
    object_distance : float
        Distance from object to lens in mm
    focal_length : float
        Focal length of the lens in mm
        
    Returns:
    --------
    system : OpticalSystem
        The optical system with proper extension
    """
    system = OpticalSystem()
    system.add_free_space(object_distance)  # Object to lens
    system.add_lens(focal_length)
    
    # Calculate where image forms and extend the system beyond it
    v = (object_distance * focal_length) / (object_distance - focal_length)
    system.add_free_space(abs(v) + 10)  # Beyond image position for visualization
    
    return system


# Example usage
if __name__ == "__main__":
    # Problem parameters
    object_height = 1.0  # mm
    object_distance = 20.0  # mm
    focal_length = 10.0  # mm
    
    # Create optical system (automatically handles image distance)
    system = create_optical_system_auto(object_distance, focal_length)
    
    # Analyze
    analyze_system(system, object_height, object_distance, focal_length)
    
    # Plot
    fig, ax = plot_optical_system(system, object_height, object_distance, focal_length,
                                  title=f"Ray Trace: Object@{object_distance}mm, Lens f={focal_length}mm")
    plt.tight_layout()
    plt.show()
