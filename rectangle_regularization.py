import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional
from curvetopia_utils import Point, Polyline, Path, CurvetopiaProblem, distance, angle

def find_corners(points: np.ndarray, angle_threshold: float = np.pi/6) -> List[int]:
    """
    Find corner indices in a set of points.
    
    Args:
    points (np.ndarray): Array of shape (n, 2) containing the points.
    angle_threshold (float): Minimum angle to consider a point as a corner.
    
    Returns:
    List[int]: Indices of corner points.
    """
    n = len(points)
    corners = []
    for i in range(n):
        prev = (i - 1) % n
        next = (i + 1) % n
        v1 = points[prev] - points[i]
        v2 = points[next] - points[i]
        ang = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        if ang < np.pi - angle_threshold:
            corners.append(i)
    return corners

def is_rectangle(polyline: Polyline, tolerance: float = 0.1) -> bool:
    """
    Check if a polyline is approximately a rectangle.
    
    Args:
    polyline (Polyline): The polyline to check.
    tolerance (float): Maximum allowed deviation from a perfect rectangle.
    
    Returns:
    bool: True if the polyline is approximately a rectangle, False otherwise.
    """
    points = np.array([p.to_tuple() for p in polyline.points])
    corners = find_corners(points)
    
    if len(corners) != 4:
        return False
    
    corner_points = points[corners]
    
    # Check if opposite sides are parallel and equal in length
    for i in range(2):
        side1 = np.linalg.norm(corner_points[i] - corner_points[(i+1)%4])
        side2 = np.linalg.norm(corner_points[(i+2)%4] - corner_points[(i+3)%4])
        if abs(side1 - side2) / max(side1, side2) > tolerance:
            return False
    
    # Check if angles are approximately 90 degrees
    for i in range(4):
        ang = angle(Point(*corner_points[i]), Point(*corner_points[(i+1)%4]), Point(*corner_points[(i+2)%4]))
        if abs(ang - np.pi/2) > tolerance:
            return False
    
    return True

def is_rounded_rectangle(polyline: Polyline, tolerance: float = 0.1, corner_ratio: float = 0.2) -> bool:
    """
    Check if a polyline is approximately a rounded rectangle.
    
    Args:
    polyline (Polyline): The polyline to check.
    tolerance (float): Maximum allowed deviation from a perfect rounded rectangle.
    corner_ratio (float): Expected ratio of corner arc length to total perimeter.
    
    Returns:
    bool: True if the polyline is approximately a rounded rectangle, False otherwise.
    """
    points = np.array([p.to_tuple() for p in polyline.points])
    
    # Check if the shape is closed
    if np.linalg.norm(points[0] - points[-1]) > tolerance * np.linalg.norm(points[1] - points[0]):
        return False
    
    # Estimate corner regions
    perimeter = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    corner_length = perimeter * corner_ratio
    
    # Find potential corners
    curvature = []
    for i in range(len(points)):
        prev = (i - 1) % len(points)
        next = (i + 1) % len(points)
        v1 = points[prev] - points[i]
        v2 = points[next] - points[i]
        curvature.append(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    
    potential_corners = [i for i, c in enumerate(curvature) if c < np.pi - tolerance]
    
    # Check if we have exactly 4 corner regions
    if len(potential_corners) != 4:
        return False
    
    # Check if corner regions are evenly spaced
    corner_distances = np.diff(potential_corners + [potential_corners[0] + len(points)])
    if max(corner_distances) - min(corner_distances) > tolerance * len(points):
        return False
    
    # Check if sides are approximately straight
    for i in range(4):
        start = (potential_corners[i] + int(corner_length/2)) % len(points)
        end = (potential_corners[(i+1)%4] - int(corner_length/2)) % len(points)
        side_points = points[start:end] if start < end else np.vstack((points[start:], points[:end]))
        
        # Fit a line to the side points
        vx, vy, x0, y0 = cv2.fitLine(side_points, cv2.DIST_L2, 0, 0.01, 0.01)
        distances = np.abs((vy*(side_points[:,0]-x0) - vx*(side_points[:,1]-y0)) / np.sqrt(vx*vx + vy*vy))
        
        if np.max(distances) > tolerance * np.linalg.norm(side_points[-1] - side_points[0]):
            return False
    
    return True

def regularize_rectangle(polyline: Polyline) -> Polyline:
    """
    Regularize a rectangle by adjusting its corners.
    
    Args:
    polyline (Polyline): The polyline to regularize.
    
    Returns:
    Polyline: A new polyline representing a perfect rectangle.
    """
    points = np.array([p.to_tuple() for p in polyline.points])
    corners = find_corners(points)
    corner_points = points[corners]
    
    # Calculate the average width and height
    width = (np.linalg.norm(corner_points[1] - corner_points[0]) + 
             np.linalg.norm(corner_points[3] - corner_points[2])) / 2
    height = (np.linalg.norm(corner_points[2] - corner_points[1]) + 
              np.linalg.norm(corner_points[0] - corner_points[3])) / 2
    
    # Calculate the center and orientation
    center = np.mean(corner_points, axis=0)
    angle = np.arctan2(corner_points[1][1] - corner_points[0][1], 
                       corner_points[1][0] - corner_points[0][0])
    
    # Generate perfect rectangle points
    perfect_corners = np.array([
        [-width/2, -height/2],
        [width/2, -height/2],
        [width/2, height/2],
        [-width/2, height/2]
    ])
    
    # Rotate and translate the perfect rectangle
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_corners = np.dot(perfect_corners, rotation_matrix.T) + center
    
    return Polyline([Point(x, y) for x, y in rotated_corners])

def regularize_rounded_rectangle(polyline: Polyline, num_points: int = 100) -> Polyline:
    """
    Regularize a rounded rectangle by adjusting its corners and sides.
    
    Args:
    polyline (Polyline): The polyline to regularize.
    num_points (int): Number of points to generate for the regularized shape.
    
    Returns:
    Polyline: A new polyline representing a perfect rounded rectangle.
    """
    points = np.array([p.to_tuple() for p in polyline.points])
    
    # Estimate corner regions
    perimeter = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    corner_length = perimeter * 0.2  # Assuming 20% of perimeter is corners
    
    # Find potential corners
    curvature = []
    for i in range(len(points)):
        prev = (i - 1) % len(points)
        next = (i + 1) % len(points)
        v1 = points[prev] - points[i]
        v2 = points[next] - points[i]
        curvature.append(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    
    potential_corners = [i for i, c in enumerate(curvature) if c < np.pi - 0.1]
    
    # Extract side midpoints
    side_midpoints = []
    for i in range(4):
        start = (potential_corners[i] + int(corner_length/2)) % len(points)
        end = (potential_corners[(i+1)%4] - int(corner_length/2)) % len(points)
        side_points = points[start:end] if start < end else np.vstack((points[start:], points[:end]))
        side_midpoints.append(np.mean(side_points, axis=0))
    
    # Calculate center, width, and height
    center = np.mean(side_midpoints, axis=0)
    width = np.linalg.norm(side_midpoints[1] - side_midpoints[3])
    height = np.linalg.norm(side_midpoints[0] - side_midpoints[2])
    
    # Calculate corner radius (assuming it's the same for all corners)
    corner_radius = min(width, height) * 0.1  # Assuming corner radius is 10% of the smaller dimension
    
    # Generate perfect rounded rectangle points
    t = np.linspace(0, 2*np.pi, num_points)
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    
    for i in range(4):
        start_angle = i * np.pi/2
        end_angle = (i + 1) * np.pi/2
        mask = (t >= start_angle) & (t < end_angle)
        x[mask] = (width/2 - corner_radius) * np.sign(np.cos(start_angle)) + corner_radius * np.cos(t[mask])
        y[mask] = (height/2 - corner_radius) * np.sign(np.sin(start_angle)) + corner_radius * np.sin(t[mask])
    
    # Rotate to match original orientation
    original_angle = np.arctan2(side_midpoints[1][1] - side_midpoints[3][1], 
                                side_midpoints[1][0] - side_midpoints[3][0])
    rotation_matrix = np.array([
        [np.cos(original_angle), -np.sin(original_angle)],
        [np.sin(original_angle), np.cos(original_angle)]
    ])
    rotated_points = np.dot(np.column_stack((x, y)), rotation_matrix.T) + center
    
    return Polyline([Point(x, y) for x, y in rotated_points])

def identify_and_regularize_rectangles(problem: CurvetopiaProblem, tolerance: float = 0.1) -> CurvetopiaProblem:
    """
    Identify and regularize rectangles and rounded rectangles in the Curvetopia problem.
    
    Args:
    problem (CurvetopiaProblem): The original problem.
    tolerance (float): Tolerance for shape detection.
    
    Returns:
    CurvetopiaProblem: A new problem with regularized rectangles and rounded rectangles.
    """
    regularized_paths = []
    
    for path in problem.paths:
        regularized_polylines = []
        for polyline in path.polylines:
            if is_rectangle(polyline, tolerance):
                regularized_polylines.append(regularize_rectangle(polyline))
            elif is_rounded_rectangle(polyline, tolerance):
                regularized_polylines.append(regularize_rounded_rectangle(polyline))
            else:
                regularized_polylines.append(polyline)
        regularized_paths.append(Path(regularized_polylines))
    
    return CurvetopiaProblem(regularized_paths)

# Example usage
if __name__ == "__main__":
    from curvetopia_utils import read_csv, visualize
    
    # Replace 'path_to_your_csv_file.csv' with the actual path to your CSV file
    path = r"problems/frag0.csv"
    original_problem = read_csv(path)
    print("Original problem:", original_problem)
    
    regularized_problem = identify_and_regularize_rectangles(original_problem)
    print("Regularized problem:", regularized_problem)
    
    print("Visualizing original problem:")
    visualize(original_problem)
    
    print("Visualizing regularized problem:")
    visualize(regularized_problem)
