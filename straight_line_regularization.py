import numpy as np
from typing import List, Tuple
from curvetopia_utils import Point, Polyline, Path, CurvetopiaProblem, distance

def is_straight_line(polyline: Polyline, tolerance: float = 0.05) -> bool:
    """
    Check if a polyline is approximately a straight line.
    
    Args:
    polyline (Polyline): The polyline to check.
    tolerance (float): Maximum allowed deviation from a straight line.
    
    Returns:
    bool: True if the polyline is approximately a straight line, False otherwise.
    """
    if len(polyline.points) < 3:
        return True
    
    points = np.array([p.to_tuple() for p in polyline.points])
    
    # Fit a line using linear regression
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Calculate the maximum distance from any point to the fitted line
    distances = np.abs(y - (m * x + c)) / np.sqrt(m**2 + 1)
    max_distance = np.max(distances)
    
    # Check if the maximum distance is within the tolerance
    return max_distance <= tolerance * np.linalg.norm(points[-1] - points[0])

def regularize_straight_line(polyline: Polyline) -> Polyline:
    """
    Regularize a straight line by adjusting its endpoints.
    
    Args:
    polyline (Polyline): The polyline to regularize.
    
    Returns:
    Polyline: A new polyline with regularized endpoints.
    """
    if len(polyline.points) < 2:
        return polyline
    
    start = polyline.points[0]
    end = polyline.points[-1]
    
    # Create a new polyline with only the start and end points
    return Polyline([start, end])

def identify_and_regularize_lines(problem: CurvetopiaProblem, tolerance: float = 0.05) -> CurvetopiaProblem:
    """
    Identify and regularize straight lines in the Curvetopia problem.
    
    Args:
    problem (CurvetopiaProblem): The original problem.
    tolerance (float): Tolerance for line detection.
    
    Returns:
    CurvetopiaProblem: A new problem with regularized straight lines.
    """
    regularized_paths = []
    
    for path in problem.paths:
        regularized_polylines = []
        for polyline in path.polylines:
            if is_straight_line(polyline, tolerance):
                regularized_polylines.append(regularize_straight_line(polyline))
            else:
                regularized_polylines.append(polyline)
        regularized_paths.append(Path(regularized_polylines))
    
    return CurvetopiaProblem(regularized_paths)

def analyze_lines(problem: CurvetopiaProblem) -> List[Tuple[float, float, float]]:
    """
    Analyze the straight lines in the problem and return their properties.
    
    Args:
    problem (CurvetopiaProblem): The problem containing straight lines.
    
    Returns:
    List[Tuple[float, float, float]]: A list of (length, angle, midpoint_x, midpoint_y) for each straight line.
    """
    line_properties = []
    
    for path in problem.paths:
        for polyline in path.polylines:
            if len(polyline.points) == 2:  # Assuming regularized straight lines have only 2 points
                start, end = polyline.points
                length = distance(start, end)
                angle = np.arctan2(end.y - start.y, end.x - start.x)
                midpoint_x = (start.x + end.x) / 2
                midpoint_y = (start.y + end.y) / 2
                line_properties.append((length, angle, midpoint_x, midpoint_y))
    
    return line_properties

# Example usage
if __name__ == "__main__":
    from curvetopia_utils import read_csv, visualize
    
    # Replace 'path_to_your_csv_file.csv' with the actual path to your CSV file
    path = r"problems/frag0.csv"
    original_problem = read_csv(path)
    print("Original problem:", original_problem)
    
    regularized_problem = identify_and_regularize_lines(original_problem)
    print("Regularized problem:", regularized_problem)
    
    print("Visualizing original problem:")
    visualize(original_problem)
    
    print("Visualizing regularized problem:")
    visualize(regularized_problem)
    
    line_properties = analyze_lines(regularized_problem)
    print("Line properties (length, angle, midpoint_x, midpoint_y):")
    for prop in line_properties:
        print(f"Length: {prop[0]:.2f}, Angle: {prop[1]:.2f} rad, Midpoint: ({prop[2]:.2f}, {prop[3]:.2f})")
