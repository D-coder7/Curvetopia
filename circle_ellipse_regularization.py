import numpy as np
from scipy import optimize
from typing import List, Tuple, Optional
from curvetopia_utils import Point, Polyline, Path, CurvetopiaProblem, distance

def fit_circle(points: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a circle to a set of 2D points using least squares.
    
    Args:
    points (np.ndarray): Array of shape (n, 2) containing the points.
    
    Returns:
    Tuple[float, float, float]: (center_x, center_y, radius)
    """
    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(points, axis=0)
    center, _ = optimize.leastsq(f, center_estimate)
    xc, yc = center
    Ri = calc_R(xc, yc)
    R = Ri.mean()
    
    return (xc, yc, R)

# def fit_ellipse(points: np.ndarray) -> Tuple[float, float, float, float, float]:
#     """
#     Fit an ellipse to a set of 2D points using least squares.
    
#     Args:
#     points (np.ndarray): Array of shape (n, 2) containing the points.
    
#     Returns:
#     Tuple[float, float, float, float, float]: (center_x, center_y, a, b, angle)
#     """
#     def ellipse_func(params, x, y):
#         xc, yc, a, b, theta = params
#         x_centered = x - xc
#         y_centered = y - yc
#         cos_theta = np.cos(theta)
#         sin_theta = np.sin(theta)
#         x_rotated = x_centered * cos_theta + y_centered * sin_theta
#         y_rotated = -x_centered * sin_theta + y_centered * cos_theta
#         return (x_rotated / a)**2 + (y_rotated / b)**2 - 1

    # def fit_func(params):
    #     return ellipse_func(params, points[:, 0], points[:, 1])

    # # Initial guess
    # center_estimate = np.mean(points, axis=0)
    # a_estimate = np.std(points[:, 0])
    # b_estimate = np.std(points[:, 1])
    # initial_guess = [center_estimate[0], center_estimate[1], a_estimate, b_estimate, 0]

    # params, _ = optimize.leastsq(fit_func, initial_guess)
    # return tuple(params)

import numpy as np
from scipy import optimize
from typing import List, Tuple, Optional
from curvetopia_utils import Point, Polyline, Path, CurvetopiaProblem, distance

def ellipse_func(params, x, y):
    """
    Ellipse function for fitting and evaluation.
    
    Args:
    params (tuple): (center_x, center_y, a, b, theta)
    x, y (float or np.ndarray): Coordinates to evaluate
    
    Returns:
    float or np.ndarray: Ellipse equation result
    """
    xc, yc, a, b, theta = params
    x_centered = x - xc
    y_centered = y - yc
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rotated = x_centered * cos_theta + y_centered * sin_theta
    y_rotated = -x_centered * sin_theta + y_centered * cos_theta
    return (x_rotated / a)**2 + (y_rotated / b)**2 - 1

def fit_ellipse(points: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Fit an ellipse to a set of 2D points using least squares.
    
    Args:
    points (np.ndarray): Array of shape (n, 2) containing the points.
    
    Returns:
    Tuple[float, float, float, float, float]: (center_x, center_y, a, b, angle)
    """
    def fit_func(params):
        return ellipse_func(params, points[:, 0], points[:, 1])

    # Initial guess
    center_estimate = np.mean(points, axis=0)
    a_estimate = np.std(points[:, 0])
    b_estimate = np.std(points[:, 1])
    initial_guess = [center_estimate[0], center_estimate[1], a_estimate, b_estimate, 0]

    params, _ = optimize.leastsq(fit_func, initial_guess)
    return tuple(params)

def is_ellipse(polyline: Polyline, tolerance: float = 0.05) -> bool:
    """
    Check if a polyline is approximately an ellipse.
    
    Args:
    polyline (Polyline): The polyline to check.
    tolerance (float): Maximum allowed deviation from a perfect ellipse.
    
    Returns:
    bool: True if the polyline is approximately an ellipse, False otherwise.
    """
    points = np.array([p.to_tuple() for p in polyline.points])
    params = fit_ellipse(points)
    
    deviations = np.abs(ellipse_func(params, points[:, 0], points[:, 1]))
    max_deviation = np.max(deviations)
    
    return max_deviation <= tolerance

# ... (rest of the code remains the same)

def is_circle(polyline: Polyline, tolerance: float = 0.05) -> bool:
    """
    Check if a polyline is approximately a circle.
    
    Args:
    polyline (Polyline): The polyline to check.
    tolerance (float): Maximum allowed deviation from a perfect circle.
    
    Returns:
    bool: True if the polyline is approximately a circle, False otherwise.
    """
    points = np.array([p.to_tuple() for p in polyline.points])
    xc, yc, R = fit_circle(points)
    
    distances = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)
    max_deviation = np.max(np.abs(distances - R))
    
    return max_deviation <= tolerance * R

def is_ellipse(polyline: Polyline, tolerance: float = 0.05) -> bool:
    """
    Check if a polyline is approximately an ellipse.
    
    Args:
    polyline (Polyline): The polyline to check.
    tolerance (float): Maximum allowed deviation from a perfect ellipse.
    
    Returns:
    bool: True if the polyline is approximately an ellipse, False otherwise.
    """
    points = np.array([p.to_tuple() for p in polyline.points])
    params = fit_ellipse(points)
    
    deviations = np.abs(ellipse_func(params, points[:, 0], points[:, 1]))
    max_deviation = np.max(deviations)
    
    return max_deviation <= tolerance

def regularize_circle(polyline: Polyline) -> Polyline:
    """
    Regularize a circle by fitting a perfect circle to the points.
    
    Args:
    polyline (Polyline): The polyline to regularize.
    
    Returns:
    Polyline: A new polyline representing a perfect circle.
    """
    points = np.array([p.to_tuple() for p in polyline.points])
    xc, yc, R = fit_circle(points)
    
    # Generate 100 points for the regularized circle
    theta = np.linspace(0, 2*np.pi, 100)
    x = xc + R * np.cos(theta)
    y = yc + R * np.sin(theta)
    
    regularized_points = [Point(xi, yi) for xi, yi in zip(x, y)]
    return Polyline(regularized_points)

def regularize_ellipse(polyline: Polyline) -> Polyline:
    """
    Regularize an ellipse by fitting a perfect ellipse to the points.
    
    Args:
    polyline (Polyline): The polyline to regularize.
    
    Returns:
    Polyline: A new polyline representing a perfect ellipse.
    """
    points = np.array([p.to_tuple() for p in polyline.points])
    xc, yc, a, b, theta = fit_ellipse(points)
    
    # Generate 100 points for the regularized ellipse
    t = np.linspace(0, 2*np.pi, 100)
    x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    
    regularized_points = [Point(xi, yi) for xi, yi in zip(x, y)]
    return Polyline(regularized_points)

def identify_and_regularize_circles_ellipses(problem: CurvetopiaProblem, tolerance: float = 0.05) -> CurvetopiaProblem:
    """
    Identify and regularize circles and ellipses in the Curvetopia problem.
    
    Args:
    problem (CurvetopiaProblem): The original problem.
    tolerance (float): Tolerance for circle and ellipse detection.
    
    Returns:
    CurvetopiaProblem: A new problem with regularized circles and ellipses.
    """
    regularized_paths = []
    
    for path in problem.paths:
        regularized_polylines = []
        for polyline in path.polylines:
            if is_circle(polyline, tolerance):
                regularized_polylines.append(regularize_circle(polyline))
            elif is_ellipse(polyline, tolerance):
                regularized_polylines.append(regularize_ellipse(polyline))
            else:
                regularized_polylines.append(polyline)
        regularized_paths.append(Path(regularized_polylines))
    
    return CurvetopiaProblem(regularized_paths)

def analyze_circles_ellipses(problem: CurvetopiaProblem) -> List[Tuple[str, Tuple]]:
    """
    Analyze the circles and ellipses in the problem and return their properties.
    
    Args:
    problem (CurvetopiaProblem): The problem containing circles and ellipses.
    
    Returns:
    List[Tuple[str, Tuple]]: A list of ('circle', (center_x, center_y, radius)) or 
                             ('ellipse', (center_x, center_y, a, b, angle)) for each shape.
    """
    shape_properties = []
    
    for path in problem.paths:
        for polyline in path.polylines:
            points = np.array([p.to_tuple() for p in polyline.points])
            if len(points) == 100:  # Assuming regularized shapes have 100 points
                if is_circle(polyline):
                    xc, yc, R = fit_circle(points)
                    shape_properties.append(('circle', (xc, yc, R)))
                else:
                    params = fit_ellipse(points)
                    shape_properties.append(('ellipse', params))
    
    return shape_properties

# Example usage
if __name__ == "__main__":
    from curvetopia_utils import read_csv, visualize
    
    # Replace 'path_to_your_csv_file.csv' with the actual path to your CSV file
    path = r"problems/frag0.csv"
    original_problem = read_csv(path)
    print("Original problem:", original_problem)
    
    regularized_problem = identify_and_regularize_circles_ellipses(original_problem)
    print("Regularized problem:", regularized_problem)
    
    print("Visualizing original problem:")
    visualize(original_problem)
    
    print("Visualizing regularized problem:")
    visualize(regularized_problem)
    
    shape_properties = analyze_circles_ellipses(regularized_problem)
    print("Shape properties:")
    for shape_type, props in shape_properties:
        if shape_type == 'circle':
            print(f"Circle: Center ({props[0]:.2f}, {props[1]:.2f}), Radius {props[2]:.2f}")
        else:
            print(f"Ellipse: Center ({props[0]:.2f}, {props[1]:.2f}), Semi-major axis {props[2]:.2f}, Semi-minor axis {props[3]:.2f}, Angle {props[4]:.2f} rad")
