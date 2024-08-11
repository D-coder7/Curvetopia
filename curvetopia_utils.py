import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

class Polyline:
    def __init__(self, points: List[Point]):
        self.points = points

    def __repr__(self):
        return f"Polyline({len(self.points)} points)"

    def to_numpy(self) -> np.ndarray:
        return np.array([p.to_tuple() for p in self.points])

class Path:
    def __init__(self, polylines: List[Polyline]):
        self.polylines = polylines

    def __repr__(self):
        return f"Path({len(self.polylines)} polylines)"

class CurvetopiaProblem:
    def __init__(self, paths: List[Path]):
        self.paths = paths

    def __repr__(self):
        return f"CurvetopiaProblem({len(self.paths)} paths)"

def read_csv(csv_path: str) -> CurvetopiaProblem:
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    paths = []
    
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        polylines = []
        
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            points = [Point(x, y) for x, y in XY]
            polylines.append(Polyline(points))
        
        paths.append(Path(polylines))
    
    return CurvetopiaProblem(paths)

def visualize(problem: CurvetopiaProblem):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(problem.paths)))

    for path, color in zip(problem.paths, colors):
        for polyline in path.polylines:
            points = polyline.to_numpy()
            ax.plot(points[:, 0], points[:, 1], c=color, linewidth=2)

    ax.set_aspect('equal')
    plt.show()

# Utility function to calculate distance between two points
def distance(p1: Point, p2: Point) -> float:
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Utility function to calculate the angle between three points
def angle(p1: Point, p2: Point, p3: Point) -> float:
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# Example usage
if __name__ == "__main__":
    # Replace 'path_to_your_csv_file.csv' with the actual path to your CSV file
    path = path = r"problems/frag0.csv"
    problem = read_csv(path)
    print(problem)
    visualize(problem)
