import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def circle_intersections(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    if d > r1 + r2 or d < abs(r1 - r2):
        return []  # No intersections
    
    a = (r1**2 - r2**2 + d**2) / (2*d)
    h = np.sqrt(max(0, r1**2 - a**2))  # Use max to avoid negative values due to float precision
    
    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d
    
    x4 = x3 + h * (y2 - y1) / d
    y4 = y3 - h * (x2 - x1) / d
    
    x5 = x3 - h * (y2 - y1) / d
    y5 = y3 + h * (x2 - x1) / d
    
    return [(x4, y4), (x5, y5)]

def is_point_inside_circle(point, circle):
    return np.linalg.norm(np.array(point) - np.array(circle[:2])) < circle[2]

def angle_between_points(center, p1, p2):
    v1 = np.array(p1) - np.array(center)
    v2 = np.array(p2) - np.array(center)
    angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
    return angle if angle >= 0 else angle + 2*np.pi

def is_arc_on_perimeter(circle, start, end, other_circles):
    center = circle[:2]
    radius = circle[2]
    start_angle = np.arctan2(start[1] - center[1], start[0] - center[0])
    end_angle = np.arctan2(end[1] - center[1], end[0] - center[0])
    
    if end_angle < start_angle:
        end_angle += 2 * np.pi
    
    mid_angle = (start_angle + end_angle) / 2
    mid_point = (center[0] + radius * np.cos(mid_angle), center[1] + radius * np.sin(mid_angle))
    
    return not any(is_point_inside_circle(mid_point, other) for other in other_circles)

def find_outer_perimeter(circles):
    all_intersections = []
    for i, c1 in enumerate(circles):
        circle_intersects = []
        for j, c2 in enumerate(circles):
            if i != j:
                intersects = circle_intersections(c1, c2)
                circle_intersects.extend([(p, j) for p in intersects])
        all_intersections.append((c1, circle_intersects))

    outer_arcs = []
    for circle, intersections in all_intersections:
        center = circle[:2]
        radius = circle[2]
        points = [p for p, _ in intersections]
        
        if len(points) >= 2:
            sorted_points = sorted(points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
            for i in range(len(sorted_points)):
                p1, p2 = sorted_points[i], sorted_points[(i+1) % len(sorted_points)]
                if is_arc_on_perimeter(circle, p1, p2, [c for c in circles if not np.array_equal(c, circle)]):
                    angle = angle_between_points(center, p1, p2)
                    outer_arcs.append((circle, p1, p2, angle))
        elif len(points) == 1:
            # Handle the case where a circle is tangent to the perimeter
            p = points[0]
            other_circles = [c for c in circles if not np.array_equal(c, circle)]
            if not any(is_point_inside_circle(p, other) for other in other_circles):
                angle = 2 * np.pi
                outer_arcs.append((circle, p, p, angle))

    return outer_arcs

def generate_perimeter_points(outer_arcs, num_points_per_arc=100):
    perimeter_points = []
    for arc in outer_arcs:
        circle, start, end, angle = arc
        center = circle[:2]
        radius = circle[2]
        start_angle = np.arctan2(start[1] - center[1], start[0] - center[0])
        end_angle = start_angle + angle
        
        if start == end:  # Full circle case
            angles = np.linspace(0, 2*np.pi, num_points_per_arc, endpoint=False)
        else:
            angles = np.linspace(start_angle, end_angle, num_points_per_arc)
        
        arc_points = [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]
        perimeter_points.extend(arc_points)
    
    return perimeter_points

def plot_circles_and_perimeter(circles, perimeter_points):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot original circles
    for circle in circles:
        c = plt.Circle((circle[0], circle[1]), circle[2], fill=False, color='r', linestyle='--')
        ax.add_artist(c)
    
    # Plot perimeter points
    perimeter_x, perimeter_y = zip(*perimeter_points)
    ax.scatter(perimeter_x, perimeter_y, color='blue', s=10, alpha=0.5, label='Perimeter Points')
    
    # Set limits and aspect
    all_coords = np.vstack((circles[:, :2], circles[:, :2] + circles[:, 2][:, np.newaxis], circles[:, :2] - circles[:, 2][:, np.newaxis]))
    ax.set_xlim(all_coords[:, 0].min() - 50, all_coords[:, 0].max() + 50)
    ax.set_ylim(all_coords[:, 1].min() - 50, all_coords[:, 1].max() + 50)
    ax.set_aspect('equal', adjustable='box')
    
    plt.title("Circles and Outer Perimeter Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.show()

# Your input data
circles = np.array([
    [300, 8, 151],
    [210, 2, 139],
    [466, 28, 114],
    [386, 10, 147]
])

# Find outer perimeter
outer_arcs = find_outer_perimeter(circles)

# Generate perimeter points
perimeter_points = generate_perimeter_points(outer_arcs)

# Output the number of perimeter points
print(f"Total number of perimeter points: {len(perimeter_points)}")

# Plot the circles and perimeter points
plot_circles_and_perimeter(circles, perimeter_points)