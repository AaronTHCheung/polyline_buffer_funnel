import math
import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Circle
from typing import List, Dict, Any, Tuple, NamedTuple

# ---------------------------------------------------------------------------
# Types & Data Structures (Immutable)
# ---------------------------------------------------------------------------

class Point(NamedTuple):
    x: float
    y: float

class Portal(NamedTuple):
    left: Point
    right: Point

Polygon = List[Point]

# ---------------------------------------------------------------------------
# Pure Functions: Math & Geometry
# ---------------------------------------------------------------------------

def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def point_line_distance(pt: Point, line_start: Point, line_end: Point) -> float:
    line_mag = distance(line_start, line_end)
    if line_mag == 0.0:
        return distance(pt, line_start)
    
    numerator = abs(
        (line_end.y - line_start.y) * pt.x - 
        (line_end.x - line_start.x) * pt.y + 
        line_end.x * line_start.y - 
        line_end.y * line_start.x
    )  # cross product, and the absolute operator extracts just the area of the parallelogram

    return numerator / line_mag  # dividing the parallelogram area by the base length gives the height, which is the distance between the point to the line

def calculate_tangent_angle(apex: Point, center: Point, radius: float, is_left: bool) -> float:
    dist = distance(apex, center)
    if dist <= radius:
        return math.atan2(apex.y - center.y, apex.x - center.x)  # the vector points from the center to the apex - a safety measure to drives the apex away from the circle

    theta = math.atan2(center.y - apex.y, center.x - apex.x)
    phi = math.asin(radius / dist)  # the angle offset from theta that will graze the circle's edge
    
    # Left obstacles require tangent on their right side (-phi)
    # Right obstacles require tangent on their left side (+phi)
    return theta - phi if is_left else theta + phi

def angle_diff(angle1: float, angle2: float) -> float:
    return (angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi

def get_tangent_point(apex: Point, center: Point, radius: float, is_left: bool) -> Point:
    dist = distance(apex, center)
    if dist <= radius:
        return apex

    tangent_angle = calculate_tangent_angle(apex, center, radius, is_left)
    tangent_length = math.sqrt(dist**2 - radius**2)
    
    return Point(
        x=apex.x + math.cos(tangent_angle) * tangent_length,
        y=apex.y + math.sin(tangent_angle) * tangent_length
    )

def generate_arc_points(center: Point, radius: float, angle_in: float, angle_out: float, is_left: bool, step: float = 0.1) -> List[Point]:
    """
    Generates discretized arc points using r_safe to guarantee the linear segments 
    never cross inside the clearance circle (secant error prevention).
    """
    r_safe = radius / math.cos(step / 2.0)
    
    angle_in = angle_in % (2 * math.pi)
    angle_out = angle_out % (2 * math.pi)
    diff = angle_out - angle_in
    
    if is_left:
        # Wrap counter-clockwise
        if diff < 0: diff += 2 * math.pi
    else:
        # Wrap clockwise
        if diff > 0: diff -= 2 * math.pi
        
    steps = max(2, int(abs(diff) / step) + 1)
    
    points = []
    for i in range(1, steps + 1):
        a = angle_in + diff * (i / steps)
        points.append(Point(
            x=center.x + math.cos(a) * r_safe,
            y=center.y + math.sin(a) * r_safe
        ))
    return points

def visualize_arc_points_with_arrows(center: Point, radius: float, points: List[Point]):
    """
    Plots the clearance circle and the arc path with directional arrows to confirm CCW motion.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Plot the clearance circle
    circle = plt.Circle((center.x, center.y), radius, color='red', fill=False, linestyle='--', label='Clearance Circle')
    ax.add_patch(circle)
    
    # 2. Extract coordinates
    x = np.array([p.x for p in points])
    y = np.array([p.y for p in points])
    
    # 3. Plot the line path
    ax.plot(x, y, color='blue', marker='o', label='Discretized Arc')
    
    # 4. FIX: Add directional arrows
    # Calculate the direction of the path segments
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Use quiver to plot arrows at the midpoint of each segment
    mid_x = (x[:-1] + x[1:]) / 2
    mid_y = (y[:-1] + y[1:]) / 2
    
    # quiver(X, Y, U, V)
    ax.quiver(mid_x, mid_y, dx, dy, color='blue', angles='xy', scale_units='xy', scale=1, width=0.005)
    
    # Formatting
    ax.plot(center.x, center.y, marker='x', color='black', label='Center')
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    plt.title("Arc Discretization (Arrows indicate Direction)")
    
    # Margin
    margin = radius * 1.5
    ax.set_xlim(center.x - margin, center.x + margin)
    ax.set_ylim(center.y - margin, center.y + margin)
    
    plt.show()

if False:
    center_pt = Point(0, 0)
    clearance_radius = 5.0

    # Example: 180-degree counter-clockwise turn starting from the bottom
    # Step is set large (0.5 rad = ~28 deg) to make r_safe inflation obvious
    generated_points = generate_arc_points(
        center=center_pt,
        radius=clearance_radius,
        angle_in=math.pi/2,  
        angle_out=0,   
        is_left=True,           
        step=0.5                 
    )

    # visualize_arc_points(center_pt, clearance_radius, generated_points)
    visualize_arc_points_with_arrows(center_pt, clearance_radius, generated_points)
     
def calculate_centroid(polygon: Polygon) -> Point:
    x_coords = [p.x for p in polygon]
    y_coords = [p.y for p in polygon]
    return Point(sum(x_coords) / len(polygon), sum(y_coords) / len(polygon))

# ---------------------------------------------------------------------------
# Pipeline Algorithms
# ---------------------------------------------------------------------------

def rdp(path: List[Point], epsilon: float) -> List[Point]:
    if len(path) < 3:
        return path

    dmax = 0.0
    index = 0
    end = len(path) - 1

    for i in range(1, end):
        d = point_line_distance(path[i], path[0], path[end])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        left_rec = rdp(path[:index + 1], epsilon)
        right_rec = rdp(path[index:], epsilon)
        return left_rec[:-1] + right_rec
    else:
        return [path[0], path[end]]

def ccw(A: Point, B: Point, C: Point) -> bool:
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

def intersect(A: Point, B: Point, C: Point, D: Point) -> bool:
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def get_intersection_point(A: Point, B: Point, C: Point, D: Point) -> Point:
    a1, b1 = B.y - A.y, A.x - B.x
    c1 = a1 * A.x + b1 * A.y
    a2, b2 = D.y - C.y, C.x - D.x
    c2 = a2 * C.x + b2 * C.y
    det = a1 * b2 - a2 * b1
    if det == 0:
        return Point((A.x + B.x) / 2.0, (A.y + B.y) / 2.0)
    return Point((b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det)

def extract_portals(path: List[Point], mesh: Dict[str, np.ndarray]) -> List[Portal]:
    """Raycasts the RDP segments against the CDT edges to construct the exact valid corridor."""
    portals = [Portal(left=path[0], right=path[0])]
    vertices = [Point(x, y) for x, y in mesh['vertices']]
    
    edges = set()
    for tri in mesh['triangles']:
        for i in range(3):
            u, v = tri[i], tri[(i+1)%3]
            edges.add((min(u, v), max(u, v)))

    for i in range(len(path) - 1):
        A = path[i]
        B = path[i+1]
        
        intersections = []
        for u_idx, v_idx in edges:
            C = vertices[u_idx]
            D = vertices[v_idx]
            
            if intersect(A, B, C, D):
                pt = get_intersection_point(A, B, C, D)
                dist = distance(A, pt)
                intersections.append((dist, C, D))
        
        intersections.sort(key=lambda x: x[0])
        
        for _, C, D in intersections:
            cross = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)
            if cross > 0:
                portals.append(Portal(left=C, right=D))
            else:
                portals.append(Portal(left=D, right=C))
                
    portals.append(Portal(left=path[-1], right=path[-1]))
    return portals


def modified_funnel(portals: List[Portal], start: Point, end: Point, radius: float) -> List[Point]:
    if not portals:
        return [start, end]

    smoothed_path = [start]

    apex = start
    left_vertex = portals[0].left
    right_vertex = portals[0].right

    left_index = 0
    right_index = 0

    i = 1
    while i < len(portals):
        new_left = portals[i].left
        new_right = portals[i].right

        if new_left == end and new_right == end:
            end_angle = math.atan2(end.y - apex.y, end.x - apex.x)

            # Check if direct line to end breaks the left bound
            if apex != left_vertex:
                current_left_angle = calculate_tangent_angle(apex, left_vertex, radius, is_left=True)
                if angle_diff(end_angle, current_left_angle) > 0.0:
                    Tin = get_tangent_point(apex, left_vertex, radius, is_left=True)
                    smoothed_path.append(Tin)
                    
                    angle_in = math.atan2(Tin.y - left_vertex.y, Tin.x - left_vertex.x)
                    dist_to_next = distance(left_vertex, end)
                    
                    if dist_to_next > radius:
                        theta = math.atan2(end.y - left_vertex.y, end.x - left_vertex.x)
                        # Asymmetric bitangent from r to 0
                        angle_out = theta - math.acos(radius / dist_to_next)
                        arc_pts = generate_arc_points(left_vertex, radius, angle_in, angle_out, is_left=True)
                        smoothed_path.extend(arc_pts)
                        apex = arc_pts[-1]
                    else:
                        apex = Tin
                    
                    left_vertex = apex
                    right_vertex = apex
                    apex_index = left_index
                    left_index = apex_index
                    right_index = apex_index
                    i = apex_index + 1
                    continue
            
            # Check if direct line to end breaks the right bound
            if apex != right_vertex:
                current_right_angle = calculate_tangent_angle(apex, right_vertex, radius, is_left=False)
                if angle_diff(end_angle, current_right_angle) < 0.0:
                    Tin = get_tangent_point(apex, right_vertex, radius, is_left=False)
                    smoothed_path.append(Tin)
                    
                    angle_in = math.atan2(Tin.y - right_vertex.y, Tin.x - right_vertex.x)
                    dist_to_next = distance(right_vertex, end)
                    
                    if dist_to_next > radius:
                        theta = math.atan2(end.y - right_vertex.y, end.x - right_vertex.x)
                        # Asymmetric bitangent from r to 0
                        angle_out = theta + math.acos(radius / dist_to_next)
                        arc_pts = generate_arc_points(right_vertex, radius, angle_in, angle_out, is_left=False)
                        smoothed_path.extend(arc_pts)
                        apex = arc_pts[-1]
                    else:
                        apex = Tin
                        
                    left_vertex = apex
                    right_vertex = apex
                    apex_index = right_index
                    left_index = apex_index
                    right_index = apex_index
                    i = apex_index + 1
                    continue
            
            # Direct path is clear; break loop and append end coordinate
            break

        # -------------------------------------------------------------------
        # 1. Update Right Vertex
        # -------------------------------------------------------------------
        if apex == right_vertex:
            right_vertex = new_right
            right_index = i
        else:
            new_right_angle = calculate_tangent_angle(apex, new_right, radius, is_left=False)
            current_right_angle = calculate_tangent_angle(apex, right_vertex, radius, is_left=False)

            if angle_diff(new_right_angle, current_right_angle) >= 0.0:
                if apex == left_vertex:
                    right_vertex = new_right
                    right_index = i
                else:
                    current_left_angle = calculate_tangent_angle(apex, left_vertex, radius, is_left=True)

                    if angle_diff(new_right_angle, current_left_angle) > 0.0:
                        Tin = get_tangent_point(apex, left_vertex, radius, is_left=True)
                        smoothed_path.append(Tin)

                        angle_in = math.atan2(Tin.y - left_vertex.y, Tin.x - left_vertex.x)
                        dist_to_next = distance(left_vertex, new_right)

                        if dist_to_next > radius:
                            theta = math.atan2(new_right.y - left_vertex.y, new_right.x - left_vertex.x)
                            phi = math.asin(radius / dist_to_next)
                            angle_out = theta - phi

                            arc_pts = generate_arc_points(left_vertex, radius, angle_in, angle_out, is_left=True)
                            smoothed_path.extend(arc_pts)
                            apex = arc_pts[-1]
                        else:
                            apex = Tin

                        left_vertex = apex
                        right_vertex = apex

                        apex_index = left_index
                        left_index = apex_index
                        right_index = apex_index
                        i = apex_index + 1 
                        continue
                    else:
                        right_vertex = new_right
                        right_index = i

        # -------------------------------------------------------------------
        # 2. Update Left Vertex
        # -------------------------------------------------------------------
        if apex == left_vertex:
            left_vertex = new_left
            left_index = i
        else:
            new_left_angle = calculate_tangent_angle(apex, new_left, radius, is_left=True)
            current_left_angle = calculate_tangent_angle(apex, left_vertex, radius, is_left=True)

            if angle_diff(new_left_angle, current_left_angle) <= 0.0:
                if apex == right_vertex:
                    left_vertex = new_left
                    left_index = i
                else:
                    current_right_angle = calculate_tangent_angle(apex, right_vertex, radius, is_left=False)

                    if angle_diff(new_left_angle, current_right_angle) < 0.0:
                        Tin = get_tangent_point(apex, right_vertex, radius, is_left=False)
                        smoothed_path.append(Tin)

                        angle_in = math.atan2(Tin.y - right_vertex.y, Tin.x - right_vertex.x)
                        dist_to_next = distance(right_vertex, new_left)

                        if dist_to_next > radius:
                            theta = math.atan2(new_left.y - right_vertex.y, new_left.x - right_vertex.x)
                            phi = math.asin(radius / dist_to_next)
                            angle_out = theta + phi

                            arc_pts = generate_arc_points(right_vertex, radius, angle_in, angle_out, is_left=False)
                            smoothed_path.extend(arc_pts)
                            apex = arc_pts[-1]
                        else:
                            apex = Tin

                        left_vertex = apex
                        right_vertex = apex

                        apex_index = right_index
                        left_index = apex_index
                        right_index = apex_index
                        i = apex_index + 1 
                        continue
                    else:
                        left_vertex = new_left
                        left_index = i

        i += 1

    smoothed_path.append(end)
    return [*smoothed_path[:-2], smoothed_path[-1]]


# ---------------------------------------------------------------------------
# Mesh Preparation & Generation
# ---------------------------------------------------------------------------

def append_polygon_to_pslg(
    polygon: Polygon, 
    current_vertices: List[Point], 
    current_segments: List[Tuple[int, int]]
) -> Tuple[List[Point], List[Tuple[int, int]]]:
    new_vertices = list(current_vertices)
    new_segments = list(current_segments)
    start_idx = len(new_vertices)
    num_points = len(polygon)
    
    for i, pt in enumerate(polygon):
        new_vertices.append(pt)
        new_segments.append((start_idx + i, start_idx + ((i + 1) % num_points)))
        
    return new_vertices, new_segments

def prepare_pslg(input_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    vertices: List[Point] = []
    segments: List[Tuple[int, int]] = []
    holes: List[Point] = []
    
    boundary = [Point(p[0], p[1]) for p in input_data["boundary"]]
    obstacles = [[Point(p[0], p[1]) for p in obs] for obs in input_data["obstacles"]]
    
    vertices, segments = append_polygon_to_pslg(boundary, vertices, segments)
    
    for obs in obstacles:
        vertices, segments = append_polygon_to_pslg(obs, vertices, segments)
        holes.append(calculate_centroid(obs))
        
    pslg_dict = {
        "vertices": np.array([[p.x, p.y] for p in vertices], dtype=np.float64),
        "segments": np.array(segments, dtype=np.int32)
    }
    if holes:
        pslg_dict["holes"] = np.array([[p.x, p.y] for p in holes], dtype=np.float64)
        
    return pslg_dict

def generate_cdt(pslg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return tr.triangulate(pslg, 'p')

# ---------------------------------------------------------------------------
# Pipeline Integration
# ---------------------------------------------------------------------------

def smooth_path(raw_path: List[Point], mesh: Dict[str, np.ndarray], radius: float) -> Tuple[List[Point], List[Point]]:
    if not raw_path:
        return [], []
    
    epsilon = radius / 2.0
    rdp_path = rdp(raw_path, epsilon)
    
    portals = extract_portals(rdp_path, mesh)
    
    final_smoothed_path = modified_funnel(portals, rdp_path[0], rdp_path[-1], radius)
    
    return rdp_path, final_smoothed_path

# ---------------------------------------------------------------------------
# I/O & Visualization (Side Effects)
# ---------------------------------------------------------------------------

def close_polygon(polygon: List[List[float]]) -> np.ndarray:
    return np.array(polygon + [polygon[0]])

def visualize_navmesh(
    cdt_mesh: Dict[str, np.ndarray], 
    original_data: Dict[str, Any], 
    rdp_path: List[Point],
    smoothed_path: List[Point],
    radius: float
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.28)
    
    # 1. Base Mesh
    vertices = cdt_mesh['vertices']
    triangles = cdt_mesh['triangles']
    ax.triplot(vertices[:, 0], vertices[:, 1], triangles, color='lightgray', linewidth=0.5, zorder=1)
    
    # 2. Boundary
    b_pts = close_polygon(original_data["boundary"])
    line_boundary, = ax.plot(b_pts[:, 0], b_pts[:, 1], color='green', linewidth=2.5, label='Original Boundary', zorder=2)
    
    # 3. Obstacles
    lines_obstacles = []
    for obs in original_data["obstacles"]:
        o_pts = close_polygon(obs)
        lo, = ax.plot(o_pts[:, 0], o_pts[:, 1], color='darkred', linewidth=2.5, label='_nolegend_', zorder=2)
        lines_obstacles.append(lo)
    ax.plot([], [], color='darkred', linewidth=2.5, label='Original Obstacles')

    # 4. Unsmoothed Path
    raw_path_array = np.array(original_data["path"])
    line_unsmoothed, = ax.plot(raw_path_array[:, 0], raw_path_array[:, 1], color='blue', linestyle='--',
                               marker='o', linewidth=1.5, markersize=4, label='Unsmoothed Path', zorder=3)
                               
    # 5. RDP Path
    rdp_path_array = np.array([[p.x, p.y] for p in rdp_path])
    line_rdp, = ax.plot(rdp_path_array[:, 0], rdp_path_array[:, 1], color='orange', linestyle='-',
                             marker='^', linewidth=2.0, markersize=5, label='RDP Path', zorder=4)
                             
    # 6. Clearance Circles on Mesh Vertices
    circles = []
    for v in vertices:
        circle_patch = Circle((v[0], v[1]), radius, color='cyan', fill=False, linestyle=':', linewidth=1.0, zorder=3.5)
        ax.add_patch(circle_patch)
        circles.append(circle_patch)
    ax.plot([], [], color='cyan', linestyle=':', linewidth=1.0, label='Vertex Circles')
    
    # 7. Smoothed Path
    smooth_path_array = np.array([[p.x, p.y] for p in smoothed_path])
    line_smoothed, = ax.plot(smooth_path_array[:, 0], smooth_path_array[:, 1], color='magenta', 
                             marker='s', linewidth=2.5, markersize=5, label='Smoothed Path', zorder=5)
    
    # UI Setup
    rax = fig.add_axes([0.02, 0.35, 0.25, 0.3])
    rax.set_title('Toggle Visibility')
    labels = ['Original Boundary', 'Original Obstacles', 'Unsmoothed Path', 'RDP Path', 'Vertex Circles', 'Smoothed Path']
    visibility = [True, True, True, True, True, True]
    
    check = CheckButtons(rax, labels, visibility)
    
    def toggle_visibility(label: str) -> None:
        if label == 'Original Boundary':
            line_boundary.set_visible(not line_boundary.get_visible())
        elif label == 'Original Obstacles':
            for lo in lines_obstacles:
                lo.set_visible(not lo.get_visible())
        elif label == 'Unsmoothed Path':
            line_unsmoothed.set_visible(not line_unsmoothed.get_visible())
        elif label == 'RDP Path':
            line_rdp.set_visible(not line_rdp.get_visible())
        elif label == 'Vertex Circles':
            for c in circles:
                c.set_visible(not c.get_visible())
        elif label == 'Smoothed Path':
            line_smoothed.set_visible(not line_smoothed.get_visible())
            
        fig.canvas.draw_idle()


    check.on_clicked(toggle_visibility)
    fig.check_buttons = check 
    
    ax.set_title("Path Smoothing Pipeline & NavMesh Viewer")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.show()

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    payload = {
      "boundary": [
        [
          102.0,
          577.0
        ],
        [
          17.0,
          196.0
        ],
        [
          279.0,
          42.0
        ],
        [
          532.0,
          49.0
        ],
        [
          621.0,
          315.0
        ],
        [
          590.0,
          612.0
        ]
      ],
      "obstacles": [
        [
          [
            215.0,
            477.0
          ],
          [
            214.0,
            376.0
          ],
          [
            292.0,
            361.0
          ],
          [
            326.0,
            494.0
          ]
        ],
        [
          [
            271.0,
            263.0
          ],
          [
            313.0,
            162.0
          ],
          [
            448.0,
            175.0
          ],
          [
            462.0,
            265.0
          ]
        ]
      ],
      "path": [
        [
          132.0,
          457.0
        ],
        [
          136.0,
          323.0
        ],
        [
          133.0,
          368.0
        ],
        [
          163.0,
          288.0
        ],
        [
          294.0,
          303.0
        ],
        [
          241.0,
          306.0
        ],
        [
          429.0,
          338.0
        ],
        [
          506.0,
          352.0
        ],
        [
          534.0,
          191.0
        ]
      ]
    }

    raw_path_points = [Point(p[0], p[1]) for p in payload["path"]]

    pslg_data = prepare_pslg(payload)
    mesh = generate_cdt(pslg_data)

    radius_val = 30
    rdp_path, final_path = smooth_path(raw_path_points, mesh, radius=radius_val)

    visualize_navmesh(mesh, payload, rdp_path, final_path, radius_val)
