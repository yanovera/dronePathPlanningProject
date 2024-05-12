import numpy as np
from scipy.spatial import KDTree

from objects import Line


def line_sphere_intersection_length(line, sphere):
    """
    Calculate the length of intersection between a line and a sphere.

    Args:
        line (Line): the line object.
        sphere (Sphere): the sphere object.

    Returns:
        float: the intersection length, 0 if no intersection.
    """
    line_dir = line.end - line.start
    a = np.dot(line_dir, line_dir)
    b = 2 * np.dot(line_dir, line.start - sphere.center)
    c = np.dot(sphere.center, sphere.center) + np.dot(line.start, line.start) - 2 * np.dot(sphere.center, line.start) - sphere.radius ** 2

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        # No intersection
        return 0
    elif discriminant == 0:
        # One intersection
        t = -b / (2 * a)
        intersection = line.start + t * line_dir
        return np.linalg.norm(intersection - line.start) if np.dot(line_dir, intersection - line.start) > 0 else 0
    else:
        # Two intersections
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        intersection1 = line.start + t1 * line_dir
        intersection2 = line.start + t2 * line_dir
        intersections = [point for point in [intersection1, intersection2] if np.dot(line_dir, point - line.start) > 0 and np.dot(-line_dir, point - line.end) > 0]
        if not intersections:
            return 0
        if len(intersections) == 1:
            return np.linalg.norm(intersections[0] - line.start)
        return np.linalg.norm(intersections[1] - intersections[0])


def add_point_to_rrt_star(height_map, max_altitude, nodes, parents, costs, new_node, spheres, height_margin):
    nearest_nodes, nearest_indices = near_nodes(nodes, new_node)
    nearest_costs = [costs[i] for i in nearest_indices]
    parent_index, cost_to_come = choose_parent(height_map, nearest_nodes, nearest_indices, nearest_costs, new_node, spheres, height_margin)
    if parent_index is not None:
        nodes.append(new_node)
        parents.append(parent_index)
        costs.append(cost_to_come)
        nodes, parents, costs = rewire(height_map, nodes, parents, costs, new_node, nearest_nodes, nearest_indices, nearest_costs, cost_to_come, spheres, height_margin)
    return nodes, parents, costs


def choose_parent(height_map, nearest_nodes, nearest_indices, nearest_costs, new_node, spheres, height_margin):
    # Initialize the minimum cost to infinity
    min_cost = np.inf
    # Initialize the parent index to None
    parent_index = None
    # For each nearest node
    for nearest_node, nearest_index, nearest_cost in zip(nearest_nodes, nearest_indices, nearest_costs):
        # Calculate the cost to come to the new node from the current nearest node
        cost_to_come = nearest_cost + penalized_cost(nearest_node, new_node, spheres)
        # If the cost to come is less than the minimum cost and the path to the new node is collision-free
        if cost_to_come < min_cost and collision_free(nearest_node, new_node, height_map, height_margin):
            # Update the minimum cost and the parent index
                min_cost = cost_to_come
                parent_index = nearest_index
    return parent_index, min_cost


def rewire(height_map, nodes, parents, costs, new_node, nearest_nodes, nearest_indices, nearest_costs, cost_to_come, spheres, height_margin):
    # For each nearest node
    for nearest_node, nearest_index, nearest_cost in zip(nearest_nodes, nearest_indices, nearest_costs):
        # If the cost to come to the nearest node through the new node is less than its current cost and the path to the new node is collision-free
        if cost_to_come + penalized_cost(new_node, nearest_node, spheres) < nearest_cost and collision_free(nearest_node, new_node, height_map, height_margin):
            # Update the parent of the nearest node to be the new node
            parents[nearest_index] = len(nodes) - 1  # The index of the new node is the last one
            # Update the cost of the nearest node
            costs[nearest_index] = cost_to_come + np.linalg.norm(np.array(nearest_node) - np.array(new_node))

    return nodes, parents, costs


def generate_new_node(nearest_node, point):
    direction = (point - nearest_node) / np.linalg.norm(point - nearest_node)
    step_size = min(np.linalg.norm(point - nearest_node), 10)
    return nearest_node + direction * step_size



def generate_random_node(height_map):
    x = np.random.uniform(0, height_map.shape[1] - 1)
    y = np.random.uniform(0, height_map.shape[0] - 1)
    z = np.random.uniform(0, np.max(height_map))
    return [x, y, z]

def find_nearest_node(nodes, new_node):
    tree = KDTree(nodes)
    dist, ind = tree.query([new_node])
    return ind[0]

def steer_towards(nearest_node, random_node, max_dist):
    vector = np.array(random_node) - np.array(nearest_node)
    dist = np.linalg.norm(vector)
    if dist < max_dist:
        return random_node
    else:
        return (np.array(nearest_node) + max_dist * (vector / dist)).tolist()


def within_bounds(node, height_map):
    x, y, z = node
    return 0 <= x < height_map.shape[1] and 0 <= y < height_map.shape[0] and 0 <= z <= np.max(height_map)

def near_nodes(nodes, new_node, num_nodes=100):
    # Get the distance to the new node for each node
    distances = [np.linalg.norm(np.array(node) - np.array(new_node)) for node in nodes]
    # Get the indices of the closest nodes
    nearest_indices = np.argsort(distances)[:num_nodes]
    # Get the nearest nodes
    nearest_nodes = [nodes[i] for i in nearest_indices]
    return nearest_nodes, nearest_indices


def collision_free(node1, node2, height_map, height_margin):
    # Get the line of points between the two nodes
    line_points = bresenham_line(node1, node2)
    # Check each point on the line
    for point in line_points:
        x, y, z = point
        # If the point is out of bounds of the height map or below the height map, return False
        if x < 0 or y < 0 or x >= height_map.shape[1] or y >= height_map.shape[0] or z < height_map[int(y), int(x)] + height_margin:
            return False
    # If none of the points on the line are below the height map, return True
    return True

def bresenham_line(node1, node2, tolerance=0.5):
    x1, y1, z1 = [int(round(coord)) for coord in node1]
    x2, y2, z2 = [int(round(coord)) for coord in node2]
    points = []
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    x_inc = 1 if dx > 0 else -1
    y_inc = 1 if dy > 0 else -1
    z_inc = 1 if dz > 0 else -1
    dx = abs(dx)
    dy = abs(dy)
    dz = abs(dz)
    if dx >= dy and dx >= dz:
        dy *= 2
        dz *= 2
        err_1 = dy - dx
        err_2 = dz - dx
        while np.linalg.norm(np.array([x1, y1, z1]) - np.array([x2, y2, z2])) > tolerance:
            points.append((x1, y1, z1))
            if err_1 >= 0:
                y1 += y_inc
                err_1 -= dx * 2
            if err_2 >= 0:
                z1 += z_inc
                err_2 -= dx * 2
            err_1 += dy
            err_2 += dz
            x1 += x_inc
    elif dy >= dx and dy >= dz:
        dx *= 2
        dz *= 2
        err_1 = dx - dy
        err_2 = dz - dy
        while np.linalg.norm(np.array([x1, y1, z1]) - np.array([x2, y2, z2])) > tolerance:
            points.append((x1, y1, z1))
            if err_1 >= 0:
                x1 += x_inc
                err_1 -= dy * 2
            if err_2 >= 0:
                z1 += z_inc
                err_2 -= dy * 2
            err_1 += dx
            err_2 += dz
            y1 += y_inc
    else:
        dx *= 2
        dy *= 2
        err_1 = dy - dz
        err_2 = dx - dz
        while np.linalg.norm(np.array([x1, y1, z1]) - np.array([x2, y2, z2])) > tolerance:
            points.append((x1, y1, z1))
            if err_1 >= 0:
                y1 += y_inc
                err_1 -= dz * 2
            if err_2 >= 0:
                x1 += x_inc
                err_2 -= dz * 2
            err_1 += dy
            err_2 += dx
            z1 += z_inc
    points.append((x2, y2, z2))  # Ensure the final point is added
    return points


def penalized_cost(node1, node2, spheres):
    """
    Calculate the cost of the path from node1 to node2 considering the penalty for passing through spheres.

    Args:
        node1 (list): the coordinates of the start node (x, y, z).
        node2 (list): the coordinates of the end node (x, y, z).
        spheres (list): a list of Sphere objects representing the spheres in the environment.

    Returns:
        float: the cost of the path from node1 to node2 with penalties for passing through spheres.
    """

    node1 = np.array(node1)
    node2 = np.array(node2)
    line = Line(node1, node2)
    cost = np.linalg.norm(node2 - node1)
    for sphere in spheres:
        intersection_length = line_sphere_intersection_length(line, sphere)
        cost += intersection_length * sphere.penalty_factor
    return cost
