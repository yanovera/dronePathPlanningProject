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


def add_point_to_rrt_star(height_map, nodes, parents, costs, new_node, spheres, height_margin):
    """
    Adds a new node to the RRT* tree and updates the tree structure.

    Args:
        height_map (np.array): A 2D numpy array representing the terrain over which the drone is flying.
        nodes (list): A list of nodes in the RRT* tree. Each node is a numpy array of three coordinates [x, y, z].
        parents (list): A list of indices representing the parent node of each node in the 'nodes' list.
        costs (list): A list of costs associated with each node, representing the cost of the path from the start to that node.
        new_node (np.array): The new node to be added to the RRT* tree, represented as a numpy array of three coordinates [x, y, z].
        spheres (list): A list of Sphere objects representing moving obstacles in the 3D space.
        height_margin (float): The minimum altitude margin above the terrain.

    Returns:
        nodes (list): The updated list of nodes in the RRT* tree.
        parents (list): The updated list of parent indices.
        costs (list): The updated list of costs.

    The function finds the nodes that are near to the new node, chooses the best parent for the new node, and adds the new node to the RRT* tree. It then updates the tree structure by rewiring the tree to ensure that the path from the start to each node has the minimum cost. The function returns the updated lists of nodes, parents, and costs.
    """
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
    """
    Chooses the best parent for a new node in the RRT* tree.

    Args:
        height_map (np.array): A 2D numpy array representing the terrain over which the drone is flying.
        nearest_nodes (list): A list of nodes in the RRT* tree that are near to the new node. Each node is a numpy array of three coordinates [x, y, z].
        nearest_indices (list): A list of indices representing the position of each nearest node in the 'nodes' list.
        nearest_costs (list): A list of costs associated with each nearest node, representing the cost of the path from the start to that node.
        new_node (np.array): The new node to be added to the RRT* tree, represented as a numpy array of three coordinates [x, y, z].
        spheres (list): A list of Sphere objects representing moving obstacles in the 3D space.
        height_margin (float): The minimum altitude margin above the terrain.

    Returns:
        parent_index (int): The index of the chosen parent node in the 'nodes' list.
        min_cost (float): The cost of the path from the start to the new node through the chosen parent node.

    The function iterates over each nearest node and calculates the cost to come to the new node from the current nearest node. If the cost to come is less than the minimum cost found so far and the path to the new node is collision-free, the function updates the minimum cost and the parent index. The function returns the index of the chosen parent node and the cost of the path from the start to the new node.
    """
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
    """
        This function rewires the nodes in a graph based on the cost to come to the nearest nodes through a new node.

        Parameters:
        height_map (np.array): The 2D array representing the height map.
        nodes (list): The list of nodes in the graph.
        parents (list): The list of parent nodes corresponding to each node.
        costs (list): The list of costs corresponding to each node.
        new_node (tuple): The new node to be added to the graph.
        nearest_nodes (list): The list of nearest nodes to the new node.
        nearest_indices (list): The indices of the nearest nodes in the nodes list.
        nearest_costs (list): The costs of the nearest nodes.
        cost_to_come (float): The cost to come to the new node.
        spheres (list): The list of spheres representing obstacles.
        height_margin (float): The margin for collision check in the height dimension.

        Returns:
        nodes (list): The updated list of nodes in the graph.
        parents (list): The updated list of parent nodes.
        costs (list): The updated list of costs.
        """
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
    """
    This function generates a new node in the direction of a given point from the nearest node.

    Parameters:
    nearest_node (np.array): The coordinates of the nearest node.
    point (np.array): The coordinates of the point towards which the new node is to be generated.

    Returns:
    new_node (np.array): The coordinates of the new node. The new node is generated in the direction of the given point from the nearest node. The distance between the new node and the nearest node is the minimum of the distance between the point and the nearest node and a step size of 10.
    """
    direction = (point - nearest_node) / np.linalg.norm(point - nearest_node)
    step_size = min(np.linalg.norm(point - nearest_node), 10)
    return nearest_node + direction * step_size


def find_nearest_node(nodes, new_node):
    """
    Finds the nearest node in a given set of nodes to a new node using the KDTree method.

    Parameters:
    nodes (array-like): An array-like structure of n-dimensional points representing the nodes.
    new_node (array-like): An array-like structure representing the new node.

    Returns:
    int: The index of the nearest node in the original nodes list.

    Example:
    >> nodes = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    >> new_node = [3,4]
    >> find_nearest_node(nodes, new_node)
    0
    """
    tree = KDTree(nodes)
    dist, ind = tree.query([new_node])
    return ind[0]


def near_nodes(nodes, new_node, num_nodes=100):
    """
    Finds the nearest nodes in a given set of nodes to a new node.

    Parameters:
    nodes (list of array-like): A list of n-dimensional points representing the nodes.
    new_node (array-like): An array-like structure representing the new node.
    num_nodes (int, optional): The number of nearest nodes to return. Default is 100.

    Returns:
    tuple: A tuple containing two lists:
        - nearest_nodes (list of array-like): The nearest nodes to the new node.
        - nearest_indices (list of int): The indices of the nearest nodes in the original nodes list.

    Example:
    >> nodes = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    >> new_node = [3,4]
    >> near_nodes(nodes, new_node, 3)
    ([[2, 3], [5, 4], [4, 7]], [0, 1, 3])
    """
    # Get the distance to the new node for each node
    distances = [np.linalg.norm(np.array(node) - np.array(new_node)) for node in nodes]
    # Get the indices of the closest nodes
    nearest_indices = np.argsort(distances)[:num_nodes]
    # Get the nearest nodes
    nearest_nodes = [nodes[i] for i in nearest_indices]
    return nearest_nodes, nearest_indices


def collision_free(node1, node2, height_map, height_margin):
    """
    Checks if the path between two nodes is collision-free considering a height map and a height margin.

    Parameters:
    node1 (array-like): An array-like structure representing the first node (x, y, z).
    node2 (array-like): An array-like structure representing the second node (x, y, z).
    height_map (2D array-like): A 2D array-like structure representing the height map.
    height_margin (float): The margin to be considered above the height map for collision.

    Returns:
    bool: True if the path between the nodes is collision-free, False otherwise.
    """
    # Get the line of points between the two nodes
    line_points = bresenham_line(node1, node2)
    # Check each point on the line
    for point in line_points:
        x, y, z = point
        # If the point is out of bounds of the height map or below the height map, return False
        if x < 0 or y < 0 or x >= height_map.shape[0] or y >= height_map.shape[1] or z < get_height_map(x, y, height_map, height_margin):
            return False
    # If none of the points on the line are below the height map, return True
    return True


def get_height_map(x, y, height_map, height_margin):
    return max(height_map[int(x), int(y)],
               height_map[int(x + 1), int(y)] if x < height_map.shape[0] - 1 else 0,
               height_map[int(x - 1), int(y)] if x > 0 else 0,
               height_map[int(x), int(y + 1)] if y < height_map.shape[1] - 1 else 0,
               height_map[int(x + 1), int(y + 1)] if x < height_map.shape[0] - 1 and y < height_map.shape[1] - 1 else 0,
               height_map[int(x - 1), int(y + 1)] if x > 0 and y < height_map.shape[1] - 1 else 0,
               height_map[int(x), int(y - 1)] if y > 0 else 0,
               height_map[int(x + 1), int(y - 1)] if x < height_map.shape[0] - 1 and y > 0 else 0,
               height_map[int(x - 1), int(y - 1)] if x > 0 and y > 0 else 0,
               ) + height_margin


def bresenham_line(node1, node2, tolerance=0.5):
    """
    Generates a line between two nodes in a 3D space using the Bresenham's line algorithm.

    Parameters:
    node1 (array-like): An array-like structure representing the first node (x, y, z).
    node2 (array-like): An array-like structure representing the second node (x, y, z).
    tolerance (float, optional): The tolerance for the distance between the final point and the target node. Default is 0.5.

    Returns:
    list: A list of tuples representing the points on the line between the two nodes.

    Example:
    >> node1 = [2, 3, 4]
    >> node2 = [5, 6, 7]
    >> bresenham_line(node1, node2, 0.5)
    [(2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7)]
    """
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
