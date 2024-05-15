import sys

import numpy as np
import rasterio
from scipy.spatial.distance import cdist
from vispy import app, scene
from vispy.scene import visuals

from objects import Sphere
from utils import find_nearest_node, generate_new_node, collision_free, add_point_to_rrt_star
from parameters import *


def sample_point(height_map, max_altitude, min_altitude=0):
    """
    Generates a random point in a 3D space within the given ranges.

    Args:
        height_map (np.array): A 2D numpy array representing the terrain over which the drone is flying.
        max_altitude (float): The maximum altitude at which the drone can fly.
        min_altitude (float, optional): The minimum altitude at which the drone can fly. Defaults to 0.

    Returns:
        np.array: A numpy array containing the x, y, and z coordinates of the generated point.

    The function uses a uniform random distribution to generate a 3D point (x, y, z).
    The x and y coordinates are determined by the dimensions of the height_map, and the z coordinate (altitude) is a random value between min_altitude and max_altitude.
    """
    x = np.random.uniform(0, height_map.shape[1])
    y = np.random.uniform(0, height_map.shape[0])
    z = np.random.uniform(min_altitude, max_altitude)
    return np.array([x, y, z])


def update(ev):
    """
        Updates the drone's position and the pathfinding algorithm at each time step.

        Args:
            ev: An event trigger for the function. This is typically a timer event that triggers the function at regular intervals.

        This function is called at regular intervals to simulate the drone's movement and update the pathfinding algorithm. It checks the drone's altitude and position, updates the destination of the obstacles (spheres), generates a new RRT* tree and finds a new path if necessary, moves the drone towards the next waypoint on the path, and checks if the drone has reached the target or gone out of bounds.

        The function uses several global variables:
        - path_scatter, path: Used to visualize the path.
        - t: The current time step.
        - height_map, max_altitude: Used to generate random points and check the drone's altitude.
        - drone_speed_factor: The drone's speed factor
        - nodes, parents, costs: Used in the RRT* algorithm.
        - spheres: The moving obstacles.
        - spheres: The spheres' speed factor.
        - target, start: The start and target points for the drone.
        - current_pos, current_point: The drone's current position.
        - timer: The timer event that triggers the function.
        - steering_noise_sigma: The standard deviation of the steering noise.
        - tolerance: The distance tolerance for reaching a waypoint or the target.
        - height_margin: The minimum altitude margin above the terrain.
        """
    global path_scatter, path, t, height_map, max_altitude, drone_speed_factor, nodes, parents, costs, spheres, spheres_speed_factor, target, start,\
        current_pos, current_point, timer, steering_noise_sigma, tolerance, height_margin
    t += 1.0
    if current_pos[2] < height_map[int(current_pos[0]), int(current_pos[1])] + height_margin:
        print(f'warning: drone is below height margin at t={t}.')
        fixed_current_pos = np.array(
            [current_pos[0], current_pos[1], height_map[int(current_pos[0]), int(current_pos[1])] + height_margin])
    else:
        fixed_current_pos = current_pos
    if t % 200 == 1:
        for sphere in spheres:
            sphere.destination = sample_point(height_map, max_altitude, min_altitude=sphere.min_altitude)
        nodes, parents, costs = generate_rrt_star(height_map, max_altitude, num_nodes, fixed_current_pos, spheres, target, height_margin)

        path = find_path(nodes, parents, target)
        path_scatter.set_data(path, edge_color='white', face_color=(1, 1, 1, .5), size=10)
        path_lines.set_data(pos=path, color='red', width=2)
    if np.linalg.norm(np.array(target) - current_pos) < tolerance:
        print(f'drone reached the target at t={t}.')
        timer.stop()
    if len(path) > 1:
        waypoint = path[1]
        current_distance = np.linalg.norm(path[1] - current_pos)
        if current_distance < tolerance and len(path) > 2:
            waypoint = path[2]
            path = path[1:]

        steering_noise = np.random.normal(scale=steering_noise_sigma, size=3)
        current_direction = (waypoint - current_pos) / np.linalg.norm(waypoint - current_pos)
        current_pos = current_pos + current_direction * drone_speed_factor + steering_noise

        current_point.set_data(pos=np.array([current_pos]), edge_color='blue', face_color='blue', size=10)
        if (current_pos[0] < 0 or current_pos[1] < 0 or current_pos[0] >= height_map.shape[0] or current_pos[1] >= height_map.shape[1]):
            print(f'drone went out of bounds at t={t}.')
            timer.stop()
        if (current_pos[2] < height_map[int(current_pos[0]), int(current_pos[1])]):
            print(f'drone crashed at t={t}.')
            timer.stop()

    for sphere in spheres:
        sphere_direction = sphere.destination - sphere.center
        sphere.update_center(sphere.center + sphere_direction * spheres_speed_factor)
        if sphere.is_point_inside(current_pos):
            print(f'warning: drone hit obstacle {sphere.id} at t={t}.')


def generate_rrt_star(height_map, max_altitude, num_nodes, start, spheres, target, height_margin):
    """
    Generates an RRT* tree for pathfinding in a 3D space.

    Args:
        height_map (np.array): A 2D numpy array representing the terrain over which the drone is flying.
        max_altitude (float): The maximum altitude at which the drone can fly.
        num_nodes (int): The number of nodes to generate for the RRT* tree.
        start (list): The starting point for the drone, represented as a list of three coordinates [x, y, z].
        spheres (list): A list of Sphere objects representing moving obstacles in the 3D space.
        target (list): The target point for the drone, represented as a list of three coordinates [x, y, z].
        height_margin (float): The minimum altitude margin above the terrain.

    Returns:
        nodes (list): A list of nodes in the RRT* tree. Each node is a numpy array of three coordinates [x, y, z].
        parents (list): A list of indices representing the parent node of each node in the 'nodes' list.
        costs (list): A list of costs associated with each node, representing the cost of the path from the start to that node.

    The function generates an RRT* tree by adding nodes one by one. Each node is a random point in the 3D space, and the function checks if the path from the nearest existing node to the new node is collision-free before adding the new node. The function also updates the parent and cost of each node as it is added. The function continues adding nodes until it has added the specified number of nodes, and then it adds the target point as the final node.
    """
    nodes = [np.array(start)]
    parents = [-1]  # the parent index of the start node is -1
    costs = [0]

    while len(nodes) < num_nodes:
        random_point = sample_point(height_map, max_altitude)
        nearest_node_index = find_nearest_node(nodes, random_point)
        new_node = generate_new_node(nodes[nearest_node_index], random_point)

        if collision_free(nodes[nearest_node_index], new_node, height_map, height_margin):
            nodes, parents, costs = add_point_to_rrt_star(height_map, nodes, parents, costs, new_node, spheres, height_margin)

    # Always add the target to the RRT* tree
    add_point_to_rrt_star(height_map, nodes, parents, costs, np.array(target), spheres, height_margin)
    return nodes, parents, costs


def find_path(nodes, parents, target):
    """
    Finds the shortest path from the start to the target in an RRT* tree.

    Args:
        nodes (list): A list of nodes in the RRT* tree. Each node is a numpy array of three coordinates [x, y, z].
        parents (list): A list of indices representing the parent node of each node in the 'nodes' list.
        target (list): The target point for the drone, represented as a list of three coordinates [x, y, z].

    Returns:
        np.array: A numpy array of nodes representing the shortest path from the start to the target.

    The function computes the Euclidean distance from the target to all nodes in the RRT* tree, and finds the node that is closest to the target. It then traces back from this node to the start through the parent nodes, creating a path. The path is reversed to obtain a path from the start to the target.
    """
    # Compute the distance from the target to all nodes
    distances = cdist([target], nodes, 'euclidean')[0]

    # Find the node that is closest to the target
    closest_node_index = np.argmin(distances)

    # Initialize the path with the closest node
    path = [nodes[closest_node_index]]

    # While the parent of the current node is not the root
    while parents[closest_node_index] != -1:
        # Move to the parent node
        closest_node_index = parents[closest_node_index]
        # Add the parent node to the path
        path.append(nodes[closest_node_index])

    # Reverse the path to obtain a path from the start to the target
    path = path[::-1]

    return np.array(path)


# Load the tif file
dataset = rasterio.open(MAP_FILE)

# Define the indices for the area you want to crop
start_row, end_row = X_MIN, X_MAX
start_col, end_col = Y_MIN, Y_MAX

# Read the first band (height data)
height_map = dataset.read(1)[start_row:end_row, start_col:end_col]


# Close the dataset
dataset.close()

max_altitude = MAX_ALTITUDE  # set maximum altitude
num_nodes = NUM_NODES  # set resolution

number_of_spheres = NUMBER_OF_OBSTACLES  # Number of spheres
spheres_speed_factor = OBSTACLES_SPEED_FACTOR
spheres = [Sphere(center=sample_point(height_map, max_altitude, min_altitude=OBSTACLES_MIN_ALTITUDE), radius=OBSTACLES_SAFETY_RADIUS, visual_radius=OBSTACLES_VISUAL_RADIUS, penalty_factor=OBSTACLES_PENALTY_FACTOR, min_altitude=OBSTACLES_MIN_ALTITUDE, id=i) for i in range(number_of_spheres)]

# Redefine the starting point and generate the RRT* tree
start = np.array(DRONE_START) - np.array([X_MIN, Y_MIN, 0])  # an arbitrary starting position
target = np.array(DRONE_TARGET) - np.array([X_MIN, Y_MIN, 0])  # an arbitrary target point

steering_noise_sigma = DRONE_STEERING_NOISE_SIGMA
height_margin = MAP_HEIGHT_MARGIN
tolerance = TARGET_TOLERANCE
drone_speed_factor = DRONE_SPEED_FACTOR

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
# Create a turntable style 3D camera
cam = scene.cameras.TurntableCamera(parent=view.scene)
view.camera = cam  # Assign the camera to the view

# Create a 3D surface for the height map
surface = scene.visuals.SurfacePlot(z=height_map, color=(0.5, 0.5, 0.5, 1), shading='smooth')
view.add(surface)
cam.set_range()

# Add sphere visuals to view
for sphere in spheres:
    view.add(sphere.visual)

path = np.empty(0)

current_pos = np.array(start)

current_point = visuals.Markers()
current_point.set_data(pos=np.array([current_pos]), edge_color='blue', face_color='blue', size=10)

target_point = visuals.Markers()
target_point.set_data(pos=np.array([target]), edge_color='green', face_color='green', size=15)

path_scatter = visuals.Markers()
path_lines = visuals.Line()
view.add(path_scatter)
view.add(path_lines)
view.add(current_point)
view.add(target_point)

t = 0.0

timer = app.Timer()
timer.connect(update)
timer.start(0)
if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()
