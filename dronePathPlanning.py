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
    x = np.random.uniform(0, height_map.shape[1])
    y = np.random.uniform(0, height_map.shape[0])
    z = np.random.uniform(min_altitude, max_altitude)
    return np.array([x, y, z])


def update(ev):
    global path_scatter, path, t, height_map, max_altitude, nodes, parents, costs, spheres, target, start,\
        current_pos, current_point, timer, movement_noise_sigma, tolerance, height_margin
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

        movement_noise = np.random.normal(scale=movement_noise_sigma, size=3)
        current_direction = (waypoint - current_pos) / np.linalg.norm(waypoint - current_pos)
        current_pos = current_pos + current_direction * 0.1 + movement_noise

        current_point.set_data(pos=np.array([current_pos]), edge_color='blue', face_color='blue', size=10)
        if (current_pos[0] < 0 or current_pos[1] < 0 or current_pos[0] >= height_map.shape[1] or current_pos[1] >= height_map.shape[0]):
            print(f'drone went out of bounds at t={t}.')
            timer.stop()
        if (current_pos[2] < height_map[int(current_pos[0]), int(current_pos[1])]):
            print(f'drone crashed at t={t}.')
            timer.stop()

    for sphere in spheres:
        sphere_direction = sphere.destination - sphere.center
        sphere.update_center(sphere.center + sphere_direction * 0.001)


def generate_rrt_star(height_map, max_altitude, num_nodes, start, spheres, target, height_margin):
    nodes = [np.array(start)]
    parents = [-1]  # the parent index of the start node is -1
    costs = [0]

    while len(nodes) < num_nodes:
        random_point = sample_point(height_map, max_altitude)
        nearest_node_index = find_nearest_node(nodes, random_point)
        new_node = generate_new_node(nodes[nearest_node_index], random_point)

        if collision_free(nodes[nearest_node_index], new_node, height_map, height_margin):
            nodes, parents, costs = add_point_to_rrt_star(height_map, max_altitude, nodes, parents, costs, new_node, spheres, height_margin)

    add_point_to_rrt_star(height_map, max_altitude, nodes, parents, costs, np.array(target), spheres, height_margin)
    return nodes, parents, costs


def find_path(nodes, parents, target):
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

# Let's test the function with some example parameters
max_altitude = MAX_ALTITUDE  # an arbitrary maximum altitude
num_nodes = NUM_NODES  # an arbitrary resolution

number_of_spheres = NUMBER_OF_OBSTACLES  # Number of spheres

spheres = [Sphere(center=sample_point(height_map, max_altitude, min_altitude=OBSTACLES_MIN_ALTITUDE), radius=OBSTACLES_RADIUS, penalty_factor=OBSTACLES_PENALTY_FACTOR, min_altitude=OBSTACLES_MIN_ALTITUDE) for _ in range(number_of_spheres)]

# Redefine the starting point and generate the RRT* tree
start = DRONE_START  # an arbitrary starting position
# Let's test the function with an arbitrary target point
target = DRONE_TARGET  # an arbitrary target point

movement_noise_sigma = DRONE_NOISE_SIGMA
height_margin = MAP_HEIGHT_MARGIN
tolerance = TARGET_TOLERANCE

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
