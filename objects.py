import numpy as np

from vispy import scene
from vispy.scene import visuals


class Sphere:
    def __init__(self, center, radius, visual_radius, penalty_factor, min_altitude, id):
        # Initialize the center of the sphere
        self.center = np.empty(3)
        # Set the radius of the sphere
        self.radius = radius
        # Set the penalty factor for the sphere
        self.penalty_factor = penalty_factor
        # Create a visual representation of the sphere
        self.visual_radius = visual_radius
        self.visual = visuals.Sphere(radius=visual_radius, method='latitude',
                                     color=(0.5, 0.5, 1, 0.8))
        # Update the center of the sphere
        self.update_center(center)
        # Set the destination of the sphere to its center
        self.destination = self.center
        # Set the minimum altitude for the sphere
        self.min_altitude = min_altitude
        self.id = id

    def is_point_inside(self, point):
        # Check if a point is inside the sphere by comparing the distance from the center to the radius
        return np.linalg.norm(self.center - np.array(point)) <= self.visual_radius

    def update_center(self, center):
        # Update the center of the sphere
        self.center = center
        # Reset the transformation of the visual representation
        self.visual.transform = scene.transforms.MatrixTransform()
        # Translate the visual representation to the new center
        self.visual.transform.translate(self.center)


class Line:
    def __init__(self, start, end):
        # Set the start point of the line
        self.start = np.array(start)
        # Set the end point of the line
        self.end = np.array(end)