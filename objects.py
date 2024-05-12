import numpy as np

from vispy import scene
from vispy.scene import visuals


class Sphere:
    def __init__(self, center, radius, penalty_factor, min_altitude):
        self.center = np.empty(3)
        self.radius = radius
        self.penalty_factor = penalty_factor
        self.visual = visuals.Sphere(radius=self.radius, method='latitude',
                                     color=(0.5, 0.5, 1, 0.8))
        self.update_center(center)
        self.destination = self.center
        self.min_altitude = min_altitude

    def is_point_inside(self, point):
        return np.linalg.norm(self.center - np.array(point)) <= self.radius

    def update_center(self, center):
        self.center = center
        self.visual.transform = scene.transforms.MatrixTransform()
        self.visual.transform.translate(self.center)


class Line:
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)
