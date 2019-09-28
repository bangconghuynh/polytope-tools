"""
@module PolyhedronDrawingTools
"""

import sys
import numpy as np
# from sympy import sympify
from sympy.geometry import Point, Line, Segment
from sympy.vector import CoordSys3D

## Implementation of hidden line removal algorithm for interesecting solids -- Wei-I Hsu and J. L. Hock, Comput. & Graphics Vol. 15, No. 1, pp 67--86, 1991

class Point(object):

    def __init__(self, object_coordinates=np.array([0,0,0])):
        self.object_coordinates = np.array(object_coordinates)
        while self.object_coordinates.shape[0] < 3:
            self.object_coordinates = np.append(self.object_coordinates, axis=0)

    def get_object_coordinates(self):
        return self.object_coordinates

    def set_object_coordinates(self, coords):
        self.object_coordinates = coords

    def get_vector_to(self, other):
        return Vector(other.object_coordinates-self.object_coordinates)


class Vector(object):

    def __init__(self, object_components=np.array([0,0,0])):
        self.object_components = np.array(object_components)
        while self.object_components.shape[0] < 3:
            self.object_components = np.append(self.object_components, axis=0)

    def __add__(self, other):
        return Vector(self.object_components+other.object_components)

    def __sub__(self, other):
        return Vector(self.object_components-other.object_components)

    def dot(self, other):
        return np.dot(self.object_components, other.object_components)

    def cross(self, other):
        return  Vector(np.cross(self.object_components, other.object_components))

    def __mul__(self, other):
        return self.dot(other)

    def __neg__(self):
        return Vector(-self.object_components)

    def abs(self):
        return np.abs(self.object_components)

    def get_object_components():
        return self.object_components

    def set_object_components(coords):
        self.object_components = coords

    def get_view_components():
        return self.view_components

    def set_view_components(coords):
        self.view_components = coords


class Line(object):
    def __init__(self, anchor, direction):
        self.anchor = anchor
        self.direction = direction

    def check_parallel(self, other, thresh=1e-6):
        if self.cross(other).abs() < thresh:
            return True
        else:
            return False

    def check_contain(self, point, thresh=1e-6):
        if point.get_vector_to(self.anchor).dot(self.direction) < 1e-6:
            return True

