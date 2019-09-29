"""
@module PolyhedronDrawingTools
Module containing necessary classes and functions to draw three-dimensional
intersecting polyhedra with careful treatment of hidden lines.
"""

import sys
import numpy as np
from sympy.geometry import Point, Line, Segment
from sympy.vector import CoordSys3D

## Implementation of hidden line removal algorithm for interesecting solids -- Wei-I Hsu and J. L. Hock, Comput. & Graphics Vol. 15, No. 1, pp 67--86, 1991

# Constants
ZERO_TOLERANCE = 1e-15

class Point(object):

    def __init__(self, coordinates=np.array([0,0,0])):
        self.coordinates = np.array(coordinates)
        while self.coordinates.shape[0] < 3:
            self.coordinates = np.append(self.coordinates, axis=0)

    def __str__(self):
        return "Point({},{},{})".format(self.coordinates[0],\
                self.coordinates[1], self.coordinates[2])

    def __getitem__(self, key):
        return self.coordinates[key]

    def get_coordinates(self):
        return self.coordinates

    def set_coordinates(self, coords):
        self.coordinates = coords

    def get_vector_to(self, other):
        return Vector(other.coordinates-self.coordinates)


class Vector(object):

    def __init__(self, components=np.array([0,0,0])):
        self.components = np.array(components)
        while self.components.shape[0] < 3:
            self.components = np.append(self.components, axis=0)

    def __str__(self):
        return "Vector({},{},{})".format(self.components[0],\
                self.components[1], self.components[2])

    def __getitem__(self, key):
        return self.components[key]

    def __add__(self, other):
        return Vector(self.components+other.components)

    def __sub__(self, other):
        return Vector(self.components-other.components)

    def __neg__(self):
        return Vector(-self.components)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.dot(other)
        else:
            try:
                return Vector(other*self.components)
            except:
                raise TypeError('{} is not a number or a Vector.'. format(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        try:
            return Vector(self.components/other)
        except:
            raise TypeError('{} is not a number or a Vector.'. format(other))

    def dot(self, other):
        return np.dot(self.components, other.components)

    def cross(self, other):
        return Vector(np.cross(self.components, other.components))

    def get_norm(self):
        return np.linalg.norm(self.components)

    def normalise(self, thresh=ZERO_TOLERANCE):
        norm = self.get_norm()
        if norm < thresh:
            raise ZeroDivisionError('{} has norm {} and is a null vector.'\
                .format(self, norm))
        else:
            return self/self.get_norm()

    def get_components(self):
        return self.components

    def set_components(coords):
        self.components = coords


class Line(object):
    def __init__(self, anchor, direction):
        self.anchor = anchor
        self.direction = direction.normalise()

    def __str__(self):
        return "Line[{} + lambda*{}]".format(self.anchor, self.direction)

    def is_parallel(self, other, thresh=ZERO_TOLERANCE):
        if self.direction.cross(other.direction).get_norm() < thresh:
            return True
        else:
            return False

    def contains_point(self, point, thresh=ZERO_TOLERANCE):
        anchor_to_point = self.anchor.get_vector_to(point)
        if anchor_to_point.get_norm() < thresh:
            return True
        elif anchor_to_point.normalise().cross(self.direction).get_norm()\
                < thresh:
            return True
        else:
            return False

    def intersects_line(self, other, thresh=ZERO_TOLERANCE):
        if self.is_parallel(other, thresh):
            if self.contains_point(other.anchor, thresh):
                return (np.inf,)
            else:
                return(0,)
        else:
            interanchor = self.anchor.get_vector_to(other.anchor)
            if interanchor.get_norm() < thresh:
                return (1,)
            else:
                v = self.direction.cross(other.direction).normalise()
                if interanchor.dot(v) < thresh:
                    return (1,)
                else:
                    return (0,)
