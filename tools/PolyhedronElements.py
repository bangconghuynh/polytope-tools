# -*- coding: utf-8 -*-
"""`PolyhedronElements` contains classes inherited from general three-dimensional
geometrical objects but specialised for polyhedron descriptions.
"""

import numpy as np
from scipy.spatial import ConvexHull

from .Geometry3D import Point, Vector, Line, Segment, Plane

## Implementation of hidden line removal algorithm for interesecting solids -- Wei-I Hsu and J. L. Hock, Comput. & Graphics Vol. 15, No. 1, pp 67--86, 1991

# Constants
ZERO_TOLERANCE = 1e-15

class Vertex(Point):
    """A contour in a three-dimensional Euclidean space.

    (Hsu & Hock, 1991) "A contour is a closed planar polygon that may be one
    of ordered orientation."

    Parameters
    ----------
    edges : [Segment]
        A list of segments defining the edges of the contour. Any ordered
        orientation is implied by the order of the list.
    """

class Contour(object):
    """A contour in a three-dimensional Euclidean space.

    (Hsu & Hock, 1991) "A contour is a closed planar polygon that may be one
    of ordered orientation."

    Parameters
    ----------
    edges : [Segment]
        A list of segments defining the edges of the contour. Any ordered
        orientation is implied by the order of the list.
    """

    def __init__(self, edges):
        self.edges = edges

    @property
    def edges(self):
        """[Segment]: A list of segments defining the edges of the contour.
        """
        return self._edges

    @edges.setter
    def edges(self, edges):
        self._edges = edges

    @property
    def associated_plane(self):
        v1 = self.edges[0].associated_line.direction
        v2 = self.edges[1].associated_line.direction
        n = v1.cross(v2)
        A = self.edges[0].endpoints[0]
        return Plane(n, A)
