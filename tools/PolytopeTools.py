"""`PolytopeTools` contains classes and methods for constructing convex hulls
of polytope 3D-projections and converting them into polyhedra.
"""

import copy
import numpy as np
from scipy.spatial import ConvexHull

from .Geometry3D import Contour, ZERO_TOLERANCE

def construct_convex_hull(vertices, thresh=ZERO_TOLERANCE):
    """A method to convert a three-dimensional convex hull into a Polyhedron
    object.

    Parameters
    ----------
    vertices : [Point]
        A list of all vertices to be analysed.
    thresh : float
        Threshold to check if vertices are coplanar.
    """
    coords = np.zeros((len(vertices),3))
    for i,vertex in enumerate(vertices):
        coords[i,:] = vertex.coordinates
    hull = ConvexHull(coords)
    print(hull.simplices)
    print(hull.neighbors)

    # In 3D, all simplices of the convex hull are triangles. We need to
    # identify and merge coplanar simplices.
    simplex_indices = list(range(len(hull.simplices)))
    coplanar = []
    coplanar_detected = False
    while len(simplex_indices) > 0:
        if not coplanar_detected:
            coplanar.append([])
            i_simplex = simplex_indices.pop()
            coplanar[-1].append(i_simplex)
        else:
            coplanar_detected = False
            i_simplex = coplanar[-1][-1]
        simplex_vertices = hull.simplices[i_simplex]
        simplex_plane = Contour.from_vertices([vertices[i]\
                                for i in simplex_vertices]).associated_plane
        neighbor_indices = hull.neighbors[i_simplex]
        for i_neighbor in neighbor_indices:
            neighbor_vertices = hull.simplices[i_neighbor]
            for i in neighbor_vertices:
                if i in simplex_vertices:
                    continue
                else:
                    unshared_neighbor_vertex = i
                    break
            if simplex_plane.contains_point(vertices[unshared_neighbor_vertex]):
                if i_neighbor in simplex_indices:
                    simplex_indices.remove(i_neighbor)
                    coplanar[-1].append(i_neighbor)
                    coplanar_detected = True
    print(coplanar)
