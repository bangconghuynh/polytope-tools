"""`PolytopeTools` contains classes and methods for constructing convex hulls
of polytope 3D-projections and converting them into polyhedra.
"""

import numpy as np
from pyhull import qconvex

from .Geometry3D import Contour, Facet, Polyhedron

def construct_convex_hull(vertices):
    """A method to construct a convex hull from a list of vertices as a
    Polyhedron object.

    Parameters
    ----------
    vertices : [Point]
        A list of all vertices to be analysed.

    Returns
    -------
    polyhedron : Polyhedron
        A Polyhedron object whose facets are the non-simplicial facets of the
        convex hull of `vertices`.
    """
    coords = np.zeros((len(vertices),3))
    for i,vertex in enumerate(vertices):
        coords[i,:] = vertex.coordinates
    hull = qconvex("i", coords)
    n_facets = int(hull[0])
    facets = []
    for facet_vertices_str in hull[1:]:
        facet_vertices_idx = [int(x) for x in facet_vertices_str.split(' ')]
        facet_vertices = [vertices[i] for i in facet_vertices_idx]
        facet = Facet([Contour.from_vertices(facet_vertices)])
        facets.append(facet)
    polyhedron = Polyhedron(facets)
    return polyhedron


def construct_convex_hull_from_coords(coords):
    """A method to construct a convex hull from a coordinate matrix.

    Parameters
    ----------
    coords : `array_like`
        An array of coordinates

    Returns
    -------
    polyhedron : Polyhedron
        A Polyhedron object whose facets are the non-simplicial facets of the
        convex hull of `vertices`.
    """
    K = coords.shape[1]
    print(coords)
    hull = qconvex("i", coords)
    n_facets = int(hull[0])
    facets = []
    for facet_vertices_str in hull[1:]:
        facet_vertices_idx = [int(x) for x in facet_vertices_str.split(' ')]
        facet_vertices = [vertices[i] for i in facet_vertices_idx]
        facet = Facet([Contour.from_vertices(facet_vertices)])
        facets.append(facet)
    polyhedron = Polyhedron(facets)
    return polyhedron
