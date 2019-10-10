"""`PolytopeTools` contains classes and methods for constructing convex hulls
of polytope 3D-projections and converting them into polyhedra.
"""

import numpy as np
import re
from pyhull import qconvex

from .Geometry3D import Point, Segment, Contour, Facet, Polyhedron

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
    dim = coords.shape[1]
    assert dim <= 3, 'We cannot visualise anything with more than three\
            dimensions.'
    vertices = []
    for i in range(coords.shape[0]):
        if dim == 2:
            vertices.append(Point(np.array([coords[i,0],coords[i,1],0])))
        else:
            vertices.append(Point(coords[i,:]))
    if dim == 3:
        hull = qconvex("i", coords)
        n_facets = int(hull[0])
        facets = []
        for facet_vertices_str in hull[1:]:
            facet_vertices_idx = [int(x) for x in facet_vertices_str.split(' ')]
            facet_vertices = [vertices[i] for i in facet_vertices_idx]
            facet = Facet([Contour.from_vertices(facet_vertices)])
            facets.append(facet)
        polyhedron = Polyhedron(facets)
    elif dim == 2:
        hull = qconvex("Fx", coords)
        n_hull_vertices = int(hull[0])
        hull_vertices = []
        for vertex_str in hull[1:]:
            vertex_idx = int(vertex_str)
            hull_vertices.append(vertices[vertex_idx])
        contour = Contour.from_vertices(hull_vertices)
        polyhedron = Polyhedron([Facet([contour])])
    return polyhedron


def polyhedra_from_xyz(xyz_file):
    """Method to parse an xyz file of vertices and construct convex polyhedra.
    Vertices belonging to the same polyhedron share the same atomic label.
    """

    polyhedron_dict = {}
    polyhedron_list = []
    type_order = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if i == 0:
                l = re.search("\d+$", line)
                n_points = int(l.group())
            elif i == 1:
                l = re.search("\d+$", line)
                dim = int(l.group())
                assert dim <= 3, 'We cannot visualise the fourth dimension and\
                        above.'
            else:
                if line == '':
                    continue
                l = re.search("([A-Za-z]+[0-9]*)[\s\t]+", line)
                assert l is not None
                point_type = l.group(1)
                l = re.findall("[+-]?\d+\.\d*", line)
                point_coordinates = []
                for coordinate in l:
                    point_coordinates.append(float(coordinate))
                assert len(point_coordinates) == dim
                if point_type not in polyhedron_dict:
                    polyhedron_dict[point_type] = []
                polyhedron_dict[point_type].append(point_coordinates)
                if point_type not in type_order:
                    type_order.append(point_type)

        for point_type in type_order:
            n_vertices = len(polyhedron_dict[point_type])
            if dim == 3 and n_vertices == 3:
                vertices = []
                for point_coords in polyhedron_dict[point_type]:
                    vertices.append(Point(np.array(point_coords)))
                contour = Contour.from_vertices(vertices)
                facet = Facet([contour])
                polyhedron_list.append(Polyhedron([facet])) 
            else:
                assert n_vertices > dim, 'There are too few vertices for a\
                        meaningful description of a polygon or a polyhedron.'
                polyhedron_dict[point_type] = np.array(polyhedron_dict[point_type])
                polyhedron_list.append(construct_convex_hull_from_coords\
                                        (polyhedron_dict[point_type]))

    return polyhedron_list
