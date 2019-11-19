"""`ConstructionTools` contains classes and methods for constructing\
convex hulls of polytope 3D-projections and converting them into\
polyhedra or vertex collections.
"""

from __future__ import annotations

import re

import numpy as np # type: ignore
from pyhull import qconvex # type: ignore
from typing import Sequence, List, Dict, Tuple, Union

from .Geometry3D import Point, Segment, Contour, Facet, Polyhedron,\
                        VertexCollection


def construct_convex_hull(vertices: Sequence[Point]) -> Polyhedron:
    """Construct a polyhedron as a convex hull from a list of vertices.

    :param vertices: A list of all vertices to be analysed.

    :returns: A polyhedron whose facets are the non-simplicial facets of the
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

def construct_convex_hull_from_coords(coords: np.ndarray) -> Polyhedron:
    """Construct a polyhedron as a convex hull from a coordinate matrix.

    :param coords: An array of coordinates.

    :returns: A polyhedron whose facets are the non-simplicial facets of the
              convex hull of the vertices defined by `coords`.
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

def construct_vertex_collection_from_coords(coords: np.ndarray,
                                            cutoff: float) -> VertexCollection:
    """Construct a vertex collection from a coordinate matrix.

    :param coords: An array of coordinates.
    :param cutoff: A cut-off for the inter-vertex distances to be considered as\
                   an edge.

    :returns: A vertex collection with vertices given by `coords` and edges\
              with lengths of at most `cutoff`.
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
    vertex_collection = VertexCollection(vertices, cutoff)
    return vertex_collection

def polyhedra_from_xyz(xyz_file: str,
                       try_convex_hull: bool = True)\
                      -> Tuple[List[Polyhedron],\
                               List[VertexCollection],\
                               List[Union[Polyhedron,VertexCollection]]]:
    """Parse an ``xyz`` file of vertices and construct convex polyhedra or\
    vertex collections.

    Vertices belonging to the same polyhedron or vertex collection share\
            the same atomic label.

    :param xyz_file: Path to an ``xyz`` file.
    :param try_convex_hull: Flag to indicate if an attempt to construct\
                            a convex hull should be made. If `False`, all sets\
                            will be treated as vertex collections.

    Returns
    -------
    polyhedron_list:
        List of all polyhedra constructable from the sets of vertices.
    vertex_collection_list:
        List of all vertex collections constructed from the sets of vertices.
    object_list:
        List of all polyhedra and vertex collections in the order they appear\
                in the ``xyz`` file.
    """

    object_coordinates_dict: Dict[str, List[List[float]]] = {}
    polyhedron_list: List[Polyhedron] = []
    vertex_collection_list: List[VertexCollection] = []
    object_list: List[Union[Polyhedron,VertexCollection]] = []
    type_order: List[str] = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if i == 0:
                l = re.search("\d+$", line)
                assert l is not None
                n_points = int(l.group())
            elif i == 1:
                l = re.search("\d+$", line)
                assert l is not None
                dim = int(l.group())
                assert dim <= 3, 'We cannot visualise the fourth dimension and\
                        above.'
            else:
                if line == '':
                    continue
                l = re.search("([A-Za-z]+[0-9]*)[\s\t]+", line)
                assert l is not None
                point_type = l.group(1)
                l2 = re.findall("[+-]?\d+\.\d*", line)
                point_coordinates = []
                for coordinate in l2:
                    point_coordinates.append(float(coordinate))
                assert len(point_coordinates) == dim
                if point_type not in object_coordinates_dict:
                    object_coordinates_dict[point_type] = []
                object_coordinates_dict[point_type].append(point_coordinates)
                if point_type not in type_order:
                    type_order.append(point_type)

    for point_type in type_order:
        object_coordinates = np.array(object_coordinates_dict[point_type])
        if try_convex_hull:
            try:
                print("Attempting to construct a convex hull for {}..."\
                        .format(point_type))
                polyhedron = construct_convex_hull_from_coords\
                                                    (object_coordinates)
                polyhedron_list.append(polyhedron)
                object_list.append(polyhedron) 
            except:
                print("Failed to construct a convex hull for {}."\
                        .format(point_type))
                print("Falling back to vertex collection for {}..."\
                        .format(point_type))
                vertex_collection = construct_vertex_collection_from_coords\
                                                    (object_coordinates, 2)
                vertex_collection_list.append(vertex_collection)
                object_list.append(vertex_collection) 
        else:
            print("Constructing a vertex collection for {}..."\
                    .format(point_type))
            vertex_collection = construct_vertex_collection_from_coords\
                                                    (object_coordinates, 2)
            vertex_collection_list.append(vertex_collection)
            object_list.append(vertex_collection) 

    return polyhedron_list,vertex_collection_list,object_list
