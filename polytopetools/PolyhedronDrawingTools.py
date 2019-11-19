"""`PolyhedronDrawingTools` contains classes and methods for drawing realistic\
two-dimensional projections of three-dimensional polyhedra. This implementation\
is based loosely on :cite:`article:Hsu1991`.
"""

from __future__ import annotations

import numpy as np # type: ignore
import copy, textwrap
from typing import Sequence, List, Tuple, Union, Optional

from .Geometry3D import Point, Vector, Segment, Contour, Facet,\
                        Polyhedron, VertexCollection, rotate_point,\
                        ZERO_TOLERANCE
from .ConstructionTools import construct_convex_hull_from_coords


polyhedron_color_list = ['black!20!orange', 'violet', 'green', 'DarkViolet',\
                         'DarkCyan', 'FireBrick', 'NavyBlue', 'Fuchsia']


class Scene():
    """A class to contain, manage, and draw multiple polyhedra and
    vertex collections in a three-dimensional scene.
    """

    def __init__(self,
                 polyhedra: List[Polyhedron],
                 vertexcollections: List[VertexCollection],
                 objects: List[Union[Polyhedron,VertexCollection]],
                 a: float = np.arctan(2),
                 thresh: float = ZERO_TOLERANCE) -> None:
        """
        :param polyhedra: A list of polyhedra in the scene.
        :param vertexcollections: A list of vertex collections in the scene.
        :param objects: An ordered list of polyhedra and vertex collections\
                        in the scene.
        :param a: Cabinet projection angle (radians).
        :param thresh: Threshold for various comparisons.
        """
        self.polyhedra = polyhedra
        self.vertexcollections = vertexcollections
        self.cabinet_angle = a
        if len(objects) == 0:
            self._objects = polyhedra + vertexcollections # type: ignore
        else:
            self._objects = objects # type: ignore
        self._thresh = thresh

    @property
    def polyhedra(self) -> List[Polyhedron]:
        """A list of all polyhedra in the scene.
        """
        return self._polyhedra

    @polyhedra.setter
    def polyhedra(self, polyhedra: Sequence[Polyhedron]) -> None:
        for polyhedron in polyhedra:
            assert isinstance(polyhedron, Polyhedron)
        self._polyhedra = list(polyhedra)

    @property
    def vertexcollections(self) -> List[VertexCollection]:
        """A list of all non-polyhedron vertex collections in the scene.
        """
        return self._vertexcollections

    @vertexcollections.setter
    def vertexcollections(self,
                          vertexcollections: Sequence[VertexCollection])\
                         -> None:
        for vertexcollection in vertexcollections:
            assert isinstance(vertexcollection, VertexCollection)
        self._vertexcollections = list(vertexcollections)

    @property
    def all_vertices(self) -> List[Point]:
        """A list of all vertices in the scene.
        """
        from_polyhedra = [vertex for polyhedron in self.polyhedra\
                          for vertex in polyhedron.vertices]
        from_collections = [vertex for collection in self.vertexcollections\
                            for vertex in collection.vertices]
        return from_polyhedra + from_collections

    @property
    def all_edges(self) -> List[Segment]:
        """A list of all edges in the scene.
        """
        from_polyhedra = [edge for polyhedron in self.polyhedra\
                          for edge in polyhedron.edges]
        from_collections = [edge for collection in self.vertexcollections\
                            for edge in collection.edges]
        return from_polyhedra + from_collections

    @property
    def all_facets(self) -> List[Facet]:
        """A list of all facets in the scene.
        """
        return [facet for polyhedron in self.polyhedra\
                      for facet in polyhedron.facets]

    @property
    def cabinet_angle(self) -> float:
        """The cabinet projection angle (radians) of the view of this scene.

        This angle lies within the :math:`[0,2\\pi)` interval.
        """
        return self._cabinet_angle

    @cabinet_angle.setter
    def cabinet_angle(self, a: float) -> None:
        angle = a
        while angle >= 2*np.pi:
            angle = angle - 2*np.pi
        while angle < 0:
            angle = angle + 2*np.pi
        self._cabinet_angle = angle

    @property
    def viewing_vector(self) -> Vector:
        """The viewing vector of the view of this scene.

        This vector is proportional to\
        :math:`(\\frac{1}{2}\\cos\\alpha,\\frac{1}{2}\\sin\\alpha,1)`\
        where :math:`\\alpha` is the cabinet projection angle.
        """
        a = self.cabinet_angle
        return Vector([0.5*np.cos(a),0.5*np.sin(a),1]).normalise()

    @property
    def visible_segments(self) -> List[List[Segment]]:
        """The visible segments of the edges w.r.t. the current viewing vector.
        """
        return self._visible_segments

    @visible_segments.setter
    def visible_segments(self, visible: Sequence[Sequence[Segment]]) -> None:
        self._visible_segments = [list(v) for v in visible]

    @property
    def visible_vertices(self) -> List[List[Point]]:
        """The visible vertices w.r.t. the current viewing vector.
        """
        return self._visible_vertices

    @visible_vertices.setter
    def visible_vertices(self, visible: Sequence[Sequence[Point]]) -> None:
        self._visible_vertices = [list(v) for v in visible]

    @property
    def hidden_segments(self) -> List[List[Segment]]:
        """The hidden segments of the edges w.r.t. the current viewing vector.
        """
        return self._hidden_segments

    @hidden_segments.setter
    def hidden_segments(self, hidden: Sequence[Sequence[Segment]]) -> None:
        self._hidden_segments = [list(h) for h in hidden]

    @property
    def hidden_vertices(self) -> List[List[Point]]:
        """The hidden vertices w.r.t. the current viewing vector.
        """
        return self._hidden_vertices

    @hidden_vertices.setter
    def hidden_vertices(self, hidden: Sequence[Sequence[Point]]) -> None:
        self._hidden_vertices = [list(h) for h in hidden]

    @property
    def visible_segments_2D(self) -> List[List[Segment]]:
        """The visible segments of the edges w.r.t. the current viewing vector\
        in the cabinet projection.
        """
        return list(map(lambda sublist :\
                [s.get_cabinet_projection(self.cabinet_angle)\
                 for s in sublist], self.visible_segments))

    @property
    def visible_vertices_2D(self) -> List[List[Point]]:
        """The visible vertices w.r.t. the current viewing vector in the\
        cabinet projection.
        """
        return list(map(lambda sublist :\
                [v.get_cabinet_projection(self.cabinet_angle)\
                 for v in sublist], self.visible_vertices))

    @property
    def hidden_segments_2D(self) -> List[List[Segment]]:
        """The hidden segments of the edges w.r.t. the current viewing vector\
        in the cabinet projection.
        """
        return list(map(lambda sublist :\
                [s.get_cabinet_projection(self.cabinet_angle)\
                 for s in sublist], self.hidden_segments))

    @property
    def hidden_vertices_2D(self) -> List[List[Point]]:
        """[Point]: The hidden vertices w.r.t. the current viewing vector
        in the cabinet projection.
        """
        return list(map(lambda sublist :\
                [v.get_cabinet_projection(self.cabinet_angle)\
                 for v in sublist], self.hidden_vertices))

    def centre_scene(self) -> None:
        """Translate all polyhedra in the scene such that the geometric centre
        is at the origin.
        """
        vecsum = Vector()
        for vertex in self.all_vertices:
            vecsum = vecsum + Vector.from_point(vertex)
        centre_vector = vecsum/len(self.all_vertices)
        centred_polyhedra: List[Polyhedron] = []
        centred_vertexcollections: List[VertexCollection] = []
        centred_objects: List[Union[Polyhedron,VertexCollection]] = []

        for o in self._objects:
            if isinstance(o, Polyhedron):
                centred_facets = []
                for facet in o.facets:
                    centred_contours = []
                    for contour in facet.contours:
                        centred_edges = []
                        for edge in contour.edges:
                            centred_edges.append(Segment(\
                                list(map(lambda\
                                x : Point.from_vector(\
                                Vector.from_point(x)-centre_vector),\
                                edge.endpoints))))
                        centred_contours.append(Contour(centred_edges))
                    centred_facets.append(Facet(centred_contours))
                centred_polyhedra.append(Polyhedron(centred_facets))
                centred_objects.append(centred_polyhedra[-1])

            elif isinstance(o, VertexCollection):
                centred_vertices = []
                for vertex in o.vertices:
                    centred_vertex = Point.from_vector(\
                                        Vector.from_point(vertex)-centre_vector)
                    centred_vertices.append(centred_vertex)
                cutoff = o._cutoff
                centred_vertexcollections.append(\
                                    VertexCollection(centred_vertices, cutoff))
                centred_objects.append(centred_vertexcollections[-1])

            else:
                raise TypeError('Invalid object.')

        self.polyhedra = centred_polyhedra
        self.vertexcollections = centred_vertexcollections
        self._objects = centred_objects # type: ignore

    def rotate_scene(self, angle: float, direction: Vector) -> None:
        """Rotate the entire scene through `angle` about the vector
        `direction`.

        :param angle: Angle of rotation (radians).
        :param direction: Vector indicating the direction of rotation.
        """
        rotated_polyhedra: List[Polyhedron] = []
        rotated_vertexcollections: List[VertexCollection] = []
        rotated_objects: List[Union[Polyhedron,VertexCollection]] = []
        for o in self._objects:
            if isinstance(o, Polyhedron):
                rotated_facets = []
                for facet in o.facets:
                    rotated_contours = []
                    for contour in facet.contours:
                        rotated_edges = []
                        for edge in contour.edges:
                            rotated_edges.append(Segment(\
                                list(map(lambda\
                                x : rotate_point(x, angle, direction),\
                                edge.endpoints))))
                        rotated_contours.append(Contour(rotated_edges))
                    rotated_facets.append(Facet(rotated_contours))
                rotated_polyhedra.append(Polyhedron(rotated_facets))
                rotated_objects.append(rotated_polyhedra[-1])

            elif isinstance(o, VertexCollection):
                rotated_vertices = []
                for vertex in o.vertices:
                    rotated_vertex = rotate_point(vertex, angle, direction) 
                    rotated_vertices.append(rotated_vertex)
                cutoff = o._cutoff
                rotated_vertexcollections.append(\
                                    VertexCollection(rotated_vertices, cutoff))
                rotated_objects.append(rotated_vertexcollections[-1])

        self.polyhedra = rotated_polyhedra
        self.vertexcollections = rotated_vertexcollections
        self._objects = rotated_objects # type: ignore

    def write_to_tikz(self,
                      output: str,
                      bb: Optional[List[Tuple[float,float]]] = None,
                      vertex_size: str = '1pt',
                      scale: float = 1.0) -> None:
        """Write the current scene to a standalone TeX file.

        :param output: Output file name.
        :param bb: The bounding box of the TikZ picture in the format\
                   [(xmin,xmax),(ymin,ymax)].
        :param vertex_size: Size of the circle nodes used to depict vertices.
        :param scale: Scale factor for the entire TikZ picture.
        """
        self.visible_segments,self.hidden_segments =\
                self._get_visible_hidden_segments(self._thresh)
        self.visible_vertices,self.hidden_vertices =\
                self._get_visible_hidden_vertices(self._thresh)

        preamble_str =\
                """\
                \\documentclass[tikz]{standalone}
                \\PassOptionsToPackage{svgnames}{xcolor}
                \\usepackage{pgfplots}
                \\pgfplotsset{compat=1.16}
                \\usetikzlibrary{calc, luamath, positioning}

                \\begin{document}
                \\begin{tikzpicture}
                """
        if bb is not None:
            (xmin,xmax) = bb[0]
            (ymin,ymax) = bb[1]
            preamble_str = preamble_str +\
                """
                    \\useasboundingbox ({},{}) rectangle ({},{});
                """.format(xmin*scale,ymin*scale,xmax*scale,ymax*scale)

        preamble_str = preamble_str +\
                """
                    \\scalebox{{ {} }}{{
                """.format(scale)
        all_str = preamble_str

        for pi,polyhedron in enumerate(self._objects):
            color = polyhedron_color_list[pi]

            vertices_str =\
                """
                    % Coordinate definition\
                """
            vertices_str = vertices_str + \
                """
                    %%%%%%%%%%%%%%%%
                    % Polyhedron {}\
                """.format(pi)
            vi = 0
            visible_vertices_str =\
                """
                    % Visible vertices\
                """
            for vertex_2D in self.visible_vertices_2D[pi]:
                vi += 1
                x,y = vertex_2D[0],vertex_2D[1]
                visible_vertices_str = visible_vertices_str + \
                """
                \\node at ({},{})\
                    [circle, fill = {}, very thin,\
                     draw = black, inner sep={}]{{}};
                """.format(x, y, color, vertex_size)


            hidden_vertices_str =\
                """
                    % Hidden vertices\
                """
            for vertex_2D in self.hidden_vertices_2D[pi]:
                vi += 1
                x,y = vertex_2D[0],vertex_2D[1]
                hidden_vertices_str = hidden_vertices_str + \
                """
                \\node at ({},{})\
                    [circle, fill = {}!50!white, very thin,\
                     draw = black!50!white, inner sep={}]{{}};
                """.format(x, y, color, vertex_size)
            hidden_vertices_str = hidden_vertices_str + "\n"

            visible_str =\
                """
                    % Visible segments\
                """
            for segment_2D in self.visible_segments_2D[pi]:
                xstart,ystart = segment_2D.endpoints[0][0],segment_2D.endpoints[0][1]
                xend,yend = segment_2D.endpoints[1][0],segment_2D.endpoints[1][1]
                visible_str = visible_str + \
                """
                    \\draw[{}] ({},{}) -- ({},{});\
                """.format(color, xstart, ystart, xend, yend)
            visible_str = visible_str + "\n"

            hidden_str =\
                """
                    % Hidden segments\
                """
            for segment_2D in self.hidden_segments_2D[pi]:
                xstart,ystart = segment_2D.endpoints[0][0],segment_2D.endpoints[0][1]
                xend,yend = segment_2D.endpoints[1][0],segment_2D.endpoints[1][1]
                hidden_str = hidden_str + \
                """
                    \\draw[{}, opacity=0.25] ({},{}) -- ({},{});\
                """.format(color, xstart, ystart, xend, yend)
            hidden_str = hidden_str + "\n"

            all_str = all_str +\
                    hidden_str +\
                    visible_str +\
                    hidden_vertices_str +\
                    visible_vertices_str

        end_str =\
                """
                    }
                \\end{tikzpicture}
                \\end{document}
                """

        all_str = all_str + end_str

        with open(output, 'w') as f:
            f.write(textwrap.dedent(all_str))


    def _get_visible_hidden_segments(self,
                                     thresh: float = ZERO_TOLERANCE)\
                                    -> Tuple[List[List[Segment]],\
                                             List[List[Segment]]]:
        # Visibility is only meaningful in the projected view, so we check for
        # visibility  of any back edges or segments in the projected view.
        # However, we store all segments in the full object space.
        visible: List[List[Segment]] = []
        hidden: List[List[Segment]] = []
        for polyhedron in self._objects:
            visible.append([])
            hidden.append([])
            for edge in polyhedron.edges:
                previous_segments_visible = [copy.deepcopy(edge)]
                for test_facet in self.all_facets:
                    updated_segments_visible = []
                    updated_segments_hidden = []
                    front_normal = test_facet.get_front_normal(self.viewing_vector)
                    facet_point = test_facet.get_a_point()
                    test_facet_2D = test_facet.\
                                        get_cabinet_projection(self.cabinet_angle)
                    while len(previous_segments_visible) > 0:
                        test_segment = previous_segments_visible.pop()
                        back_segment = False
                        back_segment_parallel = False
                        test_segment_2D = test_segment.\
                                        get_cabinet_projection(self.cabinet_angle)
                        if not test_segment_2D.intersects_bounding_box(test_facet_2D):
                            # test_segment not covered by test_facet in 2D projection
                            updated_segments_visible.append(test_segment)
                            continue
                        else:
                            n_plane,intersection_plane =\
                                    test_segment.intersects_plane\
                                            (test_facet.associated_plane, thresh)
                            if n_plane == 0:
                                # Segment parallel to plane of facet
                                edge_point = test_segment.get_a_point()
                                fe_vector = facet_point.get_vector_to(edge_point)
                                if fe_vector.dot(front_normal) > 0:
                                    # Segment in front of facet
                                    updated_segments_visible.append(test_segment)
                                    continue
                                else:
                                    # Segment on the back of facet
                                    # Needs further testing
                                    back_test_segment = test_segment
                                    back_segment = True
                                    back_segment_parallel = True
                            elif n_plane == np.inf:
                                # Segment on plane
                                updated_segments_visible.append(test_segment)
                                continue
                            elif n_plane == 1:
                                # Segment intersects plane
                                # Front fraction always visible
                                # Back fraction needs testing
                                assert isinstance(intersection_plane, Point)
                                for endpoint in test_segment.endpoints:
                                    fe_vector = facet_point.get_vector_to\
                                                    (endpoint)
                                    if fe_vector.dot(front_normal) > thresh:
                                        front_test_segment = Segment([\
                                                    endpoint,\
                                                    intersection_plane])
                                        updated_segments_visible.append(\
                                                front_test_segment)
                                    elif fe_vector.dot(front_normal) < -thresh:
                                        back_test_segment = Segment([\
                                                    endpoint,\
                                                    intersection_plane])
                                        back_segment = True
                                    else:
                                        # This endpoint lies on the plane.
                                        continue

                        if not back_segment:
                            continue

                        # If get to this point, back_test_segment exists.
                        # We can only handle facets consisting of one contour
                        # at the moment.
                        # assert len(test_facet.contours) == 1,\
                        #     'We can only handle facets consisting of\
                        #      one contour at the moment.'
                        back_test_segment_2D = back_test_segment\
                                .get_cabinet_projection(self.cabinet_angle)
                        # test_contour_2D = test_facet.contours[0]\
                        #         .get_cabinet_projection(self.cabinet_angle)
                        if back_test_segment_2D.intersects_bounding_box\
                                (test_facet_2D):
                            if back_segment_parallel:
                                anchor = back_test_segment.endpoints[0]
                            else:
                                assert isinstance(intersection_plane, Point)
                                anchor = intersection_plane
                            anchor_2D = anchor\
                                    .get_cabinet_projection(self.cabinet_angle)
                            n,J_points,segments_2D_hidden,segments_2D_visible =\
                                    back_test_segment_2D.intersects_facet\
                                            (test_facet_2D, anchor_2D, thresh)
                            # We need to convert the 2D information back to 3D data.
                            # We need 3D data for further testing with other facets.
                            # For parallel projection, mu is the same in both 3D
                            # and 2D.
                            for (segments_2D, mu_start, mu_end) in\
                                    segments_2D_hidden:
                                A = back_test_segment.get_fraction_of_segment\
                                            (mu_start, anchor)[0]
                                B = back_test_segment.get_fraction_of_segment\
                                            (mu_end, anchor)[0]
                                updated_segments_hidden.append(Segment([A,B]))
                            for (segments_2D, mu_start, mu_end) in\
                                    segments_2D_visible:
                                A = back_test_segment.get_fraction_of_segment\
                                            (mu_start, anchor)[0]
                                B = back_test_segment.get_fraction_of_segment\
                                            (mu_end, anchor)[0]
                                updated_segments_visible.append(Segment([A,B]))
                        else:
                            updated_segments_visible.append(back_test_segment)
                    # End of while; no more segments left in
                    # previous_segments_visible for this facet

                    # Preparing for the next facet
                    previous_segments_visible = updated_segments_visible

                    # Saving all segments from this edge hidden by this facet.
                    for segment in updated_segments_hidden:
                        hidden[-1].append(segment)

                # End of for; no more facets left for this edge.
                # Saving all visible segments from this edge.
                for segment in previous_segments_visible:
                    visible[-1].append(segment)

        return visible,hidden


    def _get_visible_hidden_vertices(self, thresh: float = ZERO_TOLERANCE)\
                                    -> Tuple[List[List[Point]],\
                                             List[List[Point]]]:
        # Visibility is only meaningful in the projected view, so we check for
        # visibility  of any back vertices in the projected view.
        # However, we store all vertices in the full object space.
        visible: List[List[Point]] = []
        hidden: List[List[Point]] = []
        for polyhedron in self._objects:
            visible.append([])
            hidden.append([])
            for vertex in polyhedron.vertices:
                previous_vertex_visible = [copy.deepcopy(vertex)]
                for test_facet in self.all_facets:
                    updated_vertex_visible = []
                    front_normal = test_facet.get_front_normal(self.viewing_vector)
                    facet_point = test_facet.get_a_point()
                    test_facet_2D = test_facet.\
                                    get_cabinet_projection(self.cabinet_angle)
                    while len(previous_vertex_visible) > 0:
                        test_vertex = previous_vertex_visible.pop()
                        test_vertex_2D = test_vertex.\
                                        get_cabinet_projection(self.cabinet_angle)
                        if not test_vertex_2D.intersects_bounding_box(test_facet_2D):
                            # test_vertex not covered by test_facet in 2D projection
                            updated_vertex_visible.append(test_vertex)
                            continue
                        else:
                            # test_vertex covered by test_facet in 2D projection
                            if test_facet.associated_plane.contains_point\
                                    (test_vertex, thresh):
                                # Vertex on facet's plane and is visible w.r.t.
                                # this facet
                                updated_vertex_visible.append(test_vertex)
                                continue
                            else:
                                # Vertex off-plane
                                fv_vector = facet_point.get_vector_to\
                                                (test_vertex)
                                if fv_vector.dot(front_normal) > thresh:
                                    # Vertex lies in front
                                    updated_vertex_visible.append(test_vertex)
                                    continue
                                elif fv_vector.dot(front_normal) < -thresh:
                                    # Vertex lies behind
                                    back_vertex = True
                                else:
                                    # This vertex lies on the plane.
                                    updated_vertex_visible.append(test_vertex)
                                    continue

                        if not back_vertex:
                            continue

                        # If get to this point, back_vertex is true.
                        if test_vertex_2D.is_inside_facet(test_facet_2D):
                            hidden[-1].append(test_vertex)
                            break
                        else:
                            updated_vertex_visible.append(test_vertex)

                    # Preparing for the next facet
                    previous_vertex_visible = updated_vertex_visible

                # End of for; no more facets left for this vertex.
                for vertex in previous_vertex_visible:
                    visible[-1].append(vertex)

        return visible,hidden
