"""`PolyhedronDrawing` contains classes and methods for drawing realistic
two-dimensional projections of three-dimensional polyhedra.
"""

import numpy as np
import re
import copy, textwrap

from .Geometry3D import Point, Vector, Segment, Contour, Facet,\
                        Polyhedron, rotate_point, ZERO_TOLERANCE
from .PolytopeTools import construct_convex_hull_from_coords

## Implementation of hidden line removal algorithm for interesecting solids -- Wei-I Hsu and J. L. Hock, Comput. & Graphics Vol. 15, No. 1, pp 67--86, 1991

polyhedron_color_list = ['blue', 'red', 'green', 'DarkViolet', 'DarkCyan',\
                         'FireBrick', 'NavyBlue', 'Fuchsia']

class Scene():
    """A class to contain, manage, and draw multiple polyhedra in a
    three-dimensional scene.

    Parameters
    ----------
    polyhedra : [Polyhedron]
        A list of polyhedra in the scene.
    a : float
        Cabinet projection angle (radians).
    thresh : float
        Threshold for various comparisons.
    """

    def __init__(self, polyhedra, a=np.arctan(2), thresh=ZERO_TOLERANCE):
        self.polyhedra = polyhedra
        self.cabinet_angle = a
        self._thresh = thresh

    @property
    def polyhedra(self):
        """[Polyhedron]: A list of all polyhedra in the scene.
        """
        return self._polyhedra

    @polyhedra.setter
    def polyhedra(self, polyhedra):
        for polyhedron in polyhedra:
            assert isinstance(polyhedron, Polyhedron)
        self._polyhedra = polyhedra

    @property
    def all_vertices(self):
        """[Point]: A list of all vertices in the scene.
        """
        return [vertex for polyhedron in self.polyhedra\
                     for vertex in polyhedron.vertices]

    @property
    def all_edges(self):
        """[Segment]: A list of all edges in the scene.
        """
        return [edge for polyhedron in self.polyhedra\
                     for edge in polyhedron.edges]

    @property
    def all_facets(self):
        """[Facet]: A list of all facets in the scene.
        """
        return [facet for polyhedron in self.polyhedra\
                      for facet in polyhedron.facets]

    @property
    def cabinet_angle(self):
        """`float`: The cabinet projection angle (radian) of the view of
        this scene.

        This angle lies between 0 (inclusive) and 2*pi (exclusive).
        """
        return self._cabinet_angle

    @cabinet_angle.setter
    def cabinet_angle(self, a):
        angle = a
        while angle >= 2*np.pi:
            angle = angle - 2*np.pi
        while angle < 0:
            angle = angle + 2*np.pi
        self._cabinet_angle = angle

    @property
    def viewing_vector(self):
        """Vector: The viewing vector of the view of this scene.

        This vector is proportional to (0.5*cosa,0.5*sina,1) where a is the
        cabinet projection angle.
        """
        a = self.cabinet_angle
        return Vector([0.5*np.cos(a),0.5*np.sin(a),1]).normalise()
        # return Vector([1, 0.5*np.cos(a),0.5*np.sin(a)]).normalise()

    @property
    def visible_segments(self):
        """[Segment]: The visible segments of the edges w.r.t. the current
        viewing vector.
        """
        return self._visible_segments

    @visible_segments.setter
    def visible_segments(self, visible):
        self._visible_segments = visible

    @property
    def visible_vertices(self):
        """[Point]: The visible vertices w.r.t. the current viewing vector.
        """
        return self._visible_vertices

    @visible_vertices.setter
    def visible_vertices(self, visible):
        self._visible_vertices = visible

    @property
    def hidden_segments(self):
        """[Segment]: The hidden segments of the edges w.r.t. the current
        viewing vector.
        """
        return self._hidden_segments

    @hidden_segments.setter
    def hidden_segments(self, hidden):
        self._hidden_segments = hidden

    @property
    def hidden_vertices(self):
        """[Point]: The hidden vertices w.r.t. the current viewing vector.
        """
        return self._hidden_vertices

    @hidden_vertices.setter
    def hidden_vertices(self, hidden):
        self._hidden_vertices = hidden

    @property
    def visible_segments_2D(self):
        """[Segment]: The visible segments of the edges w.r.t. the current
        viewing vector in the cabinet projection.
        """
        return list(map(lambda sublist :\
                [s.get_cabinet_projection(self.cabinet_angle)\
                 for s in sublist], self.visible_segments))

    @property
    def visible_vertices_2D(self):
        """[Point]: The visible vertices w.r.t. the current viewing vector
        in the cabinet projection.
        """
        return list(map(lambda sublist :\
                [v.get_cabinet_projection(self.cabinet_angle)\
                 for v in sublist], self.visible_vertices))

    @property
    def hidden_segments_2D(self):
        """[Segment]: The hidden segments of the edges w.r.t. the current
        viewing vector in the cabinet projection.
        """
        return list(map(lambda sublist :\
                [s.get_cabinet_projection(self.cabinet_angle)\
                 for s in sublist], self.hidden_segments))

    @property
    def hidden_vertices_2D(self):
        """[Point]: The hidden vertices w.r.t. the current viewing vector
        in the cabinet projection.
        """
        return list(map(lambda sublist :\
                [v.get_cabinet_projection(self.cabinet_angle)\
                 for v in sublist], self.hidden_vertices))

    def centre_scene(self):
        """Translate all polyhedra in the scene such that the geometric centre
        is at the origin.
        """
        vecsum = Vector()
        for vertex in self.all_vertices:
            vecsum = vecsum + Vector.from_point(vertex)
        centre_vector = vecsum/len(self.all_vertices)
        centred_polyhedra = []
        for polyhedron in self.polyhedra:
            centred_facets = []
            for facet in polyhedron.facets:
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
        self.__init__(centred_polyhedra, self.cabinet_angle, self._thresh)

    def rotate_scene(self, angle, direction):
        """Rotate the entire scene through an angle `angle` about the vector
        `direction`.
        """
        rotated_polyhedra = []
        for polyhedron in self.polyhedra:
            rotated_facets = []
            for facet in polyhedron.facets:
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
        self.__init__(rotated_polyhedra, self.cabinet_angle, self._thresh)

    def _get_visible_hidden_segments(self, thresh=ZERO_TOLERANCE):
        # Visibility is only meaningful in the projected view, so we check for
        # visibility  of any back edges or segments in the projected view.
        # However, we store all segments in the full object space.
        visible = []
        hidden = []
        for polyhedron in self.polyhedra:
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
                                anchor = intersection_plane
                            anchor_2D = anchor\
                                    .get_cabinet_projection(self.cabinet_angle)
                            n,J_points,segments_2D_hidden,segments_2D_visible =\
                                    back_test_segment_2D.intersects_facet\
                                            (test_facet_2D, anchor_2D)
                            # We need to convert the 2D information back to 3D data.
                            # We need 3D data for further testing with other facets.
                            # For parallel projection, mu is the same in both 3D
                            # and 2D.
                            for (segments_2D, mu_start, mu_end) in segments_2D_hidden:
                                A = back_test_segment.get_fraction_of_segment\
                                            (mu_start, anchor)[0]
                                B = back_test_segment.get_fraction_of_segment\
                                            (mu_end, anchor)[0]
                                updated_segments_hidden.append(Segment([A,B]))
                            for (segments_2D, mu_start, mu_end) in segments_2D_visible:
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
                for segment in updated_segments_visible:
                    visible[-1].append(segment)

        return visible,hidden

    def _get_visible_hidden_vertices(self, thresh=ZERO_TOLERANCE):
        # Visibility is only meaningful in the projected view, so we check for
        # visibility  of any back vertices in the projected view.
        # However, we store all vertices in the full object space.
        visible = []
        hidden = []
        for polyhedron in self.polyhedra:
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
                for vertex in updated_vertex_visible:
                    visible[-1].append(vertex)

        return visible,hidden

    def write_to_tikz(self, output, bb=None):
        """Write the current scene to a standalone TeX file.

        Parameters
        ----------
        output : `str`
            Output file name.
        bb : list of tuples
            [(xmin,xmax),(ymin,ymax)]
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
                """.format(xmin,ymin,xmax,ymax)

        all_str = preamble_str

        for pi,polyhedron in enumerate(self.polyhedra):
            color = polyhedron_color_list[pi]

            vertex_labels = []
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
                    [circle, fill = {}, draw = black, inner sep=1pt]{{}};
                """.format(x, y, color)


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
                    [circle, fill = {}!50!white, draw = black!50!white,\
                    inner sep=1pt]{{}};
                """.format(x, y, color)
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
                \\end{tikzpicture}
                \\end{document}
                """

        all_str = all_str + end_str

        with open(output, 'w') as f:
            f.write(textwrap.dedent(all_str))


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
                K = int(l.group())
            else:
                if line == '':
                    continue
                l = re.search("([^ \t\d]+)\s+", line)
                assert l is not None
                point_type = l.group(1)
                l = re.findall("[+-]?\d+\.\d+", line)
                point_coordinates = []
                for coordinate in l:
                    point_coordinates.append(coordinate)
                assert len(point_coordinates) == K
                if point_type not in polyhedron_dict:
                    polyhedron_dict[point_type] = []
                polyhedron_dict[point_type].append(point_coordinates)
                if point_type not in point_order:
                    point_order.append(point_type)

        for point_type in polyhedron_dict.keys():
            polyhedron_dict[point_type] = np.array(polyhedron_dict[point_type])

        for point_type in point_order:
            polyhedron_list.append(construct_convex_hull_from_coords\
                                        (polyhedron_dict[point_type]))
