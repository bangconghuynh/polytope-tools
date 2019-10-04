"""`PolyhedronDrawing` contains classes and methods for drawing realistic
two-dimensional projections of three-dimensional polyhedra.
"""

import numpy as np
import copy

from .Geometry3D import Vector, Polyhedron, Segment, ZERO_TOLERANCE

## Implementation of hidden line removal algorithm for interesecting solids -- Wei-I Hsu and J. L. Hock, Comput. & Graphics Vol. 15, No. 1, pp 67--86, 1991

class PolyhedronDrawing():
    """A class to contain, manage, and draw multiple polyhedra in three dimensions.

    Parameters
    ----------
    polyhedra : [Polyhedron]
        A list of polyhedra in the scene.
    a : float
        Cabinet projection angle (radians).
    """

    def __init__(self, polyhedra, a=np.arctan(2)):
        self.polyhedra = polyhedra
        self.cabinet_angle = a

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

        This vector is propertional to (0.5*cosa,0.5*sina,1) where a is the
        cabinet projection angle.
        """
        a = self.cabinet_angle
        return Vector([0.5*np.cos(a),0.5*np.sin(a),1]).normalise()

    def get_visible_hidden_segments(self, thresh=ZERO_TOLERANCE):
        """
        """
        # Visibility is only meaningful in the projected view, so we check for
        # visibility in the projected view. However, we store all segments in
        # the full object space.
        visible = []
        hidden = []
        for edge in self.all_edges:
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
                                elif fe_vector.dot(front_normal) < -thresh:
                                    back_test_segment = Segment([\
                                                endpoint,\
                                                intersection_plane])
                                    back_segment = True
                                else:
                                    # This endpoint lies on the plane.
                                    continue
                            updated_segments_visible.append(front_test_segment)

                        if not back_segment:
                            continue

                        # If get to this point, back_test_segment exists.
                        # We can only handle facets consisting of one contour
                        # at the moment.
                        assert len(test_facet.contours) == 1,\
                            'We can only handle facets consisting of\
                             one contour at the moment.'
                        back_test_segment_2D = back_test_segment\
                                .get_cabinet_projection(self.cabinet_angle)
                        test_contour_2D = test_facet.contours[0]\
                                .get_cabinet_projection(self.cabinet_angle)
                        intersection_plane_2D = intersection_plane\
                                .get_cabinet_projection(self.cabinet_angle)
                        n,J_points,segments_2D_hidden,segments_2D_visible =\
                                back_test_segment_2D.intersects_contour\
                                        (test_contour_2D, intersection_plane_2D)

                        # We need to convert the 2D information back to 3D data.
                        # We need 3D data for further testing with other facets.
                        # For parallel projection, mu is the same in both 3D
                        # and 2D.
                        for (segments_2D, mu_start, mu_end) in segments_2D_hidden:
                            A = back_test_segment.get_fraction_of_segment\
                                        (mu_start, intersection_plane)[0]
                            B = back_test_segment.get_fraction_of_segment\
                                        (mu_end, intersection_plane)[0]
                            updated_segments_hidden.append(Segment([A,B]))
                        for (segments_2D, mu_start, mu_end) in segments_2D_visible:
                            A = back_test_segment.get_fraction_of_segment\
                                        (mu_start, intersection_plane)[0]
                            B = back_test_segment.get_fraction_of_segment\
                                        (mu_end, intersection_plane)[0]
                            updated_segments_visible.append(Segment([A,B]))
                # End of while; no more segments left in previous_segments_visible for
                # this facet

                # Preparing for the next facet
                previous_segments_visible = updated_segments_visible

                # Saving all segments from this edge hidden by this facet.
                for segment in updated_segments_hidden:
                    hidden.append(segment)

            # End of for; no more facets left for this edge.
            # Saving all visible segments from this edge.
            for segment in updated_segments_visible:
                visible.append(segment)

        return visible,hidden
