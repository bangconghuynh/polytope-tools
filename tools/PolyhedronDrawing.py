"""`PolyhedronDrawing` contains classes and methods for drawing realistic
two-dimensional projections of three-dimensional polyhedra.
"""

import numpy as np

from .Geometry3D import Vector, Polyhedron

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

    def get_visible_hidden_segments(self):
        """
        """
        # Visibility is only meaningful in the projected view, so we check for
        # visibility in the projected view. However, we store all segments in
        # the full object space.
        visible = []
        hidden = []
        for edge in self.all_edges:
            edge_visible = [edge]
            for test_facet in self.all_facets:
                test_facet_2D = test_facet.\
                                    get_cabinet_projection(self.cabinet_angle)
                for test_edge in edge_visible:
                    test_edge_2D = test_edge.\
                                    get_cabinet_projection(self.cabinet_angle)
                    if not test_edge_2D.intersects_bounding_box(test_facet_2D):
                        # test_edge not covered by test_facet in 2D projection
                        continue
                    else:
                        pass

