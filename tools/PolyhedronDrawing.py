"""`PolyhedronDrawing` contains classes and methods for drawing realistic
two-dimensional projections of three-dimensional polyhedra.
"""

import numpy as np

## Implementation of hidden line removal algorithm for interesecting solids -- Wei-I Hsu and J. L. Hock, Comput. & Graphics Vol. 15, No. 1, pp 67--86, 1991

class PolyhedronDrawing():
    """A class to contain, manage, and draw multiple polyhedra in three dimensions.

    Parameters
    ----------
    polyhedra : [Polyhedron]
       A list of polyhedra in the scene.
    """

    def __init__(self, polyhedra):
        self.polyhedra = polyhedra

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
        return [edge for edge in polyhedron.edges\
                     for polyhedron in self.polyhedra]

    @property
    def all_facets(self):
        """[Facet] A list of all facets in the scene.
        """
        return [facet for facet in polyhedron.facets\
                     for polyhedron in self.polyhedra]

