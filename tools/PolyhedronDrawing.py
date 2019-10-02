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
        return self._polyhedra

    @polyhedra.setter
    def polyhedra(self, polyhedra):
        for polyhedron in polyhedra:
            assert isinstance(polyhedron, Polyhedron)
        self._polyhedra = polyhedra


