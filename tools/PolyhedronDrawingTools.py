# -*- coding: utf-8 -*-
"""`PolyhedronDrawingTools` contains necessary classes and functions to draw
three-dimensional intersecting polyhedra with careful treatment of hidden lines.
"""

import sys
import numpy as np

## Implementation of hidden line removal algorithm for interesecting solids -- Wei-I Hsu and J. L. Hock, Comput. & Graphics Vol. 15, No. 1, pp 67--86, 1991

# Constants
ZERO_TOLERANCE = 1e-15

class Point(object):
    """A point in a three-dimensional Euclidean space.

    Parameters
    ----------
    coordinates : `array_like`, optional
        A sequence of three coordinate values.
    """

    def __init__(self, coordinates=np.array([0,0,0])):
        """Create a point from a set of coordinate values.
        """
        self.coordinates = np.array(coordinates)

    @property
    def coordinates(self):
        """`array_like`: A sequence of three coordinate values.

        When fewer than three values are supplied, the missing coordinates will
        default to zero.

        When more than three values are supplied, a `ValueError` exception will
        be raised.
        """
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords):
        if coords.shape[0] > 3:
            raise ValueError("{} has more than three dimensions.".format(coords))
        self._coordinates = coords
        while self._coordinates.shape[0] < 3:
            self._coordinates = np.append(self._coordinates, [0], axis=0)

    @classmethod
    def from_vector(cls, vector):
        """Create a point from a position vector.

        Parameters
        ----------
        vector : Vector
            Position vector of the desired point.

        Returns
        -------
        Point
            A point corresponding to `vector`.
        """
        return cls(vector.components)

    def __str__(self):
        return "Point({},{},{})".format(self[0], self[1], self[2])

    __repr__ = __str__

    def __getitem__(self, key):
        return self.coordinates[key]

    def get_vector_to(self, other):
        """Find the displacement vector to another point.

        Parameters
        ----------
        other : Point
            A point in the same Euclidean space.

        Returns
        -------
        Vector
            Displacement from the current point to `other`
        """
        return Vector(other.coordinates-self.coordinates)


class Vector(object):
    """A vector in a three-dimensional Euclidean space.

    Parameters
    ----------
    components : `array_like`, optional
        A sequence of three component values.
    """

    def __init__(self, components=np.array([0,0,0])):
        self.components = np.array(components)

    @property
    def components(self):
        """`array_like`: A sequence of three coordinate values.

        When fewer than three values are supplied, the missing components will
        default to zero.

        When more than three values are supplied, a `ValueError` exception will
        be raised.
        """
        return self._components

    @components.setter
    def components(self, comps):
        if comps.shape[0] > 3:
            raise ValueError("{} has more than three dimensions.".format(comps))
        self._components = comps
        while self._components.shape[0] < 3:
            self._components = np.append(self._components, [0], axis=0)

    @classmethod
    def from_point(cls, point):
        """Create a position vector from a point.

        Parameters
        ----------
        point : Point
            Point whose position vector will be returned.

        Returns
        -------
        Vector
            The position vector corresponding to `point`.
        """
        return cls(point.coordinates)

    def __str__(self):
        return "Vector({},{},{})".format(self[0], self[1], self[2])

    __repr__ = __str__

    def __getitem__(self, key):
        return self.components[key]

    def __add__(self, other):
        return Vector(self.components+other.components)

    def __sub__(self, other):
        return Vector(self.components-other.components)

    def __neg__(self):
        return Vector(-self.components)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.dot(other)
        else:
            try:
                return Vector(other*self.components)
            except:
                raise TypeError('{} is not a number or a Vector.'. format(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        try:
            return Vector(self.components/other)
        except:
            raise TypeError('{} is not a number or a Vector.'. format(other))

    def dot(self, other):
        """Dot product between the current vector and `other`.

        Parameters
        ----------
        other : Vector
            Another vector with which the dot product is to be calculated.

        Returns
        -------
        `number`
            The dot product between `self` and `other`.
        """
        return np.dot(self.components, other.components)

    def cross(self, other):
        """Cross product between the current vector and `other`.

        Parameters
        ----------
        other : Vector
            Another vector with which the cross product is to be calculated.

        Returns
        -------
        Vector
            The cross product between `self` and `other`.
        """
        return Vector(np.cross(self.components, other.components))

    def get_norm(self):
        """Find the norm of the current vector`.

        Returns
        -------
        float
            The norm of `self`.
        """
        return np.linalg.norm(self.components)

    def normalise(self, thresh=ZERO_TOLERANCE):
        """Normalise the current vector.

        Parameters
        ----------
        thresh : float
            Threshold to consider if the current vector is the null vector.

        Returns
        -------
        Vector
            A unit vector parallel to the current vector.
        """
        norm = self.get_norm()
        if norm < thresh:
            raise ZeroDivisionError('{} has norm {} and is a null vector.'\
                .format(self, norm))
        else:
            return self/self.get_norm()


class Line(object):
    """A line in a three-dimensional Euclidean space.

    Mathematically, this line is a collection of all points satisfying
    the vector equation

        point = `anchor` + param * `direction`

    where param is a real parameter.

    Parameters
    ----------
    anchor : Point
        A point on the line.
    direction : Vector
        A vector parallel to the line.
    """

    def __init__(self, anchor, direction):
        self.anchor = anchor
        self.direction = direction.normalise()

    @property
    def anchor(self):
        """Point: A point on the line.
        """
        return self._anchor

    @anchor.setter
    def anchor(self, point):
        self._anchor = point

    @property
    def direction(self):
        """Vector: A vector parallel to the line.
        """
        return self._direction

    @direction.setter
    def direction(self, vector):
        self._direction = vector

    def __str__(self):
        return "Line[{} + lambda*{}]".format(self.anchor, self.direction)

    __repr__ = __str__

    def get_point(self, param):
        """Obtain a point on the line satisfying

            point = `anchor` + `param` * `direction`

        for a particular value of `param`.

        Parameters
        ----------
        param : float
            A real parameter specifying the point to be returned.

        Returns
        -------
        Point
            A point on the line satisfying the above condition.
        """
        v = Vector.from_point(self.anchor) + param*self.direction
        return Point.from_vector(v)

    def is_parallel(self, other, thresh=ZERO_TOLERANCE):
        """Check if the current line is parallel to `other`.

        Parameters
        ----------
        other : Line
            A line to check for parallelism with the current line.
        thresh : float
            Threshold to determine if the cross product between the two
            vectors is the null vector.

        Returns
        -------
        bool
            `True` if the two lines are parallel, `False` if not.
        """
        if self.direction.cross(other.direction).get_norm() < thresh:
            return True
        else:
            return False

    def contains_point(self, point, thresh=ZERO_TOLERANCE):
        """Check if `point` lines on the current line.

        Parameters
        ----------
        point : Point
            A point to check for collinearity with the current line.
        thresh : float
            Threshold to determine if the perpendicular distance between
            `point` and the current line is zero.

        Returns
        -------
        bool
            `True` if `point` lies on the current line, `False` if not.
        """
        anchor_to_point = self.anchor.get_vector_to(point)
        if anchor_to_point.cross(self.direction).get_norm()\
                < thresh:
            return True
        else:
            return False

    def intersects_line(self, other, thresh=ZERO_TOLERANCE):
        """Check if the current line intersects `other`.

        Parameters
        ----------
        other : Line
            A line to check for parallelism with the current line.
        thresh : float
            Threshold to determine if the shortest distance between the two
            lines is zero.

        Returns
        -------
        n : int
            Number of points of intersection.
        l : list of Point. If `n` is zero or inf, `l` is an empty list.
        """
        if self.is_parallel(other, thresh):
            if self.contains_point(other.anchor, thresh):
                return np.inf,[]
            else:
                return 0,[]
        else:
            interanchor = self.anchor.get_vector_to(other.anchor)
            if interanchor.get_norm() < thresh:
                return 1,[self.anchor]
            else:
                v = self.direction.cross(other.direction).normalise()
                if abs(interanchor.dot(v)) < thresh: # shortest distance
                    anchors = self.anchor.get_vector_to(other.anchor)
                    coeffs = []
                    try:
                        directions_xy = np.array(\
                            [[self.direction[0], -other.direction[0]],\
                             [self.direction[1], -other.direction[1]]])
                        anchors_xy = anchors[0:2] 
                        coeffs.append(np.dot(np.linalg.inv(directions_xy),\
                                             anchors_xy))
                    except:
                        pass
                    try:
                        directions_yz = np.array(\
                            [[self.direction[1], -other.direction[1]],\
                             [self.direction[2], -other.direction[2]]])
                        anchors_yz = anchors[1:3] 
                        coeffs.append(np.dot(np.linalg.inv(directions_yz),\
                                             anchors_yz))
                    except:
                        pass
                    try:
                        directions_xz = np.array(\
                            [[self.direction[0], -other.direction[0]],\
                             [self.direction[2], -other.direction[2]]])
                        anchors_xz = anchors[[0,2]] 
                        coeffs.append(np.dot(np.linalg.inv(directions_xz),\
                                             anchors_xz))
                    except:
                        pass
                    assert len(coeffs) >= 1
                    point_on_self = self.get_point(coeffs[0][0])
                    point_on_other = other.get_point(coeffs[0][1])
                    point_diff = point_on_self.get_vector_to(point_on_other)
                    assert point_diff.get_norm() < thresh
                    return 1,[point_on_self]
                else:
                    return 0,[]
