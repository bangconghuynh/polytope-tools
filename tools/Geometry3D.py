# -*- coding: utf-8 -*-
"""`Geometry3D` contains classes for general three-dimensional
geometrical objects.
"""

import numpy as np
from scipy.spatial import ConvexHull

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

    @property
    def norm(self):
        """`float`: Frobenius norm of the current vector.
        """
        return np.linalg.norm(self.components)

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

    def normalise(self, thresh=ZERO_TOLERANCE):
        """Normalise the current vector.

        Parameters
        ----------
        thresh : `float`
            Threshold to consider if the current vector is the null vector.

        Returns
        -------
        Vector
            A unit vector parallel to the current vector.
        """
        norm = self.norm
        if norm < thresh:
            raise ZeroDivisionError('{} has norm {} and is a null vector.'\
                                    .format(self, norm))
        else:
            return self/self.norm


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
        self.direction = direction

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

        When a non-unit vector is supplied, it will be normalised automatically.

        When a vector with fewer positive coefficients than negative components
        is supplied, its negative will be taken.
        """
        return self._direction

    @direction.setter
    def direction(self, vector):
        pos_count = len([x for x in vector.components if x > 0])
        neg_count = len([x for x in vector.components if x < 0])
        if pos_count >= neg_count:
            self._direction = vector.normalise()
        else:
            self._direction = -vector.normalise()

    def __str__(self):
        return "Line[{} + lambda*{}]".format(self.anchor, self.direction)

    __repr__ = __str__

    def get_point(self, param):
        """Obtain a point on the line satisfying

            point = `anchor` + `param` * `direction`

        for a particular value of `param`.

        Parameters
        ----------
        param : `float`
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
        thresh : `float`
            Threshold to determine if the cross product between the two
            vectors is the null vector.

        Returns
        -------
        `bool`
            `True` if the two lines are parallel, `False` if not.
        """
        if self.direction.cross(other.direction).norm < thresh:
            return True
        else:
            return False

    def contains_point(self, point, thresh=ZERO_TOLERANCE):
        """Check if `point` lies on the current line.

        Parameters
        ----------
        point : Point
            A point to check for collinearity with the current line.
        thresh : `float`
            Threshold to determine if the perpendicular distance between
            `point` and the current line is zero.

        Returns
        -------
        bool
            `True` if `point` lies on the current line, `False` if not.
        """
        anchor_to_point = self.anchor.get_vector_to(point)
        if anchor_to_point.cross(self.direction).norm\
                < thresh:
            return True
        else:
            return False

    def intersects_line(self, other, thresh=ZERO_TOLERANCE):
        """Check if the current line intersects `other`.

        Parameters
        ----------
        other : Line
            A line to check for intersection with the current line.
        thresh : `float`
            Threshold to determine if the shortest distance between the two
            lines is zero.

        Returns
        -------
        n : `int`
            Number of points of intersection.
        intersection : None if `n` is zero, Point if `n` == 1, Line if `n` == `inf`
            Intersection object.
        """
        if self.is_parallel(other, thresh):
            if self.contains_point(other.anchor, thresh):
                return np.inf,self
            else:
                return 0,None
        else:
            interanchor = self.anchor.get_vector_to(other.anchor)
            if interanchor.norm < thresh:
                return 1,self.anchor
            else:
                v = self.direction.cross(other.direction).normalise()
                if abs(interanchor.dot(v)) < thresh: # shortest distance
                    # Solving for lambda and mu in
                    #     a1 + lambda*d1 = a2 + mu*d2
                    # <=> lambda*d1 - mu*d2 = a2 - a1
                    # <=> [d1 -d2] [lambda mu]T = a2 - a1
                    # Three equations, two unknowns, i.e.
                    # [d1 -d2] is a 3x2 rectangular matrix.
                    # We solve for lambda and mu in all three pairs of
                    # dimensions and check for consistency.
                    # In some pairs, there might be no or infinitely
                    # many solutions, hence singular matrices, hence
                    # try-except blocks.
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
                    assert point_diff.norm < thresh
                    return 1,point_on_self
                else:
                    return 0,None


class Segment(object):
    """A line segment in a three-dimensional Euclidean space.

    Mathematically, this line is a collection of all points satisfying
    the vector equation

        point = `anchor` + param * `direction`

    where param is a real parameter in a specified interval.

    Parameters
    ----------
    endpoints : [Point]
        A list of two points defining the endpoints of the segment.
    """

    def __init__(self, endpoints):
        self.endpoints = endpoints

    @property
    def endpoints(self):
        """[Point]: A list of two points defining the endpoints of the segment.
        """
        return self._endpoints

    @endpoints.setter
    def endpoints(self, endpoints):
        self._endpoints = endpoints

    @property
    def length(self):
        """`float`: Length of the current segment.
        """
        return np.linalg.norm(self.endpoints[0].get_vector_to(self.endpoints[1]))

    @property
    def associated_line(self):
        """Line: The line containing this segment.
        """
        direction = self.endpoints[0].get_vector_to(self.endpoints[1])
        return Line(self.endpoints[0], direction)

    def __str__(self):
        return "Segment[{}, {}]".format(self.endpoints[0], self.endpoints[1])

    __repr__ = __str__

    def contains_point(self, point, thresh=ZERO_TOLERANCE):
        """Check if `point` lines on the current segment.

        Parameters
        ----------
        point : Point
            A point to check for membership of the current segment.
        thresh : `float`
            Threshold to determine if the perpendicular distance between
            `point` and the line containing this segment is zero.

        Returns
        -------
        `bool`
            `True` if `point` lies on the current segment, `False` if not.
        """
        if self.associated_line.contains_point(point, thresh):
            AP = self.endpoints[0].get_vector_to(point)
            BP = self.endpoints[1].get_vector_to(point)
            if AP.norm < thresh or BP.norm < thresh:
                return True
            AB = self.endpoints[0].get_vector_to(self.endpoints[1])
            for i in range(3):
                if abs(AB[i]) >= thresh:
                    ratio = AP[i]/AB[i]
                    break
            for i in range(3):
                assert (abs(AP[i]-ratio*AB[i]) < thresh)
            if -thresh <= ratio and ratio <= 1.0+thresh:
                return True
            else:
                return False
        else:
            return False

    def intersects_line(self, line, thresh=ZERO_TOLERANCE):
        """Check if the current segment intersects `line`.

        Parameters
        ----------
        line : Line
            A line to check for intersection with the current segment.
        thresh : `float`
            Threshold to determine if the shortest distance between the line
            and the current segment is zero.

        Returns
        -------
        n : int
            Number of points of intersection.
        intersection : None if `n` is zero, Point if `n` == 1, Segment if `n` == `inf`
            Intersection object.
        """
        n_line,intersection_line = self.associated_line.intersects_line(line, thresh)
        if n_line == 0:
            return 0,None
        elif n_line == 1:
            if self.contains_point(intersection_line, thresh):
                return 1,intersection_line
            else:
                return 0,None
        else:
            return np.inf,self

    def intersects_segment(self, other, thresh=ZERO_TOLERANCE):
        """Check if the current segment intersects `other`.

        Parameters
        ----------
        other : Segment
            A segment to check for intersection with the current segment.
        thresh : `float`
            Threshold to determine if the shortest distance between the two
            segments is zero.

        Returns
        -------
        n : int
            Number of points of intersection.
        intersection : None if `n` is zero, Point if `n` == 1, Segment if `n` == `inf`
            Intersection object.
        """
        n_line,intersection_line = self.intersects_line(other.associated_line, thresh)
        if n_line == 0:
            return n_line,intersection_line
        elif n_line == 1:
            if self.contains_point(intersection_line, thresh)\
                    and other.contains_point(intersection_line, thresh):
                return 1,intersection_line
            else:
                return 0,None
        else:
            # Both segments on the same line.
            # - No overlap.
            # - Intersect at endpoints.
            # - Partial overlap.
            # - One segment lies entirely within the other.
            if self.contains_point(other.endpoints[0], thresh):
                if self.contains_point(other.endpoints[1], thresh):
                    return np.inf,other
                else:
                    if other.contains_point(self.endpoints[0], thresh):
                        intersection = Segment([self.endpoints[0],other.endpoints[0]])
                        if intersection.length < thresh:
                            return 1,self.endpoints[0]
                        else:
                            return np.inf,intersection
                    else:
                        assert other.contains_point(self.endpoints[1], thresh)
                        intersection = Segment([self.endpoints[1],other.endpoints[0]])
                        if intersection.length < thresh:
                            return 1,self.endpoints[1]
                        else:
                            return np.inf,intersection
            elif self.contains_point(other.endpoints[1], thresh):
                if other.contains_point(self.endpoints[0], thresh):
                    intersection = Segment([self.endpoints[0],other.endpoints[1]])
                    if intersection.length < thresh:
                        return 1,self.endpoints[0]
                    else:
                        return np.inf,intersection
                else:
                    assert other.contains_point(self.endpoints[1], thresh)
                    intersection = Segment([self.endpoints[1],other.endpoints[1]])
                    if intersection.length < thresh:
                        return 1,self.endpoints[1]
                    else:
                        return np.inf,intersection
            else:
                return 0,None


class Plane(object):
    """A plane in a three-dimensional Euclidean space.

    Mathematically, this plane is a collection of all points satisfying
    the vector equation

        `point` . `normal` = `constant`

    Parameters
    ----------
    normal : Vector
        A vector normal to the plane.
    point : Point
        A point on the plane.
    """

    def __init__(self, normal, point):
        self.normal = normal
        self.constant = Vector.from_point(point).dot(self.normal)

    @property
    def normal(self):
        """Vector: The unit vector normal to the plane with the largest number
        of positive coefficients.

        When a non-unit vector is supplied, it will be normalised automatically.

        When a vector with fewer positive coefficients than negative components
        is supplied, its negative will be taken.
        """
        return self._normal

    @normal.setter
    def normal(self, vector):
        pos_count = len([x for x in vector.components if x > 0])
        neg_count = len([x for x in vector.components if x < 0])
        if pos_count >= neg_count:
            self._normal = vector.normalise()
        else:
            self._normal = -(vector.normalise())

    @property
    def constant(self):
        """`float`: The constant in the vector equation corresponding to the
        unit vector normal to the plane with the largest number of positive
        coefficients.
        """
        return self._constant

    @constant.setter
    def constant(self, value):
        self._constant = value

    def __str__(self):
        return "Plane[r.{}={}]".format(self.normal, self.constant)

    __repr__ = __str__

    def contains_point(self, point, thresh=ZERO_TOLERANCE):
        """Check if `point` lies on the current plane.

        Parameters
        ----------
        point : Point
            A point to check for membership with the current plane.
        thresh : `float`
            Threshold to determine if the coordinates of `point` satisfy the
            the plane equation.

        Returns
        -------
        bool
            `True` if `point` lies on the current plane, `False` if not.
        """
        rdotn = Vector.from_point(point).dot(self.normal)
        if abs(rdotn - self.constant) < thresh:
            return True
        else:
            return False

    def intersects_line(self, line, thresh=ZERO_TOLERANCE):
        """Check if the current plane intersects `line`.

        Parameters
        ----------
        line : Line
            A line to check for intersection with the current plane.
        thresh : `float`
            Threshold to determine if `line` is parallel to the current plane.

        Returns
        -------
        n : int
            Number of points of intersection.
        intersection : None if `n` is zero, Point if `n` == 1, Line if `n` == `inf`
            Intersection object.
        """
        ddotn = line.direction.dot(self.normal)
        if ddotn < thresh:
            if self.contains_point(line.anchor):
                return np.inf,line
            else:
                return 0,None
        else:
            adotn = Vector.from_point(line.anchor).dot(self.normal)
            lamb = (self.constant-adotn)/ddotn
            intersection = line.get_point(lamb)
            return 1,intersection

    def intersects_plane(self, other, thresh=ZERO_TOLERANCE):
        """Check if the current plane intersects `other`.

        Parameters
        ----------
        other : Plane
            A plane to check for intersection with the current plane.
        thresh : `float`
            Threshold to determine if two planes are parallel.

        Returns
        -------
        n : int
            Number of points of intersection.
        intersection : None if `n` is zero, Line or Plane if `n` == `inf`
            Intersection object.
        """
        n1crossn2 = self.normal.cross(other.normal)
        if n1crossn2.norm < thresh:
            if abs(self.constant - other.constant) < thresh:
                return np.inf,self
            else:
                return 0,None
        else:
            constants = np.array([[self.constant],\
                                  [other.constant]])
            # If n1crossn2 is perpendicular to a Cartesian axis, then
            # all points on the line have a fixed component w.r.t. that
            # axis and we cannot choose that component arbitrarily.
            # There must therefore be a unique solution in that component.
            perpendicular_directions = []
            for i in range(3):
                if abs(n1crossn2[i]) < thresh:
                    perpendicular_directions.append(i)
            print(perpendicular_directions)
            if len(perpendicular_directions) == 0:
                # No restrictions. We pick z=0, then solve for x and y.
                i,j,k = 0,1,2
            elif len(perpendicular_directions) == 1:
                # One perpendicular direction. We must include this
                # in the 2D problem.
                i = perpendicular_directions[0]
                [j,k] = list(set(perpendicular_directions)^set(range(3)))
            else:
                # Two perpendicular directions. We must include both
                # in the 2D problem.
                i,j = perpendicular_directions[0],perpendicular_directions[1]
                [k] = list(set(perpendicular_directions)^set(range(3)))
            # Solving the 2D problem
            normals_ij = np.array([[self.normal[i], self.normal[j]],\
                                   [other.normal[i], other.normal[j]]])
            ij = np.dot(np.linalg.inv(normals_ij), constants)
            print(ij)
            coords = np.array([0,0,0])
            coords[i] = ij[0]
            coords[j] = ij[1]
            point_on_line = Point(coords)
            assert self.contains_point(point_on_line, thresh) and\
                   other.contains_point(point_on_line, thresh)
            return np.inf,Line(point_on_line, n1crossn2)


class Contour(object):
    """A contour in a three-dimensional Euclidean space.

    (Hsu & Hock, 1991) "A contour is a closed planar polygon that may be one
    of ordered orientation."

    Parameters
    ----------
    edges : [Segment]
        A list of segments defining the edges of the contour. Any ordered
        orientation is implied by the order of the list.
    """

    def __init__(self, edges):
        self.edges = edges

    @property
    def edges(self):
        """[Segment]: A list of segments defining the edges of the contour.
        """
        return self._edges

    @edges.setter
    def edges(self, edges):
        self._edges = edges

    @property
    def associated_plane(self):
        v1 = self.edges[0].associated_line.direction
        v2 = self.edges[1].associated_line.direction
        n = v1.cross(v2)
        A = self.edges[0].endpoints[0]
        return Plane(n, A)
