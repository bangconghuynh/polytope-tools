"""`Geometry3D` contains classes for general three-dimensional
geometrical objects.
"""

from __future__ import annotations

import operator
from typing import Sequence, List, Tuple, Union
import numpy as np # type: ignore
from scipy.spatial.transform import Rotation # type: ignore

# Constants
ZERO_TOLERANCE = 1e-14

# Types
Scalar = Union[int, float] # Geometry over R (and not C).

class FiniteObject(object):
    """A generic finite geometrical object in a three-dimensional Euclidean\
    space. Each finite object has a bounding box.
    """

    @property
    def bounding_box(self) -> List[Tuple[Scalar,Scalar]]:
        """A list of three tuples for the :math:`x`, :math:`y`, and :math:`z`\
        dimensions. Each tuple contains the minimum and maximum coordinates\
        in the corresponding dimension.
        """
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, bb: List[Tuple[Scalar,Scalar]]) -> None:
        assert len(bb) == 3, 'A list of three tuples must be supplied.'
        for i in range(3):
            assert type(bb[i]) is tuple,\
                   'A list of three tuples must be supplied.'
        self._bounding_box = bb

    @property
    def bounding_box_planes(self) -> List[Tuple[Plane,Plane]]:
        """A list of three tuples for the :math:`x`, :math:`y`, and :math:`z`\
        dimensions. Each tuple contains the planes at the minimum and maximum\
        coordinates in the corresponding dimension.
        """
        [(xmin,xmax),(ymin,ymax),(zmin,zmax)] = self.bounding_box
        xmin_plane = Plane(Vector([1,0,0]), Point([xmin,0,0]))
        xmax_plane = Plane(Vector([1,0,0]), Point([xmax,0,0]))
        ymin_plane = Plane(Vector([0,1,0]), Point([0,ymin,0]))
        ymax_plane = Plane(Vector([0,1,0]), Point([0,ymax,0]))
        zmin_plane = Plane(Vector([0,0,1]), Point([0,0,zmin]))
        zmax_plane = Plane(Vector([0,0,1]), Point([0,0,zmax]))
        return [(xmin_plane,xmax_plane),(ymin_plane,ymax_plane),\
                (zmin_plane,zmax_plane)]

    def _find_bounding_box(self) -> None:
        """Find the bounding box of the current geometrical object."
        """
        print("find_bounding_box must be implemented in the child class.")

    def intersects_bounding_box(self,
                                other: FiniteObject,
                                thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current bounding box intersects with `other`'s
        bounding box.

        :param other: An object that has a bounding box.
        :param thresh: The bounding box of `other` is increased by this amount\
                       in each dimension.
        :returns: `True` if the two bounding boxes intersect, `False` if not.
        """
        if self.bounding_box[0][0] > other.bounding_box[0][1]+thresh or\
           self.bounding_box[0][1] < other.bounding_box[0][0]-thresh or\
           self.bounding_box[1][0] > other.bounding_box[1][1]+thresh or\
           self.bounding_box[1][1] < other.bounding_box[1][0]-thresh or\
           self.bounding_box[2][0] > other.bounding_box[2][1]+thresh or\
           self.bounding_box[2][1] < other.bounding_box[2][0]-thresh:
            return False
        else:
            return True


class Point(FiniteObject):
    """A point in a three-dimensional Euclidean space.

    Two points are geometrically equal if and only if the distance between\
    them is smaller than the global constant `ZERO_TOLERANCE`.

    Two points can be compared using lexicographical comparison of the tuple\
    :math:`(x,y,z)`.
    """

    def __init__(self, coordinates: Sequence[Scalar] = [0,0,0]) -> None:
        """
        :param coordinates: A sequence of three coordinate values.
        """
        self.coordinates = np.array(coordinates)
        self._find_bounding_box()

    @property
    def coordinates(self) -> np.ndarray:
        """An array of three coordinate values.

        When fewer than three values are supplied, the missing coordinates will\
        default to zero.

        :raises ValueError: When more than three values are supplied.
        """
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords: np.ndarray) -> None:
        if coords.shape[0] > 3:
            raise ValueError("{} has more than three dimensions."\
                    .format(coords))
        self._coordinates = coords
        while self._coordinates.shape[0] < 3:
            self._coordinates = np.append(self._coordinates, [0], axis=0)

    @classmethod
    def from_vector(cls, vector: Vector) -> Point:
        """Create a point from a position vector.

        :param vector: Position vector of the desired point.

        :returns: A point corresponding to `vector`.
        """
        return cls(vector.components)

    def __str__(self) -> str:
        return "Point({},{},{})".format(self[0], self[1], self[2])

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.get_vector_to(other).norm < ZERO_TOLERANCE

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.get_vector_to(other).norm >= ZERO_TOLERANCE

    def __lt__(self, other: Point) -> bool:
        return (self.coordinates[0],self.coordinates[1],self.coordinates[2])\
             < (other.coordinates[0],other.coordinates[1],other.coordinates[2])

    def __gt__(self, other: Point) -> bool:
        return (self.coordinates[0],self.coordinates[1],self.coordinates[2])\
             > (other.coordinates[0],other.coordinates[1],other.coordinates[2])

    def __le__(self, other: Point) -> bool:
        return (self.coordinates[0],self.coordinates[1],self.coordinates[2])\
            <= (other.coordinates[0],other.coordinates[1],other.coordinates[2])

    def __ge__(self, other: Point) -> bool:
        return (self.coordinates[0],self.coordinates[1],self.coordinates[2])\
            >= (other.coordinates[0],other.coordinates[1],other.coordinates[2])

    def __getitem__(self, key: int) -> Scalar:
        return self.coordinates[key]

    def get_vector_to(self, other: Point) -> Vector:
        """Find the displacement vector to another point.

        :param other: A point in the same Euclidean space.

        :returns: Displacement vector from the current point to `other`.
        """
        return Vector(other.coordinates-self.coordinates)

    def get_cabinet_projection(self,
                               a: float = np.arctan(2)) -> Point:
        """The cabinet projection of the current point onto the :math:`xy`-plane.

        Let `a` = :math:`\\alpha` such that :math:`0 \\leq \\alpha < 2\\pi`.

        In a right-handed coordinate system, the cabinet projection projects\
        the point :math:`(x,y,z)` onto\
                :math:`(x-0.5z\cos\\alpha,y-0.5z\sin\\alpha,0)`.\
        Orthogonality between the :math:`x`- and :math:`y`-axes is maintained,\
        while the projected :math:`z`-axis makes an angle of `a` w.r.t. the\
        math:`x`-axis. In addition, the length of the receding lines is cut in\
        half, hence the factor of :math:`0.5`.

        The viewing vector is :math:`(0.5z\cos\\alpha,0.5z\sin\\alpha,z)`,\
        which is projected onto the origin. The projection lines are all\
        parallel to this vector since this projection is a parallel projection.

        :param a: Angle :math:`\\alpha` of projection (radians).\
                  :math:`0 \\leq \\alpha < 2\\pi`.

        :returns: The projected point.
        """
        x = self[0]
        y = self[1]
        z = self[2]
        return Point([x-0.5*z*np.cos(a), y-0.5*z*np.sin(a), 0])

    def is_same_as(self, other: object, thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current point is the same as `other`.

        :param point: A point in the same Euclidean space.
        :param thresh: Threshold to consider if the distance between the two\
                       points is zero.

        :returns: `True` if the two points are the same, `False` if not.
        """
        if not isinstance(other, Point):
            return NotImplemented
        return self.get_vector_to(other).norm < thresh

    def is_inside_contour(self, contour: Contour,
                          thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current point lies inside the contour `contour`.

        :param contour: A contour in the same Euclidean space.
        :param thresh: Threshold for various comparisons.

        :returns: `True` if the current point lies inside `contour`,\
                  `False` if not.
        """
        if contour.associated_plane.contains_point(self, thresh):
            if self.intersects_bounding_box(contour, thresh):
                for edge in contour.edges:
                    if edge.contains_point(self, thresh):
                        return True
                # Transform into the xy-plane
                n = contour.associated_plane.normal
                z = Vector([0,0,1])
                redges_xy = []
                if n.cross(z).norm >= thresh:
                    # Contour not approximately parallel to xy-plane
                    if n[2] < 0:
                        n = -n
                    rvec = n.cross(z).normalise()
                    rang = np.arccos(n.dot(z))
                    rself = rotate_point(self, rang, rvec)
                    rself_xy = Point([rself[0], rself[1], 0])
                    for edge in contour.edges:
                        rendpoint0 = rotate_point(edge.endpoints[0], rang, rvec)
                        rendpoint1 = rotate_point(edge.endpoints[1], rang, rvec)
                        rendpoint0_xy = Point([rendpoint0[0], rendpoint0[1], 0])
                        rendpoint1_xy = Point([rendpoint1[0], rendpoint1[1], 0])
                        redges_xy.append(Segment([rendpoint0_xy, rendpoint1_xy]))
                else:
                    rself_xy = Point([self[0], self[1], 0])
                    for edge in contour.edges:
                        rendpoint0 = edge.endpoints[0]
                        rendpoint1 = edge.endpoints[1]
                        rendpoint0_xy = Point([rendpoint0[0], rendpoint0[1], 0])
                        rendpoint1_xy = Point([rendpoint1[0], rendpoint1[1], 0])
                        redges_xy.append(Segment([rendpoint0_xy, rendpoint1_xy]))

                rcontour_xy = Contour(redges_xy)
                xmax = rcontour_xy.bounding_box[0][1]
                test_segment = Segment([rself_xy,\
                                        Point([xmax+1,rself_xy[1]])])
                n_intersections_with_edges = 0
                n_intersections_with_vertices = 0
                for i,test_edge in enumerate(redges_xy):
                    n_intersections,intersection =\
                            test_segment.intersects_segment(test_edge)
                    if n_intersections == 1:
                        n_intersections_with_edges += 1
                        assert isinstance(intersection, Point)
                        if intersection.is_same_as(test_edge.endpoints[0]) or\
                           intersection.is_same_as(test_edge.endpoints[1]):
                            prev_edge = redges_xy[i-1]
                            if intersection.is_same_as(prev_edge.endpoints[0],\
                                    thresh)\
                            or intersection.is_same_as(prev_edge.endpoints[1],\
                                    thresh):
                                shared_edge = prev_edge
                            else:
                                continue
                            test_segment_vec = test_segment.endpoints[0]\
                                    .get_vector_to(test_segment.endpoints[1])
                            test_edge_vec = rself_xy\
                                    .get_vector_to(test_edge.get_a_point())
                            shared_edge_vec = rself_xy\
                                    .get_vector_to(shared_edge.get_a_point())
                            value = test_segment_vec.cross(test_edge_vec).dot\
                                    (test_segment_vec.cross(shared_edge_vec))
                            if value < 0:
                                n_intersections_with_vertices += 1
                n_intersections_with_edges -= n_intersections_with_vertices
                if n_intersections_with_edges % 2 == 0:
                    return False
                else:
                    return True
            else:
                return False
        else:
            return False

    def is_inside_facet(self,
                        facet: Facet,
                        thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current point lies inside the facet `facet`.\
        If the facet consists of multiple contours, then the inside\
        of the facet is defined by the XOR operation (:cite:`article:Hsu1991`).

        :param facet: A facet in the same Euclidean space.
        :param thresh: Threshold for various comparisons.

        :returns: `True` if the current point lies inside `facet`,\
                  `False` if not.
        """
        n_inside_contours = 0
        for contour in facet.contours:
            if self.is_inside_contour(contour, thresh):
                n_inside_contours += 1
        if n_inside_contours % 2 == 0:
            return False
        else:
            return True

    def _find_bounding_box(self) -> None:
        self.bounding_box = [(self[0],self[0]),(self[1],self[1]),\
                             (self[2],self[2])]


class Vector(object):
    """A vector in a three-dimensional Euclidean space.

    Two vectors are equal if and only if each pair of corresponding components\
    differ by less than `ZERO_TOLERANCE`.
    """

    def __init__(self, components: Sequence[Scalar] = [0,0,0]) -> None:
        """
        :param components: A sequence of three component values.
        """
        self.components = np.array(components)

    @property
    def components(self) -> np.ndarray:
        """An array of three coordinate values.

        When fewer than three values are supplied, the missing components will\
        default to zero.

        :raises ValueError: When more than three values are supplied.
        """
        return self._components

    @components.setter
    def components(self, comps: np.ndarray) -> None:
        if comps.shape[0] > 3:
            raise ValueError("{} has more than three dimensions.".format(comps))
        self._components = comps
        while self._components.shape[0] < 3:
            self._components = np.append(self._components, [0], axis=0)

    @property
    def norm(self) -> float:
        """Frobenius norm of the current vector.
        """
        return np.linalg.norm(self.components)

    @classmethod
    def from_point(cls, point: Point) -> Vector:
        """Create a position vector from a point.

        :param point: Point whose position vector will be returned.

        :returns: The position vector corresponding to `point`.
        """
        return cls(point.coordinates)

    def __str__(self) -> str:
        return "Vector({},{},{})".format(self[0], self[1], self[2])

    __repr__ = __str__

    def __getitem__(self, key: int) -> Scalar:
        return self.components[key]

    def __add__(self, other: Vector) -> Vector:
        return Vector(self.components+other.components)

    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.components-other.components)

    def __neg__(self) -> Vector:
        return Vector(-self.components)

    def __mul__(self, scalar: Scalar) -> Vector:
        try:
            return Vector(scalar*self.components)
        except:
            raise TypeError('{} is not a scalar.'. format(scalar))

    __rmul__ = __mul__

    def __truediv__(self, scalar: Scalar) -> Vector:
        try:
            return Vector(self.components/scalar)
        except:
            raise TypeError('{} is not a scalar.'. format(scalar))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self.is_same_as(other, ZERO_TOLERANCE)

    def is_same_as(self,
                   other: object,
                   thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current vector is the same as `other`.

        :param vector: A vector in the same Euclidean space.
        :param thresh: Threshold to consider if the difference between the\
                       corresponding components is zero.

        :returns: `True` if the two vectors are the same, `False` if not.
        """
        if not isinstance(other, Vector):
            return NotImplemented
        return abs(self[0]-other[0])<thresh and\
               abs(self[1]-other[1])<thresh and\
               abs(self[2]-other[2])<thresh

    def get_cabinet_projection(self, a: float = np.arctan(2)) -> Vector:
        """The cabinet projection of the current vector onto the\
        :math:`xy`-plane.

        For a description of the cabinet projection, see\
                :meth:`Point.get_cabinet_projection`.

        :param a: Angle (radians) of projection.

        :returns: The projected vector.
        """
        x = self[0]
        y = self[1]
        z = self[2]
        return Vector([x-0.5*z*np.cos(a), y-0.5*z*np.sin(a), 0])

    def dot(self, other: Vector) -> Scalar:
        """Dot product between the current vector and `other`.

        :param other: Another vector with which the dot product is to be\
                      calculated.

        :returns: The dot product between `self` and `other`.
        """
        return np.dot(self.components, other.components)

    def cross(self, other: Vector) -> Vector:
        """Cross product between the current vector and `other`.

        :param other: Another vector with which the cross product is to be\
                      calculated.

        :returns: The cross product between `self` and `other`.
        """
        return Vector(np.cross(self.components, other.components))

    def normalise(self, thresh: float = ZERO_TOLERANCE) -> Vector:
        """Normalise the current vector.

        :raises ZeroDivisionError: if the current vector is a null vector.

        :param thresh: Threshold to consider if the current vector is a\
                       null vector.

        :returns: A unit vector parallel to the current vector.
        """
        norm = self.norm
        if norm < thresh:
            raise ZeroDivisionError('{} has norm {} and is a null vector.'\
                                    .format(self, norm))
        else:
            return self/self.norm


class Line:
    """A line in a three-dimensional Euclidean space.

    Mathematically, this line is a collection of all points satisfying
    the vector equation

    .. math:: \\boldsymbol{r} = \\boldsymbol{r}_0 + \\lambda \\boldsymbol{d}

    where :math:`\\lambda` is a scalar parameter.
    """

    def __init__(self, anchor: Point, direction: Vector) -> None:
        """
        :param anchor: A point on the line corresponding to the position vector\
                       :math:`\\boldsymbol{r}_0`.
        :param direction: A vector :math:`\\boldsymbol{d}` parallel to the line.
        """
        self.anchor = anchor
        self.direction = direction

    @property
    def anchor(self) -> Point:
        """A point on the line.
        """
        return self._anchor

    @anchor.setter
    def anchor(self, point: Point) -> None:
        self._anchor = point

    @property
    def direction(self) -> Vector:
        """A unit vector parallel to the line.

        When a non-unit vector is supplied, it will be normalised automatically.

        When a vector with fewer positive coefficients than negative components
        is supplied, its negative will be taken.
        """
        return self._direction

    @direction.setter
    def direction(self, vector: Vector) -> None:
        pos_count = len([x for x in vector.components if x > 0])
        neg_count = len([x for x in vector.components if x < 0])
        if pos_count >= neg_count:
            self._direction = vector.normalise()
        else:
            self._direction = -vector.normalise()

    def get_a_point(self) -> Point:
        """Return a point on the line. The easiest point to return
        is the anchor.

        :returns: A point on the line, chosen to be the anchor.
        """
        return self.anchor

    def __str__(self) -> str:
        return "Line[{} + lambda*{}]".format(self.anchor, self.direction)

    __repr__ = __str__

    def get_cabinet_projection(self, a: float = np.arctan(2)) -> Line:
        """The cabinet projection of the current line onto the\
        :math:`xy`-plane.

        For a description of the cabinet projection, see\
                :meth:`Point.get_cabinet_projection`.

        :param a: Angle (radians) of projection.

        :returns: The projected line.
        """
        return Line(self.anchor.get_cabinet_projection(a),\
                    self.direction.get_cabinet_projection(a))

    def get_point(self, param: Scalar) -> Point:
        """Obtain a point on the line with position vector\
        :math:`\\boldsymbol{r}` satisfying

        .. math:: \\boldsymbol{r} = \\boldsymbol{r}_0 + \\lambda \\boldsymbol{d}

        for a particular value of `param` :math:`\\equiv \\lambda`.

        :param param: A scalar parameter specifying the point to be returned.

        :returns: A point on the line satisfying the above condition.
        """
        v = Vector.from_point(self.anchor) + param*self.direction
        return Point.from_vector(v)

    def is_parallel(self, other: Line, thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current line is parallel to `other`.

        Let :math:`\\boldsymbol{d}_1` and :math:`\\boldsymbol{d}_2` be the\
        normalised direction vectors of the two lines. The two lines are\
        parallel if and only if :math:`\\lVert \\boldsymbol{d}_1 \\times\
        \\boldsymbol{d}_2 \\rVert < \\epsilon`, where :math:`\\epsilon\
        \\equiv` `thresh`.

        :param other: A line to check for parallelism with the current line.
        :param thresh: Threshold to determine if the cross product between\
                       the two vectors is a null vector.

        :returns: `True` if the two lines are parallel, `False` if not.
        """
        if self.direction.cross(other.direction).norm < thresh:
            return True
        else:
            return False

    def contains_point(self,
                       point: Point,
                       thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if `point` lies on the current line.

        :param point: A point to check for collinearity with the current line.
        :param thresh: Threshold to determine if the perpendicular distance\
                       between `point` and the current line is zero.

        :returns: `True` if `point` lies on the current line, `False` if not.
        """
        anchor_to_point = self.anchor.get_vector_to(point)
        if anchor_to_point.cross(self.direction).norm\
                < thresh:
            return True
        else:
            return False

    def find_parameter_for_point(self,
                                 point: Point,
                                 thresh: float = ZERO_TOLERANCE) -> Scalar:
        """Determine the parameter of the line equation corresponding
        to `point`.

        :param point: A point in space.
        :param thresh: Threshold to determine if `point` lies on the current\
                       line.

        :returns: Parameter of the current line corresponding to `point.`
                  If `point` does not lie on the current line,\
                  :const:`numpy.nan` will be returned.
        """
        if self.contains_point(point, thresh):
            anchor_to_point = self.anchor.get_vector_to(point)
            lambdas = []
            for i in range(3):
                if self.direction[i] >= thresh:
                    lambdas.append(anchor_to_point[i]/self.direction[i])
            assert len(lambdas) >= 1
            if len(lambdas) > 1:
                for i in range(len(lambdas)-1):
                    assert abs(lambdas[i]-lambdas[i+1]) < thresh
            return lambdas[0]
        else:
            return np.nan

    def intersects_line(self,
                        other: Line,
                        thresh: float = ZERO_TOLERANCE)\
                       -> Tuple[int,Union[None,Point,Line]]:
        """Check if the current line intersects `other`.

        :param other: A line to check for intersection with the current line.
        :param thresh: Threshold to determine if the shortest distance\
                       between the two lines is zero.

        Returns
        -------
        n:
            number of points of intersection
        intersection:
            :const:`None` if `n` is zero, :class:`Point` if `n` = 1,\
            :class:`Line` if `n` = :const:`numpy.inf`.
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
                    # As we have established that the shortest distance
                    # between the two lines is zero, there is guaranteed
                    # to be a solution. We thus seek the pseudo-inverse
                    # of [d1 -d2].
                    anchors =self.anchor.get_vector_to(other.anchor)\
                                                    .components
                    directions = np.array(\
                            [[self.direction[0], -other.direction[0]],\
                             [self.direction[1], -other.direction[1]],\
                             [self.direction[2], -other.direction[2]]])
                    sols = np.linalg.lstsq(directions, anchors[:], rcond=None)[0]
                    self_param = sols[0]
                    other_param = sols[1]
                    point_on_self = self.get_point(self_param)
                    point_on_other = other.get_point(other_param)
                    # assert point_on_self.is_same_as(point_on_other, thresh)
                    return 1,point_on_self
                else:
                    return 0,None

    def intersects_plane(self,
                         plane: Plane,
                         thresh: float = ZERO_TOLERANCE)\
                         -> Tuple[int,Union[None,Point,Line]]:
        """Check if the current line intersects `plane`.

        :param plane: A plane to check for intersection with the current line.
        :param thresh: Threshold to determine if `plane` is parallel to the\
                       current line.

        Returns
        -------
        n:
            Number of points of intersection.
        intersection:
            :const:`None` if `n` is zero, :class:`Point` if `n` = 1,\
            :class:`Line` if `n` = :const:`numpy.inf`.
        """
        return plane.intersects_line(self, thresh)

    def __eq__(self, other: object) -> bool:
        if not isinstance(object, Line):
            return NotImplemented
        return self.is_same_as(other, ZERO_TOLERANCE)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Line):
            return NotImplemented
        return not self.is_same_as(other, ZERO_TOLERANCE)

    def is_same_as(self, other: object, thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current line is the same line as `other`.

        :param other: A line to check for identicality with the current line.
        :param thresh: Threshold to determine if the shortest distance between\
                the two lines is zero.

        :returns: `True` if `other` is the same as the current line,\
                  `False` if not.
        """
        if not isinstance(other, Line):
            return NotImplemented
        if np.isinf(self.intersects_line(other, thresh)[0]):
            return True
        else:
            return False


class Segment(FiniteObject):
    """A line segment in a three-dimensional Euclidean space.

    Mathematically, this segment is a collection of all points satisfying
    the vector equation

    .. math:: \\boldsymbol{r} = \\boldsymbol{r}_0 + \\lambda \\boldsymbol{d}

    where :math:`\\lambda` is a scalar parameter in a specified interval.

    Two segments are equal if and only if their endpoints are equal.
    """

    def __init__(self, endpoints: Sequence[Point]) -> None:
        """
        :param endpoints: A sequence of two points defining the endpoints of\
                          the segment. The order does not matter as segments\
                          are non-directional.
        """
        self.endpoints = endpoints # type: ignore
        self._find_bounding_box()

    @property
    def endpoints(self) -> List[Point]:
        """A list of two points defining the endpoints of the segment.
        The two endpoints are sorted.
        """
        return self._endpoints

    @endpoints.setter
    def endpoints(self, endpoints: Sequence[Point]) -> None:
        assert len(endpoints) == 2
        for point in endpoints:
            assert isinstance(point, Point)
        self._endpoints = sorted(endpoints)

    @property
    def midpoint(self) -> Point:
        """Midpoint of the segment.
        """
        midvec = 0.5*(Vector.from_point(self.endpoints[0]) +\
                      Vector.from_point(self.endpoints[1]))
        return Point.from_vector(midvec)

    @property
    def length(self) -> Scalar:
        """Length of the current segment.
        """
        return self.endpoints[0].get_vector_to(self.endpoints[1]).norm

    @property
    def vector(self) -> Vector:
        """Vector corresponding to the current segment.

        The direction of this vector is chosen to be from the first endpoint\
        towards the second endpoint.
        """
        return self.endpoints[0].get_vector_to(self.endpoints[1])

    @property
    def associated_line(self) -> Line:
        """The line containing this segment.
        """
        direction = self.endpoints[0].get_vector_to(self.endpoints[1])
        return Line(self.endpoints[0], direction)

    def get_a_point(self) -> Point:
        """Return a point on the segment. The easiest point to return\
        is the midpoint.

        :returns: A point on the segment, chosen to be the midpoint.
        """
        return self.midpoint

    def __str__(self) -> str:
        return "Segment[{}, {}]".format(self.endpoints[0], self.endpoints[1])

    __repr__ = __str__

    def get_fraction_of_segment(self,
                                mu: Scalar,
                                point: Point,
                                thresh: float = ZERO_TOLERANCE)\
                               -> Tuple[Point,Segment]:
        """Return a fraction of this segment.

        Let :math:`\\mathrm{A}` be `point`. :math:`\mathrm{A}` must be one of\
                the endpoints of the segment.
        Let :math:`\\mathrm{B}` be the other endpoint of the segment, and let\
                `mu` :math:`\\equiv\\mu`.
        The segment :math:`\mathrm{AP}` is returned such that\
                :math:`\\mathrm{AP}/\\mathrm{AB} = \\mu`.

        :param mu: Fraction of the segment length. :math:`0 \leq \\mu \leq 1`.
        :param point: The endpoint from which the fraction length is measured.
        :param thresh: Threshold to determine if `point` corresponds to one of\
                       the two endpoints.

        Returns
        -------
        P
            Point :math:`\\mathrm{P}`.
        AP
            Fraction :math:`\\mathrm{AP}` of this segment as defined above.
        """
        assert point.is_same_as(self.endpoints[0], thresh) or\
               point.is_same_as(self.endpoints[1], thresh),\
               '{} does not correspond to either endpoint of {}.'\
               .format(point, self)

        assert 0 <= mu and mu <= 1, '{} lies outside [0,1].'.format(mu)

        for endpoint in self.endpoints:
            if point.is_same_as(endpoint, thresh):
                A = endpoint
            else:
                B = endpoint
        OA = Vector.from_point(A)
        AB = A.get_vector_to(B)
        AP = mu*AB
        P = Point.from_vector(OA + AP)
        return P,Segment([A, P])

    def find_fraction(self,
                      other: Segment,
                      thresh: float = ZERO_TOLERANCE) -> Scalar:
        """Return the fraction corresponding to `other` relative to the
        current segment.

        Mathematically, let :math:`\\mathrm{AP}` be `other` and\
                :math:`\\mathrm{AB}` the current segment. Both\
                :math:`\\mathrm{AP}` and :math:`\\mathrm{AB}` must share an\
                endpoint, and :math:`\\mathrm{P}` must lie on\
                :math:`\\mathrm{AB}`. We seek\
                :math:`\\mu = \\mathrm{AP}/\\mathrm{AB}`.

        :raises AssertionError: When the current segment and `other` do not\
                                share an endpoint.

        :param other: Segment :math:`\\mathrm{AP}` as defined above.
        :param thresh: Threshold to determine if the two segments share an\
                       endpoint and if :math:`\\mathrm{P}` lies on\
                       :math:`\\mathrm{AB}`.

        :returns: The value of :math:`\\mu` as defined above.
        """
        if self.endpoints[0].is_same_as(other.endpoints[0], thresh):
            A = self.endpoints[0]
            P = other.endpoints[1]
        elif self.endpoints[0].is_same_as(other.endpoints[1], thresh):
            A = self.endpoints[0]
            P = other.endpoints[0]
        elif self.endpoints[1].is_same_as(other.endpoints[0], thresh):
            A = self.endpoints[1]
            P = other.endpoints[1]
        elif self.endpoints[1].is_same_as(other.endpoints[1], thresh):
            A = self.endpoints[1]
            P = other.endpoints[0]
        else:
            raise AssertionError('{} and {} do not share an endpoint.'\
                                 .format(self, other))
        assert self.contains_point(P, thresh), '{} does not lie within {}.'\
                                               .format(P, self)
        return Segment([A,P]).length/self.length


    def get_cabinet_projection(self, a: float = np.arctan(2)) -> Segment:
        """The cabinet projection of the current segment onto the\
        :math:`xy`-plane.

        For a description of the cabinet projection, see\
                :meth:`Point.get_cabinet_projection`.

        :param a: Angle (radians) of projection.

        :returns: The projected segment.
        """
        A = self.endpoints[0]
        B = self.endpoints[1]
        return Segment([A.get_cabinet_projection(a), B.get_cabinet_projection(a)])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return NotImplemented
        return self.is_same_as(other, ZERO_TOLERANCE)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return NotImplemented
        return not self.is_same_as(other, ZERO_TOLERANCE)

    def __lt__(self, other: Segment) -> bool:
        comparison_tuple_self = (self.length, self.endpoints[0],\
                                              self.endpoints[1])
        comparison_tuple_other = (other.length, other.endpoints[0],\
                                                other.endpoints[1])
        return comparison_tuple_self < comparison_tuple_other

    def __gt__(self, other: Segment) -> bool:
        comparison_tuple_self = (self.length, self.endpoints[0],\
                                              self.endpoints[1])
        comparison_tuple_other = (other.length, other.endpoints[0],\
                                                other.endpoints[1])
        return comparison_tuple_self > comparison_tuple_other

    def __le__(self, other: Segment) -> bool:
        comparison_tuple_self = (self.length, self.endpoints[0],\
                                              self.endpoints[1])
        comparison_tuple_other = (other.length, other.endpoints[0],\
                                                other.endpoints[1])
        return comparison_tuple_self <= comparison_tuple_other

    def __ge__(self, other: Segment) -> bool:
        comparison_tuple_self = (self.length, self.endpoints[0],\
                                              self.endpoints[1])
        comparison_tuple_other = (other.length, other.endpoints[0],\
                                                other.endpoints[1])
        return comparison_tuple_self >= comparison_tuple_other

    def is_same_as(self,
                   other: object,
                   thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current segment is the same segment as `other`.

        :param other: A segment to check for identicality with\
                      the current segment.
        :param thresh: Threshold to determine if two endpoints\
                       are identical.

        :returns: `True` if `other` is the same as the current segment,\
                  `False` if not.
        """
        if not isinstance(other, Segment):
            return NotImplemented
        if (self.endpoints[0].is_same_as(other.endpoints[0], thresh) and\
            self.endpoints[1].is_same_as(other.endpoints[1], thresh)):
            return True
        else:
            return False

    def contains_point(self,
                       point: Point,
                       thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if `point` lies on the current segment.

        :param point: A point to check for membership of the current segment.
        :param thresh: Threshold to determine if the perpendicular distance\
                       between `point` and the line containing this segment\
                       is zero.

        :returns: `True` if `point` lies on the current segment, `False` if not.
        """
        if self.associated_line.contains_point(point, thresh):
            AP = self.endpoints[0].get_vector_to(point)
            BP = self.endpoints[1].get_vector_to(point)
            if AP.norm < thresh or BP.norm < thresh:
                return True
            AB = self.endpoints[0].get_vector_to(self.endpoints[1])
            if abs(AB.norm - AP.norm - BP.norm) < thresh:
                return True
            else:
                return False
        else:
            return False

    def intersects_line(self,
                        line: Line,
                        thresh: float = ZERO_TOLERANCE)\
                       -> Tuple[int,Union[None,Point,Segment]]:
        """Check if the current segment intersects `line`.

        :param line: A line to check for intersection with the current segment.
        :param thresh: Threshold to determine if the shortest distance between\
                       the line and the current segment is zero.

        Returns
        -------
        n:
            Number of points of intersection.
        intersection:
            :const:`None` if `n` is zero, :class:`Point` if `n` = 1,\
            :class:`Segment` if `n` = :const:`numpy.inf`.
        """
        n_line,intersection_line = self.associated_line.intersects_line(line, thresh)
        if n_line == 0:
            return 0,None
        elif n_line == 1:
            assert isinstance(intersection_line, Point)
            if self.contains_point(intersection_line, thresh):
                return 1,intersection_line
            else:
                return 0,None
        else:
            return np.inf,self

    def intersects_segment(self,
                           other: Segment,
                           thresh: float = ZERO_TOLERANCE)\
                          -> Tuple[int,Union[None,Point,Segment]]:
        """Check if the current segment intersects `other`.

        :param other: A segment to check for intersection with the current
                      segment.
        :param thresh: Threshold to determine if the shortest distance between\
                       the two segments is zero.

        Returns
        -------
        n:
            Number of points of intersection.
        intersection:
            :const:`None` if `n` is zero, :class:`Point` if `n` = 1,\
            :class:`Segment` if `n` = :const:`numpy.inf`.
        """
        n_line,intersection_line = self.intersects_line(other.associated_line, thresh)
        if n_line == 0:
            return n_line,intersection_line
        elif n_line == 1:
            assert isinstance(intersection_line, Point)
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
                        intersection = Segment([self.endpoints[0],\
                                                other.endpoints[0]])
                        if intersection.length < thresh:
                            return 1,self.endpoints[0]
                        else:
                            return np.inf,intersection
                    else:
                        assert other.contains_point(self.endpoints[1], thresh)
                        intersection = Segment([self.endpoints[1],\
                                                other.endpoints[0]])
                        if intersection.length < thresh:
                            return 1,self.endpoints[1]
                        else:
                            return np.inf,intersection
            elif self.contains_point(other.endpoints[1], thresh):
                if other.contains_point(self.endpoints[0], thresh):
                    intersection = Segment([self.endpoints[0],\
                                            other.endpoints[1]])
                    if intersection.length < thresh:
                        return 1,self.endpoints[0]
                    else:
                        return np.inf,intersection
                else:
                    assert other.contains_point(self.endpoints[1], thresh)
                    intersection = Segment([self.endpoints[1],\
                                            other.endpoints[1]])
                    if intersection.length < thresh:
                        return 1,self.endpoints[1]
                    else:
                        return np.inf,intersection
            else:
                return 0,None

    def intersects_plane(self,
                         plane: Plane,
                         thresh: float = ZERO_TOLERANCE)\
                        -> Tuple[int,Union[None,Point,Segment]]:
        """Check if the current segment intersects `plane`.

        :param plane: A plane to check for intersection with the current\
                      segment.
        :param thresh: Threshold to determine if the segment and `plane`\
                      are parallel.

        Returns
        -------
        n:
            Number of points of intersection.
        intersection:
            :const:`None` if `n` is zero, :class:`Point` if `n` = 1,\
            :class:`Segment` if `n` = :const:`numpy.inf`.
        """
        n_plane,intersection_plane = self.associated_line.intersects_plane\
                                        (plane, thresh)
        if n_plane == 0:
            return 0,None
        elif n_plane == 1:
            assert isinstance(intersection_plane, Point)
            if self.contains_point(intersection_plane, thresh):
                return 1,intersection_plane
            else:
                return 0,None
        else:
            return np.inf,self

    def intersects_contour(self,
                           contour: Contour,
                           anchor: Point,
                           thresh: float = ZERO_TOLERANCE)\
                          -> Tuple[int,\
                                   List[Tuple[Point,Scalar]],\
                                   List[Tuple[Segment,Scalar,Scalar]],\
                                   List[Tuple[Segment,Scalar,Scalar]]]:
        """Check if the current segment intersects `contour` and find
        J-points and J-segments as defined in :cite:`article:Hsu1991`.

        :param contour: A contour to check for intersection with the current\
                        segment.
        :param anchor: A point corresponding to one of the two endpoints of\
                       the current segment. All J-points will be given a\
                       fraction relative to this point.
        :param thresh: Threshold to determine if the segment and `contour`\
                       are parallel.

        Returns
        -------
        n:
            Number of intersection points.
        J_points:
            List of J-points and their associated fraction values.
        segments_inside:
            List of segments inside `contour` and the associated start and
            end fraction values.
        segments_outside:
            List of segments outside `contour` and the associated start and
            end fraction values.
        """
        if not self.intersects_bounding_box(contour, thresh):
            return 0,[],[],[]
        else:
            n_plane,intersection_plane = self.associated_line.intersects_plane\
                                            (contour.associated_plane, thresh)
            assert n_plane != 0
            if n_plane == 1:
                assert isinstance(intersection_plane, Point)
                if self.contains_point(intersection_plane, thresh) and\
                        intersection_plane.is_inside_contour(contour, thresh):
                    mu = self.find_fraction(\
                            Segment([anchor,intersection_plane]))
                    return 1,[(intersection_plane,mu)],[],[]
                else:
                    return 0,[],[],[]
            else:
                J_points = []
                edge_intersection_points = []
                J_segments_inside = []
                J_segments_outside = []
                for edge in contour.edges:
                    n,J_point = self.intersects_segment(edge, thresh)
                    if n == 1:
                        assert isinstance(J_point, Point)
                        mu = self.find_fraction(Segment([anchor,J_point]))
                        edge_intersection_points.append((J_point,mu))
                for endpoint in self.endpoints:
                    mu = self.find_fraction(Segment([anchor,endpoint]))
                    edge_intersection_points.append((endpoint,mu))
                edge_intersection_points = sorted(edge_intersection_points,\
                                                  key = lambda t : t[1])
                J_points.append(edge_intersection_points[0])
                for i,point in enumerate(edge_intersection_points[0:-1]):
                    current_segment = Segment([edge_intersection_points[i][0],\
                                               edge_intersection_points[i+1][0]])
                    mu_start = edge_intersection_points[i][1]
                    mu_end = edge_intersection_points[i+1][1]
                    if current_segment.length < thresh:
                        continue
                    J_points.append(edge_intersection_points[i+1])
                    if current_segment.midpoint.is_inside_contour(contour):
                        J_segments_inside.append((current_segment,\
                                                  mu_start,mu_end))
                    else:
                        J_segments_outside.append((current_segment,\
                                                   mu_start,mu_end))
                return np.inf,J_points,J_segments_inside,J_segments_outside

    def intersects_facet(self,
                         facet: Facet,
                         anchor: Point,
                         thresh: float = ZERO_TOLERANCE)\
                        -> Tuple[int,\
                                 List[Tuple[Point,Scalar]],\
                                 List[Tuple[Segment,Scalar,Scalar]],\
                                 List[Tuple[Segment,Scalar,Scalar]]]:
        """Check if the current segment intersects `facet` and find the\
        J-points and J-segments as defined in :cite:`article:Hsu1991`.

        :param facet: A facet to check for intersection with the current\
                        segment.
        :param anchor: A point corresponding to one of the two endpoints of\
                       the current segment. All J-points will be given a\
                       fraction relative to this point.
        :param thresh: Threshold to determine if the segment and `facet`\
                       are parallel.

        Returns
        -------
        n:
            Number of intersection points.
        J_points:
            List of J-points and their associated fraction values.
        segments_inside:
            List of segments inside `facet` and the associated start and
            end fraction values.
        segments_outside:
            List of segments outside `facet` and the associated start and
            end fraction values.
        """
        if not self.intersects_bounding_box(facet, thresh):
            return 0,[],[],[]
        else:
            n_plane,intersection_plane = self.associated_line.intersects_plane\
                                            (facet.associated_plane, thresh)
            assert n_plane != 0
            if n_plane == 1:
                assert isinstance(intersection_plane, Point)
                if self.contains_point(intersection_plane, thresh) and\
                        intersection_plane.is_inside_facet(facet, thresh):
                    mu = self.find_fraction(\
                            Segment([anchor,intersection_plane]))
                    return 1,[(intersection_plane,mu)],[],[]
                else:
                    return 0,[],[],[]
            else:
                # Segment lies on the same plane as facet and possibly
                # overlaps with facet
                J_points = []
                edge_intersection_points = []
                J_segments_inside = []
                J_segments_outside = []
                for contour in facet.contours:
                    for edge in contour.edges:
                        n,J_point = self.intersects_segment(edge, thresh)
                        if n == 1:
                            assert isinstance(J_point, Point)
                            mu = self.find_fraction(Segment([anchor,J_point]))
                            edge_intersection_points.append((J_point,mu))
                    for endpoint in self.endpoints:
                        mu = self.find_fraction(Segment([anchor,endpoint]))
                        edge_intersection_points.append((endpoint,mu))
                edge_intersection_points = sorted(edge_intersection_points,\
                                                key = lambda t : t[1])
                J_points.append(edge_intersection_points[0])
                for i,point in enumerate(edge_intersection_points[0:-1]):
                    current_segment = Segment([edge_intersection_points[i][0],\
                                            edge_intersection_points[i+1][0]])
                    mu_start = edge_intersection_points[i][1]
                    mu_end = edge_intersection_points[i+1][1]
                    if current_segment.length < thresh:
                        continue
                    J_points.append(edge_intersection_points[i+1])
                    if current_segment.midpoint.is_inside_facet(facet):
                        J_segments_inside.append((current_segment,\
                                                mu_start,mu_end))
                    else:
                        J_segments_outside.append((current_segment,\
                                                mu_start,mu_end))
                return np.inf,J_points,J_segments_inside,J_segments_outside

    def _find_bounding_box(self) -> None:
        xmin = min(self.endpoints[0][0], self.endpoints[1][0])
        ymin = min(self.endpoints[0][1], self.endpoints[1][1])
        zmin = min(self.endpoints[0][2], self.endpoints[1][2])
        xmax = max(self.endpoints[0][0], self.endpoints[1][0])
        ymax = max(self.endpoints[0][1], self.endpoints[1][1])
        zmax = max(self.endpoints[0][2], self.endpoints[1][2])
        self.bounding_box = [(xmin,xmax),(ymin,ymax),(zmin,zmax)]


class Plane:
    """A plane in a three-dimensional Euclidean space.

    Mathematically, this plane is a collection of all points satisfying\
    the vector equation

    .. math:: \\boldsymbol{r} \\cdot \\boldsymbol{n} = c

    where :math:`\\boldsymbol{n}` is a unit vector normal to the plane, and\
            :math:`c` is a scalar constant.
    """

    def __init__(self, normal: Vector, point: Point) -> None:
        """
        :param normal: A vector (not necessarily normalised) normal\
                       to the plane.
        :param point: A point on the plane.
        """
        self.normal = normal
        self.constant = Vector.from_point(point).dot(self.normal)
        self._apoint = point

    @property
    def normal(self) -> Vector:
        """The unit vector normal to the plane with the largest number
        of positive coefficients.

        Any non-unit vector will be normalised automatically.

        When a vector with fewer positive coefficients than negative components
        is supplied, its negative will be taken.
        """
        return self._normal

    @normal.setter
    def normal(self, vector: Vector) -> None:
        pos_count = len([x for x in vector.components if x > 0])
        neg_count = len([x for x in vector.components if x < 0])
        if pos_count >= neg_count:
            self._normal = vector.normalise()
        else:
            self._normal = -(vector.normalise())

    @property
    def constant(self) -> Scalar:
        """The scalar constant in the vector equation corresponding to the
        unit vector normal to the plane with the largest number of positive
        coefficients.
        """
        return self._constant

    @constant.setter
    def constant(self, value: Scalar) -> None:
        self._constant = value

    def get_a_point(self) -> Point:
        """Return a point on the plane.

        :returns: A point on the plane.
        """
        return self._apoint

    def __str__(self) -> str:
        return "Plane[r.{}={}]".format(self.normal, self.constant)

    __repr__ = __str__

    def get_front_normal(self,
                         viewing_vector: Vector,
                         thresh: float = ZERO_TOLERANCE) -> Vector:
        """The 'front-face' normal relative to `viewing_vector`.

        The 'front-face' normal has a positive dot product with\
                `viewing_vector`. If this dot product is zero, the plane\
                is parallel to `viewing_vector` and either normal can serve\
                as the 'front-face' normal.

        :param viewing_vector: The viewing vector corresponding to a certain\
                               parallel projection.
        :param thresh: Threshold to determine if the plane is parallel to the\
                       viewing vector.

        :returns: The 'front-face' normal relative to `viewing_vector`.
        """
        if self.normal.dot(viewing_vector.normalise()) >= thresh:
            return self.normal
        elif self.normal.dot(viewing_vector.normalise()) <= -thresh:
            return -self.normal
        else:
            return self.normal

    def contains_point(self,
                       point: Point,
                       thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if `point` lies on the current plane.

        :param point: A point to check for membership of the current segment.
        :param thresh: Threshold to determine if the the coordinates of\
                       `point` satisfy the plane equation.

        :returns: `True` if `point` lies on the current plane, `False` if not.
        """
        rdotn = Vector.from_point(point).dot(self.normal)
        if abs(rdotn - self.constant) < thresh:
            return True
        else:
            return False

    def intersects_line(self,
                        line: Line,
                        thresh: float = ZERO_TOLERANCE)\
                       -> Tuple[int,Union[None,Point,Line]]:
        """Check if the current plane intersects `line`.

        :param line: A line to check for intersection with the current segment.
        :param thresh: Threshold to determine if the shortest distance between\
                       the line and the current segment is zero.

        Returns
        -------
        n:
            Number of points of intersection.
        intersection:
            :const:`None` if `n` is zero, :class:`Point` if `n` = 1,\
            :class:`Line` if `n` = :const:`numpy.inf`.
        """
        ddotn = line.direction.dot(self.normal)
        if abs(ddotn) < thresh:
            if self.contains_point(line.anchor):
                return np.inf,line
            else:
                return 0,None
        else:
            adotn = Vector.from_point(line.anchor).dot(self.normal)
            lamb = (self.constant-adotn)/ddotn
            intersection = line.get_point(lamb)
            return 1,intersection

    def intersects_plane(self,
                         other: Plane,
                         thresh: float = ZERO_TOLERANCE)\
                        -> Tuple[int,Union[None,Line,Plane]]:
        """Check if the current plane intersects `other`.

        :param other: A plane to check for intersection with the current\
                      segment.
        :param thresh: Threshold to determine if two planes are parallel.

        Returns
        -------
        n:
            Number of points of intersection.
        intersection:
            :const:`None` if `n` is zero, :class:`Line` or :class:`Plane`\
            if `n` = :const:`numpy.inf`.
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
            ij = np.matmul(np.linalg.inv(normals_ij), constants)
            coords = [0,0,0]
            coords[i] = ij[0,0]
            coords[j] = ij[1,0]
            point_on_line = Point(np.array(coords))
            assert self.contains_point(point_on_line, thresh) and\
                   other.contains_point(point_on_line, thresh)
            return np.inf,Line(point_on_line, n1crossn2)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Plane):
            return NotImplemented
        return self.is_same_as(other, ZERO_TOLERANCE)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Plane):
            return NotImplemented
        return not self.is_same_as(other, ZERO_TOLERANCE)

    def is_same_as(self, other: object, thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current plane is the same plane as `other`.

        :param other: A plane to check for identicality with the current plane.
        :param thresh: Threshold to determine if two planes are parallel.

        :returns: `True` if `other` is the same as the current plane,\
                  `False` if not.
        """
        if not isinstance(other, Plane):
            return NotImplemented
        n,intersection = self.intersects_plane(other, thresh)
        if n == np.inf and isinstance(intersection,Plane):
            return True
        else:
            return False


class Contour(FiniteObject):
    """A contour in a three-dimensional Euclidean space.

    (:cite:`article:Hsu1991`) "A contour is a closed planar polygon that may be one
    of ordered orientation."

    Two contours are equal if and only if their edges are equal.
    """

    def __init__(self, edges: Sequence[Segment]) -> None:
        """
        :param edges: A list of segments defining the edges of the contour.\
                      Any ordered orientation is implied by the order\
                      of the list.
        """
        self.edges = edges # type: ignore
        self._find_bounding_box()

    @property
    def edges(self) -> List[Segment]:
        """A list of segments defining the edges of the contour.
        """
        return self._edges

    @edges.setter
    def edges(self, edges: Sequence[Segment]) -> None:
        assert len(edges) >= 3,\
                'A contour in 3D requires a minimum of three edges.'
        v1 = edges[0].vector
        v2 = edges[1].vector
        p = Plane(v1.cross(v2), edges[0].endpoints[0])
        for i,edge in enumerate(edges):
            assert edges[i].endpoints[0] == edges[i-1].endpoints[0] or\
                   edges[i].endpoints[0] == edges[i-1].endpoints[1] or\
                   edges[i].endpoints[1] == edges[i-1].endpoints[0] or\
                   edges[i].endpoints[1] == edges[i-1].endpoints[1],\
                   'Consecutive edges must share a vertex.'
            assert np.isinf(p.intersects_line(edge.associated_line)[0]),\
                   'All edges must be coplanar.'
        self._edges = list(edges)

    @property
    def vertices(self) -> List[Point]:
        """A list of all unique vertices in this contour.
        """
        unique_vertices = []
        for i in range(len(self.edges)):
            if self.edges[i].endpoints[0] == self.edges[i-1].endpoints[0] or\
               self.edges[i].endpoints[0] == self.edges[i-1].endpoints[1]:
                unique_vertices.append(self.edges[i].endpoints[0])
            else:
                unique_vertices.append(self.edges[i].endpoints[1])
        return unique_vertices

    @property
    def associated_plane(self) -> Plane:
        """The plane containing this contour.
        """
        v1 = self.edges[0].associated_line.direction
        v2 = self.edges[1].associated_line.direction
        n = v1.cross(v2)
        A = self.edges[0].endpoints[0]
        return Plane(n, A)

    @classmethod
    def from_vertices(cls, vertices: Sequence[Point]) -> Contour:
        """Create a contour whose edges are from consecutive vertices.

        :param vertices: List of points that form the vertices of the contour.

        :returns: Contour in which each consecutive pair of vertices in\
                  `vertices` is an edge.
        """
        edges = []
        for i,vertex in enumerate(vertices[:-1]):
            edges.append(Segment([vertex,vertices[i+1]])) 
        edges.append(Segment([vertices[-1],vertices[0]]))
        return cls(edges)

    def get_front_normal(self,
                         viewing_vector: Vector,
                         thresh: float = ZERO_TOLERANCE) -> Vector:
        """The 'front-face' normal relative to `viewing_vector`.

        The definition of 'front-face' normal is given in\
                :meth:`Plane.get_front_normal`.

        :param viewing_vector: The viewing vector corresponding to a certain\
                               parallel projection.
        :param thresh: Threshold to determine if the contour is parallel to the\
                       viewing vector.

        :returns: The 'front-face' normal relative to `viewing_vector`.
        """
        return self.associated_plane.get_front_normal(viewing_vector, thresh)

    def __str__(self) -> str:
        return "Contour{}".format(self.vertices)

    def __repr__(self) -> str:
        return "\nContour{}".format(self.vertices)

    # __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Contour):
            return NotImplemented
        return self.is_same_as(other, ZERO_TOLERANCE)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Contour):
            return NotImplemented
        return not self.is_same_as(other, ZERO_TOLERANCE)

    def get_a_point(self) -> Point:
        """Return a point on the contour. The easiest point to return
        is the geometric centre.

        :returns: A point on the contour, chosen to be the geometric centre.
        """
        all_vertices = self.vertices
        vec_sum = Vector.from_point(all_vertices[0])
        for vertex in all_vertices[1:]:
            vec_sum = vec_sum + Vector.from_point(vertex)
        centre = Point.from_vector(vec_sum/len(all_vertices))
        assert centre.is_inside_contour(self)
        return centre

    def is_same_as(self,
                   other: Contour,
                   thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current contour is the same contour as `other`.

        :param other: A contour to check for identicality with the current\
                      contour.
        :param thresh: Threshold to determine if two edges are identical.

        :returns: `True` if `other` is the same as the current contour,\
                  `False` if not.
        """
        if len(self.edges) != len(other.edges):
            return False
        else:
            self_edges_sorted = sorted(self.edges)
            other_edges_sorted = sorted(other.edges)
            for i in range(len(self_edges_sorted)):
                if self_edges_sorted[i] == other_edges_sorted[i]:
                    continue
                else:
                    return False
            return True

    def get_cabinet_projection(self, a: float = np.arctan(2)) -> Contour:
        """The cabinet projection of the current contour onto the\
        :math:`xy`-plane.

        For a description of the cabinet projection, see\
                :meth:`Point.get_cabinet_projection`.

        :param a: Angle (radians) of projection.

        :returns: The projected contour.
        """
        return Contour([edge.get_cabinet_projection(a) for edge in self.edges])

    def is_coplanar(self,
                    other: Contour,
                    thresh: float = ZERO_TOLERANCE) -> bool:
        """Check if the current contour is coplanar with `other`.

        :param other: A contour to check for coplanarity with the current\
                      contour.
        :param thresh: Threshold for various comparisons.

        :returns: `True` if `other` is coplanar with the current contour,
                  `False` if not.
        """
        self_plane = self.associated_plane
        other_plane = other.associated_plane
        n1crossn2 = self_plane.normal.cross(other_plane.normal)
        if n1crossn2.norm < thresh:
            if abs(self_plane.constant - other_plane.constant) < thresh:
                return True
            else:
                return False
        else:
            return False

    def intersects_contour(self,
                           other: Contour,
                           thresh: float = ZERO_TOLERANCE)\
                          -> Tuple[int,List[Point],List[Segment]]:
        """Check if the current contour intersects `other` and find
        J-points and J-segments as defined in :cite:`article:Hsu1991`.

        Parameters
        ----------
        other : Contour
            A contour to check for intersection with the current contour.
        thresh : `float`
            Threshold to determine if two contours are parallel.

        Returns
        -------
        n : int
            Number of J-segments.
        J_points : [Point]
            List of J-points.
        J_segments : [Segment]
            List of J-segments.
        """
        if not self.intersects_bounding_box(other, thresh):
            return 0,[],[]
        else:
            n_plane,intersection_plane = self.associated_plane.intersects_plane\
                                            (other.associated_plane, thresh)
            if n_plane == 0:
                return 0,[],[]
            elif isinstance(intersection_plane, Line):
                J_line = intersection_plane
                J_points = []
                J_segments = []
                for edge in self.edges+other.edges:
                    n,J_point = edge.intersects_line(J_line, thresh)
                    if n == 1:
                        assert isinstance(J_point, Point)
                        assert J_line.contains_point(J_point)
                        lamb = J_line.find_parameter_for_point(J_point, thresh)
                        J_points.append((J_point,lamb))
                J_points = sorted(J_points, key=operator.itemgetter(1))
                for i,point in enumerate(J_points[0:-1]):
                    current_segment = Segment([J_points[i][0], J_points[i+1][0]])
                    if current_segment.length < thresh:
                        continue
                    if current_segment.midpoint.is_inside_contour(self) and\
                       current_segment.midpoint.is_inside_contour(other):
                        J_segments.append(current_segment)
                J_points_returned = [t[0] for t in J_points]
                return len(J_segments),J_points_returned,J_segments
            else:
                # Coplanar contours
                if self.is_same_as(other, thresh):
                    return len(self.edges),self.vertices,self.edges
                else:
                    J_points_returned = []
                    J_segments = []
                    for self_edge in self.edges:
                        for other_edge in other.edges:
                            n,intersection_segment = self_edge\
                                    .intersects_segment(other_edge, thresh)
                            assert n > 0
                            if n == 1:
                                assert isinstance(intersection_segment, Point)
                                J_points_returned.append(intersection_segment)
                                for endpoint in self_edge.endpoints:
                                    cur_seg = Segment([intersection_segment,\
                                                       endpoint])
                                    if cur_seg.midpoint\
                                            .is_inside_contour(other, thresh):
                                        J_segments.append(cur_seg)
                                for endpoint in other_edge.endpoints:
                                    cur_seg = Segment([intersection_segment,\
                                                       endpoint])
                                    if cur_seg.midpoint\
                                            .is_inside_contour(self, thresh):
                                        J_segments.append(cur_seg)
                            else:
                                assert isinstance(intersection_segment, Segment)
                                J_points_returned\
                                        .extend(intersection_segment.endpoints)
                                J_segments.append(intersection_segment)
                    return len(J_segments),J_points_returned,J_segments

    def _find_bounding_box(self) -> None:
        xs = [edge.endpoints[i][0] for edge in self.edges for i in [0,1]]
        ys = [edge.endpoints[i][1] for edge in self.edges for i in [0,1]]
        zs = [edge.endpoints[i][2] for edge in self.edges for i in [0,1]]
        xmin = min(xs)
        ymin = min(ys)
        zmin = min(zs)
        xmax = max(xs)
        ymax = max(ys)
        zmax = max(zs)
        self.bounding_box = [(xmin,xmax),(ymin,ymax),(zmin,zmax)]


class Facet(FiniteObject):
    """A facet in a three-dimensional Euclidean space.

    (:cite:`article:Hsu1991`) "A facet is specified by one or more contours that
    are coplanar."
    """

    def __init__(self, contours: Sequence[Contour]) -> None:
        """
        :param countours: A sequence of contours defining the facet.
        """
        self.contours = contours # type: ignore
        self._find_bounding_box()

    @property
    def contours(self) -> List[Contour]:
        """[Contour]: A list of contours defining the facet.
        """
        return self._contours

    @contours.setter
    def contours(self, contours: Sequence[Contour]) -> None:
        if len(contours) > 1:
            for contour in contours[1:]:
                assert contours[0].is_coplanar(contour),\
                    "All contours must be coplanar."
        self._contours = list(contours)

    @property
    def associated_plane(self) -> Plane:
        """The plane containing this facet.
        """
        return self.contours[0].associated_plane

    def __str__(self) -> str:
        return "Facet{}".format(self.contours)

    def __repr__(self) -> str:
        return "\nFacet{}".format(self.contours)

    # __repr__ = __str__

    def get_a_point(self) -> Point:
        """Return a point on the facet. The easiest point to return\
        is the first contour vertex that lies inside the facet.

        :raises RuntimeError: When none of the contours has vertices that lie\
                              inside the facet.

        :returns: A point inside the facet as described above.
        """
        for contour in self.contours:
            for point in contour.vertices:
                if point.is_inside_facet(self):
                    return point
        raise RuntimeError('A point inside the facet could not be found.')

    def get_front_normal(self,
                         viewing_vector: Vector,
                         thresh: float = ZERO_TOLERANCE) -> Vector:
        """The 'front-face' normal relative to `viewing_vector`.

        The definition of 'front-face' normal is given in\
                :meth:`Plane.get_front_normal`.

        :param viewing_vector: The viewing vector corresponding to a certain\
                               parallel projection.
        :param thresh: Threshold to determine if the contour is parallel to the\
                       viewing vector.

        :returns: The 'front-face' normal relative to `viewing_vector`.
        """
        return self.associated_plane.get_front_normal(viewing_vector, thresh)

    def get_cabinet_projection(self, a: float = np.arctan(2)) -> Facet:
        """The cabinet projection of the current facet onto the\
        :math:`xy`-plane.

        For a description of the cabinet projection, see\
                :meth:`Point.get_cabinet_projection`.

        :param a: Angle (radians) of projection.

        :returns: The projected facet.
        """
        return Facet([contour.get_cabinet_projection(a) for contour in\
                      self.contours])

    def _find_bounding_box(self) -> None:
        xmin = min([contour.bounding_box[0][0] for contour in self.contours])
        ymin = min([contour.bounding_box[1][0] for contour in self.contours])
        zmin = min([contour.bounding_box[2][0] for contour in self.contours])
        xmax = max([contour.bounding_box[0][1] for contour in self.contours])
        ymax = max([contour.bounding_box[1][1] for contour in self.contours])
        zmax = max([contour.bounding_box[2][1] for contour in self.contours])
        self.bounding_box = [(xmin,xmax),(ymin,ymax),(zmin,zmax)]


class Polyhedron(FiniteObject):
    """A polyhedron in a three-dimensional Euclidean space.
    """

    def __init__(self, facets: Sequence[Facet]) -> None:
        """
        :param facets: A sequence of facets defining the polyhedron.
        """
        self.facets = facets # type: ignore
        self._find_bounding_box()

    @property
    def facets(self) -> List[Facet]:
        """A list of facets defining the polyhedron.
        """
        return self._facets

    @facets.setter
    def facets(self, facets: Sequence[Facet]) -> None:
        self._facets = list(facets)

    @property
    def edges(self) -> List[Segment]:
        """A list of unique edges in the polyhedron.
        """
        edges = [contour.edges for facet in self.facets\
                               for contour in facet.contours]
        edges_flattened = [edge for edge_list in edges for edge in edge_list]
        edges_sorted = sorted(edges_flattened)
        edges_unique = [edges_sorted[0]]
        for edge in edges_sorted[1:]:
            if edge == edges_unique[-1]:
                continue
            else:
                edges_unique.append(edge)
        return edges_unique

    @property
    def vertices(self) -> List[Point]:
        """A list of unique vertices in the polyhedron.
        """
        vertices = [vertex for edge in self.edges for vertex in edge.endpoints]
        vertices_sorted = sorted(vertices)
        vertices_unique = [vertices_sorted[0]]
        for vertex in vertices_sorted[1:]:
            if vertex == vertices_unique[-1]:
                continue
            else:
                vertices_unique.append(vertex)
        return vertices_unique

    def __str__(self) -> str:
        return "Polyhedron{}".format(self.facets)

    __repr__ = __str__

    def get_cabinet_projection(self, a: float = np.arctan(2)) -> Polyhedron:
        """The cabinet projection of the current polyhedron onto the\
        :math:`xy`-plane.

        For a description of the cabinet projection, see\
                :meth:`Point.get_cabinet_projection`.

        :param a: Angle (radians) of projection.

        :returns: The projected polyhedron.
        """
        return Polyhedron([facet.get_cabinet_projection(a) for facet in\
                           self.facets])

    def _find_bounding_box(self) -> None:
        xmin = min([facet.bounding_box[0][0] for facet in self.facets])
        ymin = min([facet.bounding_box[1][0] for facet in self.facets])
        zmin = min([facet.bounding_box[2][0] for facet in self.facets])
        xmax = max([facet.bounding_box[0][1] for facet in self.facets])
        ymax = max([facet.bounding_box[1][1] for facet in self.facets])
        zmax = max([facet.bounding_box[2][1] for facet in self.facets])
        self.bounding_box = [(xmin,xmax),(ymin,ymax),(zmin,zmax)]


class VertexCollection(FiniteObject):
    """A vertex collection in a three-dimensional Euclidean space is\
    essentially a polyhedron but only with vertices defined. They might not be\
    convex, and so no facets or contours are constructed. Edges are segments\
    joining all pairs of vertices.
    """

    def __init__(self, vertices: Sequence[Point], cutoff: float) -> None:
        """
        :param vertices: A sequence of vertices defining the vertex collection.
        :param cutoff: A cut-off for the inter-vertex distances to be\
                       considered as an edge.
        """
        self.vertices = vertices # type: ignore
        self._cutoff = cutoff
        self._find_bounding_box()

    @property
    def vertices(self) -> List[Point]:
        """A list of vertices defining the vertex collection.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: Sequence[Point]) -> None:
        self._vertices = list(vertices)

    @property
    def edges(self) -> List[Segment]:
        """A list of segments joining all pairs of vertices.
        """
        edges = []
        vertices = self.vertices
        for i in range(len(vertices)-1):
            for j in range(i+1, len(vertices)):
                edge_ij = Segment([vertices[i], vertices[j]])
                if edge_ij.length <= self._cutoff:
                    edges.append(edge_ij)
        return edges

    def __str__(self) -> str:
        return "VertexCollection{}".format(self.vertices)

    __repr__ = __str__

    def get_cabinet_projection(self, a: float = np.arctan(2))\
                              -> VertexCollection:
        """The cabinet projection of the current vertex collection onto the\
        :math:`xy`-plane.

        For a description of the cabinet projection, see\
                :meth:`Point.get_cabinet_projection`.

        :param a: Angle (radians) of vertex collection.

        :returns: The projected vertex collection.
        """
        return VertexCollection([vertex.get_cabinet_projection(a) for vertex in\
                                 self.vertices], self._cutoff)

    def _find_bounding_box(self) -> None:
        xmin = min([vertex.bounding_box[0][0] for vertex in self.vertices])
        ymin = min([vertex.bounding_box[1][0] for vertex in self.vertices])
        zmin = min([vertex.bounding_box[2][0] for vertex in self.vertices])
        xmax = max([vertex.bounding_box[0][1] for vertex in self.vertices])
        ymax = max([vertex.bounding_box[1][1] for vertex in self.vertices])
        zmax = max([vertex.bounding_box[2][1] for vertex in self.vertices])
        self.bounding_box = [(xmin,xmax),(ymin,ymax),(zmin,zmax)]


def rotate(vector: Vector, angle: float, direction: Vector) -> Vector:
    """Wrapper method to apply a scipy Rotation object to a vector.

    :param vector: A vector to which the rotation is applied.
    :param angle: The angle of rotation (radians).
    :param direction: A normalised vector indicating the rotation axis.\
                      The angle-axis parametrisation for rotation applies.

    :returns: The rotated result for `vector`.
    """
    direction_array = direction.components
    vector_array = vector.components
    R = Rotation.from_rotvec(angle*direction_array)
    rotated_vector_array = R.apply(vector_array)
    return Vector(rotated_vector_array)


def rotate_point(point: Point, angle: float, direction: Vector) -> Point:
    """Wrapper method to apply a scipy Rotation object to a point\
    about the origin.

    :param point: A point to which the rotation is applied.
    :param angle: The angle of rotation (radians).
    :param direction: A normalised vector indicating the rotation axis.\
                      The angle-axis parametrisation for rotation applies.

    :returns: The rotated result for `point`.
    """
    vector = Vector.from_point(point)
    return Point.from_vector(rotate(vector, angle, direction))
