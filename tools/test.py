#!/usr/bin/python3 -m

import sys
import numpy as np
from .Geometry3D import Point, Vector, Line, Segment, Plane, Contour, Facet, Polyhedron, rotate
from .PolyhedronDrawing import PolyhedronDrawing
import pprint

def main():
    pp = pprint.PrettyPrinter(indent=4)
    A = Point([0,0,0])
    B = Point([2,0,0])
    C = Point([0,2,0])
    D = Point([1,1,1])
    O = Point()
    E = Point([-1.5,-2,0])
    F = Point([1,2,0])
    G = Point([1,4,0])
    H = Point([0,4,0])


    fABC = Facet([Contour.from_vertices([A,B,C])])
    fABD = Facet([Contour.from_vertices([A,B,D])])
    fACD = Facet([Contour.from_vertices([A,C,D])])
    fBCD = Facet([Contour.from_vertices([B,C,D])])

    pABCD = Polyhedron([fABC, fABD, fACD, fBCD])

    scene = PolyhedronDrawing([pABCD])
    visible,hidden = scene.get_visible_hidden_segments()

    print(scene.viewing_vector)
    pp.pprint(visible)
    pp.pprint(hidden)

    # A = Point([2,0,0])
    # B = Point([0,2,0])
    # C = Point([-2,0,0])
    # D = Point([0,-2,0])
    # O = Point()

    # cABCD = Contour.from_vertices([A,B,C,D])
    # print(O.is_inside_contour(cABCD))

if __name__ == '__main__':
    main()
