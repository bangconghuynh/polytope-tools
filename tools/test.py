#!/usr/bin/python3 -m

from .Geometry3D import Point, Vector, Line, Segment, Plane, Contour, Facet, Polyhedron
from .PolyhedronDrawing import PolyhedronDrawing

def main():
    A = Point([0,0,0])
    B = Point([3,0,0])
    C = Point([3,4,0])
    D = Point([2,4,0])
    E = Point([2,2,0])
    F = Point([1,2,0])
    G = Point([1,4,0])
    H = Point([0,4,0])


    # fABC = Facet([Contour.from_vertices([A,B,C])])
    # fABD = Facet([Contour.from_vertices([A,B,D])])
    # fACD = Facet([Contour.from_vertices([A,C,D])])
    # fBCD = Facet([Contour.from_vertices([B,C,D])])

    # pABCD = Polyhedron([fABC, fABD, fACD, fBCD])

    # scene = PolyhedronDrawing([pABCD])

    cABCDEFGH = Contour.from_vertices([A,B,C,D,E,F,G,H])
    I = Point([2,1,0])
    J = Point([5,1,0])
    sIJ = Segment([G,D])
    print(sIJ.intersects_contour(cABCDEFGH, G))

if __name__ == '__main__':
    main()
