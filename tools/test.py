#!/usr/bin/python3 -m

from .Geometry3D import Point, Vector, Line, Segment, Plane, Contour, Facet, Polyhedron
import pprint

def main():
    A = Point([0,0,0])
    B = Point([2,0,0])
    C = Point([0,2,0])
    D = Point([1,1,1])

    fABC = Facet([Contour.from_vertices([A,B,C])])
    fABD = Facet([Contour.from_vertices([A,B,D])])
    fACD = Facet([Contour.from_vertices([A,C,D])])
    fBCD = Facet([Contour.from_vertices([B,C,D])])

    pABCD = Polyhedron([fABC, fABD, fACD, fBCD])
    print(pABCD.edges)
    print(pABCD.get_cabinet_projection())

if __name__ == '__main__':
    main()
