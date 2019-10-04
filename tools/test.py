#!/usr/bin/python3 -m

from .Geometry3D import Point, Vector, Line, Segment, Plane, Contour, Facet, Polyhedron
from .PolyhedronDrawing import PolyhedronDrawing

def main():
    A = Point([0,0,0])
    B = Point([2,0,0])
    C = Point([0,2,0])
    D = Point([1,1,1])
    E = Point([2,2,0])
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

    print(visible)
    print(hidden)


if __name__ == '__main__':
    main()
