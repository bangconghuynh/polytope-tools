#!/usr/bin/python3 -m

from .Geometry3D import Point, Vector, Line, Segment, Plane, Contour, Facet

def main():
    A = Point([0,0,0])
    B = Point([3,0,0])
    C = Point([3,2,0])
    D = Point([2,2,0])
    E = Point([2,1,0])
    F = Point([1,1,0])
    G = Point([1,2,0])
    H = Point([0,2,0])
    c1 = Contour.from_vertices([A,B,C,D,E,F,G,H])
    I = Point([1.0,1.5,1])
    J = Point([2.0,1.5,1])
    K = Point([2.0,1.5,-1])
    L = Point([1.0,1.5,-1])
    c2 = Contour.from_vertices([I,J,K,L])
    print(c1.intersects_contour(c2))
if __name__ == '__main__':
    main()
