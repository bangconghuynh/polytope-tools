#!/usr/bin/python3 -m

from .Geometry3D import Point, Vector, Line, Segment, Plane, Contour, Facet

def main():
    A = Point([0,0,0])
    B = Point([1,0,0])
    C = Point([1,1,0])
    D = Point([0,1,0])
    sAB = Segment([A,B])
    sBC = Segment([B,C])
    sCD = Segment([C,D])
    sDA = Segment([D,A])
    cABCDA = Contour([sAB,sBC,sCD,sDA])
    E = Point([5,6,0])
    F = Point([1,2,0])
    G = Point([0,3,0])
    H = Point([-4,2,0])
    sEF = Segment([E,F])
    sFG = Segment([F,G])
    sGH = Segment([G,H])
    sHE = Segment([H,E])
    cEFGHE = Contour([sEF,sFG,sGH,sHE])
    f = Facet([cABCDA,cEFGHE])
    print(cABCDA.intersects_bounding_box(cEFGHE))

if __name__ == '__main__':
    main()
