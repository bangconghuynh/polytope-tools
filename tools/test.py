#!/usr/bin/python3 -m

from .Geometry3D import Point, Vector, Line, Segment, Plane, Contour

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
    print(cABCDA.associated_plane)

if __name__ == '__main__':
    main()
