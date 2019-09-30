#!/usr/bin/python3 -m

from .PolyhedronDrawingTools import Point, Vector, Line, Segment, Plane

def main():
    A = Point([0,0,0])
    B = Point([1,2,1])
    n = Vector([0,0,1])
    pnA = Plane(n, A)
    l = Line(A, Vector([1,1,0]))
    print(pnA.intersects_line(l))

if __name__ == '__main__':
    main()
