#!/usr/bin/python3 -m

from .PolyhedronDrawingTools import Point, Vector, Line

def main():
    A = Point([0,0,0])
    B = Point([1,0,0])
    C = Point([0,0,-1])
    D = Point([0,1,0])
    AB = A.get_vector_to(B)
    CD = C.get_vector_to(D) 
    lAB = Line(A, AB)
    lCD = Line(C, CD)
    print(lAB.intersects_line(lCD))

if __name__ == '__main__':
    main()
