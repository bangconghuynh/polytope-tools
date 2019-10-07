#!python3

import sys, copy
sys.path.append('/home/dorebase2006/code/bang/polytope-tools')
import numpy as np
from tools.Geometry3D import Point, Vector, Line, Segment, Plane, Contour, Facet, Polyhedron, rotate
from tools.PolyhedronDrawing import Scene
from tools.PolytopeTools import construct_convex_hull
from scipy.spatial import ConvexHull
import pprint

def main():
    pp = pprint.PrettyPrinter(indent=4)
    A = Point([2,2,0])
    B = Point([-2,2,0])
    C = Point([0,-2,2])
    D = Point([0,-2,-2])
    E = Point([1,2,1])
    F = Point([2,0,2])
    G = Point([2,2,2])
    H = Point([0,2,2])
    polyhedron = construct_convex_hull([A,B,C,D])

    I = Point([1,1,0])
    J = Point([2,0,0])
    K = Point([1,-1,0])
    L = Point([-1,-1,0])
    M = Point([-2,0,0])
    N = Point([-1,1,0])
    O = Point([0,0,1])
    # polyhedron = construct_convex_hull([I,J,K,L,M,N,O])
    for a in range(10):
        print(0.5+a*0.1)
        scene = Scene([polyhedron], 0.5+a*0.1)
        scene.write_to_tikz('test.{}.tex'.format(str(0.5+a*0.1)))

if __name__ == '__main__':
    main()
