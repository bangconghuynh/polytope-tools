#!/usr/bin/python3 -m

import sys, copy
import numpy as np
from .Geometry3D import Point, Vector, Line, Segment, Plane, Contour, Facet, Polyhedron, rotate
from .PolyhedronDrawing import PolyhedronDrawing
from .PolytopeTools import construct_convex_hull
from scipy.spatial import ConvexHull
import pprint

def main():
    pp = pprint.PrettyPrinter(indent=4)
    A = Point([0,0,0])
    B = Point([2,0,0])
    C = Point([2,2,0])
    D = Point([0,2,0])
    E = Point([0,0,2])
    F = Point([2,0,2])
    G = Point([2,2,2])
    H = Point([0,2,2])
    polyhedron = construct_convex_hull([A,B,C,D,E,F,G,H])
    scene = PolyhedronDrawing([polyhedron])
    pp.pprint(scene.visible_segments)
    pp.pprint(scene.hidden_segments)

    # I = Point([1,1,0])
    # J = Point([2,0,0])
    # K = Point([1,-1,0])
    # L = Point([-1,-1,0])
    # M = Point([-2,0,0])
    # N = Point([-1,1,0])
    # O = Point([0,0,1])
    # construct_convex_hull([I,J,K,L,M,N,O])

if __name__ == '__main__':
    main()
