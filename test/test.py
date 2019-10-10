#!python3

import sys, copy
sys.path.append('..')
import numpy as np
from tools.Geometry3D import Point, Vector, Line, Segment, Plane, Contour, Facet, Polyhedron
from tools.PolyhedronDrawing import PolyhedronScene
from tools.PolytopeTools import construct_convex_hull, polyhedra_from_xyz
from scipy.spatial import ConvexHull
import pprint

# def main():
#     pp = pprint.PrettyPrinter(indent=4)
#     for i in range(10):
#         print(i)
#         A = Point([-1,0,-1])
#         B = Point([1,0,-1])
#         C = Point([1,0,1])
#         D = Point([-1,0,1])
#         E = Point([0,1+0.1*i,0])
#         F = Point([0,-1-0.1*i,0])
#         polyhedron1 = construct_convex_hull([A,B,C,D,E,F])

#         G = Point([1.5,1.5,0])
#         H = Point([-1.5,1.5,0])
#         I = Point([0,-1.5,1.5])
#         J = Point([0,-1.5,-1.5])
#         polyhedron2 = construct_convex_hull([G,H,I,J])

#         scene = Scene([polyhedron1,polyhedron2], 1.0)
#         scene.centre_scene()
#         scene.rotate_scene(-0.05*i, Vector([0,1,0]))
#         scene.write_to_tikz('test.{}.tex'.format(str(i)), [(-2.2,2.2),(-2.3,2.1)])

def main():
    for i in range(1, 11):
        print("i is:", i)
        p = polyhedra_from_xyz('AAdash.Egu.elongation.UHF.g{}.xc0.scf.transformed.xyz'.format(str(i)))
        scene = PolyhedronScene(p)
        scene.centre_scene()
        scene.rotate_scene(-np.pi/2, Vector([1,0,0]))
        scene.rotate_scene(-np.pi/2, Vector([0,1,0]))
        scene.write_to_tikz('test.xyz.{}.tex'.format(str(i)), vertex_size='0.2pt', bb=[(-1,1),(-1,1)], scale=5)

if __name__ == '__main__':
    main()
