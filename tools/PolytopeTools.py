#!/usr/bin/env python2

"""
@module PolytopeTools
"""

import sys
import numpy as np
import sympy as sp
from itertools import compress
from sympy import sympify
from sympy.geometry import Point
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers.solveset import nonlinsolve

from ..QCMagicRC import GlobalOptions


def expand(x):
    # endows point vectors in the sequence x with an additional zero component

    for i,xi in enumerate(x):
        x[i] = np.append(x[i], 0.0)


def square_distance(x1, x2):
    diff = x2 - x1
    sqdiff = map(lambda x: x**2, diff)
    return sum(sqdiff)


def intersectSpheres(centres, sqradii, thresh=1e-6):
    # returns the list of solutions of the intersections of spheres with centres in list centres and corresponding squared radii in list sqradii

    K = centres[0].shape[0] # dimension of the affine span of the centres
    r = sp.symbols('r0:%d'%(K+1))

    system=[]
    for icentre,centre in enumerate(centres):
        strExpr = ""
        for icomponent,component in enumerate(centre):
            strExpr += "({}-{})**2".format(r[icomponent], component)
            if icomponent != K-1:
                strExpr += " + "
            else:
                strExpr += " + ({})**2 - {}".format(r[icomponent+1], sqradii[icentre])

        Expr = parse_expr(strExpr, evaluate=False)
        system.append(Expr)

    rawSols = nonlinsolve(system, list(r)).args
    if len(rawSols) == 2:
        if sp.Abs(rawSols[0][-1]) < thresh and sp.Abs(rawSols[1][-1]) < thresh:
            allSols = [tuple(list(rawSols[0])[0:-1])]
        elif sp.Abs(rawSols[0][-1]) > thresh and sp.Abs(rawSols[1][-1]) > thresh:
            allSols = [sol for sol in rawSols]
    elif len(rawSols) == 1:
        allSols = [tuple(list(rawSols[0])[0:-1])]
    realMask = map(lambda x: all(map(lambda xi: xi.is_real, x)), allSols)
    realSols = list(compress(allSols, realMask))

    return realSols


def edmsph(D, thresh=1e-6):

    I = [0, 1] # Indices of existing vertices
    K = 1
    n = D.shape[0]
    x = [np.array([0.0]), np.array([np.sqrt(D[1,0])])] # Existing vertices
    for i in range(2, n):
        sqradiiconsidered = [D[i,j] for j in I]
        centresconsidered = [x[m] for m in I]
        gamma = intersectSpheres(centresconsidered, sqradiiconsidered, thresh)

        if len(gamma) == 0:
            print 'No solutions found.'
            sys.exit(1)
        elif len(gamma) == 1:
            x.append(np.array(gamma[0], dtype=float))
        elif len(gamma) == 2:
            expand(x)
            K += 1
            for sol in gamma:
                if sol[-1] >= 0:
                    x.append(np.array(sol, dtype=float))
            I.append(i)
        else:
            print 'Something wrong!'

    print "Embedded dimension found:", K
    print "Threshold:", thresh

    # points = []
    # for i in x:
    #     points.append(i)

    return x, K
