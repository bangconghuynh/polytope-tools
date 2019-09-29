'''
@module PrintTools
'''

import numpy as np
import sys


def printOutputHead(filename):
    with open(filename, 'w') as f:
        f.write('***********************************************************\n')
        f.write('SUMMARY OF CHARACTER AND REPRESENTATION ANALYSIS\n')
        f.write('***********************************************************\n')
        f.write('Below shows a readable summary of the representations of \n')
        f.write('the SCF states at each geometry.\n')
        f.write('***********************************************************\n\n')


def printGeometryHead(f, GI):
    f.write('***********************************************************\n')
    f.write('Geometry {}\n'.format(GI))
    f.write('***********************************************************\n\n')


def printMat(f, Mat, Title, Type, isComplex, maxnumCols=6, maxlenWhole=1, maxlenFrac=3, maxlenType=12, shiftIndex=0, header=True):
    nFullBatches = Mat.shape[1]/maxnumCols
    nExtraCols = Mat.shape[1]%maxnumCols
    f.write('\n{}\n'.format(Title))
    assert len(Type) <= maxlenType, "Type {} has more than {} character(s).".format(Type, maxlenType)
    if isComplex:
        maxlenFull = 2*(maxlenWhole+maxlenFrac)+5
    else:
        maxlenFull = (maxlenWhole+maxlenFrac)+2

    # Full batches
    for batch in range(nFullBatches):
        if header:
            f.write('{:>{lenType}}\t'.format(Type, lenType=maxlenType))
            for SJ in range(batch*maxnumCols, (batch+1)*maxnumCols):
                f.write('{:>{lenFull}}\t'.format(SJ+shiftIndex, lenFull=maxlenFull))
            f.write('\n')
        for SI in range(Mat.shape[0]):
            f.write('{:>{lenType}}\t'.format(SI+shiftIndex, lenType=maxlenType))
            for SJ in range(batch*maxnumCols, (batch+1)*maxnumCols):
                f.write('{:> {lenFull}.{lenFrac}f}\t'.format(Mat[SI, SJ], lenFull=maxlenFull, lenFrac=maxlenFrac))
            f.write('\n')
        f.write('\n')

    # Extra columns
    if nExtraCols > 0:
        if header:
            f.write('{:>{lenType}}\t'.format(Type, lenType=maxlenType))
            for SJ in range(nFullBatches*maxnumCols, nFullBatches*maxnumCols+nExtraCols):
                f.write('{:>{lenFull}}\t'.format(SJ+shiftIndex, lenFull=maxlenFull))
            f.write('\n')
        for SI in range(Mat.shape[0]):
            f.write('{:>{lenType}}\t'.format(SI+shiftIndex, lenType=maxlenType))
            for SJ in range(nFullBatches*maxnumCols, nFullBatches*maxnumCols+nExtraCols):
                f.write('{:> {lenFull}.{lenFrac}f}\t'.format(Mat[SI, SJ], lenFull=maxlenFull, lenFrac=maxlenFrac))
            f.write('\n')
        f.write('\n')


def printCoordinates(f, Points, Dims, Title, Type, PointLabel='', maxnumCols=6, maxlenWhole=1, maxlenFrac=3, maxlenType=12, shiftIndex=0, header=True):

    # Dims: 1-based dimensions to be printed out

    nFullBatches = len(Dims)/maxnumCols
    nExtraCols = len(Dims)%maxnumCols

    f.write('\n{}\n'.format(Title))
    assert len(Type) <= maxlenType, "Type {} has more than {} character(s).".format(Type, maxlenType)
    assert len(PointLabel) <= maxlenType, "Point label {} has more than {} character(s).".format(Type, maxlenType)
    maxlenFull = (maxlenWhole+maxlenFrac)+2

    # Full batches
    for batch in range(nFullBatches):
        if header:
            f.write('{:>{lenType}}\t'.format(Type, lenType=maxlenType))
            for dim in Dims[batch*maxnumCols:(batch+1)*maxnumCols]:
                f.write('{:>{lenFull}}\t'.format(dim, lenFull=maxlenFull))
            f.write('\n')

        for i,point in enumerate(Points):
            if len(PointLabel) == 0:
                f.write('{:>{lenType}}\t'.format(i+1, lenType=maxlenType))
            else:
                f.write('{:>{lenType}}\t'.format(PointLabel, lenType=maxlenType))
            for dim in Dims[batch*maxnumCols:(batch+1)*maxnumCols]:
                if dim <= point.shape[0]:
                    f.write('{:> {lenFull}.{lenFrac}f}\t'.format(point[dim-1], lenFull=maxlenFull, lenFrac=maxlenFrac))
                else:
                    f.write('{:> {lenFull}.{lenFrac}f}\t'.format(0.0, lenFull=maxlenFull, lenFrac=maxlenFrac))
            f.write('\n')
        f.write('\n')

    # Extra columns
    if nExtraCols > 0:
        if header:
            f.write('{:>{lenType}}\t'.format(Type, lenType=maxlenType))
            for dim in Dims[nFullBatches*maxnumCols:nFullBatches*maxnumCols+nExtraCols]:
                f.write('{:>{lenFull}}\t'.format(dim, lenFull=maxlenFull))
            f.write('\n')

        for i,point in enumerate(Points):
            if len(PointLabel) == 0:
                f.write('{:>{lenType}}\t'.format(i+1, lenType=maxlenType))
            else:
                f.write('{:>{lenType}}\t'.format(PointLabel, lenType=maxlenType))
            for dim in Dims[batch*maxnumCols:(batch+1)*maxnumCols]:
                if dim <= point.shape[0]:
                    f.write('{:> {lenFull}.{lenFrac}f}\t'.format(point[dim-1], lenFull=maxlenFull, lenFrac=maxlenFrac))
                else:
                    f.write('{:> {lenFull}.{lenFrac}f}\t'.format(0.0, lenFull=maxlenFull, lenFrac=maxlenFrac))
            f.write('\n')
        f.write('\n')
