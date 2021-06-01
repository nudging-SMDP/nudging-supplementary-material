""" Utils for computing conic intersection points

Helper functions for the script intersectConics.py

This is the Python implementation of the code published by:

Pierluigi Taddei (2021). Conics intersection (https://www.mathworks.com/matlabcentral/fileexchange/28318-conics-intersection), MATLAB Central File Exchange. 

This script requires that `numpy` be installed within the Python
environment you are running this script in.

This file is imported as a module in intersectConics.py and contains the following
functions:
    * completeIntersection 
    * decomposeDegenerateConic 

"""

import numpy as np
from numpy.linalg import matrix_rank, inv, det


def completeIntersection(E1, E2):
    """ Exploits determinant multilinearity 
    Args:
        E1 (np.array): homogeneous matrix of conic section 1
        E2 (np.array): homogeneous matrix of conic section 2
    """

    # characteristic polynom: C1*(-C2)^-1 - lambda I
    EE = np.matmul(E1, inv(-E2))
    k = np.array([  -1, 
                    np.trace(EE),
                    -( det(EE[0:2,0:2]) + det(EE[1:3,1:3]) + det(EE[[[0],[2]],[0,2]]) ),
                    det(EE) ])

    r = np.roots(k)
    m = np.array([])
    if (np.isreal(r[0])):
        E0 = E1 + r[0]*E2
        m, l = decomposeDegenerateConic(E0)
    if (m.size==0 and np.isreal(r[1])):
        E0 = E1 + r[1]*E2
        m, l = decomposeDegenerateConic(E0)
    if (m.size==0 and np.isreal(r[2])):
        E0 = E1 + r[2]*E2
        m, l = decomposeDegenerateConic(E0)

    if (m.size==0):
        print('[CONICS] no intersecting lines detected')
        P = np.array([])
        return P
  
    P1 = intersectConicLine(E1,m)
    P2 = intersectConicLine(E1,l)
    P = np.array([P1, P2])
    return P

def decomposeDegenerateConic(c):
    """ Decompose in two lines the given degenerate conic

    Args:
        c : degenerate symmetric 3x3 matrix of the conic
    
    Returns:
        l, m: homogeneous line coordinates
    """

    if (matrix_rank(c) == 1): #c is rank 1: direct split is possible
        C = c
    else: #rank 2: need to detect the correct rank 1 matrix
        # use the dual conic of c
        B = -adjointSym3(c)
        # detect intersection point p
        maxV = np.max(np.abs(np.diag(B)))
        di = np.argmax(np.abs(np.diag(B)))
        i = di
        if (B[i,i] < 0):
            l = np.array([])
            m = np.array([])
            return l, m
        b = np.sqrt(B[i,i])
        p = B[:,i]/b

        # detect lines product
        Mp = crossMatrix(p)
        C = c + Mp

    # recover lines
    maxV = np.max(np.abs( C.flatten(order='F') ))
    ci = np.argmax(np.abs( C.flatten(order='F') )) + 1.0
    j = int(np.floor((ci-1.0) / 3.0)+1.0)
    i = int(ci - (j-1.0)*3.0)
    l = C[i-1,:].T 
    m = C[:,j-1]
    return l, m

def adjointSym3(M):
    A = np.zeros([3,3], dtype=complex)
    a = M[0,0] 
    b = M[0,1] 
    c = M[1,1] 
    d = M[0,2]   
    e = M[1,2]
    f = M[2,2]

    A[0,0] = c*f - e*e
    A[0,1] = -b*f + e*d
    A[0,2] = b*e - c*d

    A[1,0] = A[0,1]
    A[1,1] = a*f - d*d
    A[1,2] = -a*e + b*d

    A[2,0] = A[0,2]
    A[2,1] = A[1,2]
    A[2,2] = a*c - b*b

    return A

def intersectConicLine(C, l):
    P = np.array([])
    p1, p2 = getPointsOnLine(l)

    p1Cp1 = np.matmul(p1.T, np.matmul(C, p1) )
    p2Cp2 = np.matmul(p2.T, np.matmul(C, p2) )
    p1Cp2 = np.matmul(p1.T, np.matmul(C, p2) )

    if (p2Cp2 == 0): #linear
       k1 = -0.5*p1Cp1 / p1Cp2;
       P = np.array([p1 + k1*p2])
    else:
        delta = p1Cp2**2 - p1Cp1*p2Cp2
        if (delta >= 0):
            deltaSqrt = np.sqrt(delta)
            k1 = (-p1Cp2 + deltaSqrt)/p2Cp2
            k2 = (-p1Cp2 - deltaSqrt)/p2Cp2
            P = np.array( [p1 + k1*p2, p1 + k2*p2] )
    return P

def crossMatrix(p):
    Mp = np.zeros([3,3], dtype=complex)
    Mp[0,1] = p[2]
    Mp[0,2] = -p[1]
    Mp[1,0] = -p[2]
    Mp[1,2] = p[0]
    Mp[2,0] = p[1]
    Mp[2,1] = -p[0]
    return Mp

def getPointsOnLine(l):
    if (l[0] == 0 and l[1] == 0): #line at infinity
        p1 = np.array([1, 0, 0])
        p2 = np.array([0, 1, 0])
    else:
        p2 = np.array([-l[1], l[0], 0])
        if (np.abs(l[0]) < np.abs(l[1])):
            p1 = np.array([0, -l[2], l[1]])
        else:
            p1 = np.array([-l[2], 0, l[0]])
    return p1, p2

