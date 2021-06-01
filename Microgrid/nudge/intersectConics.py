""" Conics Intersection

Given the homogeneous matrices of two conics, it returns up to four intersection points.

This is the Python implementation of the code published by:

Pierluigi Taddei (2021). Conics intersection (https://www.mathworks.com/matlabcentral/fileexchange/28318-conics-intersection), MATLAB Central File Exchange. 

This script requires that `numpy` and `matplotlib`  be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:
    * intersectConics - returns the intersection points.
    * plotConic - shows a matplotlib plot of two conics and their intersection points.

"""


import numpy as np
from numpy.linalg import matrix_rank, inv, det
from nudge.utils_intersectConics import completeIntersection, decomposeDegenerateConic, intersectConicLine
import matplotlib.pyplot as plt


def intersectConics(E1, E2):
    """ Intersects two non degenerate conics

    Args:
        E1 (np.array): homogeneous matrix of conic section 1
        E2 (np.array): homogeneous matrix of conic section 2

    Returns:
        points_x: a list of x's coordinates of intersection points
        points_y: a list of y's coordinates of intersection points
    """

    P = np.array([])
    r1 = matrix_rank(E1)
    r2 = matrix_rank(E2)
    
    if(r1==3 and r2==3):
        P = completeIntersection(E1,E2)        
    else:
        if (r2 < 3): #E2 is degenerate
            defE = E2
            fullE = E1
        else:
            defE = E1 #E1 is degenerate
            fullE = E2
        m, l = decomposeDegenerateConic(defE)
        P1 = intersectConicLine(fullE,m)
        P2 = intersectConicLine(fullE,l)
        P = np.array([P1, P2])
    points_x = []
    points_y = []
    for i in range(2):
        P1 = P[i]
        if(P1.size!=0):
            for j in range(P1.shape[0]):
                points_x.append(P1[j,0]/P1[j,2])
                points_y.append(P1[j,1]/P1[j,2])
    return points_x, points_y

def plotConic(L, R, points_x, points_y, xBounds=[-50,50], yBounds=[-50,50]):
    """ Plots two conic sections based on their homogeneous representation and 
        their intersection points

    The homogeneous representation of a conic is:
    Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0

    And the matrix form:
            [ A    B/2     D/2
        M =  B/2    C      E/2
             D/2   E/2      F ]

    Args:
        L (np.array): homogeneous matrix of conic section 1
        R (np.array): homogeneous matrix of conic section 2
        points_x: a list of x's coordinates of intersection points
        points_y: a list of y's coordinates of intersection points
        xBounds, yBounds = a list of maximum x and y upper and lower bounds, by default [-50,50]
    """ 
    
    x = np.linspace(xBounds[0],xBounds[1],5000)
    y = np.linspace(yBounds[0],yBounds[1],5000)
    x, y = np.meshgrid(x, y)
    #assert B**2 - 4*A*C == 0   
    A = L[0,0]
    B = 2*L[0,1]
    C = L[1,1]
    D = 2*L[0,2]
    E = 2*L[1,2]
    F = L[2,2] 
    plt.contour(x, y,(A*x**2 + B*x*y + C*y**2 + D*x + E*y + F), [0], colors='g')
    A = R[0,0]
    B = 2*R[0,1]
    C = R[1,1]
    D = 2*R[0,2]
    E = 2*R[1,2]
    F = R[2,2] 
    plt.contour(x, y,(A*x**2 + B*x*y + C*y**2 + D*x + E*y + F), [0], colors='r')
    plt.scatter(points_x, points_y, marker='o', color='k')
    plt.show()

