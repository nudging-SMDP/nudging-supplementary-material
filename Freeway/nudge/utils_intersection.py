""" Utils for nudge functions

Helper functions for the script nudge_functions.py

This script requires that `numpy` and `matplotlib`  be installed within the Python
environment you are running this script in.

"""

from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import rc

#rc('text', usetex=True)


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    """ Returns x,y coordinates where line L1 intersects with L2 """

    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def intersection_AC(rho_i, D, A, C, value):
    """ Computes intersection point of line passing through (rho_i.w,rho_i.w) with slope
         m=(D-V*)/(D+V*) with the line segment AC of the enclosing triangle

    Args:
        rho_i: approximated gain at iteration i of the average reward problem
        D: a bound on unsigned, unnudged reward 
        A: w,l coordinates for vertice A of enclosing triangle
        C: w,l coordinates for vertice C of enclosing triangle
        value: value of recurrent state sI according to policy at iteration i
    
    Returns:
        (x,y): tuple with intersection point coordinates
        m: slope 
    """

    m = (D-value)/(D+value)
    b = rho_i.l - m*rho_i.w
    L1 = line([rho_i.w, rho_i.l], [D, m*D+b])
    L2 = line([A.w, A.l],[C.w, C.l])
    return intersection(L1,L2), m

def intersection_BC(rho_i, D, B, C, value):
    """ Computes intersection point of line passing through (rho_i.w,rho_i.w) with slope
         m=(D-V*)/(D+V*) with the line segment BC of the enclosing triangle

    Args:
        rho_i: approximated gain at iteration i of the average reward problem
        D: a bound on unsigned, unnudged reward 
        B: w,l coordinates for vertice B of enclosing triangle
        C: w,l coordinates for vertice C of enclosing triangle
        value: value of recurrent state sI according to policy at iteration i
    
    Returns:
        (x,y): tuple with intersection point coordinates
    """

    m = (D-value)/(D+value)
    b = rho_i.l - m*rho_i.w
    L1 = line([rho_i.w, rho_i.l], [D, m*D+b])
    L2 = line([B.w, B.l],[C.w, C.l])
    return intersection(L1,L2)

def intersection_AB(rho_i, D, A, B, value):
    """ Computes intersection point of line passing through (rho_i.w,rho_i.w) with slope
         m=(D-V*)/(D+V*) with the line segment AB of the enclosing triangle

    Args:
        rho_i: approximated gain at iteration i of the average reward problem
        D: a bound on unsigned, unnudged reward 
        A: w,l coordinates for vertice A of enclosing triangle
        B: w,l coordinates for vertice B of enclosing triangle
        value: value of recurrent state sI according to policy at iteration i
    
    Returns:
        (x,y): tuple with intersection point coordinates
    """

    m = (D-value)/(D+value)
    b = rho_i.l - m*rho_i.w
    L1 = line([rho_i.w, rho_i.l], [D, m*D+b])
    L2 = line([A.w, A.l],[B.w, B.l])
    return intersection(L1,L2)

def intersection_S_BC(Anew, D, B, C, value):
    """ Computes intersection of point S with line segment BC of the enclosing triangle

    Args:
        Anew:  w,l coordinates for new vertice A  of enclosing triangle
        D: a bound on unsigned, unnudged reward 
        B: w,l coordinates for vertice B of enclosing triangle
        C: w,l coordinates for vertice C of enclosing triangle
        value: value of recurrent state sI according to policy at iteration i
    
    Returns:
        (x,y): tuple with intersection point coordinates
    """

    b = Anew.l - Anew.w
    L1 = line([Anew.w, Anew.l],[D, D+b])
    L2 = line([B.w, B.l],[C.w, C.l])
    return intersection(L1, L2)

def intersection_T_AC(Bnew, D, A, C, value):
    """ Computes intersection of point T with line segment AC of the enclosing triangle

    Args:
        Bnew:  w,l coordinates for new vertice B of enclosing triangle
        D: a bound on unsigned, unnudged reward 
        A: w,l coordinates for vertice A of enclosing triangle
        C: w,l coordinates for vertice C of enclosing triangle
        value: value of recurrent state sI according to policy at iteration i
    
    Returns:
        (x,y): tuple with intersection point coordinates
    """

    b = Bnew.l - Bnew.w
    L1 = line([Bnew.w, Bnew.l],[D, D+b])
    L2 = line([A.w, A.l],[C.w, C.l])
    return intersection(L1, L2)


def plot_enclosing_triangle(D, A, B, C, Aant, Bant, Cant, P, Q, rho_i, rho_value, value,
                            iteration, directory):
    """ Saves the plot of the current enclosing triangle

    Args:
        D: a bound on unsigned, unnudged reward 
        A: w,l coordinates for vertice A of enclosing triangle
        B: w,l coordinates for vertice B of enclosing triangle
        C: w,l coordinates for vertice C of enclosing triangle
        Aant: older w,l coordinates for vertice A of enclosing triangle
        Bant: older w,l coordinates for vertice B of enclosing triangle
        Cant: older w,l coordinates for vertice C of enclosing triangle
        P: w,l coordinates for point P. w component of the slope-one projection 
           of the vertice A to the w = −l line
        Q: w,l coordinates for point Q. w component of the slope-one projection 
           of the vertice C to the w = −l line
        rho_i: w,l coordinates for gain rho at iteration i
        rho_value: gain at iterarion i
        value: value of recurrent state sI according to policy at iteration i
        iteration: number of current iteration. Used for set the file name
        directory: path where to save the .png image
    """

    plt.figure()
    enclosing_pts = np.array([[A.w, A.l],[B.w, B.l],[C.w,C.l]]) 
    ant_enclosing_pts = np.array([[Aant.w, Aant.l],[Bant.w, Bant.l],[Cant.w,Cant.l]])   
    wl_pts = np.array([[0,0],[0,D],[D,0]])
    wl_line = plt.Line2D((0, D/2), (0, -D/2), lw=1.0, ls='--', color='black')
    Q_line = plt.Line2D((C.w, Q.w), (C.l, Q.l), lw=1.0, ls='--', color='black')
    P_line = plt.Line2D((A.w, P.w), (A.l, P.l), lw=1.0, ls='--', color='black')
    m = (D-value)/(D+value)
    b = rho_i.l - m*rho_i.w
    rho_line = plt.Line2D((rho_i.w, D), (rho_i.l, m*D+b), lw=2.0, ls='-', color='black')
    enclosing_triangle = Polygon(enclosing_pts, closed=True, color='gray')
    ant_enclosing_triangle = Polygon(ant_enclosing_pts, closed=True, color='yellow')
    wl_space = Polygon(wl_pts, closed=True, color='black', fill=False)
    ax = plt.gca()    
    ax.add_line(wl_line)
    ax.add_line(Q_line)
    ax.add_line(P_line)
    ax.add_line(rho_line)
    ax.add_patch(ant_enclosing_triangle)
    ax.add_patch(enclosing_triangle)
    #ax.add_patch(wl_space)
    plt.scatter(enclosing_pts[:, 0], enclosing_pts[:, 1], s = 10, color ='black')
    plt.scatter([rho_i.w],[rho_i.l], s = 15, color ='red')
    plt.scatter([P.w, Q.w],[P.l, Q.l], s = 15, color ='black')
    ax.text(rho_i.w, rho_i.l, '$p_i$', fontsize=8)
    ax.text(Q.w, Q.l, '$Q$', fontsize=8)
    ax.text(P.w, P.l, '$P$', fontsize=8)
    #plt.title(r'$\rho =$'+str(round(rho_value,3))+'   $v^{*}=$'+str(round(value,3)))
    if B.w > C.w:
	    ax.set(xlim=(0,1.1*B.w), ylim=(-1.1*B.l,1.1*B.l))
    else:
	    ax.set(xlim=(0,1.1*C.w), ylim=(-1.1*B.l,1.1*B.l))
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(directory+'enclosing_triangle_'+str(iteration)+'.png')
    plt.close()