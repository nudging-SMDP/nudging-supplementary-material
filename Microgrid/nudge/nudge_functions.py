""" Nudging Functions

Funtions used in optimal/alpha nudging algorithm to update gain (rho) and
the enclosing triangle

This file can also be imported as a module and contains the following
functions:
	* set_initial_enclosing_triangle - set initial vertices A, B, C for enclosing triangle, given D
    * updatePQ - updates points P and Q for enclosing triangle, given new vertices A and C
    * get_alpha_rho_value - get next rho value for a given alpha, between 0 and 1
	* get_optimal_rho_value - get optimal rho value by the intersection of the conic sections of the
	  left and right unsertainty	
	* get_r -  get an aproximation for the next optimal rho value, comparing the left and right
	  uncertainty in several points.
	* rho - returns w,l coordinates for rho in line w=-l
	* projr - slope-one projection 
	* update_enclosing_triangle - update vertices A, B, C for enclosing triangle

This script also contains the class coord, for setting w,l coordinates of vertices in enclosing 
triangle
"""

from nudge.utils_intersection import *
from nudge.intersectConics import intersectConics

# global variables: initialization of vertices and key-points of enclosing triangle
D = None; A = None; B = None; C = None; P = None; Q = None


class coord:
	"""
	A class used to set w,l coordinates of vertices in enclosing triangles

	Attributes
	----------
		w - coordinate w in space w-l
		l - coordinate l in space w-l
	
	Methods
	-------
		copy() - returns a coord object with the same attributes w and l
	"""

	def __init__(self, w, l):
		self.w = w
		self.l = l
	def copy(self):
		return coord(self.w, self.l)

Anew = coord(0,0)
Bnew = coord(0,0)
Cnew = coord(0,0)

Aant = coord(0,0)
Bant = coord(0,0)
Cant = coord(0,0)

def updatePQ(A,C):
	""" Updates slope-on projections for vertices A and C

	Args:
		A: w,l coordinates for vertice A of enclosing triangle
        C: w,l coordinates for vertice C of enclosing triangle

	Returns:
		P: coord object for point P. w and l components of the slope-one projection 
           of the vertice A to the w = −l line
        Q: coord object for point Q. w and l components of the slope-one projection 
           of the vertice C to the w = −l line
	"""

	P = coord((A.w-A.l)/2.0, -(A.w-A.l)/2.0)
	Q = coord((C.w-C.l)/2.0, -(C.w-C.l)/2.0)
	return P, Q


def set_initial_enclosing_triangle(D_value):
	""" Initialize the A,B,C vertices of enclosign triangle given D
	Args:
		D_value: a bound on unsigned, unnudged reward 
	
	Modifies global variables A,B,C,P,Q,D
	"""

	global A,B,C,P,Q,D
	D = D_value
	A = coord(0.0,   0.0)
	B = coord(D/2.0, D/2.0)
	C = coord(D, 0)
	P,Q = updatePQ(A,C)

def get_alpha_rho_value(alpha=0.5):
	""" Computes the new gain proportional to alpha between the line segment PQ """
	
	B1 = (B.w - B.l)/2.0
	C1 = (C.w - C.l)/2.0
	rho = (1-alpha)*B1 + alpha*C1
	return 2*rho

def get_optimal_rho_value():
	""" Computes the optimal rho value, as the intersection point of the homogeneous for the 
	    conic sections of the left and right uncertainty
	
	Returns:
		rho
	""" 
	m_r = (C.l-A.l) / (C.w-A.w)
	m_b = (C.l-B.l) / (C.w-B.w)
	Br_Cr = (m_r*(B.w-C.w)-B.l+C.l)/(m_r + 1.0)
	Br = (m_r*B.w-B.l) / (m_r+1.0)
	B1 = (B.w-B.l) / 2.0
	C1 = (C.w-C.l) / 2.0
	a = (1.0 - m_b)*(1.0 + m_r)
	b = (1.0 + m_b)*(1.0 - m_r)
	c = m_r - m_b
	aCr_bCb = (1.0-m_b)*(m_r*C.w-C.l) + (1.0-m_r)*(m_b*C.w-C.l)
	matA = np.array([[0.0, 0.5, Br_Cr], 
				 	[0.5, 0.0, -Br], 
				 	[Br_Cr, -Br, -4*B1*Br_Cr]])
	matB = np.array([[c, -(a+b), -2*C1*c], 
				 	[-(a+b), c, aCr_bCb], 
				 	[-2*C1*c, aCr_bCb, 4*c*(C1**2)]])
	pX, pY = intersectConics(matA,matB)
	rhos = []
	for i in range(len(pX)):
		pi = np.real(pX[i]/2.0)
		if (np.round(pi,4)>=np.round(B1,4) and np.round(pi,4)<=np.round(C1,4)):
			rhos.append(pi*2.0)
	return np.max(rhos)


def get_r(points=1000):
	""" Computes an approximation for the optimal rho value, evaluating over several points in the 
		left and right uncertainty and looking for the one where both are almost the same.
	
	Args:
		points: number of point to evaluate
	Returns:
		rho: an approximation to the optimal rho value
	""" 

	s = B.l - A.l
	t = B.w - A.w
	U = A.l - C.l
	V = A.w - C.w
	m = B.l - C.l
	n = B.w - C.w
	P = C.l*A.w - A.l*C.w
	b = C.l*B.w - B.l*C.w

	rmin = (s*B.w-t*B.l)/(s+t)
	rmax = (s*C.w-t*C.l)/(s+t)
	r = np.linspace(rmin,rmax,points)

	if(n==0):
		Q = (s*V-t*U)*(r*(m+n)+b)*(r*(U+V)+P)/(n*s-m*t)
		l_s = (-b*U - r*m*(U+V) - m*np.sqrt(Q)) / (m*V-n*U)
		w_T = (P*(n*l_s-b-m*r)+r*V*((m+n)*l_s-b)) / (m*V*(l_s+r)-U*(n*l_s-b-m*r))
		f_2 = (V*l_s*(n*s-m*t) + (m*P*t-b*s*V) - m*w_T*(s*V-t*U)) / (m*V*(s+t))
	else:
		Q = (s*V-t*U)*(r*(U+V)+P)*(r*(m+n)+b)/(n*s-m*t)
		w_s = (-b*V - r*n*(U+V) - n*np.sqrt(Q)) / (m*V-n*U)
		w_T = (n*P*(w_s-r)+r*V*((m+n)*w_s+b)) / (V*(m*w_s+n*r+b)-n*U*(w_s-r))
		f_2 = (V*w_s*(n*s-m*t) + (t*n*P-t*V*b) - n*w_T*(s*V-t*U)) / (n*V*(s+t))

	f_1 = (s*B.w-t*B.l-r*(s+t))*(U*B.w-V*B.l+P)/(s+t)/(V*(B.l+r)-U*(B.w-r))
	idx = np.nanargmin(np.abs(f_2-f_1))
	rho = r[idx]
	return rho*2.0

def rho(rho_value):
	""" Returns coord object for rho """

	return coord(rho_value/2.0, -rho_value/2.0)

def projr(X):
	""" Computes the slope-one projection of X """

	return (X.w-X.l)/2.0


def update_enclosing_triangle(rho_value,v_k,iteration,directory):
	""" Updates the vertices of the enclosing triangle, given a new rho value

	Args:
		rho_value: gain at iterarion i
		v_k: value of recurrent state sI according to policy at iteration i
		iteration: number of current iteration
        directory: path where to save the .png image of the enclosing triangle

	Returns:
		exit_code: -1 if not getting a valid enclosing triangle
		m: slope of line passing through (rho_value.w, rho_value.w)
	"""

	global A, B, C, P, Q, Anew, Bnew, Cnew, Aant, Bant, Cant
	rho_coord = None
	exit_code = -1
	tol = 10e-6 
	rho_i = rho_value/2.0
	# compute point S, which is the intersection with segment AC
	m = (D-v_k)/(D+v_k)
	m_AC = (C.l-A.l)/(C.w-A.w)
	S = coord(0,0)
	S.w = (rho_i*(m+1) - m_AC*A.w + A.l)/(m - m_AC)
	S.l = m*(S.w-rho_i) - rho_i
	# evaluate if we are in case 4
	s = projr(S)
	if(np.abs(rho_i-s)<=tol):
		Anew = S
		Bnew = S
		Cnew = S
		exit_code = 4
	else:
		# compute slope of line between rho and S
		m1 = (S.l+rho_i)/(S.w-rho_i)
		m_BC = (B.l-C.l)/(B.w-C.w)
		# compute point T, which is the intersection with segment BC
		T = coord(0,0)
		T.w = ((m1+1)*rho_i-m_BC*C.w+C.l)/(m1-m_BC)
		T.l = m1*(T.w-rho_i)-rho_i
		# evaluate if we are in cases 2, 3 or 5
		if(s>rho_i):
			# in case 5
			Anew = S
			Cnew = T
			w = (m_BC*C.w-C.l-S.w+S.l)/(m_BC-1)
			l = w - S.w + S.l
			Bnew = coord(w,l)
			exit_code = 5
		elif(projr(T)>projr(A)):
			# in case 3
			Cnew = S
			Bnew = T
			w = (m_AC*A.w-A.l-T.w+T.l)/(m_AC-1)
			l = w - T.w + T.l
			Anew = coord(w,l)
			exit_code = 3
		else:
			# in case 2
			Anew = A
			Cnew = S
			w = ((m1+1)*rho_i-A.w+A.l)/(m1-1)
			l = w-A.w+A.l
			Bnew = coord(w,l)
			exit_code = 2
	if((projr(Anew)-projr(Cnew))>tol):
		# not an enclosing triangle
		exit_code = -1
	if exit_code != -1:
		Aant = A.copy()
		Bant = B.copy()
		Cant = C.copy()
		A = Anew.copy()
		B = Bnew.copy()
		C = Cnew.copy()
		P,Q = updatePQ(A,C)
		print('New coordinates of enclosing triangle')
		print('A: '+str((A.w, A.l)))
		print('B: '+str((B.w, B.l)))
		print('C: '+str((C.w, C.l)))
		print('P: '+str((P.w, P.l)))
		print('Q: '+str((Q.w, Q.l)))		
	rho_coord = rho(rho_value)
	print('rho_coord:' + str((rho_coord.w, rho_coord.l)))
	plot_enclosing_triangle(D, A, B, C, Aant, Bant, Cant, P, Q, rho_coord, rho_value, v_k, iteration, directory)
	return exit_code, m

