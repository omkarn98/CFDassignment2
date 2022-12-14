# MTF072 Computational Fluid Dynamics
# Task 2: convection-diffusion equation
# Template prepared by:
# Gonzalo Montero Villar
# Department  of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# November 2019

# The script assumes that the folder with data is in the same path as this file

# Packages needed
import numpy as np
import matplotlib.pyplot as plt

def  ReadDataAndGeometry(caseID, grid_type):
	if caseID <= 5:
		grid_number = 1
	elif caseID <= 10:
		grid_number = 2
	elif caseID <= 15:
		grid_number = 3
	elif caseID <= 20:
		grid_number = 4
	elif caseID <= 25:
		grid_number = 5

	path = 'data/grid%d/%s_grid' % (grid_number,grid_type)

	# Read data
	xCoords_M = np.genfromtxt('%s/xc.dat' % (path)) # x node coordinates
	yCoords_M = np.genfromtxt('%s/yc.dat' % (path)) # y node coordinates
	u_data = np.genfromtxt('%s/u.dat' % (path))     # U velocity at the nodes
	v_data = np.genfromtxt('%s/v.dat' % (path))     # V veloctiy at the nodes

	# Allocate geometrical data and variables
	mI        = len(xCoords_M);      # number of mesh points in the x direction
	mJ        = len(yCoords_M);      # number of mesh points in the y direction
	nI        = mI + 1;              # number of nodes in the x direction
	nJ        = mJ + 1;              # number of nodes in the y direction
	xCoords_N = np.zeros((nI,1));    # mesh points x coordinates
	yCoords_N = np.zeros((nJ,1));    # mesh points y coordinates
	dxe_N     = np.zeros((nI,1));    # X distance to east cell
	dxw_N     = np.zeros((nI,1));    # X distance to west cell
	dyn_N     = np.zeros((nJ,1));    # Y distance to north cell
	dys_N     = np.zeros((nJ,1));    # Y distance to south cell
	dx_CV      = np.zeros((nI,1));    # X size of the cell
	dy_CV      = np.zeros((nJ,1));    # Y size of the cell

	# Fill the cell coordinates as the mid point between mesh points, and at the same
	# position at the boundaries. Compute cell sizes
	for i in range(1,nI-1):
		xCoords_N[i] = (xCoords_M[i] + xCoords_M[i-1])/2
		dx_CV[i]      = xCoords_M[i] - xCoords_M[i-1]
	
	for j in range(1,nJ-1):
		yCoords_N[j] = (yCoords_M[j] + yCoords_M[j-1])/2
		dy_CV[j]      = yCoords_M[j] - yCoords_M[j-1]
	
	xCoords_N[0]  = xCoords_M[0]
	xCoords_N[-1] = xCoords_M[-1]
	yCoords_N[0]  = yCoords_M[0]
	yCoords_N[-1] = yCoords_M[-1]

	# Compute distances between nodes
	for i in range(1,nI-1): 
		dxe_N[i] = xCoords_N[i+1] - xCoords_N[i]
		dxw_N[i] = xCoords_N[i] - xCoords_N[i-1]
	
	for j in range(1,nJ-1):
		dyn_N[j] = yCoords_N[j+1] - yCoords_N[j]
		dys_N[j] = yCoords_N[j] - yCoords_N[j-1]
	

	# Reshape the velocity data
	U = u_data.reshape(nI,nJ)
	V = v_data.reshape(nI,nJ)

	return [xCoords_M, yCoords_M, mI, mJ, nI, nJ, xCoords_N, yCoords_N, dxe_N, dxw_N, dyn_N, dys_N, dx_CV, dy_CV, U, V]

# Inputs

grid_type   = 'coarse' # either 'coarse' or 'fine'
caseID      =     3    # your case number to solve
k           =     1
rho         =     1   # density
nIterations =     80000  # number of iterations
Cp          = 500
method = 'TDMA'
plotVelocityVectors = False
resTolerance = 0.001

# Read data for velocity fields and geometrical quantities

# For all the matrices the first input makes reference to the x coordinate
# and the second input to the y coordinate, (i+1) is east and (j+1) north

[xCoords_M, yCoords_M, mI, mJ, nI, nJ, xCoords_N, yCoords_N, dxe_N, dxw_N, dyn_N, dys_N, dx_CV, dy_CV, U, V] = ReadDataAndGeometry(caseID, grid_type)

# Plot velocity vectors if required
if plotVelocityVectors:
	plt.figure()
	plt.quiver(xCoords_N, yCoords_N, U.T, V.T)
	plt.title('Velocity vectors')
	plt.xlabel('x [m]')
	plt.ylabel('y [m]')
	plt.show()

#Find boundaries to the problem
#Number for each point on boundary, 0 for wall, 1 for inlet, 2 for outlet.
B1 = np.zeros((nI, 1))
B2 = np.zeros((nJ, 1))
B3 = np.zeros((nI, 1))
B4 = np.zeros((nJ, 1))
velWall = 1e-4 #Tolerance for wall speed 
for i in range(1, nI-1):
	j = 0
	velNorm = -V[i,j] #Scalar multiplied with boundary normals
	if(velNorm < velWall and velNorm > -velNorm):
		B1[i] = 0
	elif(velNorm < -velWall):
		B1[i] = 1
	elif(velNorm > velWall):
		B1[i] = 2

	j = nJ-1
	velNorm = V[i,j] #Scalar multiplied with boundary normals
	if(velNorm < velWall and velNorm > -velNorm):
		B3[i] = 0
	elif(velNorm < -velWall):
		B3[i] = 1
	elif(velNorm > velWall):
		B3[i] = 2

for j in range(1,nJ-1):
	i = nI-1
	velNorm = U[i,j] #Scalar multiplied with boundary normals
	if(velNorm < velWall and velNorm > -velNorm):
		B2[j] = 0
	elif(velNorm < -velWall):
		B2[j] = 1
	elif(velNorm > velWall):
		B2[j] = 2
	i = 0
	velNorm = -U[i,j] #Scalar multiplied with boundary normals
	if(velNorm < velWall and velNorm > -velNorm):
		B4[j] = 0
	elif(velNorm < -velWall):
		B4[j] = 1
	elif(velNorm > velWall):
		B4[j] = 2

# print(B2)
# Allocate needed vairables
T = np.zeros((nI, nJ))        # temperature matrix
D = np.zeros((nI, nJ,4))      # diffusive coefficients e, w, n and s
F = np.zeros((nI, nJ,4))      # convective coefficients e, w, n and s
coeffsT = np.zeros((nI,nJ,5)) # hybrid scheme coefficients E, W, N, S, P

residuals = []

# Code

gamma = k/Cp

# Diffusive and convective coefficient calculations
for i in range(1,nI-1):
	for j in range(1,nJ-1):
		D[i,j,0] =  gamma * dy_CV[i] / dxe_N[i] # east diffusive
		D[i,j,1] =  gamma * dy_CV[i] / dxw_N[i] # west diffusive
		D[i,j,2] =  gamma * dx_CV[j] / dyn_N[j] # north diffusive
		D[i,j,3] =  gamma * dx_CV[j] / dys_N[j] # south diffusive
	# if(U[i,j] and V[i,j] > 0):
			
		Fx_e = 0.5 * dx_CV[i] / dxe_N[i]
		Fx_w = 0.5 * dx_CV[i] / dxw_N[i]
		Fx_n = 0.5 * dy_CV[j] / dyn_N[j]
		Fx_s = 0.5 * dy_CV[j] / dys_N[j]
		
		
		F[i,j,0] =  Fx_e * (rho * U[i+1,j]) * dy_CV[i] + (1-Fx_e) * rho * U[i,j] * dy_CV[i]  # east convective
		F[i,j,1] =  Fx_w * (rho * U[i-1,j]) * dy_CV[i] + (1-Fx_w) * rho * U[i,j] * dy_CV[i]  # weast convective
		F[i,j,2] =  Fx_n * (rho * V[i,j+1]) * dx_CV[j] + (1-Fx_n) * rho * V[i,j] * dx_CV[j]  # north convective
		F[i,j,3] =  Fx_s * (rho * V[i,j-1]) * dx_CV[j] + (1-Fx_s) * rho * V[i,j] * dx_CV[j]  # south convective

# Hybrid scheme coefficients calculations (taking into account boundary conditions)
for i in range(1,nI-1):
	for j in range(1,nJ-1):
		coeffsT[i,j,0] = np.max([-F[i,j,0], D[i,j,0] - F[i,j,0]/2, 0]) #East
		coeffsT[i,j,1] = np.max([ F[i,j,1], D[i,j,1] + F[i,j,1]/2, 0]) #West
		coeffsT[i,j,2] = np.max([-F[i,j,2], D[i,j,2] - F[i,j,2]/2, 0]) #North
		coeffsT[i,j,3] = np.max([ F[i,j,3], D[i,j,3] + F[i,j,3]/2, 0]) #South

#Boundary conditions

for j in range(1, nJ-1):
	i = 1
	if(B4[j] == 1):
		T[0,j] = 293
	else:
		coeffsT[i,j,1] = 0

	i = nI-2
	if(B2[j] == 2):
		coeffsT[i,j,0] = 0
	else:
		T[nI-1,j] = 283


for i in range(1,nI-1):
	j = 1
	coeffsT[i,j,3] = 0
	# T[i,j] = 273

	j = nJ-2
	coeffsT[i,j,2] = 0
	
for i in range(1, nI-1):
	for j in range(1, nJ-1):
		S_p = 0 # -F[i,j,0] + F[i,j,1] - F[i,j,2] + F[i,j,3] 0 due to div(v) = 0
		coeffsT[i,j,4] = np.sum(coeffsT[i,j,0:4]) - S_p

for iter in range(nIterations): 
    # Impose boundary conditions
    
    # Solve for T using Gauss-Seidel or TDMA (both results need to be presented)
	if (method == 'Gauss'):
		for i in range(1, nI-1):
			for j in range(1, nJ-1):
				RHS = coeffsT[i,j,0] * T[i+1,j] + coeffsT[i,j,1] * T[i-1,j] \
					+ coeffsT[i,j,2] * T[i,j+1] + coeffsT[i,j,3] * T[i,j-1]
				T[i,j] = RHS/ coeffsT[i,j,4]
		
	elif(method == 'TDMA'):
		#Pre-TDMA coefficients horizontal
		a = np.zeros([nI,nJ])
		b = np.zeros([nI,nJ])
		c = np.zeros([nI,nJ])
		d = np.zeros([nI,nJ])
		P = np.zeros([nI,nJ])
		Q = np.zeros([nI,nJ])

		if(np.mod(iter,2)==0):
			#Solve horizontally
			for j in range(1,nJ-1):
				for i in range(1,nI-1):
					a[i,j] = coeffsT[i,j,4] #a_p
					b[i,j] = coeffsT[i,j,0] #a_e
					c[i,j] = coeffsT[i,j,1] #a_w
					d[i,j] = coeffsT[i,j,2] * T[i,j+1] + coeffsT[i,j,3] * T[i,j-1] 
				#Construct P and Q terms
				i = 1
				P[i,j] = b[i,j] / a[i,j]
				Q[i,j] = (d[i,j] + c[i,j] * T[i-1,j])/a[i,j] 
				for i in range(2,nI-2):
					P[i,j] = b[i,j] / (a[i,j] - c[i,j] * P[i-1,j])
					Q[i,j] = (d[i,j] + c[i,j] * Q[i-1,j]) / (a[i,j] - c[i,j] * P[i-1,j])
				i=nI-2
				P[i,j] = 0
				Q[i,j] = (d[i,j] + c[i,j] * Q[i-1,j] + b[i,j] * T[i+1,j]) / (a[i,j] - c[i,j] * P[i-1,j])

				for i in reversed(range(1, nI-1)):
					T[i,j] = P[i,j] * T[i+1,j] + Q[i,j]
		else:
			#Solve horizontally
			for i in range(1,nI-1):
				for j in range(1,nJ-1):
					a[i,j] = coeffsT[i,j,4] #a_p
					b[i,j] = coeffsT[i,j,2] #a_n
					c[i,j] = coeffsT[i,j,3] #a_s
					d[i,j] = coeffsT[i,j,0] * T[i+1,j] + coeffsT[i,j,1] * T[i-1,j] 
				#Construct P and Q terms
				j = 1
				P[i,j] = b[i,j] / a[i,j]
				Q[i,j] = (d[i,j] + c[i,j] * T[i,j-1])/a[i,j] 
				for j in range(2,nJ-2):
					P[i,j] = b[i,j] / (a[i,j] - c[i,j] * P[i,j-1])
					Q[i,j] = (d[i,j] + c[i,j] * Q[i,j-1]) / (a[i,j] - c[i,j] * P[i,j-1])
				j=nJ-2
				P[i,j] = 0
				Q[i,j] = (d[i,j] + c[i,j] * Q[i,j-1] + b[i,j] * T[i,j+1]) / (a[i,j] - c[i,j] * P[i,j-1])

				for j in reversed(range(1, nJ-1)):
					T[i,j] = P[i,j] * T[i,j+1] + Q[i,j]

    # Copy temperatures to boundaries
	for i in range(0, nI):
		j = 1
		T[i,j-1] = T[i,j]

		j = nJ-1
		T[i,j] = T[i,j-1]

	for j in range(1, nJ-1):
		i = 1
		if(B4[j] != 1):
			T[i-1,j] = T[i,j]

		i = nI-1
		if(B2[j] != 0):
			T[i,j] = T[i-1,j]

    # # Compute residuals (taking into account normalization)
	F_res_xi = 0
	F_res_xo = 0
	for j in range(1, nJ-1):
		i = 0
		if(B4[j] != 0):
			F_res_xi = F_res_xi + (abs(rho * U[i,j] * dy_CV[j]) * T[i,j]) + (abs(rho * V[i,j] * dx_CV[i]) * T[i,j]) 

		i = nI-1
		if(B2[j] != 0):
			F_res_xo = F_res_xo + (abs(rho * U[i,j] * dy_CV[j]) * T[i,j]) + (abs(rho * V[i,j] * dx_CV[i]) * T[i,j]) 
	
	F_res_total = abs(F_res_xi - F_res_xo)

	R = 0
	r = 0
	for i in range(1, nI-1):
		for j in range(1, nJ-1):
			R = R + abs(coeffsT[i,j,4] * T[i,j] - coeffsT[i,j,0] * T[i+1, j] \
				- coeffsT[i,j,1] * T[i-1, j] - coeffsT[i,j,2] * T[i,j+1] \
				- coeffsT[i,j,3] * T[i, j-1]) 

	r = R / F_res_total	



	residuals.append(r) # fill with your residual value for the 
                       # current iteration
    
	print('iteration: %d\nresT = %.5e\n\n' % (iter, residuals[-1]))
    
    # Check convergence
    
	if resTolerance>residuals[-1]:	
		break

# Compute heat fluxes
q = np.zeros((nI, nJ, 2))
for i in range(1,nI-1):
	for j in range(1,nJ-1):
		q[i,j,0] = -k*(T[i+1,j]-T[i-1,j])/(dxe_N[i]+dxw_N[i])
		q[i,j,1] = -k*(T[i,j+1]-T[i,j-1])/(dyn_N[j]+dys_N[j])

#Boundary 2 non-outlet part
i = nI-1
for j in range(1, nJ-1):
	q[i,j,0] = -k * (T[i,j] - T[i-1,j])/dxw_N[i-1]
	q[i,j,1] = -k*(T[i,j+1]-T[i,j-1])/(dyn_N[j]+dys_N[j])

#Heat flux out of dirichlet
F_dirich = 0
for j in range(1, nJ-1):
	if(B2[j] == 0):
		F_dirich = F_dirich + q[nI-1,j,0] * dy_CV[j]

#Total heat flux
F_total = F_res_xo * Cp + F_dirich - F_res_xi * Cp #W, Correct units
print('The total heat flux of the system is ', str(round(F_total[0], 2)), ' Watts.')

F_ratio = (F_total)/(F_res_xo * Cp + F_dirich + F_res_xi * Cp)
print('The ratio between the heat flux and the sum of the magnitude of the heat fluxes is ', str(round(F_ratio[0] * 100, 2)), ' percent.')

# Plotting (these are some examples, more plots might be needed)
xv, yv = np.meshgrid(xCoords_N, yCoords_N)

#plt.figure()
#plt.plot()

# plt.figure()
# plt.quiver(xv, yv, U.T, V.T)
# plt.title('Velocity vectors')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')

plt.figure()
plt.quiver(xv, yv, q[:,:,0].T, q[:,:,1].T)
plt.title('Heat flux vectors')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.figure()
plt.plot(residuals)
plt.yscale('log')
plt.title('Residual convergence')
plt.xlabel('iterations')
plt.ylabel('residuals [-]')
plt.title('Residual')

plt.figure()
plt.plot(xCoords_N, T[:,0])
plt.xlabel('x [m]')
plt.ylabel('T [K]')
plt.title('Temperature on boundary 1')

plt.figure()
plt.contourf(xv, yv, T.T) #contourf for smooth plot
plt.colorbar()
#plt.quiver(xv, yv, U.T, V.T)
plt.title('Temperature')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show(block = True)