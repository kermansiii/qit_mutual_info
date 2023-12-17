import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import linalg
from scipy import interpolate

# Configure Matplotlib to export the figure to latex
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size':10,
})

# Approximation to zero
eps = 10e-8

#Definition of the theta states
def theta0_matrix(theta):
    return np.array([[np.cos(theta)**2,np.cos(theta)*np.sin(theta)],[np.cos(theta)*np.sin(theta),np.sin(theta)**2]])

def theta1_matrix(theta):
    return np.array([[np.cos(theta)**2,-np.cos(theta)*np.sin(theta)],[-np.cos(theta)*np.sin(theta),np.sin(theta)**2]])

#Definition of the psi matrices
def psi(theta,y):
    if 0 == y:
        state = np.kron(theta0_matrix(theta),np.kron(theta0_matrix(theta),theta0_matrix(theta)))
    if 1 == y:
        state = np.kron(theta0_matrix(theta),np.kron(theta1_matrix(theta),theta1_matrix(theta)))
    if 2 == y:
        state = np.kron(theta1_matrix(theta),np.kron(theta0_matrix(theta),theta1_matrix(theta)))
    if 3 == y:
        state = np.kron(theta1_matrix(theta),np.kron(theta1_matrix(theta),theta0_matrix(theta)))
    return state

#Definition of the POVMs
def povm(theta, y):
    return 1/4*(sqroot.dot(psi(theta,y).dot(sqroot)))

#Entropy for the H(B)_\rho^3
def entropy_b():
    #Diagonalization of the matrix
    w, vr = linalg.eig(t) 
    entropy = 0
    # Assume that close values to zero are null
    w[np.abs(np.real_if_close(w)) < eps] = 0 
    for elem in w:
        if np.real_if_close(elem) !=0:
            #Compute the entropy using the definition
            entropy += np.real_if_close(elem)*np.log2(np.real_if_close(elem))
    return entropy

#Entropy for the H(Y)_\rho
def entropy_y():
    entropy = 0
    for i in range(4):
        for j in range(4):
            #Apply the definition, after the measurement is done
            val = np.real_if_close(np.trace(povm(theta,i).dot(psi(theta,j))))/4
            entropy += val*np.log2(val)
    return entropy

#Values of the x-axis (angle theta)
x = np.linspace(0,np.pi,30)
temp_c = np.power(np.cos(x/2),2)
temp_s = np.power(np.sin(x/2),2)
#Analytical results from the first section
entropy_ar_b = - temp_c*np.log2(temp_c)-temp_s*np.log2(temp_s)
entropy_ar_y = np.sin(x)/2*np.log2( (1+np.sin(x)) / (1-np.sin(x)) ) + 1/2*np.log2(np.power(np.cos(x),2))
#Fix the limit of log(0)
entropy_ar_b[0]=0

#Define the figures and their sizes
fig ,ax  = plt.subplots()
fig2,ax2 = plt.subplots()
fig3,ax3 = plt.subplots()

fig.set_size_inches(6,4)
fig2.set_size_inches(6,4)
fig3.set_size_inches(6,4)

#Use interpolation to make the figures smooth
#and polot I(X;B), I(X;Y) from the analytical results
xnew = np.linspace(0,np.pi, 300)
spl = interpolate.make_interp_spline(x, entropy_ar_y, k=3)
smooth = spl(xnew)

ax.plot(xnew,smooth,label= r'$I(X;Y)$')

spl = interpolate.make_interp_spline(x, entropy_ar_b, k=3)
smooth = spl(xnew)
ax.plot(xnew,smooth,label= r'$I(X;B)$')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$I$')
ax.legend()

#Compute the values for I(X;B^3) and I(X;Y^3)
minfo_ar_b3 = np.zeros(x.shape)
minfo_ar_y3 = np.zeros(x.shape)

#Compute the entropies for different values of theta
for i in range(len(x)):
    theta = x[i]/2
    #Compute the components needed for the POVM
    t = (psi(theta,0)+psi(theta,1)+psi(theta,2)+psi(theta,3))/4
    tinv = linalg.fractional_matrix_power(t,0.5)
    sqroot = linalg.pinv(tinv)
    #Entropies of the POVM, from analytical results, we now that
    #H(X) and H(Y) = 2
    minfo_ar_b3[i] = -1.0*entropy_b()
    minfo_ar_y3[i] = 4+1.0*entropy_y()

#Plot using interpolation
spl = interpolate.make_interp_spline(x, minfo_ar_y3, k=3)
smooth = spl(xnew)
ax2.plot(xnew,smooth,label= r'$I_3(X;Y)$')
spl = interpolate.make_interp_spline(x, minfo_ar_b3, k=3)
smooth = spl(xnew)
ax2.plot(xnew,smooth,label= r'$I_3(X;B^3)$')

y_ar_y = minfo_ar_y3-3*entropy_ar_y
spl = interpolate.make_interp_spline(x, y_ar_y, k=3)
smooth_y = spl(xnew)
y_ar_b = minfo_ar_b3-3*entropy_ar_b
spl = interpolate.make_interp_spline(x, y_ar_b, k=3)
smooth_b = spl(xnew)
ax3.plot(xnew,smooth_y,label= r'$I_3(X;Y)-3I(X;Y)$')
ax3.plot(xnew,smooth_b,label= r'$I_3(X;B^3)-3I(X;B)$')
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel(r'$I$')
ax3.set_xlabel(r'$\theta$')
ax3.set_ylabel(r'$I$')
ax2.legend()
ax3.legend()


#Save he figures in the appropiate format
fig.savefig('./qinfo/plot1.pgf')
fig2.savefig('./qinfo/plot2.pgf')
fig3.savefig('./qinfo/plot3.pgf')
