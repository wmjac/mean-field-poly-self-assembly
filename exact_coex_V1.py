import numpy as np
import scipy.optimize, scipy.spatial
import math
import matplotlib.pyplot as plt
import random
import sys


### m is the degree of complexation (n in the paper)
m = 4

### Excluded volume in unit of blob
Np = 6
Ns = 24


### dw is \beta \delta G
dw = -27.

### ep is \epsilon, dpp is the Boltzmann factor
ep = -math.log(56)
dpp = math.e**(-ep)


### consider no interaction between polymer and complex in this work
dsp = 0
### consider no complex-complex interaction in this work
dss = 0

### internal energy per complex, from eq.(6) in the main text
us = dw+m*(1-Np)


### initial guesses for phi_s in the range (0,1) for the optimizer, used in solving eq.(7)
### We need multiple initial guesses to find all the minima in Fig. 2a.
bfw = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]


### SAFT, solving for X in eq.(5)
def solveX(phis,phip):
    rhos = phis/Ns
    rhop = phip/Np
#    print(x1,rhos,rhop)
    tol = 10**(-11)
    X = np.array([0.9,0.9])
    Xn = np.zeros(2)
    while True:
        Xn[0] = 1/(1+rhos*X[0]*dss+Np*rhop*X[1]*dsp/m)
        Xn[1] = 1/(1+rhos*X[0]*dsp+Np*rhop*X[1]*dpp)
        if np.linalg.norm(Xn-X)<tol and np.all(Xn<=1):
            return Xn[0],Xn[1]
        else:
            X = np.copy(Xn)


### chemical potential for polymer(P) and complex(S)
def muevs(phis,phip):
    mu = -math.log(1-phis)+Ns*math.log((1-phis)/(1-phis-phip))-Ns*(1-1/Np)*phip/(1-phis)
    return mu

def muevp(phis,phip):
    mu = -Np*math.log(1-phis-phip)+(Np-1)*math.log(1-phis)
    return mu


def mus(phis,phip):
    Xs, Xp = solveX(phis,phip)
    # print('s=',Xs,Xp)
    ms = math.log(phis)+us+muevs(phis,phip)+m*math.log(Xs)
    return ms

def mup(phis,phip):
    Xs, Xp = solveX(phis,phip)
    # print('p=',phis,phip,Xp)
    # if Xp<=0:
    #     return np.inf
    mp = math.log(phip)+1-Np+muevp(phis,phip)+Np*math.log(Xp)
    return mp



### solving for phi_s using eq.(7)
### three versions: mudiff0 works better when the solution is close to 0 while
### mudiff1 works better when the solution is close to 1. Otherwise, use mudiff.
def mudiff(phis,phit):
    # print('phis=',phis)
    if phis<=0 or phis>=1 or np.isnan(phis):
        return np.inf
    phip = phit-phis*m*Np/Ns
    if phip<=0 or (phip+phis)>=1:
        return np.inf
    dif = mus(phis,phip)-m*mup(phis,phip)
    # print(dif)
    return dif

def mudiff0(lnb,phit):
    # print('x1=',x1)
    # print(lnb)
    # if lnb>=0:
    #     return np.inf
    phip = (1-math.exp(lnb))*phit
    if phip<=0:
        return np.inf
    phis = math.exp(lnb)*Ns/m*phit/Np
    # phip = phit-phis*4*Np/Ns
    if phis<=0 or (phip+phis)>=1:
        return np.inf
    dif = mus(phis,phip)-m*mup(phis,phip)
    # print(dif)
    return dif

def mudiff1(lnunb,phit):
    # print('x1=',x1)
    if lnunb>=0:
        return np.inf
    phip = math.exp(lnunb)*phit
    phis = (1-math.exp(lnunb))*Ns/m*phit/Np
    # phip = phit-phis*4*Np/Ns
    # print(phis,phip)
    if phis<=0 or (phip+phis)>=1:
        return np.inf
    dif = mus(phis,phip)-m*mup(phis,phip)
    # print(dif)
    return dif


### free energy density as a function of phi_s and phi_p, eq.(1) in the paper
def fphi(phis,phip):
    rhos = phis/Ns
    rhop = phip/Np
    Xs, Xp = solveX(phis,phip)
    f = rhos*math.log(phis)+rhop*math.log(phip)+(1-phis-phip)*math.log(1-phis-
        phip)+m*rhos*(math.log(Xs)-Xs/2+1/2)+Np*rhop*(math.log(Xp)-Xp/2+
        1/2)+rhos*us+(-(1-phis)*(1-1/Ns)+phip*(1-1/Np))*math.log(1-phis)
    return f

### free energy density as a function of phi_s and phi_t
### phi_t = rho*Np*v0, proportional to total concentration
def fphis(phis,phit):
    phip = phit-phis*m*Np/Ns
    if phis==0 or phip<=0 or phis+phip>=1:
        return 100
    else:
        return fphi(phis, phip)

### chemical equlibrium calculation
### at a fixed phi_t, free energy is a function of phi_s (fig.2a in the paper)
### finding all minima (at most 3) and their corresponding free eergy
def allsol(phit):
    tol = 10**(-10)
    phisl = np.ones(3)*10
    j = 0
    res = scipy.optimize.root(mudiff0, -15, args=phit, method='lm',
                                  options={'xtol': tol})
    phis = math.exp(res.x)*Ns/m*phit/Np
    if abs(res.fun) < 0.00001 and min((phisl-phis)**2) > 10**(-10):
        print(phis)
        print('left:',fphis(phis,phit))
        phisl[j] = phis
        j = j+1
    
    if Ns/m*phit/Np<1:
        res = scipy.optimize.root(mudiff1, -15, args=phit, method='lm',
                                  options={'xtol': tol})
        phis = (1-math.exp(res.x))*Ns/m*phit/Np
        if abs(res.fun) < 0.00001 and min((phisl-phis)**2) > 10**(-10):
            print(phis)
            print('right:',fphis(phis,phit))
            phisl[j] = phis
            j = j+1
    for item in bfw:
        if phit*Ns/m/Np*item<1:
            res = scipy.optimize.root(mudiff, phit*Ns/m/Np*item, args=phit, method='lm',
                                      options={'xtol': tol})
            if abs(res.fun) < 0.00001 and min((phisl-res.x)**2) > 10**(-10):
                print(res.x)
                print('middle:',fphis(res.x,phit))
                phisl[j] = res.x
                j = j+1            
    nsol = np.count_nonzero(phisl)
    # print('number of solutions:',nsol)
    phis = np.zeros(3)
    fgy = np.zeros(3)
    for i in range(3):
        if i<nsol:
            phis[i] = phisl[i]
            fgy[i] = fphis(phis[i],phit)
        else:
            fgy[i] = 100
    asdf = np.argsort(fgy)
    phis = phis[asdf]
    fgy = fgy[asdf]
    return phis[0],fgy[0]

### exact phase-coexistence calculation
### grand potential difference as a function of chemical potential
### the zero of the function is the chemical potential at phase coexistence
def gpdiff(mu,phi1t,phi2t):
    global phi1,phi2
    tol = 10**(-7)
    def Omega(phi):
        if phi<=0 or phi>=1:
            return np.inf
        phis,fgy = allsol(phi)
        omega = fgy-mu*phi/Np
        return omega
    res1 = scipy.optimize.minimize(Omega,
                                       phi1t, method='Nelder-Mead',
                                       options={'xatol': tol, 'fatol': tol})
    phi1 = res1.x
    print('rho1={}'.format(res1.x))
    omega1 = res1.fun
    res2 = scipy.optimize.minimize(Omega,
                                       phi2t, method='Nelder-Mead',
                                       options={'xatol': tol, 'fatol': tol})
    phi2 = res2.x
    print('rho2={}'.format(res2.x))
    omega2 = res2.fun
    print('diff={}'.format(omega2-omega1))
    return omega2-omega1



phimax = 0.99
nptt = 300

### lscape: free energy density as a function of phi_t
### 2D-array of shape (2,nptt)
### Try loading the landscape first. Calculate the landscape if it does not exist.
try:
    lscape = np.load('chem{}_{}_{}_{:.3f}.npy'.format(Np,Ns,dw,ep))
    phit = lscape[0]
    f = lscape[2]
except:
    
    phit = np.logspace(math.log(phimax/90000), math.log(phimax),base=math.e,num=nptt)

    phis = np.zeros([nptt])
    f = np.zeros([nptt])
    for i in range(nptt):
        phis[i],f[i] = allsol(phit[i])
        print('i=',i,flush=True)
    lscape = np.zeros([3,nptt])
    lscape[0] = phit
    lscape[1] = phis
    lscape[2] = f
    np.save('chem{}_{}_{}_{}_{:.3f}.npy'.format(m,Np,Ns,dw,ep),lscape)


### Convex-hull calculation
points = np.array([[phit[i],f[i]] for i in range(nptt)])
points = np.append(points,[[phimax*1.01,1000]],axis=0)
hull = scipy.spatial.ConvexHull(points)
bidx = []
phull = np.array(hull.vertices)
for item in hull.vertices:
    if ((item-1)%(nptt+1) not in phull) or ((item+1)%(nptt+1) not in phull):
        bidx.append(item)
print("number of points",len(bidx))
bidx.sort()


### if len(bidx)==0: no phase separation
### if len(bidx)==2: two-phase region
### if len(bidx)==4: two two-phase regions


### Start exact coexistence calculation
if bidx:
    resu = np.zeros([len(bidx)//2,5])
    for i in range(len(bidx)//2):
        idx1 = bidx[2*i]
        idx2 = bidx[2*i+1]


        phi1 = phit[idx1]
        phi2 = phit[idx2]
        muguess = (f[idx1]-f[idx2])/(phi1-phi2)*Np
        res = scipy.optimize.root(gpdiff, muguess, args=(phi1,phi2),method='hybr')
        
        phi1n = phi1
        phi2n = phi2
        
        phis1,f1 = allsol(phi1n)
        phis2,f2 = allsol(phi2n)
        resu[i,:] = np.array([phi1n,phi2n,m*phis1/Ns*Np/phi1n,m*phis2/Ns*Np/phi2n,(f2-f1)/(phi2n-phi1n)]).reshape(-1)
    
### final results: 2D-array
### number of rows are number of coexistence regions
### In each row, the five items are: phase1 concentration, phase2 concentration,
### phase1 S fraction, phase2 S fraction, coexistence chemical potential.

np.savetxt('exact{}_{}_{}_{}_{:.3f}.txt'.format(m,Np,Ns,dw,ep),resu,fmt='%.4e')

