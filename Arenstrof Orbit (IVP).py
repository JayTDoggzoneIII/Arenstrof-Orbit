"""
x'' = x + 2y' - v * (x + u)/D1 - u * (x - v)/D2

y'' = y - 2x' - v * y/D1 - u * y/D2

D1 = ((x + u)**2 + y**2)**(3/2)
D2 = ((x - v)**2 + y**2)**(3/2)

v = 1 - u
u = 0.012277471

x(0) = (0.994, 0)
x'(0) = (0, -2.00158510637908252240537862224)
T = 17.0652165601579625588917206249

"""


'''
z_1 = y
z_2 = x'
z_3 = x
z_4 = y'


z_1' = z_4
z_2' = z_3 + 2*z_4 - v * (z_3 + u)/D_1 - u * (z_3 - v)/D_2
z_3' = z_2
z_4' = z_1 - 2*z_2 - v * z_1/D_1 - u * z_1/D_2

D_1 = ((z_3 + u)**2 + z_1**2)**(3/2)
D_2 = ((z_3 - v)**2 + z_1**2)**(3/2)

v = 1 - u
u = 0.012277471


z_3(0) = 0.994
z_1(0) = 0
z_2(0) = 0
z_4(0) = -2.00158510637908252240537862224

T = 17.0652165601579625588917206249
'''

"""
y[n+1] = y[n] + h * sum(k[i] * b[i] for i in range(n))

k[i] = f(t[n] + c[i]*h, y[n] + h*(a[i,1]*k[1] + a[i,2]*k[2] + ... + a[i,i-1]*k[i-1]))

"""

from mpmath import mp, mpf
from decimal import Decimal as D, getcontext
from numpy import array, log
from numpy.linalg import norm
from time import perf_counter

TIME = [perf_counter()]

Digits = 20
mp.dps = Digits
getcontext().prec = Digits

def D_1(z1,z3):
    return ((z3 + u)**2 + z1**2)**one_plus_half

def D_2(z1,z3):
    return ((z3 - v)**2 + z1**2)**one_plus_half

'''
z[0] = y
z[1] = x'
z[2] = x
z[3] = y'
'''


def f2(z1, z3, z4):
    return z3 + 2*z4 - v * (z3 + u)/D_1(z1, z3) - u * (z3 - v)/D_2(z1, z3)

def f4(z1, z2, z3):
    return z1 - 2*z2 - v * z1/D_1(z1, z3) - u * z1/D_2(z1, z3)

def RK3_step(z, h):
    y1,y2,y3,y4 = z
    k_11 = h * y4
    k_21 = h * f2(y1,y3,y4)
    k_31 = h * (y2 + k_21/3)
    k_41 = h * f4(y1 + k_11/3, y2 + k_21/3, y3 + k_31/3)
    
    k_12 = h * (y4 + 2*k_41/3)
    k_22 = h * f2(y1 + k_11/3 + k_12/3, y3 + 2*k_31/3, y4 + 2*k_41/3)
    k_32 = h * (y2 + k_22)
    k_42 = h * f4(y1 + k_12, y2 + k_22, y3 + k_31)
    
    y1_h = y1 + (k_11 + 3*k_12)/4
    y2_h = y2 + (k_21 + 3*k_22)/4
    y3_h = y3 + (3*k_31 + k_32)/4
    y4_h = y4 + (3*k_41 + k_42)/4
    
    return array([y1_h, y2_h, y3_h, y4_h])

def RK3_auto(z0, h = 1e-6,  tol = 1e-3, s = 3, cycles = 1, verbose = False):
    hs = [float(h)]
    h_next = h
    ans = [z0[:]]
    t = 0
    ts = [0]
    N = 0
    while (t < cycles*T):
        z = RK3_step(ans[-1], h)
        z1 = RK3_step(ans[-1], h/2)
        z2 = RK3_step(z1, h/2)
        N += 12
        #rho = norm((z2 - z)/((1 << s) - 1))
        rho = sum((z2[i] - z[i])**2 for i in range(4))**half / ((1 << s) - 1)
        if (rho > (1 << s)*tol):
            k = h
            while (rho > (1 << s)*tol):
                k /= 2
                z = RK3_step(ans[-1], k)
                z1 = RK3_step(ans[-1], k/2)
                z2 = RK3_step(z1, k/2)
                N += 12
                #rho = norm((z2 - z)/((1 << s) - 1))
                rho = sum((z2[i] - z[i])**2 for i in range(4))**half / ((1 << s) - 1)
            #print(k, flush=True)
            t += k
            ans.append(z)
            #hs.append(k)
            #ts.append(t)
            continue
        
        elif (tol < rho < (1 << s)*tol):
            h_next = h/2
        
        elif (tol < (1 << s)*rho < (1 << s)*tol):
            h_next = h
        
        #elif ((1 << s)*rho < tol):
        else:
            h_next = 2*h
        
        hs.append(float(h))
        ans.append(z)
        t += h
        ts.append(float(t))
        h = h_next
        if (verbose and len(ans) % 10_000 == 0): print(t,h, flush = True)
    return [[float(z1), float(z2), float(z3), float(z4)] for z1,z2,z3,z4 in ans], hs, ts, N

def f(y):
 
    z1, z2, f1, f2 = y
    D1 = ((z1 + u)**2 + z2**2)**one_plus_half
    D2 = ((z1 - v)**2 + z2**2)**one_plus_half
 
    f3 = z1 + 2*f2 - v * (z1 + u) / D1 - u * (z1 - v) / D2
    f4 = z2 - 2*f1 - v * z2 / D1 - u * z2 / D2
 
    return array([f1, f2, f3, f4])
 

def RK4_step(z0, h):
 
    k1 = f(z0)
    k2 = f(z0 + k1 * h/2)
    k3 = f(z0 + k2 * h/2)
    k4 = f(z0 + k3 * h)
 
    return z0 + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
 
# Error estimation and step size adjustment
def RK4_auto(z0, h = 1e-6, tol = 1e-3, s = 4, cycles = 1, verbose = False):
    hs = [float(h)]
    h_next = h
    ans = [z0[:]]
    t = 0
    ts = [0]
    N = 0
    while (t < cycles*T):
        z = RK4_step(ans[-1], h)
        z1 = RK4_step(ans[-1], h/2)
        z2 = RK4_step(z1, h/2)
        N += 24
        #rho = norm((z2 - z)/((1 << s) - 1))
        rho = sum((z2[i] - z[i])**2 for i in range(4))**half / ((1 << s) - 1)
        if (rho > (1 << s)*tol):
            k = h
            while (rho > (1 << s)*tol):
                k /= 2
                z = RK4_step(ans[-1], k)
                z1 = RK4_step(ans[-1], k/2)
                z2 = RK4_step(z1, k/2)
                N += 24
                #rho = norm((z2 - z)/((1 << s) - 1))
                rho = sum((z2[i] - z[i])**2 for i in range(4))**half / ((1 << s) - 1)
            #print(k, flush=True)
            t += k
            ans.append(z)
            #hs.append(k)
            #ts.append(t)
            continue
        
        elif (tol <= rho <= (1 << s)*tol):
            h_next = h/2
        
        elif (tol <= (1 << s)*rho <= (1 << s)*tol):
            h_next = h
        
        #elif (16*rho < tol):
        else:
            h_next = 2*h
        
        hs.append(float(h))
        ans.append(z)
        t += h
        ts.append(float(t))
        h = h_next
        
        if (verbose and len(ans) % 10_000 == 0): print(t,h, flush = True)
    return [[float(z1), float(z2), float(z3), float(z4)] for z1,z2,z3,z4 in ans], hs, ts, N

'''
one_plus_half = mpf('1.5')
half = mpf('0.5')

u = mpf('0.012277471')
v = 1 - u

T = mpf('17.0652165601579625588917206249')
z0_SRK3 = [mpf('0'), mpf('0'), mpf('0.994'), mpf('-2.00158510637908252240537862224')]
z0_RK4  = [mpf('0.994'), mpf('0'), mpf('0'), mpf('-2.00158510637908252240537862224')]

h = mpf('1e-6')
tol = mpf('1e-16')
'''

'''
one_plus_half = D('1.5')
half = D('0.5')

u = D('0.012277471')
v = 1 - u

T = D('17.0652165601579625588917206249')
z0_SRK3 = [D('0'), D('0'), D('0.994'), D('-2.00158510637908252240537862224')]
z0_RK4  = [D('0.994'), D('0'), D('0'), D('-2.00158510637908252240537862224')]

h = D('1e-6')
tol = D('1e-16')
'''


one_plus_half = 1.5
half = 0.5

u = 0.012277471
v = 1 - u

T = 17.0652165601579625588917206249
z0_SRK3 = [0, 0, 0.994, -2.00158510637908252240537862224]
z0_RK4  = [0.994, 0, 0, -2.00158510637908252240537862224]

tol = 1e-9

delta_SRK3 = (1/norm(z0_SRK3))**4 + norm(f(z0_RK4))**4
h_SRK3 = (tol / delta_SRK3)**(1/4)


delta_RK4 = (1/norm(z0_SRK3))**5 + norm(f(z0_RK4))**5
h_RK4 = (tol / delta_RK4)**(1/5)


print("eps = %e"%tol, flush=True)
print("SRK3 h0 = %e"%h_SRK3, flush=True)
print("RK4  h0 = %e"%h_RK4, flush=True)
print()

cycles = 1
X3, Y3 = [], []
X4, Y4 = [], []

solve1 = RK3_auto(z0_SRK3, h = h_SRK3, tol = tol, cycles = cycles)
TIME.append(perf_counter())
print("SRK3 done in %.6f seconds"%(TIME[-1] - TIME[-2]), flush = True)
print("ln(Nf) = %.3f"%log(solve1[3]), flush = True)

print()

solve2 = RK4_auto(z0_RK4 , h = h_RK4, tol = tol, cycles = cycles)
TIME.append(perf_counter())
print("RK4 done in %.6f seconds"%(TIME[-1] - TIME[-2]), flush = True)
print("ln(Nf) = %.3f"%log(solve2[3]), flush = True)

for y,dx,x,dy in solve1[0]:
    X3.append(x)
    Y3.append(y)

for x,y,dx,dy in solve2[0]:
    X4.append(x)
    Y4.append(y)

import matplotlib.pyplot as plt
fig1, (ax1, ax2) = plt.subplots(1,2)

ax1.set_title("Arenstrof Orbit")
ax1.plot(X3, Y3, c = 'g', label = "SRK3")
ax1.plot(X4, Y4, c = 'r', label = "RK4")
ax1.legend()
ax1.grid()

ax2.set_title("-ln(h)")
ax2.plot(solve1[2], -log(solve1[1]), c = 'g', label = "SRK3")
ax2.plot(solve2[2], -log(solve2[1]), c = 'r', label = "RK4")
ax2.legend()
ax2.grid()
plt.show()
