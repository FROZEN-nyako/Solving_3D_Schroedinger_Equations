import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from parity import P

# parameters

L = 0.1   # MeV ** (-1), lattice length
a = 0.001   # MeV ** (-1), lattice spacing
n = int(L / a)   # number of lattice per dimension
cen = (1 + n) / 2 - 1   # centre of the lattice
m = 100   # dimension of Heff
M = 938.92 / 2   # MeV, reduced mass of two-nucleon system

# kinetic matrix

diagonalsT = [3 / a ** 2 / M]
Ts = [-0.5 / a ** 2 / M]

elements = [diagonalsT,
            Ts, Ts, Ts, Ts,
            Ts, Ts, Ts, Ts,
            Ts, Ts, Ts, Ts]

positions = [0,
             -1, (n ** 3 - 1), 1, -(n ** 3 - 1),
             -n, (n ** 3 - n), n, -(n ** 3 - n),
             -n ** 2, (n ** 3 - n ** 2), n ** 2, -(n ** 3 - n ** 2)]

T = sp.sparse.diags(elements, positions, shape=(n ** 3, n ** 3), format='dia')


def r(i):
    return a * (i - cen)


# HO potential matrix

K = 100000   # MeV ** 3, spring constant

diagonalsV_HO = [0.5 * K * (r(i) ** 2 + r(j) ** 2 + r(k) ** 2)
                 for i in range(n) for j in range(n) for k in range(n)]

V_HO = sp.sparse.diags(diagonalsV_HO, 0, shape=(n ** 3, n ** 3), format='dia')

# Coulomb potential matrix

FSC = 1/137   # fine structure constant

diagonalsV_C = [-FSC / np.sqrt(r(i) ** 2 + r(j) ** 2 + r(k) ** 2)
                for i in range(n) for j in range(n) for k in range(n)]

V_C = sp.sparse.diags(diagonalsV_C, 0, shape=(n ** 3, n ** 3), format='dia')

# well potential matrix

R = 0.0106349   # MeV ** (-1) = 2.1 fm
V0 = 33.73416   # MeV

diagonalsV_w = []

for i in range(n):
    for j in range(n):
        for k in range(n):
            if r(i) ** 2 + r(j) ** 2 + r(k) ** 2 <= R ** 2:
                diagonalsV_w.append(-V0)
            else:
                diagonalsV_w.append(0)

V_w = sp.sparse.diags(diagonalsV_w, 0, shape=(n ** 3, n ** 3), format='dia')


# definition of Hamiltonian

H = T + V_w   # TODO: define Hamiltonian using V_HO, V_C or V_w


def OverlapRemoval(myv, ves):
    ovlp = np.einsum('...i,...i', ves, myv)
    red = np.einsum('i...,i', ves, ovlp)
    newv = myv - red
    return newv


def LanczosAlgorithm(mat, m):
    lth = mat.shape[0]
    v1 = np.random.rand(lth)
    v1 /= np.sqrt(np.sum(v1 * v1))
    tv = mat.dot(v1)
    alpha = np.einsum('i,i', v1, tv)
    v2 = tv - alpha * v1
    v2 = v2 - np.einsum('i,i', v1, v2) * v1
    beta = np.sqrt(np.sum(v2 * v2))
    v2 /= beta
    q = [v1, v2]
    r = [alpha, beta]

    for k in range(2, m + 1):
        cv = q[k - 1]
        tv = mat.dot(cv)
        alpha = np.einsum('i,i', cv, tv)
        v = tv - r[-1] * q[k - 2] - alpha * cv
        v = OverlapRemoval(v, q)
        beta = np.sqrt(np.sum(v * v))
        if beta == 0:
            break
        v /= beta
        q.append(v)
        r.append(alpha)
        r.append(beta)

    eigs = [r[i] for i in range(len(r) - 1)]
    vecs = np.transpose(q)
    diagonals = [eigs[2 * i - 2] for i in range(1, m + 1)]
    off_diagonals = [eigs[2 * i - 1] for i in range(1, m)]
    data = [diagonals, off_diagonals, off_diagonals]
    offsets = [0, -1, 1]
    Heff = sp.sparse.diags(data, offsets, shape=(m, m), format='dia')

    return Heff, vecs

Heff, vs = LanczosAlgorithm(H, m)   # shape(vs) = (n ** 3) * (m + 1)

num_eigenvalues = 1   # TODO: set other numbers to check excited states
eigenvalues, eigenvectors = sp.sparse.linalg.eigs(Heff, k=num_eigenvalues, which='SR')

min_eigenvalue = np.max(eigenvalues)
min_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
min_eigenvector = min_eigenvector.tolist()
min_eigenvector.append(0)   # length = m + 1

approxgroundstate = np.einsum('...i,i', vs, min_eigenvector)
approxgroundstate /= np.sqrt(np.sum(approxgroundstate * approxgroundstate))

# calculate the approximate groundstate energy

approxenergy = np.dot(approxgroundstate, H.dot(approxgroundstate)).real
print('The approximate groundstate energy is {:.4f} MeV.'.format(approxenergy))

# the ground state energy for the HO potential is expected to be 21.8923 MeV
# the ground state energy for the Coulomb potential is expected to be -0.0125 MeV (not so accurate here)
# the ground state energy for the well potential is expected to be -2.2245 MeV

# calculate the parity

vp = approxgroundstate - P(approxgroundstate, n)
vm = approxgroundstate + P(approxgroundstate, n)

if np.sum(vp * vp) < 10 ** -4:
    print('It has positive parity:', np.sum(vp * vp).real)
if np.sum(vm * vm) < 10 ** -4:
    print('It has negative parity:', np.sum(vm * vm).real)

# 2D slice plot

z0 = 50   # TODO: set position of the 2D slice to plot

x = np.arange(n)
y = np.arange(n)
x, y = np.meshgrid(x, y)

wavefunction = approxgroundstate.flatten()[n ** 2 * x + n * y + z0].real

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, wavefunction)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Wavefunction')

plt.show()

# 3D plot

x = np.arange(n)
y = np.arange(n)
z = np.arange(n)
x, y, z = np.meshgrid(x, y, z)

sizes = approxgroundstate.flatten()[n ** 2 * x + n * y + z].real

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=sizes, s=10, alpha=0.01)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.colorbar(sc, ax=ax, label='Function Value')

plt.show()