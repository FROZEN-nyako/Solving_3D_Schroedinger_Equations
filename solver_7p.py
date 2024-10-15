import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
from test_potentials import V_HO, V_well
from Lanczos import LanczosAlgorithm
from parity import P

np.set_printoptions(precision=5, suppress=True)


class Parameters:
    def __init__(self, L, n, M, P, num, m, interval):
        self.L = L   # MeV ** (-1), lattice length
        self.n = n   # number of points per dimension
        self.a = L / n   # MeV ** (-1), lattice spacing
        self.cen = int(n / 2)   # centre of the lattice
        self.M = M   # MeV, reduced mass of two-nucleon system
        self.P = P   # potential
        self.num = num   # number of wanted states
        self.m = m   # number of iterations
        self.interval = interval   # for demonstration of convergence


def V_DIY(r):
    """
    customize your potential
    """
    v = 1 / r   # for example
    return v


deuteron = Parameters(0.177371, 100, 938.92 / 2, V_well, 1, 100, 10)
DIY = Parameters(0.0724685, 100, 938.92 / 2, V_DIY, 2, 100, 10)

input = deuteron


def initialize(params):

    n = params.n
    a = params.a
    cen = params.cen
    M = params.M
    P = params.P

    # kinetic matrix

    T_diagonals = np.array([6 / a ** 2 / (2 * M)])
    T_off_diagonals = np.repeat(-1 / a ** 2 / (2 * M), 12)

    elements = np.append(T_diagonals, T_off_diagonals)

    positions = [0,
                 -n ** 2, (n ** 3 - n ** 2), n ** 2, -(n ** 3 - n ** 2),
                 -n, (n ** 3 - n), n, -(n ** 3 - n),
                 -1, (n ** 3 - 1), 1, -(n ** 3 - 1)]

    T = sp.sparse.diags(elements, positions, shape=(n ** 3, n ** 3), format='csr')

    # potential matrix

    def r(i):
        return a * (i + 1 / 2 - cen)

    V_diagonals = []

    for i in range(n):
        for j in range(n):
            for k in range(n):
                rr = np.sqrt(r(i) ** 2 + r(j) ** 2 + r(k) ** 2)
                V_diagonals.append(P(rr))

    V = sp.sparse.diags(V_diagonals, 0, shape=(n ** 3, n ** 3), format='csr')

    # definition of Hamiltonian
    H = T + V

    return H


def output(params, H, spectrum, wavefunctions, residue):

    n = params.n
    num = params.num

    for u in range(num):

        print('State number:', u)

        approxenergy = spectrum[u]
        approxstate = wavefunctions[u]

        print(f'The approximate energy is {approxenergy} MeV')

        total_error = np.linalg.norm(H.dot(approxstate) - approxenergy * approxstate)
        print(f'The total error is {total_error}')

        print(f'The theoretical total error is {residue[u]}')

        local_error = np.abs((H.dot(approxstate) / approxenergy - approxstate) / approxstate)
        max_index, max_value = max(enumerate(local_error), key=lambda x: x[1])
        i0 = max_index // (n ** 2)
        j0 = max_index % (n ** 2) // n
        k0 = max_index % n
        print(f'The biggest local relative error {max_value} appears at ({i0}, {j0}, {k0})')

        # calculate the parity

        vp = np.linalg.norm(approxstate - P(approxstate, n))
        vn = np.linalg.norm(approxstate + P(approxstate, n))

        if vp < 0.01:
            print('It has positive parity (+):', vp)
        elif vn < 0.01:
            print('It has negative parity (-):', vn)

        print('')

        # 1D slice plot
        """
        set position of the 1D slice to plot
        """
        y0, z0 = 50, 50

        x = np.arange(n)
        wavefunction = approxstate[n ** 2 * x + n * y0 + z0].real

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(x, wavefunction)

        ax.set_xlabel('x')
        ax.set_ylabel('Wavefunction')

        plt.show()

        # 2D slice plot
        """
        set position of the 2D slice to plot
        """
        z0 = 50

        x = np.arange(n)
        y = np.arange(n)
        x, y = np.meshgrid(x, y)

        wavefunction = approxstate[n ** 2 * x + n * y + z0].real

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

        sizes = approxstate[n ** 2 * x + n * y + z].real

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=sizes, s=10, alpha=0.01)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.colorbar(sc, ax=ax, label='Function Value')

        plt.show()


start_time = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

H = initialize(input)

if rank == 0:
    building_time = time.time()

spectrum, wavefunctions, residue, tm, tr = LanczosAlgorithm(H, input.m, input.num, input.interval)

if rank == 0:
    Lanczos_time = time.time()
    print("Building time:", building_time - start_time)
    print("Lanczos time:", Lanczos_time - building_time)
    print("Matrix product time:", tm)
    print("Overlap removal time:", tr)
    print('')
    output(input, H, spectrum, wavefunctions, residue)
    end_time = time.time()
    print("Total time:", end_time - start_time)