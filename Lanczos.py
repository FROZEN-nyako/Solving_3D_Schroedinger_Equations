import numpy as np
import scipy as sp
from mpi4py import MPI
import time


# INFO: Terminology

#   solve an eigenvalue problem using Lanczos Algorithm
#   inputs:
#      mat - matrix to solve
#      m - number of iterations
#      num - number of wanted states
#      interval - for demonstration of convergence
#   implementation employs MPI


def LanczosAlgorithm(mat, m, num, interval):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    lth = mat.shape[1]

    chunk_size = lth // size
    xtra = lth % size

    if rank < xtra:
        mycocnt = chunk_size + 1
        costart = rank * mycocnt
    else:
        mycocnt = chunk_size
        costart = rank * mycocnt + xtra

    colim = costart + mycocnt
    local_mat = mat[:, costart:colim]

    sendcounts = np.array(comm.gather(mycocnt, root=0))

    v1 = np.random.rand(mycocnt)
    v1 /= np.linalg.norm(v1) * np.sqrt(size)

    if rank == 0:
        cu = np.empty(lth, dtype=np.float64)
        tu = np.empty(lth, dtype=np.float64)
    else:
        cu = None
        tu = None
        u = None
        beta = None

    comm.Gatherv(sendbuf=v1, recvbuf=(cu, sendcounts), root=0)

    local_result = local_mat.dot(v1)
    comm.Reduce(local_result, tu, op=MPI.SUM, root=0)

    if rank == 0:
        alpha = np.einsum('i,i', cu, tu)
        u = tu - alpha * cu
        u -= np.einsum('i,i', cu, u) * cu
        beta = np.linalg.norm(u)
        u = u / beta
        ou = cu
        cu = u
        r = [alpha, beta]

    cv = np.empty((mycocnt))
    comm.Scatterv(sendbuf=(cu, sendcounts), recvbuf=cv, root=0)
    v2 = cv
    p = np.array([v1, v2])

    t_m = 0
    t_r = 0

    for k in range(2, m + 1):

        t0 = time.time()
        local_result = local_mat.dot(cv)
        comm.Reduce(local_result, tu, op=MPI.SUM, root=0)
        t1 = time.time()
        t_m += (t1 - t0)   # matrix product time
        if rank == 0:
            alpha = np.einsum('i,i', cu, tu)
            u = tu - r[-1] * ou - alpha * cu
        comm.Scatterv(sendbuf=(u, sendcounts), recvbuf=cv, root=0)

        t0 = time.time()
        local_overlap = np.einsum('...i,i', p, cv)
        overlap = comm.allreduce(local_overlap, op=MPI.SUM)
        cv -= np.einsum('i...,i', p, overlap)
        t1 = time.time()
        t_r += (t1 - t0)   # overlap removal time

        comm.Gatherv(sendbuf=cv, recvbuf=(u, sendcounts), root=0)

        if rank == 0:
            beta = np.linalg.norm(u)
            u /= beta
            ou = cu
            cu = u
            r.append(alpha)
            r.append(beta)

        beta = comm.bcast(beta, root=0)
        cv /= beta
        p = np.append(p, [cv], axis=0)

    vecs = p.T   # shape(vecs) = lth * (m + 1)

    if rank == 0:

        dia = [r[2 * i] for i in range(m)]   # diagonals
        off = [r[2 * i + 1] for i in range(m)]   # off-diagonals
        eigenvalues, eigenvectors = sp.linalg.eigh_tridiagonal(dia, off[:-1], select='i', select_range=(0, num - 1))

        # demonstration
        converge = []
        print('Spectrum:')
        for i in range(m // interval):
            spec, conv = convergence(dia, off, interval * (i + 1), num)
            converge.append(conv)
            print(f'm = {interval * (i + 1)}: {spec}')
        print('')
        print('Residue:')
        for i in range(m // interval):
            print(f'm = {interval * (i + 1)}: {converge[i]}')
        print('')

        wavefunctions = []
        spectrum = []
        residue = converge[-1]

    else:
        eigenvector = None
        approxstate = None

    for u in range(num):

        if rank == 0:

            spectrum.append(eigenvalues[u].real)

            eigenvector = eigenvectors[:, u]
            eigenvector = np.append(eigenvector, 0)   # length = m + 1

        eigenvector = comm.bcast(eigenvector, root=0)
        local_approxstate = np.einsum('...i,i', vecs, eigenvector)

        if rank == 0:
            approxstate = np.empty(lth, dtype=np.float64)

        comm.Gatherv(sendbuf=local_approxstate, recvbuf=(approxstate, sendcounts), root=0)

        if rank == 0:

            approxstate /= np.linalg.norm(approxstate)
            wavefunctions.append(approxstate)

    if rank == 0:
        return spectrum, wavefunctions, residue, t_m, t_r
    else:
        return None, None, None, None, None


def LanczosAlgorithmC(mat, m, num, interval):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    lth = mat.shape[1]

    chunk_size = lth // size
    xtra = lth % size

    if rank < xtra:
        mycocnt = chunk_size + 1
        costart = rank * mycocnt
    else:
        mycocnt = chunk_size
        costart = rank * mycocnt + xtra

    colim = costart + mycocnt
    local_mat = mat[:, costart:colim]

    sendcounts = np.array(comm.gather(mycocnt, root=0))

    v1 = np.random.rand(mycocnt) + 1j * np.random.rand(mycocnt)
    v1 /= np.linalg.norm(v1) * np.sqrt(size)

    if rank == 0:
        cu = np.empty(lth, dtype=np.complex128)
        tu = np.empty(lth, dtype=np.complex128)
    else:
        cu = None
        tu = None
        u = None
        beta = None

    comm.Gatherv(sendbuf=v1, recvbuf=(cu, sendcounts), root=0)

    local_result = local_mat.dot(v1)
    comm.Reduce(local_result, tu, op=MPI.SUM, root=0)

    if rank == 0:
        alpha = np.einsum('i,i', np.conj(cu), tu).real
        u = tu - alpha * cu
        u -= np.einsum('i,i', np.conj(cu), u) * cu
        beta = np.linalg.norm(u)
        u = u / beta
        ou = cu
        cu = u
        r = [alpha, beta]

    cv = np.empty(mycocnt, dtype=np.complex128)
    comm.Scatterv(sendbuf=(cu, sendcounts), recvbuf=cv, root=0)
    v2 = cv
    p = np.array([v1, v2])

    t_m = 0
    t_r = 0

    for k in range(2, m + 1):

        t0 = time.time()
        local_result = local_mat.dot(cv)
        comm.Reduce(local_result, tu, op=MPI.SUM, root=0)
        t1 = time.time()
        t_m += (t1 - t0)   # matrix product time
        if rank == 0:
            alpha = np.einsum('i,i', np.conj(cu), tu).real
            u = tu - r[-1] * ou - alpha * cu
        comm.Scatterv(sendbuf=(u, sendcounts), recvbuf=cv, root=0)

        t0 = time.time()
        local_overlap = np.einsum('...i,i', np.conj(p), cv)
        overlap = comm.allreduce(local_overlap, op=MPI.SUM)
        cv -= np.einsum('i...,i', p, overlap)
        t1 = time.time()
        t_r += (t1 - t0)   # overlap removal time

        comm.Gatherv(sendbuf=cv, recvbuf=(u, sendcounts), root=0)

        if rank == 0:
            beta = np.linalg.norm(u)
            u /= beta
            ou = cu
            cu = u
            r.append(alpha)
            r.append(beta)

        beta = comm.bcast(beta, root=0)
        cv /= beta
        p = np.append(p, [cv], axis=0)

    vecs = p.T   # shape(vecs) = lth * (m + 1)

    if rank == 0:

        dia = [r[2 * i] for i in range(m)]   # diagonals
        off = [r[2 * i + 1] for i in range(m)]   # off-diagonals
        eigenvalues, eigenvectors = sp.linalg.eigh_tridiagonal(dia, off[:-1], select='i', select_range=(0, num - 1))

        # demonstration
        converge = []
        print('Spectrum:')
        for i in range(m // interval):
            spec, conv = convergence(dia, off, interval * (i + 1), num)
            converge.append(conv)
            print(f'm = {interval * (i + 1)}: {spec}')
        print('')
        print('Residue:')
        for i in range(m // interval):
            print(f'm = {interval * (i + 1)}: {converge[i]}')
        print('')

        wavefunctions = []
        spectrum = []
        residue = converge[-1]

    else:
        eigenvector = None
        approxstate = None

    for u in range(num):

        if rank == 0:

            spectrum.append(eigenvalues[u].real)

            eigenvector = eigenvectors[:, u]
            eigenvector = np.append(eigenvector, 0)   # length = m + 1

        eigenvector = comm.bcast(eigenvector, root=0)
        local_approxstate = np.einsum('...i,i', vecs, eigenvector)

        if rank == 0:
            approxstate = np.empty(lth, dtype=np.complex128)

        comm.Gatherv(sendbuf=local_approxstate, recvbuf=(approxstate, sendcounts), root=0)

        if rank == 0:

            approxstate /= np.linalg.norm(approxstate)
            wavefunctions.append(approxstate)

    if rank == 0:
        return spectrum, wavefunctions, residue, t_m, t_r
    else:
        return None, None, None, None, None


def convergence(d, e, p, num):
    dpart = d[:p]
    epart = e[:p - 1]
    eigenvalues, eigenvectors = sp.linalg.eigh_tridiagonal(dpart, epart, select='i', select_range=(0, num - 1))
    beta = e[p - 1]
    res = np.abs(beta * eigenvectors[p - 1, :])
    return eigenvalues.tolist(), res.tolist()