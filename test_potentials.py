def V_HO(r):
    """
    the ground state energy for the HO potential is expected to be 21.8923 MeV
    """
    k = 100000   # MeV ** 3, spring constant
    v = 1 / 2 * k * r ** 2
    return v


def V_well(r):
    """
    the ground state energy for the well potential is expected to be -2.2245 MeV
    """
    R = 0.0106422   # MeV ** (-1) = 2.1 fm
    v0 = 33.73416   # MeV
    if r < R:
        v = -v0
    else:
        v = 0
    return v