import numpy as np
from scipy import constants as const
from numpy import linalg as LA
from numpy.testing import assert_allclose
from woodsaxon.woodsaxon import create_ham_proton

def test_woodsaxon():
    
# all necessary constants
    hbar = const.hbar
    hbarc=197.326968
    e = const.e
    pi = const.pi
    epsilon_0 = const.epsilon_0
    joul_to_eV = e

    m_proton = 938.27231

    Z=8
    N=8
    A=Z+N

    a=0.67
    r_0=1.27
    R=r_0*A**(1/3)

    V0proton=51+33*(N-Z)/A
    
    rmin = 0.01
    rmax=20
    Number=2000
    dr=(rmax-0)/Number
    g=1
    m=1
    alpha=1

    #r scale
    r = np.zeros(Number)

    for i in np.arange(Number):
        r[i] = rmin + i*dr

    hamiltonian_proton = create_ham_proton(1/2,0,r,R,a,V0proton,r_0,Z,m_proton)

    eigenvalues_proton, eigenvectors_proton = LA.eig(hamiltonian_proton)
    eigenvectors_proton= np.array([x for _, x in sorted(zip(eigenvalues_proton, eigenvectors_proton.T), key=lambda pair: pair[0])])
    eigenvalues_proton = np.sort(eigenvalues_proton)
    exp=-26.452093521424523
    assert_allclose(eigenvalues_proton[0], exp)
