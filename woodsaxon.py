from traceback import print_tb
import numpy as np
from scipy import constants as const
from scipy import sparse as sparse
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt

from numpy import linalg as LA

hbar = const.hbar
hbarc = 197.3269680000
e = const.e
pi = const.pi
epsilon_0 = const.epsilon_0
joul_to_eV = e

m_neutron = 939.565630000000
m_proton = 938.272310000000

Z = 8
N = 8
A = Z + N

a = 0.67
r_0 = 1.27
R = r_0 * A ** (1 / 3)

V0proton = 51 + 33 * (N - Z) / A
V0neutron = 51 - 33 * (N - Z) / A

rmin = 0.01
rmax = 20
Number = 2000
dr = (rmax - 0) / Number
g = 1
m = 1
alpha = 1


r = np.zeros(Number)

for i in np.arange(Number):
    r[i] = rmin + i * dr  # fm


def yukawa(r, g, alpha, m):
    yukawa = np.zeros(Number)
    for i in np.arange(len(r)):
        yukawa[i] = (-1 * g * g * np.exp(-1 * alpha * m * r[i])) / (r[i] ** 2)
    return yukawa


plt.plot(r, yukawa(r, g, alpha, m))
plt.show()


def woodsaxon(r, R, a, V0):
    ws = np.zeros(Number)
    for i in np.arange(len(r)):

        ws[i] = -V0 / (1 + np.exp((r[i] - R) / a))
    return ws


def columb(r, Z, R):
    col = np.zeros(Number)
    for i in np.arange(len(r)):
        if r[i] <= R:
            col[i] = (Z * 197 / (137)) * (3 - (r[i] / R) ** 2) / (2 * R)
        if r[i] > R:
            col[i] = (Z * 197 / (137)) * (1 / r[i])
    return col


def angular(r, l):
    angular = np.zeros(Number)
    for i in np.arange(len(r)):
        angular[i] = (l * (l + 1)) / (r[i] ** 2)
    return angular


def laplace(r):
    h = r[1] - r[0]
    main_diag = -2.0 / h**2 * np.ones(Number)
    off_diag = 1.0 / h**2 * np.ones(Number - 1)
    laplace_matice = (
        np.diag(off_diag, k=1) + np.diag(off_diag, k=-1) + np.diag(main_diag)
    )
    return laplace_matice


def spinorbit(r, R, a, V0, r_0):
    so = np.zeros(Number)
    for i in range(len(r)):
        so[i] = (
            -0.44
            * V0
            * (r_0 / hbar)
            * (r_0 / hbar)
            * (1 / r[i])
            * np.exp((r[i] - R) / a)
            / (a * (np.exp((r[i] - R) / a) + 1) ** 2)
        )
    return so


def scalarLS(j, l, r, R, a, V0, r_0):
    return (
        0.5
        * (j * (j + 1) - l * (l + 1) - 3 / 4)
        * hbar
        * hbar
        * spinorbit(r, R, a, V0, r_0)
    )


def create_ham_proton(j, l, r, R, a, V0proton, r_0, Z, m_proton):
    columbvmev_matice = np.diag(columb(r, Z, R))
    woodsaxon_term_proton = np.diag((woodsaxon(r, R, a, V0proton)))
    scalarLS_term_proton = np.diag(scalarLS(j, l, r, R, a, V0proton, r_0))
    angular_matice = np.diag(angular(r, l))
    laplace_term = laplace(r)
    hamiltonian_proton = (-(hbarc**2) / (2.0 * m_proton)) * (
        laplace_term - angular_matice
    ) + (woodsaxon_term_proton + columbvmev_matice + scalarLS_term_proton)
    return hamiltonian_proton


def create_ham_neutron(j, l, r, R, a, V0neutron, r_0, m_neutron):
    woodsaxon_term_neutron = np.diag((woodsaxon(r, R, a, V0neutron)))
    scalarLS_term_neutron = np.diag(scalarLS(j, l, r, R, a, V0neutron, r_0))
    angular_matice = np.diag(angular(r, l))
    laplace_term = laplace(r)
    hamiltonian_neutron = (-(hbarc**2) / (2.0 * m_neutron)) * (
        laplace_term - angular_matice
    ) + (woodsaxon_term_neutron + scalarLS_term_neutron)
    return hamiltonian_neutron


def plot_potencial(j, l, r, R, a, V0neutron, r_0, V0proton):
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))

    ax[0].plot(r, woodsaxon(r, R, a, V0proton), color="y", label="Wood-Saxon potenciál")
    ax[0].plot(r, scalarLS(j, l, r, R, a, V0proton, r_0), color="r", label="Spin-Orbit")
    ax[0].plot(r, columb(r, Z, R), color="b", label="Coulomb potenciál")
    ax[0].plot(
        r,
        columb(r, Z, R)
        + woodsaxon(r, R, a, V0proton)
        + scalarLS(j, l, r, R, a, V0proton, r_0),
        color="g",
        label="Wood-Saxon + Coulomb potenciál + Spin-Orbit",
    )
    ax[0].set_xlabel("r[fm]")
    ax[0].set_ylabel("V[MeV]")
    ax[0].legend(loc=0)
    ax[0].set_title("Proton potenciál V(r), l=%s" % l + " j=%s" % j)

    ax[1].plot(
        r, woodsaxon(r, R, a, V0neutron), color="y", label="Wood-Saxon potenciál"
    )
    ax[1].plot(
        r, scalarLS(j, l, r, R, a, V0neutron, r_0), color="r", label="Spin-orbit"
    )
    ax[1].plot(
        r,
        woodsaxon(r, R, a, V0neutron) + scalarLS(j, l, r, R, a, V0neutron, r_0),
        color="g",
        label="Wood-Saxon + Spin-Orbit",
    )
    ax[1].set_ylim([-60, 50])
    ax[1].set_xlabel("r[fm]")
    ax[1].set_ylabel("V[MeV]")
    ax[1].set_title("Neutron potenciál V(r),l=%s" % l + " j=%s" % j)
    ax[1].legend(loc=0)

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )
    return 0


def plot_densities(
    r, densities_proton, densities_neutron, energies_proton, energies_neutron, l, j
):
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))

    for i in range(4):
        ax[0].plot(r, densities_proton[i], label="%s" % energies_proton[i])
        ax[1].plot(r, densities_neutron[i], label="%s" % energies_neutron[i])

    for i in range(2):
        ax[i].set_xlabel("r[fm]")
        ax[i].set_ylabel("probability density")
        ax[i].legend(loc=0)

    ax[0].set_title("Proton l=%s" % l + " j=%s" % j)
    ax[1].set_title("Neutron l=%s" % l + " j=%s" % j)

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )
    plt.show()
    return 0


l1 = [0, 1, 1, 2, 2]
j1 = [1 / 2, 3 / 2, 1 / 2, 5 / 2, 3 / 2]

for i in range(len(l1)):

    print(
        "----------------------------------------------------------------------------------------"
    )
    print(
        "Orbital angular momenta l: %s" % l1[i]
        + "\nTotal angular momentum j: %s \n" % j1[i]
    )

    hamiltonian_proton = create_ham_proton(
        j1[i], l1[i], r, R, a, V0proton, r_0, Z, m_proton
    )
    hamiltonian_neutron = create_ham_neutron(
        j1[i], l1[i], r, R, a, V0neutron, r_0, m_neutron
    )

    eigenvalues_proton, eigenvectors_proton = LA.eig(hamiltonian_proton)
    eigenvalues_neutron, eigenvectors_neutron = LA.eig(hamiltonian_neutron)
    eigenvectors_proton = np.array(
        [
            x
            for _, x in sorted(
                zip(eigenvalues_proton, eigenvectors_proton.T), key=lambda pair: pair[0]
            )
        ]
    )
    eigenvalues_proton = np.sort(eigenvalues_proton)

    eigenvectors_neutron = np.array(
        [
            x
            for _, x in sorted(
                zip(eigenvalues_neutron, eigenvectors_neutron.T),
                key=lambda pair: pair[0],
            )
        ]
    )
    eigenvalues_neutron = np.sort(eigenvalues_neutron)

    print(
        "E_0proton = %s " % eigenvalues_proton[0]
        + " E_0neutron = %s \n" % eigenvalues_neutron[0]
    )

    densities = [
        np.absolute(eigenvectors_proton[i, :]) ** 2
        for i in range(len(eigenvalues_proton))
    ]

    if j1[i] == 1 / 2 and l1[i] == 1:
        densities_proton = [
            np.absolute(eigenvectors_proton[i, :]) ** 2
            for i in range(len(eigenvalues_proton))
        ]
        densities_neutron = [
            np.absolute(eigenvectors_neutron[i, :]) ** 2
            for i in range(len(eigenvalues_neutron))
        ]
        energies_proton = [
            "E_%s" % i + " = {: >5.3f} MeV".format(eigenvalues_proton[i].real)
            for i in range(30)
        ]
        energies_neutron = [
            "E_%s =" % i + " {: >5.3f} MeV".format(eigenvalues_neutron[i].real)
            for i in range(30)
        ]
        j = j1[i]
        l = l1[i]

    hamiltonian_proton = 0
    hamiltonian_neutrom = 0

    print(
        "----------------------------------------------------------------------------------------"
    )

plot_potencial(j, l, r, R, a, V0neutron, r_0, V0proton)
plot_densities(
    r, densities_proton, densities_neutron, energies_proton, energies_neutron, l, j
)

plt.show()
