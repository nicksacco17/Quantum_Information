
from PauliInteraction import PauliInteraction
from Ising import Ising


import qutip as qp
import numpy as np
import random as rand

X_HAT = "X"
Y_HAT = "Y"
Z_HAT = "Z"

J = [0.450874, -1.368073, 0.194027, 0.383486, 1.940685, 1.258323,
            -0.812024, -0.206118, 0.066754, -0.878955, 1.156949, 1.032118, 
            0.201304, -0.770762, 0.116750, -0.851013, -0.974961, -0.392041,
            1.772177, -0.227679, 1.223752]
h = [1.095761, -0.263176, -1.195265, -0.741288, -0.235239, -0.248207, -0.008947]

def create_ferromagnetic_driver(num_qubits):

    num_interactions = int((num_qubits * (num_qubits - 1)) / 2)
    J = -1 * np.ones(num_interactions, dtype = float)
    h = np.zeros(num_qubits, dtype = float)

    ferromagnetic_H = Ising(num_qubits, J, h, X_HAT)

    return ferromagnetic_H

def create_antiferromagnetic_driver(num_qubits):

    num_interactions = int((num_qubits * (num_qubits - 1)) / 2)
    J = np.ones(num_interactions, dtype = float)
    h = np.zeros(num_qubits, dtype = float)

    anti_ferromagnetic_H = Ising(num_qubits, J, h, X_HAT)
    
    return anti_ferromagnetic_H

def create_mixed_magnetic_driver(num_qubits):
    
    num_interactions = int((num_qubits * (num_qubits - 1)) / 2)
    rand.seed(1)
    J = np.ones(num_interactions, dtype = float)
    h = np.zeros(num_qubits, dtype = float)

    for i in range(len(J)):
        rij = rand.randint(0, 1)
        if rij == 0:
            J[i] = -1

    mixed_magnetic_H = Ising(num_qubits, J, h, X_HAT)

    return mixed_magnetic_H
        
def create_resonator_driver(num_qubits, Jij, hi):

    resonator_H = Ising(num_qubits, Jij, hi, Y_HAT)
    return resonator_H
    

'''def main():
    
    H_F = create_ferromagnetic_driver(N)
    H_A = create_antiferromagnetic_driver(N)
    H_M = create_mixed_magnetic_driver(N)

    H_F.print_couplings()

    H_A.print_couplings()

    H_M.print_couplings()
    '''

if __name__ == "__main__":
    #main()
    print("MAIN")
