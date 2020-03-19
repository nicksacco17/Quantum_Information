
from PauliInteraction import PauliInteraction
from Ising import Ising
from CoefficientGenerator import CoefficientGenerator
import Driver as Driver_H

from MasterEquation import MasterEquation
from QuantumAnnealer import QuantumAnnealer

import qutip as qp
import numpy as np
import matplotlib.pyplot as plt

X_HAT = "X"
Y_HAT = "Y"
Z_HAT = "Z"

N = 7
T = 100
NUM_INTERACTIONS = int((N * (N - 1)) / 2)
RAND_COEF = False
DRIVER_FLAG = True

# MAIN TEST CASES FROM SAMPLE CODE

def main():

#---------------------------- STEP 1: COEFFICIENTS -----------------------------

    if RAND_COEF:

        ising_coefficients = CoefficientGenerator(0.0, 1.0, NUM_INTERACTIONS, N)
        J = ising_coefficients.get_coupling_coefs()
        h = ising_coefficients.get_field_coefs()
    
    else:
        J = [0.450874, -1.368073, 0.194027, 0.383486, 1.940685, 1.258323,
            -0.812024, -0.206118, 0.066754, -0.878955, 1.156949, 1.032118, 
            0.201304, -0.770762, 0.116750, -0.851013, -0.974961, -0.392041,
            1.772177, -0.227679, 1.223752]
        h = [1.095761, -0.263176, -1.195265, -0.741288, -0.235239, -0.248207, -0.008947]

#-------------------------------------------------------------------------------

#---------------------------- STEP 2: HAMILTONIANS -----------------------------
    
    basic_network = PauliInteraction(N)

    H_START = qp.Qobj()
    for i in range(N):
        H_START += basic_network.get_ztensor(i)

    if DRIVER_FLAG:
        H_DRIVE = Driver_H.create_resonator_driver(N, J, h)
    elif not DRIVER_FLAG:
        H_DRIVE = qp.qzero(2)
        for i in range(N - 1):
            H_DRIVE = qp.tensor(H_DRIVE, qp.qzero(2))
    
    H_PROB = Ising(N, J, h, X_HAT)

#-------------------------------------------------------------------------------

#---------------------------- STEP 3: BASE METRICS -----------------------------

    # CREATE THE GROUND STATE TO EVOLVE
    eigenvalues_H_start, eigenkets_H_start = H_START.eigenstates()
    eigenvalues_H_prob, eigenkets_H_prob = H_PROB.my_Hamiltonian.eigenstates()

    PSI_GND_i = eigenkets_H_start[0]
    E_GND_i = eigenvalues_H_start[0]

    PSI_GND_f = eigenkets_H_prob[0]
    E_GND_f = eigenvalues_H_prob[0]

    print("START: GROUND STATE ENERGY = %f" % E_GND_i)
    print("STOP: GROUND STATE ENERGY = %f" % E_GND_f)

    overlap = (qp.fidelity(PSI_GND_i, PSI_GND_f) ** 2)
    print("OVERLAP = %f" % overlap)

#-------------------------------------------------------------------------------

#----------------------- STEP 4: MASTER EQUATION ROUTINE -----------------------
    qme_solver = MasterEquation(H_START, H_PROB.my_Hamiltonian, H_DRIVE.my_Hamiltonian, PSI_GND_i, T)
    qme_solver.solve()
#-------------------------------------------------------------------------------

#--------------------- STEP 5: ADIABATIC EVOLUTION ROUTINE ---------------------
    computer = QuantumAnnealer(H_START, H_PROB.my_Hamiltonian, H_DRIVE.my_Hamiltonian, T)
    computer.evolve()
#-------------------------------------------------------------------------------

#----------------------------- STEP 6: COMPARISON ------------------------------

    print(len(computer.current_ground_state_eigenkets))
    print(len(qme_solver.m_states_t))
    overlap = np.ndarray(100, dtype = float)
    t = np.linspace(0, 100, 100)
    for i in range(len(qme_solver.m_states_t)):
        
        overlap[i] = (qp.fidelity(computer.current_ground_state_eigenkets[i], qme_solver.m_states_t[i]) ** 2)
        
        print("INDEX %d, OVERLAP = %f" % (i, overlap[i]))
        
    fig = plt.figure()
    plt.axis([0, 100, 0, 1.0])
    plt.title('Overlap between QME and AQC')
    plt.xlabel('Index (t)')
    plt.ylabel('Overlap')
        # Need to go [0, T] inclusive - continuous time evolution not discrete

       
    plt.plot(t, overlap, 'r-', markersize = 1.0, label = 'OVERLAP')
    plt.show()



    #for i in range(len(computer.current_ground_state_eigenkets)):
        #print(i)


#-------------------------------------------------------------------------------

#---------------------------- STEP 7: VISUALIZATION ----------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()


