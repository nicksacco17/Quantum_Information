
# Imports
import numpy as np
import qutip as qp
import matplotlib.pyplot as plt

# CUSTOM CLASSES
from PauliInteraction import PauliInteraction
from Ising import Ising

import Callbacks as cb

T = 100
R = True           # Resonator Flag

T_CB = {'T' : T, 'R' : R}

N = 7
NUM_INTERACTIONS = int((N * (N - 1)) / 2)
DEBUG_PLOT = True

w = 100

TIME_STEPS = 100
QME_NUM_STEPS = 1e9
QME_ARG_LIST = {'w': w}

QME_OPTIONS = qp.Options(nsteps = QME_NUM_STEPS)

class QuantumAnnealer:

    def __init__(self, start_Hamiltonian, stop_Hamiltonian, driver_Hamiltonian, annealing_t):

        self.start_H = start_Hamiltonian
        self.stop_H = stop_Hamiltonian
        self.driver_H = driver_Hamiltonian
        self.annealing_time = annealing_t
        self.current_H = self.start_H

        self.current_eigenvalues = np.ndarray(self.annealing_time + 1, dtype = np.ndarray)
        self.current_eigenkets = np.ndarray(self.annealing_time + 1, dtype = qp.Qobj)

        self.current_ground_state_eigenvalues = np.ndarray(self.annealing_time + 1, dtype = float)
        self.current_ground_state_eigenkets = np.ndarray(self.annealing_time + 1, dtype = qp.Qobj)
       

    def evolve(self):

        # Need to go [0, T] inclusive - continuous time evolution not discrete
        for t in range(self.annealing_time + 1):
          
            # Evolving correctly now
            self.current_H = (cb.start_H_time_coeff_cb(t, T_CB) * self.start_H) + \
                            (cb.driver_H_time_coeff_cb(t, T_CB) * self.driver_H) + \
                            (cb.stop_H_time_coeff_cb(t, T_CB) * self.stop_H)
            self.measure(t)
                       
    def measure(self, time_index):
        
        temp_eigval, temp_eigket = self.current_H.eigenstates()
        self.current_eigenvalues[time_index] = temp_eigval
        self.current_ground_state_eigenvalues[time_index] = temp_eigval[0]

        self.current_eigenkets[time_index] = temp_eigket
        self.current_ground_state_eigenkets[time_index] = temp_eigket[0]

def adiabatic_evolution_routine():
    
    H_START = qp.Qobj()
    H_DRIVE = qp.qzero(2)
    for i in range(6):
        H_DRIVE = qp.tensor(H_DRIVE, qp.qzero(2))

    H_PROB = Ising(N, J, h)

    basic_network = PauliInteraction(N)
    for i in range(N):
        H_START += basic_network.get_ztensor(i)

    #print(H_START.shape)
    #print(H_DRIVE.shape)
    #print(H_PROB.my_Hamiltonian.shape)


    # CREATE THE GROUND STATE TO EVOLVE
    eigenvalues_H_start, eigenkets_H_start = H_START.eigenstates()
    eigenvalues_H_prob, eigenkets_H_prob = H_PROB.my_Hamiltonian.eigenstates()

    #print(type(eigenvalues_H_prob[0]))
    #print(type(eigenkets_H_start[0]))

    print(len(eigenkets_H_start))

    PSI_GND = eigenkets_H_start[0]
    print("START: GROUND STATE ENERGY = %f" % eigenvalues_H_start[0])
    print("PROB: GROUND STATE ENERGY = %f" % eigenvalues_H_prob[0])

    overlap = (qp.fidelity(eigenkets_H_start[0], eigenkets_H_prob[0]) ** 2)
    print("OVERLAP = %f" % overlap)

    computer = QuantumAnnealer(H_START, H_PROB.my_Hamiltonian, H_DRIVE, T)
    computer.evolve()



if __name__ == "__main__":
    
    #master_equation_routine()
    adiabatic_evolution_routine()

