
from PauliInteraction import PauliInteraction
from Ising import Ising

import Callbacks as cb
import qutip as qp
import numpy as np

N = 7
NUM_INTERACTIONS = int((N * (N - 1)) / 2)

T = 100
R = True           # Resonator Flag
TIME_STEPS = 100
QME_NUM_STEPS = 1e9
T_CB = {'T' : T}
QME_OPTIONS = qp.Options(nsteps=QME_NUM_STEPS)

J = [0.450874, -1.368073, 0.194027, 0.383486, 1.940685, 1.258323, -0.812024, -0.206118, 0.066754, \
    -0.878955, 1.156949, 1.032118, 0.201304, -0.770762, 0.116750, -0.851013, -0.974961, -0.392041, \
    1.772177, -0.227679, 1.223752]
h = [1.095761, -0.263176, -1.195265, -0.741288, -0.235239, -0.248207, -0.008947]

class MasterEquation:

    def __init__(self, start_Hamiltonian, stop_Hamiltonian, driver_Hamiltonian, start_state, evolution_time):

        self.start_H = start_Hamiltonian
        self.stop_H = stop_Hamiltonian
        self.driver_H = driver_Hamiltonian
        self.m_start_state = start_state
        self.m_evolved_state_result = None
        self.m_evolution_time = evolution_time
        self.m_states_t = np.ndarray(self.m_evolution_time, dtype = qp.Qobj)
        
        self.time_dependent_H = [
            [self.start_H, cb.start_H_time_coeff_cb],
            [self.driver_H, cb.driver_H_time_coeff_cb],
            [self.stop_H, cb.stop_H_time_coeff_cb]
        ]

        #self.current_eigenvalues = np.ndarray(self.annealing_time + 1, dtype = np.ndarray)
        #self.current_ground_state_eigenvalues = np.ndarray(self.annealing_time + 1, dtype = float)
        #self.current_eigenkets= np.ndarray(self.annealing_time + 1, dtype = qp.Qobj)

    def solve(self):
        
        t = np.linspace(0, self.m_evolution_time, 100)
        self.m_evolved_state = qp.mesolve(self.time_dependent_H, self.m_start_state, t, c_ops = None, e_ops = None, options = QME_OPTIONS, args = T_CB)
        self.m_states_t = self.m_evolved_state.states
       
        #overlap = (qp.fidelity(self.m_start_state, self.m_states_t[-1]) ** 2)
        #print("OVERLAP = %f" % overlap)

def main():
    
    basic_network = PauliInteraction(N)

    H_START = qp.Qobj()
    for i in range(N):
        H_START += basic_network.get_ztensor(i)

    eigenvalues_H_start, eigenkets_H_start = H_START.eigenstates()
    PSI_GND = eigenkets_H_start[0]

    H_DRIVE = qp.qzero(2)
    for i in range(N - 1):
        H_DRIVE = qp.tensor(H_DRIVE, qp.qzero(2))
    
    H_PROB = Ising(N, J, h, "X")

    qme_solver = MasterEquation(H_START, H_PROB.my_Hamiltonian, H_DRIVE, PSI_GND, T)
    qme_solver.solve()

if __name__ == "__main__":
    main()

