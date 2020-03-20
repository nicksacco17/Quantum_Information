
from Ising import Ising
from QuantumAnnealer import QuantumAnnealer
from MasterEquation import MasterEquation

import qutip as qp
import numpy as np
import matplotlib.pyplot as plt

class Evaluator:

    def __init__(self, quantum_annealer_gnd_states, master_equation_gnd_states, routine_time, expected_state):

        self.m_quantum_annealer_gnd_states = quantum_annealer_gnd_states
        self.m_master_equation_gnd_states = master_equation_gnd_states
        self.m_routine_time = routine_time
        self.m_expected_state = expected_state

        self.overlap_aqc_qme = np.ndarray(self.m_routine_time, dtype = float)
        self.overlap_aqc_psi = np.ndarray(self.m_routine_time, dtype = float)
        self.overlap_qme_psi = np.ndarray(self.m_routine_time, dtype = float)

    def evaluate(self):
        for i in range(self.m_routine_time):
            self.overlap_aqc_qme[i] = (qp.fidelity(self.m_quantum_annealer_gnd_states[i], self.m_master_equation_gnd_states[i]) ** 2)
            self.overlap_aqc_psi[i] = (qp.fidelity(self.m_quantum_annealer_gnd_states[i], self.m_expected_state) ** 2)
            self.overlap_qme_psi[i] = (qp.fidelity(self.m_master_equation_gnd_states[i], self.m_expected_state) ** 2)
        
        print("OVERLAP: %lf" % self.overlap_aqc_qme[self.m_routine_time - 1])

    def display(self):
        t = np.linspace(0, self.m_routine_time, self.m_routine_time)
        t_norm = t / self.m_routine_time

        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.axis([0, 1.0, 0, 1.0])
        plt.title('Overlap between QME and AQC')
        plt.xlabel('Index (t)')
        plt.ylabel('Overlap')
        plt.plot(t_norm, self.overlap_aqc_qme, 'r-', markersize = 1.0, label = 'OVERLAP_AQC_QME')

        plt.subplot(1, 3, 2)
        plt.axis([0, 1.0, 0, 1.0])
        plt.title('Overlap between AQC and PSI')
        plt.xlabel('Index (t)')
        plt.ylabel('Overlap')
        plt.plot(t_norm, self.overlap_aqc_psi, 'g-', markersize = 1.0, label = 'OVERLAP_AQC_PSI')

        plt.subplot(1, 3, 3)
        plt.axis([0, 1.0, 0, 1.0])
        plt.title('Overlap between QME and PSI')
        plt.xlabel('Index (t)')
        plt.ylabel('Overlap')
        plt.plot(t_norm, self.overlap_qme_psi, 'b-', markersize = 1.0, label = 'OVERLAP_QME_PSI')

        plt.show()

    
        