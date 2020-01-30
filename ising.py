#!/usr/bin/env python

from qutip import *
import numpy as np

# Number of qubits
N = 7

class IsingHamiltonian():

    def __init__(self):
        self.num_qubits = N
        self.pairwise_couplings = []
        self.individual_fields = []

    def populatePairwiseCouplings(self, J_ij):
        self.pairwise_couplings = J_ij.copy()

    def populateIndividualFields(self, h_i):
        self.individual_fields = h_i.copy()

    def printPairwiseCouplings(self):
        print("********** PAIRWISE COUPLINGS **********")
        for i in range(len(self.pairwise_couplings)):
            print("J[%d] = %f" % (i, self.pairwise_couplings[i]))

    def printIndividualFields(self):
        print("********** INDIVIDUAL FIELDS **********")
        for i in range(len(self.individual_fields)):
            print("h[%d] = %f" % (i, self.individual_fields[i]))

if __name__ == '__main__':

    hard7 = IsingHamiltonian()
    hard7.num_qubits = N

    J=[0.315595,0.969839,-0.650295,-0.058989,-0.380561,-0.237262,1.546382,0.494935,-1.073902,\
    0.261010,2.938737,1.314844,-0.610274,-0.616883,2.923590,0.964480,-0.428509,1.169902,\
    -0.285724,2.460704,-0.783270]
    h=[0.717564,0.894829,0.623232,0.079529,1.635336,-0.061965,0.752149]

    hard7.populatePairwiseCouplings(J)
    hard7.populateIndividualFields(h)
    hard7.printPairwiseCouplings()
    hard7.printIndividualFields()




