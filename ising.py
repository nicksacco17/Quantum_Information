#!/usr/bin/env python
# Optimization of Quantum Annealing Routines using Non-Stoquastic Drivers
# Nicholas Sacco, UMass Lowell, '19, '10
# Advisors: Dr. Archana Kamal (Physics), Dr. Seung Woo Son (EECE)
# Version: 0.5

# Imports
import qutip as qp
import numpy as np
import random as rand
import matplotlib.pyplot as plt

#---------------------------------- CONSTANTS ----------------------------------

N = 7                               # Number of qubits
T = 100                             # Total annealing time
DEBUG = 0                           # Debugging flag

# SIGMA-Z TENSORS
sz1 = qp.tensor(qp.sigmaz(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))
sz2 = qp.tensor(qp.qeye(2), qp.sigmaz(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))
sz3 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.sigmaz(), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2))
sz4 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmaz(), qp.qeye(2), qp.qeye(2), qp.qeye(2))
sz5 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmaz(), qp.qeye(2), qp.qeye(2))
sz6 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmaz(), qp.qeye(2))
sz7 = qp.tensor(qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.qeye(2), qp.sigmaz())

#----------------------------------- CLASSES -----------------------------------

# Tensor interactions describing an all-to-all connected graph
class Interaction:

    def __init__(self, num_q):
        self.num_qubits = num_q
        self.x_tensor = []
        self.y_tensor = []
        self.z_tensor = []

    def getXTensor(self, index):
        return self.x_tensor[index]
    
    def getYTensor(self, index):
        return self.y_tensor[index]

    def getZTensor(self, index):
        return self.z_tensor[index]

    def setXTensor(self):
        # Begin tensor product construction - N total tensor products

        # For each tensor product
        for i in range(N):
            # Build the tensor product 
            for j in range(N):
                # Initialize the first element in the tensor product
                if j == 0:
                    # If first tensor product, initialize element with sigma
                    if i == 0:
                        sx = qp.sigmax()
                    # Else initialize element with identity
                    else:
                        sx = qp.qeye(2)
                # Populate the rest of the elements - tensor product is 'square'
                # The nth element of nth tensor product should be a Pauli matrix
                else:
                    # If index of element matches index of tensor product
                    if j == i:
                        # Add sigma to the tensor product
                        sx = qp.tensor(sx, qp.sigmax())
                    else:
                        # Else add identity to the product
                        sx = qp.tensor(sx, qp.qeye(2))
            # Push the tensor product to the list
            self.x_tensor.insert(i, sx)
        print("X TENSORS INITIALIZED, SIZE = %d" % (len(self.x_tensor)))

    def setYTensor(self):

        # Begin tensor product construction - N total tensor products

        # For each tensor product
        for i in range(N):
            # Build the tensor product 
            for j in range(N):
                # Initialize the first element in the tensor product
                if j == 0:
                    # If first tensor product, initialize element with sigma
                    if i == 0:
                        sy = qp.sigmay()
                    # Else initialize element with identity
                    else:
                        sy = qp.qeye(2)
                # Populate the rest of the elements - tensor product is 'square'
                # The nth element of nth tensor product should be a Pauli matrix
                else:
                    # If index of element matches index of tensor product
                    if j == i:
                        # Add sigma to the tensor product
                        sy = qp.tensor(sy, qp.sigmay())
                    else:
                        # Else add identity to the product
                        sy = qp.tensor(sy, qp.qeye(2))
            # Push the tensor product to the list
            self.y_tensor.insert(i, sy)
        print("Y TENSORS INITIALIZED, SIZE = %d" % (len(self.y_tensor)))

    def setZTensor(self):
        # Begin tensor product construction - N total tensor products

        # For each tensor product
        for i in range(N):
            # Build the tensor product 
            for j in range(N):
                # Initialize the first element in the tensor product
                if j == 0:
                    # If first tensor product, initialize element with sigma
                    if i == 0:
                        sz = qp.sigmaz()
                    # Else initialize element with identity
                    else:
                        sz = qp.qeye(2)
                # Populate the rest of the elements - tensor product is 'square'
                # The nth element of nth tensor product should be a Pauli matrix
                else:
                    # If index of element matches index of tensor product
                    if j == i:
                        # Add sigma to the tensor product
                        sz = qp.tensor(sz, qp.sigmaz())
                    else:
                        # Else add identity to the product
                        sz = qp.tensor(sz, qp.qeye(2))
            # Push the tensor product to the list
            self.z_tensor.insert(i, sz)
        print("Z TENSORS INITIALIZED, SIZE = %d" % (len(self.z_tensor)))

# Ising Hamiltonian encoded on all-to-all connected graph
class IsingHamiltonian():

    def __init__(self, num_q, annealing_t):
        # System parameters
        self.num_qubits = num_q
        self.num_interactions = int(((num_q - 1) * (num_q)) / 2)
        self.annealing_parameter = (1.0 / annealing_t)
        self.annealing_time = annealing_t

        # Couplings and coefficients
        self.pairwise_couplings = []
        self.individual_fields = []

        # Basic tensor network
        self.network = Interaction(num_q)

        # Hamiltonians
        self.H_problem = qp.Qobj()              # HP - DONE
        self.H_start = qp.Qobj()                # HB - DONE
        self.H_free = qp.Qobj()                 # H0 - DONE
                                                # --> EQN[H0 = HB + HP]
        self.H_intermediate = qp.Qobj()         # HI - DONE
        self.H_magnetic = qp                    # HM - NOT DONE
                                                # --> EQN[HM = HB + HI + HP]
        self.H_cat = qp.Qobj()                  # HC - NOT DONE
        self.H_driven = qp.Qobj()               # HD - NOT DONE
                                                # --> EQN[HD = HB + HI + HC + HP]
        # Eigenstates
        # TODO(Evaluate which of these eigenstates are actually necessary)
        self.eigval_problem = []                # HP
        self.eigket_problem = []                # HP

        self.eigval_start = []                  # HB
        self.eigket_start = []                  # HB

        self.eigval_free = []                   # H0
        self.eigket_free = []                   # H0

        self.eigval_intermediate = qp.Qobj()    # HI
        self.eigket_intermediate = qp.Qobj()    # HI

        self.eigval_magnetic = qp.Qobj()        # HM
        self.eigket_magnetic = qp.Qobj()        # HM

        self.eigval_cat = qp.Qobj()             # HC
        self.eigket_cat = qp.Qobj()             # HC

        self.eigval_driven = qp.Qobj()          # HD
        self.eigket_driven = qp.Qobj()          # HD

        self.me_eval0 = 0.0
        self.me_fidelity = 0.0

        # Populate default tensors
        self.network.setXTensor()
        self.network.setYTensor()
        self.network.setZTensor()
     
    def populatePairwiseCouplings(self, J_ij):
        self.pairwise_couplings = J_ij.copy()

    def populateIndividualFields(self, h_i):
        self.individual_fields = h_i.copy()

    # Encode the coefficients of the problem to solve in the Ising Hamiltonian
    # TODO(Determine the problem to solve)
    # TODO(Determine the initial state - should it be Z tensors?)
    def createProblemHamiltonian(self):
        
        Jij = 0
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                #print('i = %d, j = %d' % (i, j))
                self.H_problem += (self.pairwise_couplings[Jij] \
                    * self.network.getZTensor(i) * self.network.getZTensor(j))
                Jij += 1

        for i in range(self.num_qubits):
            self.H_problem += self.individual_fields[i] * self.network.getZTensor(i)  

        assert Jij == self.num_interactions, "[ERROR] Num connections don't agree"
        print('--> HP INITIALIZED')

    # Initialize the starting Hamiltonian
    # TODO(Determine initial state for Hamiltonian - should it be X Tensors?)
    def createStartHamiltonian(self):
        for i in range(self.num_qubits):
            self.H_start += self.network.getXTensor(i)
        print('--> HB INITIALIZED')

    # Create the Driver Hamiltonian - Non-Stoquastic Hamiltonian
    # Three different forms - Ferromagnetic form, Antiferromagnetic form, Mixed form
    # TODO(Determine form for these Hamiltonians)
    # TODO(Rewrite...)
    def createIntermediateHamiltonian(self, interaction_type):

        coef = []
        for i in range(self.num_interactions):
            if interaction_type == "MIXED":
                sign = rand.randint(0, 1)
                if sign == 0:
                    sign = -1
            else:
                sign = 1
            coef.insert(i, sign)

        rij = 0
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                self.driver_Hamiltonian += (coef[rij] \
                    * self.network.getXTensor(i) * self.network.getXTensor(j))
                rij += 1
        
        if interaction_type == "FERROMAGNETIC":
            self.intermediate_Hamiltonian *= -1

        print('--> HI INITIALIZED')

    # Create the Free Adiabatic Hamiltonian - No driver term!
    # TODO(VALIDATION)
    # TODO(Verify annealing parameters make sense- ensure they function as booleans)
    def initializeFreeAdiabaticHamiltonian(self):
        # System starts in the [START] state
        self.H_free = self.H_start + (0.0 * self.H_problem) 
        print('--> H0 INITIALIZED')

    def evolveFreeAdiabaticHamiltonian(self, t):
        self.H_free = (1.0 - (t / self.annealing_time)) * self.H_start + \
            (t / self.annealing_time) * self.H_problem

    # Create the Driven Adiabatic Hamiltonian - Driver term!
    # TODO(VALIDATION)
    # TODO(Verify annealing parametesr make sense - ensure they function as booleans)
    # TODO(Add non-stoqustic term from cat state - should be sigmay terms)
    def createDrivenAdiabaticHamiltonian(self):
        self.driver_Hamiltonian = self.free_Hamiltonian + self.annealing_parameter * \
            (1 - self.annealing_parameter) * self.intermediate_Hamiltonian
        print('--> HD INITIALIZED')

    # DONE
    def printProblemHamiltonian(self):
    
        nrows = self.problem_Hamiltonian.shape[0]
        ncols = self.problem_Hamiltonian.shape[1]
        print('<---------- PROBLEM HAMILTONIAN ---------->')
        print('Overall matrix shape - (%d %d)' % (nrows, ncols))

        # <----- LIST OF LISTS ----->
        for i in range(nrows):
            for j in range(ncols):
                if self.problem_Hamiltonian[i][0][j].real != 0 \
                    or self.problem_Hamiltonian[i][0][j].imag != 0:
                    print('ROW %d, COL %d: %f + %fj' % \
                        (i, j, self.problem_Hamiltonian[i][0][j].real, self.problem_Hamiltonian[i][0][j].imag))
        #print(self.problem_Hamiltonian[0][0][0])
        #print(self.problem_Hamiltonian.data[0][0][0])
        print('<---------- PROBLEM HAMILTONIAN ---------->')

    # DONE
    # TODO(Implement optional data file log)
    # TODO(Verify all required elements are printed - do we just print sparse?)
    def printStartHamiltonian(self):
        
        nrows = self.start_Hamiltonian.shape[0]
        ncols = self.start_Hamiltonian.shape[1]

        print('<---------- START HAMILTONIAN ---------->')
        print('Overall matrix shape - (%d %d)' % (nrows, ncols))

        for i in range(nrows):
            for j in range(ncols):
                print('ROW %d, COL %d: %f + %fj' % \
                    (i, j, self.start_Hamiltonian[i][0][j].real, self.start_Hamiltonian[i][0][j].imag))
        print('<---------- START HAMILTONIAN ---------->')

    # DONE
    # TODO(Implement optional data file log)
    # TODO(Verify all required elements are printed - do we just print sparse?)
    def printFreeAdiabaticHamiltonian(self):
        
        nrows = self.start_Hamiltonian.shape[0]
        ncols = self.start_Hamiltonian.shape[1]

        print('<---------- FREE ADIABATIC HAMILTONIAN ---------->')
        print('Overall matrix shape - (%d %d)' % (nrows, ncols))

        for i in range(nrows):
            for j in range(ncols):
                print('ROW %d, COL %d: %f + %fj' % \
                    (i, j, self.start_Hamiltonian[i][0][j].real, self.start_Hamiltonian[i][0][j].imag))
        print('<---------- FREE ADIABATIC HAMILTONIAN ---------->')

    # DONE
    # TODO(Implement optional data file log)
    # TODO(Verify all required elements are printed - do we just print sparse?)
    def printIntermediateHamiltonian(self):

        nrows = self.start_Hamiltonian.shape[0]
        ncols = self.start_Hamiltonian.shape[1]

        print('<---------- INTERMEDIATE HAMILTONIAN ---------->')
        print('Overall matrix shape - (%d %d)' % (nrows, ncols))

        for i in range(nrows):
            for j in range(ncols):
                print('ROW %d, COL %d: %f + %fj' % \
                    (i, j, self.start_Hamiltonian[i][0][j].real, self.start_Hamiltonian[i][0][j].imag))
        print('<---------- INTERMEDIATE HAMILTONIAN ---------->')

    # DONE
    # TODO(Implement optional data file log)
    # TODO(Verify all required elements are printed - do we just print sparse?)
    def printDrivenAdiabaticHamiltonian(self):

        nrows = self.start_Hamiltonian.shape[0]
        ncols = self.start_Hamiltonian.shape[1]

        print('<---------- DRIVEN ADIABATIC HAMILTONIAN ---------->')
        print('Overall matrix shape - (%d %d)' % (nrows, ncols))

        for i in range(nrows):
            for j in range(ncols):
                print('ROW %d, COL %d: %f + %fj' % \
                    (i, j, self.start_Hamiltonian[i][0][j].real, self.start_Hamiltonian[i][0][j].imag))
        print('<---------- DRIVEN ADIABATIC HAMILTONIAN ---------->')

    # DONE
    # TODO(Implement optional data file log)
    # TODO(Verify all required elements are printed - do we just print sparse?)
    def printPairwiseCouplings(self):
        print("********** PAIRWISE COUPLINGS **********")
        for i in range(len(self.pairwise_couplings)):
            print("J[%d] = %f" % (i, self.pairwise_couplings[i]))

    def solveStartHamiltonian(self):
        print('SOLVING START HAMILTONIAN...')

    def solveMasterEquation(self):

        w=100
        args={'w':w}
        tlist = np.linspace(0,w,100)

        #H_me = sz1+sz2+sz3+sz4+sz5+sz6+sz7
        H_me = self.H_start
        eval0, eket0 = H_me.eigenstates()
        gnd0 = eket0[0]
        
        opts = qp.Options(nsteps=1e9)#,atol=1e-14,rtol=1e-14)
        medata = qp.mesolve(self.H_free, gnd0, tlist, [], [], options = opts, args = args)

        # Success probability
        #print(qp.fidelity(medata.states[-1],eket0[0])**2)


    # ?????
    def solveFreeHamiltonian(self):

        A = []
        B = []
    
        self.initializeFreeAdiabaticHamiltonian()
        
        for t in range(self.annealing_time):
            # Evolve the free Hamiltonian
            self.evolveFreeAdiabaticHamiltonian(t)
            # Calculate eigenstates
            eigval_free, eigket_free = self.H_free.eigenstates()

            print(eigval_free[0])

            

        # Find the eigenvalues and eigenkets for the free Hamiltonian
        #eig_val0,eig_ket0 = self.free_Hamiltonian.eigenstates()
        #self.eigval0, self.eigket0 = self.free_Hamiltonian.eigenstates()
        # And assert that Schrodinger's Equation Holds - H0|e> = e|e>
        #for i in range(len(self.eigval0)):
            #assert self.free_Hamiltonian * self.eigket0[i] == self.eigval0[i] * self.eigket0[i]
        print('H0 SOLVED')
        


        #for i in range(len(eig_val0)):
            #print('EIG VAL %i = %f' % (i, eig_val0[i]))
        #print(H0 * eket0[0])

        #for i in range(len(eket0)):
            #for j in range(eket0[i].shape[0]):
                #if eket0[i][j] != 0:
                    #print('i = %d, j = %d, val = %f + %fj' % (i, j, eket0[i][j].real, eket0[i][j].imag))

        #print(eket0[127])

    # DONE
    # TODO(Implement optional data file log)
    # TODO(Verify all required elements are printed - do we just print sparse?)
    def printIndividualFields(self):
        print("********** INDIVIDUAL FIELDS **********")
        for i in range(len(self.individual_fields)):
            print("h[%d] = %f" % (i, self.individual_fields[i]))

# Main Method
if __name__ == '__main__':

    hard7 = IsingHamiltonian(N, T)

    J=[0.315595,0.969839,-0.650295,-0.058989,-0.380561,-0.237262,1.546382,0.494935,-1.073902,\
    0.261010,2.938737,1.314844,-0.610274,-0.616883,2.923590,0.964480,-0.428509,1.169902,\
    -0.285724,2.460704,-0.783270]
    h=[0.717564,0.894829,0.623232,0.079529,1.635336,-0.061965,0.752149]

    hard7.populatePairwiseCouplings(J)
    hard7.populateIndividualFields(h)


    hard7.createProblemHamiltonian()
    hard7.createStartHamiltonian()
    #hard7.createIntermediateHamiltonian("ANTIFERROMAGNETIC")
    #hard7.createFreeAdiabaticHamiltonian()
    #hard7.createDrivenAdiabaticHamiltonian()

    if DEBUG:
        hard7.printProblemHamiltonian()
        hard7.printStartHamiltonian()
        hard7.printIntermediateHamiltonian()
        hard7.printFreeAdiabaticHamiltonian()
        hard7.printDrivenAdiabaticHamiltonian()

    w=100
    hard7.solveFreeHamiltonian()
    #hard7.solveMasterEquation()
  
    #tlist = np.linspace(1,w,100)
    #print(len(tlist))
    #for i in range(len(tlist)):
        #print(tlist[i])




