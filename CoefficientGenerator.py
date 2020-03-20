
# Imports
import numpy as np
import time as time

class CoefficientGenerator:

    def __init__(self, lower_bound, upper_bound, num_Jij, num_hi):

        self.m_lower_bound = lower_bound
        self.m_upper_bound = upper_bound
        self.m_num_Jij = num_Jij
        self.m_num_hi = num_hi

        self.m_coupling_coefs = np.ndarray(self.m_num_Jij, dtype = float)
        self.m_field_coefs = np.ndarray(self.m_num_hi, dtype = float)

    def generateCoefficients(self):

        self.m_coupling_coefs = np.random.uniform(low = self.m_lower_bound, high = self.m_upper_bound, size = self.m_num_Jij)
        self.m_fields_coefs = np.random.uniform(low = self.m_lower_bound, high = self.m_upper_bound, size = self.m_num_hi)

        #self.m_Jij_coefs = np.random.randint(0, high = 10, size = 5)
    def printCoefficients(self):
        print(self.m_coupling_coefs)
        print(self.m_field_coefs)

    def optimizeCoefficients(self):
        print("Beep boop doing some machine learning beep boop")

    def get_coupling_coefs(self):
        return self.m_coupling_coefs

    def get_field_coefs(self):
        return self.m_fields_coefs

def main():
    
    couplings = CoefficientGenerator(0.0, 1.0, 21, 7)
    couplings.generateCoefficients()
    couplings.printCoefficients()
    couplings.optimizeCoefficients()


if __name__ == "__main__":
    main()