
from PauliInteraction import PauliInteraction
from Ising import Ising
from CoefficientGenerator import CoefficientGenerator
from Evaluator import Evaluator

#from DataVisualizer import DataVisualizer
#from DataLogger import DataLogger

import Driver as Driver_H

from MasterEquation import MasterEquation
from QuantumAnnealer import QuantumAnnealer
from Test import test

import qutip as qp
import numpy as np
import time as time

X_HAT = "X"
Y_HAT = "Y"
Z_HAT = "Z"

N = 7
T = 100
NUM_INTERACTIONS = int((N * (N - 1)) / 2)
RAND_COEF = True

# MAIN TEST CASES FROM SAMPLE CODE

def main():

    for i in range(10):
        test(RAND_COEF, i)


if __name__ == "__main__":
    main()


