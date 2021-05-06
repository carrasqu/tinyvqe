import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer


# heisenberg Hamiltonian in 1d with periodic boundary conditions
def Ham(Jz,Jp,qubits):

    coeffs = [] 
    obs = []
    for i in range(qubits-1):
        coeffs.append(Jz)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(i+1))      
        coeffs.append(Jp)
        obs.append(qml.PauliX(i) @ qml.PauliX(i+1)) 
        coeffs.append(Jp)  
        obs.append(qml.PauliY(i) @ qml.PauliY(i+1)) 

    # periodic boundary conditions
    coeffs.append(Jz)
    obs.append(qml.PauliZ(qubits-1) @ qml.PauliZ(0))
    coeffs.append(Jp)
    obs.append(qml.PauliX(qubits-1) @ qml.PauliX(0)) 
    coeffs.append(Jp)
    obs.append(qml.PauliY(qubits-1) @ qml.PauliY(0))   
 
    return  qml.Hamiltonian(coeffs, obs)



qubits = 4 # number of spins

nlayers = 4 # number of layers in the quantum circuits ansatz 
np.random.seed(0)
params = np.random.normal(0, np.pi, (qubits*nlayers, 3)) # parameter initialization

# This is the choice of device. Right now it is a simulator (default.qubit), but it can be a real device
dev = qml.device('default.qubit', wires=qubits)


# this is the quantum circuit, ie the ansatz
#@qml.qnode(dev)
def circuit(params,wires):
     # initial state 
     #qml.BasisState(np.array(qubits*[0], requires_grad=False), wires=wires)
     for j in range(nlayers):
         for i in range(qubits):
             qml.Rot(*params[qubits*j+i], wires=[i])
         for i in range(qubits-1):
             qml.CNOT(wires=[i, i+1])
         qml.CNOT(wires=[qubits-1, 0])
 
#     return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# Parameters in the Hamiltonian
Jz = 1.0
Jp = 1.0 

h = Ham(Jz,Jp,qubits)

# Evaluates the expectation value h over the circuit ansatz on the device dev
cost_fn = qml.ExpvalCost(circuit, h, dev)

#opt = qml.GradientDescentOptimizer(stepsize=0.05)
opt = qml.AdamOptimizer(0.1) # we use gradient descent or a variant of it to optimize the cost function iteratively


# number of iteration
max_iterations = 150


for n in range(max_iterations):
    params, prev_energy = opt.step_and_cost(cost_fn, params) # does one step of gradient descent optimization updating parameters
    energy = cost_fn(params) # evaluate the new energy
    conv = np.abs(energy - prev_energy)
    
    print(n, energy)
