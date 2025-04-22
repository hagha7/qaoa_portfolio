import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

def convert_qubo_to_hamiltonian(Q):
    """Convert QUBO matrix to a PennyLane Hamiltonian (Z, ZZ terms)."""
    num_qubits = Q.shape[0]
    coeffs = []
    ops = []

    for i in range(num_qubits):
        coeffs.append(Q[i, i])
        ops.append(qml.PauliZ(i))
        for j in range(i + 1, num_qubits):
            if Q[i, j] != 0:
                coeffs.append(Q[i, j])
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    return qml.Hamiltonian(coeffs, ops)

def qaoa_circuit(params, cost_h, mixer_h, p):
    """Define the full QAOA circuit with p layers."""
    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    for layer in range(p):
        qml.expval.Hamiltonian(cost_h, wires=range(num_qubits)).queue()
        qml.Hamiltonian(cost_h.coeffs, cost_h.ops)(params[layer][0])
        qml.Hamiltonian(mixer_h.coeffs, mixer_h.ops)(params[layer][1])

def run_qaoa_with_pennylane(Q, num_qubits, steps=1, shots=1000):
    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
    cost_h = convert_qubo_to_hamiltonian(Q)

    # Mixer: sum of X over all qubits
    mixer_h = qml.Hamiltonian(
        [-1.0] * num_qubits,
        [qml.PauliX(i) for i in range(num_qubits)]
    )

    def qaoa_layer(params):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        for l in range(steps):
            # Cost unitary
            qml.templates.ApproxTimeEvolution(cost_h, params[l, 0], 1)
            # Mixer unitary
            qml.templates.ApproxTimeEvolution(mixer_h, params[l, 1], 1)

    @qml.qnode(dev)
    def circuit(params):
        qaoa_layer(params)
        return qml.expval(cost_h)

    # Training
    np.random.seed(42)
    params = 0.01 * np.random.rand(steps, 2, requires_grad=True)
    optimizer = NesterovMomentumOptimizer(0.5)

    steps_opt = 50
    for i in range(steps_opt):
        params, loss = optimizer.step_and_cost(circuit, params)
        if i % 10 == 0:
            print(f"Step {i}: Cost = {loss:.6f}")

    # Sampling
    @qml.qnode(dev)
    def sampler(params):
        qaoa_layer(params)
        return qml.sample()

    samples = sampler(params)
    return samples
