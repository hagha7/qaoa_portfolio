import numpy as np
import networkx as nx
from quantum_solver import run_qaoa_with_pennylane
import matplotlib.pyplot as plt
from collections import Counter

def build_portfolio_qubo(mu, sigma, lam=0.5):
    """
    Convert portfolio optimization into a QUBO matrix.
    """
    n = len(mu)
    Q = np.zeros((n, n))

    for i in range(n):
        Q[i, i] = -mu[i] + lam * sigma[i][i]
        for j in range(i + 1, n):
            Q[i, j] = lam * sigma[i][j]
            Q[j, i] = Q[i, j]

    return Q

def format_samples(samples):
    return [''.join(str(bit) for bit in row) for row in samples]

def plot_histogram(samples):
    formatted = format_samples(samples)
    counts = Counter(formatted)
    top = counts.most_common(10)

    labels, values = zip(*top)
    plt.bar(labels, values)
    plt.xticks(rotation=45)
    plt.title("Top 10 Portfolio Selections (Bitstrings)")
    plt.xlabel("Portfolio (bitstring)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def main():
    # Dummy example data
    mu = np.array([0.12, 0.10, 0.07, 0.03])  # expected returns
    sigma = np.array([                     # covariance matrix
        [0.10, 0.02, 0.01, 0.0],
        [0.02, 0.08, 0.01, 0.0],
        [0.01, 0.01, 0.05, 0.0],
        [0.0,  0.0,  0.0,  0.02]
    ])

    Q = build_portfolio_qubo(mu, sigma, lam=0.5)
    num_qubits = len(mu)

    print("⚛️ Running QAOA with PennyLane...")
    samples = run_qaoa_with_pennylane(Q, num_qubits)
    print("🧪 Sampled result:", samples)

    # After sampling
    plot_histogram(samples)

if __name__ == '__main__':
    main()
