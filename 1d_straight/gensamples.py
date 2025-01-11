'''Generate Samples'''
import numpy as np
import pandas as pd
import random

class StraightIsingSamplesGenerator():
    def __init__(self, num_nodes, num_edges):
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    # Method to solve the spin configuration
    def spin_config_solver(self, edge_weights): # input: the "edge_weights" list
        spins = [random.choice([-1, 1])] # choose the first spin randomly from -1 and 1
        for i in range(self.num_edges):
            if edge_weights[i] >= 0:
                spins.append(spins[i]) # if the i-th edge-weight >= 0, then the next spin is set equal to the current spin
            else:
                spins.append(-spins[i]) # if the i-th edge-weith is less than 0, then the next spin is set equal to the opposite of the current spin
        return spins
    
    # Method to calculate the Hamiltonian
    def hamiltonian_solver(self, edge_weights, spins): # input: "edge_weights" list, "spins" list
        hamiltonian = 0
        for i in range(self.num_edges):
            hamiltonian += -edge_weights[i] * spins[i] * spins[i + 1]
        return hamiltonian
    
    # Method to generate samples
    def sample_generator(self, num_samples):
        samples = []
        for _ in range(num_samples):
            # Randomly assign weights to the edges (between -1 and 1)
            edge_weights = np.random.uniform(-1, 1, self.num_edges)
            # Solve for spin configuration
            spins = self.spin_config_solver(edge_weights)
            # Calculate the Hamiltonian
            hamiltonian = self.hamiltonian_solver(edge_weights, spins)
            # Store the result
            samples.append({
                "edge_weights": edge_weights,
                "spins": spins,
                "hamiltonian": hamiltonian
            })
        return samples

if __name__ == "__main__":
    # create an object
    num_nodes = 10
    num_edges = num_nodes - 1
    num_samples = 100

    straight_ising_generator = StraightIsingSamplesGenerator(num_nodes, num_edges)

    # get samples
    samples = straight_ising_generator.sample_generator(num_samples)

    # Convert samples to a pandas DataFrame
    samples_df = pd.DataFrame([{**{
        f"edge_{i}-{i+1}": weights[i] for i in range(len(weights))
    }, **{
        f"spin_{i}": spins[i] for i in range(len(spins))
    }, "hamiltonian": hamiltonian} for sample in samples for weights, spins, hamiltonian in [(sample["edge_weights"], sample["spins"], sample["hamiltonian"])]])

    # Save the DataFrame to a CSV file
    samples_df.to_csv("./1d_straight/1d_straight_ising_model_samples.csv", index=False)
