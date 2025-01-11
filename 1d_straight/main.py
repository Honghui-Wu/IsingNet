import pandas as pd


from gensamples import StraightIsingSamplesGenerator

# create an object
num_nodes = 2
num_edges = num_nodes - 1
num_samples = 10

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