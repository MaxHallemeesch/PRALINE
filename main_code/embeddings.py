import argparse
import numpy as np
from sklearn.decomposition import NMF
import json

def rank_calculation(adj_matrix, emb_size):
    """
    Calculate the optimal rank of the specified dataframe.
    """
    model = NMF(n_components=emb_size, init='random', random_state=0, max_iter=500, verbose=1)
    W = model.fit_transform(adj_matrix)
    H = model.components_
    V = W @ H
    return W

def load_gene_adjacency_matrix(file):
    with open(file, 'rb') as f:
        matrix = np.load(f, allow_pickle=True)
    return matrix

def write_gene_embeddings(matrix, save_file):
    with open(save_file, 'wb') as file:
        np.save(file, matrix, allow_pickle=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--adjacency_matrix_file', type=str, help='Path to the adjacency matrix file (.npy)')
    parser.add_argument('--save_file', type=str, help='Path to the file (.npy) where the gene embeddings are stored')
    args = parser.parse_args()

    adj_matrix = load_gene_adjacency_matrix(args.adjacency_matrix_file)
    matrix = rank_calculation(adj_matrix=adj_matrix, emb_size=1024)
    write_gene_embeddings(matrix, args.save_file)