import torch
import numpy as np
import pandas as pd
from dgl import DGLGraph
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial
import dgl
import os
import hashlib


def smiles_to_dgl(smiles_list):
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
    fc = partial(smiles_to_bigraph, add_self_loop=True)
    graphs = []

    for smiles in smiles_list:
        graph = fc(smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
        graphs.append(graph)

    return graphs


def max_node_padding(graph, max_node):
    actual_node_feats = graph.ndata.pop('h')
    num_actual_nodes = actual_node_feats.shape[0]
    num_virtual_nodes = max_node - num_actual_nodes

    virtual_node_bit = torch.zeros([num_actual_nodes, 1])
    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
    graph.ndata['h'] = actual_node_feats

    virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
    graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})

    graph = graph.add_self_loop()

    return graph


def save_graph(graph, file_path):
    dgl.save_graphs(file_path, [graph])

def smiles_to_file_name(smiles):

    return hashlib.md5(smiles.encode('utf-8')).hexdigest() + '.bin'

def save_graphs(smiles_list, graphs, save_dir="graphs_global"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    graph_file_paths = {}

    for smiles, graph in zip(smiles_list, graphs):
        file_name = smiles_to_file_name(smiles)
        file_path = os.path.join(save_dir, file_name)

        save_graph(graph, file_path)

        graph_file_paths[smiles] = file_path

    np.save(os.path.join(save_dir, 'graphs_file_paths.npy'), graph_file_paths)

if __name__ == "__main__":

    df = pd.read_excel('Drugcomb_drugs.xlsx')
    smiles_list = df['smiles'].dropna().astype(str).tolist()

    graphs = smiles_to_dgl(smiles_list)

    max_node = 114
    padded_graphs = [max_node_padding(graph, max_node) for graph in graphs]
    g = padded_graphs[0]
    print("num nodes:", g.num_nodes())
    print("h.shape:", g.ndata['h'].shape)
    print("virtual_mask sum:", g.ndata['virtual_mask'].sum().item())
    print("is_real sum:", g.ndata['is_real'].sum().item())
    print("is_global indices:", torch.nonzero(g.ndata['is_global'], as_tuple=True)[0].tolist())
    print("global node feature (last dimension):", g.ndata['h'][-1, -1].item())

    save_graphs(smiles_list, padded_graphs)

    print("Graphs saved successfully.")