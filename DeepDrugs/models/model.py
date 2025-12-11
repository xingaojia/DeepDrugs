import torch
import torch.nn as nn
from .model_utlis import Encoder, EncoderCell, FCNet, TCNet
import numpy as np
import torch.nn.functional as F
import pandas as pd
import dgl
import os
from torch.nn.utils.weight_norm import weight_norm
from dgllife.model.gnn import GCN
from functools import partial
import models.Tensor as Tensor



class CellCNN(nn.Module):
    def __init__(self, in_channel=3, feat_dim=None, args=None):
        super(CellCNN, self).__init__()

        max_pool_size = [2, 2, 2]
        drop_rate = 0.2
        kernel_size = [8, 8, 8]

        in_channels = [3, 8, 16]
        out_channels = [8, 16, 32]
        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[0]),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[1]),
            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )

        self.cell_linear = nn.Linear(out_channels[2], feat_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x_cell_embed = self.cell_conv(x)
        x_cell_embed = x_cell_embed.transpose(1, 2)
        x_cell_embed = self.cell_linear(x_cell_embed)

        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        if isinstance(batch_graph, list):
            batch_graph = dgl.batch(batch_graph)
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)


class MLP(torch.nn.Module):
    def __init__(self, out_channels=256):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 30, int(out_channels * 20)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(out_channels * 20), out_channels * 10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(out_channels * 10, 1),
        )

    def forward(self, x_cell_embed, drug_embed):
        out = torch.cat((x_cell_embed, drug_embed), dim=1)
        out = self.fc(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)



class TriAttention(nn.Module):
    def __init__(self, v_dim, q_dim, a_dim, h_dim, h_out, rank, glimpse, k, dropout=[.2, .5]):
        super(TriAttention, self).__init__()
        self.glimpse = glimpse
        self.TriAtt = TCNet(v_dim, q_dim, a_dim, h_dim, h_out, rank, glimpse, dropout=dropout, k=k)
        # Projection layers for feature transformation
        self.d = h_dim  # Projection dimension
        self.v_proj = nn.Linear(v_dim, self.d)
        self.q_proj = nn.Linear(q_dim, self.d)
        self.a_proj = nn.Linear(a_dim, self.d)

        # Linear layers to update features using attention
        self.v_update = nn.Linear(glimpse * self.d, v_dim)
        self.q_update = nn.Linear(glimpse * self.d, q_dim)
        self.a_update = nn.Linear(glimpse * self.d, a_dim)

        self.dropout = nn.Dropout(dropout[1])  # Dropout for regularization

    def forward(self, v, q, a):
        batch_size = v.size(0)
        v_num = v.size(1)
        q_num = q.size(1)
        a_num = a.size(1)
        logits, z = self.TriAtt(v, q, a)
        mask = (0 == v.abs().sum(2)).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(logits.size())
        logits.data.masked_fill_(mask.data, -float('inf'))
        p = torch.softmax(logits.contiguous().view(-1, v_num * q_num * a_num, self.glimpse), 1)
        p = p.view(-1, v_num, q_num, a_num, self.glimpse)  # Shape: (batch, v, q, a, glimpse)

        q_proj = self.q_proj(q)  # (batch, q_num, d)
        a_proj = self.a_proj(a)  # (batch, a_num, d)
        v_proj = self.v_proj(v)  # (batch, v_num, d)

        qa_comb = q_proj.unsqueeze(2) + a_proj.unsqueeze(1)  # (batch, q_num, a_num, d)
        va_comb = v_proj.unsqueeze(2) + a_proj.unsqueeze(1)  # (batch, v_num, a_num, d)
        vq_comb = v_proj.unsqueeze(2) + q_proj.unsqueeze(1)  # (batch, v_num, q_num, d)

        c_v = torch.einsum('bvqag,bqad->bvdg', p, qa_comb)  # (batch, v, d, glimpse)
        c_v = c_v.permute(0, 1, 3, 2).contiguous().view(batch_size, v_num, -1)  # (batch, v, glimpse*d)
        updated_v = v + self.dropout(self.v_update(c_v))

        c_q = torch.einsum('bvqag,bvad->bqdg', p, va_comb)  # (batch, q, d, glimpse)
        c_q = c_q.permute(0, 1, 3, 2).contiguous().view(batch_size, q_num, -1)
        updated_q = q + self.dropout(self.q_update(c_q ))

        c_a = torch.einsum('bvqag,bvqd->bagd', p, vq_comb)  # (batch, a, d, glimpse)
        c_a = c_a.permute(0, 1, 3, 2).contiguous().view(batch_size, a_num, -1)
        updated_a = a + self.dropout(self.a_update(c_a ))

        return p, logits, z, updated_v, updated_q, updated_a


class DeepDrugsNet(torch.nn.Module):

    def __init__(self,
                 num_attention_heads=8,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 output_dim=2560,
                 args=None,
                 device=None):
        super(DeepDrugsNet, self).__init__()

        self.args = args
        self.device = device
        self.include_omic = args.omic.split(',')  
        self.omic_dict = {'exp': 0, 'mut': 1, 'cn': 2}
        self.in_channel = len(self.include_omic)
        self.max_length = 3

        graph_paths_file = 'data/0_drug_data/graphs/graphs_file_paths.npy'
        self.drugs = np.load(graph_paths_file, allow_pickle=True).item()

        graph_dir = "data/0_drug_data/"
        self.drug_graph_cache = {}
        for drug, path in self.drugs.items():
            full_path = os.path.join(graph_dir, path)
            self.drug_graph_cache[drug] = dgl.load_graphs(full_path)[0][0]

        self.genes_nums = 1061
        hidden_size = 128
        self.cell_conv = CellCNN(in_channel=self.in_channel, feat_dim=hidden_size, args=args)

        intermediate_size = hidden_size * 2
        self.drug_SA = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                               hidden_dropout_prob)
        self.cell_SA = EncoderCell(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                                   hidden_dropout_prob)
        self.head = MLP()

        self.cell_fc = nn.Sequential(
            nn.Linear(126 * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.drug_fc = nn.Sequential(
            nn.Linear(114 * hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.tri_attention = TriAttention(
            v_dim=128,
            q_dim=128,
            a_dim=128,
            h_dim=64,
            h_out=1,
            rank=4,
            glimpse=2,
            k=1
        )

        self.drug_extractor = MolecularGCN(
            in_feats=75,
            dim_embedding=128,
            padding=True,
            hidden_feats=[128, 128]
        )

    def forward(self, data):


        device = next(self.parameters()).device

        drugA_graphs = [self.drug_graph_cache[drug].to(device) for drug in data.drugA]
        drugB_graphs = [self.drug_graph_cache[drug].to(device) for drug in data.drugB]
        drugA = self.drug_extractor(drugA_graphs)
        drugB = self.drug_extractor(drugB_graphs)
        batch_size, patch = drugA.size(0), drugA.size(1)
        drugA = drugA.float()
        drugB = drugB.float()
        drugA, attn5 = self.drug_SA(drugA, None)
        drugB, attn6 = self.drug_SA(drugB, None)

        x_cell = data.x_cell.type(torch.float32)
        x_cell = x_cell[:, [self.omic_dict[i] for i in self.include_omic]]  # [batch*4079,len(omics)]
        cellA = x_cell.view(batch_size, self.genes_nums, -1)
        cellA = self.cell_conv(cellA)
        cellA, attn7 = self.cell_SA(cellA, None)

        _, M, z, cellA, drugA, drugB = self.tri_attention(cellA, drugA, drugB)
        drugA_embed = self.drug_fc(drugA.view(-1, drugA.shape[1] * drugA.shape[2]))
        drugB_embed = self.drug_fc(drugB.view(-1, drugB.shape[1] * drugB.shape[2]))
        cellA_embed = self.cell_fc(cellA.view(-1, cellA.shape[1] * cellA.shape[2]))
        cell_embed = cellA_embed
        drug_embed = torch.cat((drugA_embed, drugB_embed), 1)
        output = self.head(cell_embed, drug_embed)
        return output, M

    def init_weights(self):

        self.head.init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

        self.cell_conv.init_weights()
