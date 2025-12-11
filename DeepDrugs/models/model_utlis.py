import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import models.Tensor as Tensor


class TCNet(nn.Module):
    def __init__(self, v_dim, q_dim, a_dim, h_dim, h_out, rank, glimpse, act='ReLU', dropout=[.2, .5], k=1):
        super(TCNet, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.a_dim = a_dim
        self.h_out = h_out
        self.rank  = rank
        self.h_dim = h_dim*k
        self.hv_dim = int(h_dim/rank)
        self.hq_dim = int(h_dim/rank)
        self.ha_dim = int(h_dim/rank)


        self.v_tucker = FCNet([v_dim, self.h_dim], act=act, dropout=dropout[1])
        # self.q_tucker = FCNet([q_dim, self.h_dim], act=act, dropout=dropout[0])
        # self.a_tucker = FCNet([a_dim, self.h_dim], act=act, dropout=dropout[0])
        self.qa_tucker = FCNet([q_dim, self.h_dim], act=act, dropout=dropout[0])
        self.q_tucker = self.qa_tucker  # q和a共享tucker层
        self.a_tucker = self.qa_tucker
        if self.h_dim < 1024:
            self.a_tucker = FCNet([a_dim, self.h_dim], act=act, dropout=dropout[0])
            self.v_net = nn.ModuleList([FCNet([self.h_dim, self.hv_dim], act=act, dropout=dropout[1]) for _ in range(rank)])
            # self.q_net = nn.ModuleList([FCNet([self.h_dim, self.hq_dim], act=act, dropout=dropout[0]) for _ in range(rank)])
            # self.a_net = nn.ModuleList([FCNet([self.h_dim, self.ha_dim], act=act, dropout=dropout[0]) for _ in range(rank)])
            self.qa_net = nn.ModuleList(
                [FCNet([self.h_dim, self.hq_dim], act=act, dropout=dropout[0]) for _ in range(rank)])
            self.q_net = self.qa_net
            self.a_net = self.qa_net

            if h_out > 1:
                self.ho_dim = int(h_out / rank)
                h_out = self.ho_dim

            self.T_g = nn.Parameter(torch.Tensor(1, rank, self.hv_dim, self.hq_dim, self.ha_dim, glimpse, h_out).normal_())
        self.dropout = nn.Dropout(dropout[1])
        self.bn = nn.BatchNorm1d(h_dim)


    def forward(self, v, q, a):
        f_emb = 0
        v_tucker = self.v_tucker(v)
        q_tucker = self.q_tucker(q)
        a_tucker = self.a_tucker(a)
        for r in range(self.rank):
            v_ = self.v_net[r](v_tucker)
            q_ = self.q_net[r](q_tucker)
            a_ = self.a_net[r](a_tucker)
            f_emb = Tensor.ModeProduct(self.T_g[:, r, :, :, :, :, :], v_, q_, a_, None) + f_emb

        v_ = self.v_tucker(v).transpose(2, 1)  # b x d x v
        q_ = self.q_tucker(q).transpose(2, 1).unsqueeze(3)  # b x d x q x 1
        a_ = self.a_tucker(a).transpose(2, 1).unsqueeze(3)  # b x d x a

        logits = torch.einsum('bdv,bvqag,bdqi,bdaj->bdijg', [v_, f_emb, q_, a_])
        # logits = logits.mean(dim=-1)
        logits = logits.sum(dim=-1)
        logits = logits.squeeze(3).squeeze(2)
        logits = self.bn(logits)

        return f_emb.squeeze(4), logits


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        # Normalize input_tensor
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # Apply scaling and bias
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)  
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs_0


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    
        

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs_0 = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs_0


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs_0 = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs_0  


class EncoderCell(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(EncoderCell,self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)        
        self.dense = nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.ReLU(),
                    nn.Dropout(hidden_dropout_prob),
                    nn.Linear(intermediate_size, hidden_size))


    def forward(self, hidden_states, attention_mask):
        hidden_states_1 = self.LayerNorm(hidden_states)
        attention_output, attention_probs_0  = self.attention(hidden_states_1, attention_mask)
        hidden_states_2 = hidden_states_1 + attention_output
        hidden_states_3 = self.LayerNorm(hidden_states_2)

        hidden_states_4 = self.dense(hidden_states_3)

        layer_output = hidden_states_2 + hidden_states_4

        return layer_output, attention_probs_0   
