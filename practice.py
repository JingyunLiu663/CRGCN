import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import Dataset
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss


class GraphEncoder(nn.Module):
    def __init__(self, layers, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
        return x

class CRGCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(CRGCN, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.Graph_encoder = nn.ModuleDict({
            behavior: GraphEncoder(self.layers[index], self.embedding_size, self.node_dropout) for index, behavior in
            enumerate(self.behaviors)
        })

        self.reg_weight = args.reg_weight
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_all_embeddings = None

        self.apply(self._init_weights)

        self._load_model()