import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import aggr
from dataLoader import relation_encoder


class SubGL(nn.Module):
    def __init__(self, args):
        super(SubGL, self).__init__()
        self.dim = args.dim
        self.user_num = args.user_num
        self.rela_num = 3

        self.REm = relation_encoder(self.rela_num)
        self.UEm = nn.Embedding(self.user_num, self.dim, padding_idx=0)
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(self.rela_num, 1), requires_grad=True)
        self.aggr = aggr.SumAggregation()
        self.drop = nn.Dropout()

    def forward(self, sequence, seq_neighbor, rel_neigh):
        init_user = self.UEm.weight
        init_rela = self.REm

        seq_embed = F.embedding(sequence, init_user, padding_idx=0)
        neighbor_embed = F.embedding(seq_neighbor, init_user, padding_idx=0)
        relation_weight = F.embedding(rel_neigh, init_rela, padding_idx=0)

        beta = F.softmax(self.weight_b, dim=0)
        temp = torch.matmul(relation_weight.float(), beta)
        neighbor = self.drop(temp * neighbor_embed)

        aggregated_neighbors = self.aggr(neighbor).squeeze()

        cas_h = seq_embed + aggregated_neighbors
        cas_h = F.normalize(cas_h, p=2, dim=2)
        return cas_h


class PositionEmbedding(nn.Module):
    def __init__(self, args):
        super(PositionEmbedding, self).__init__()
        self.dim = args.dim
        self.pos_embed = nn.Embedding(200, self.dim)
        self.Drop = nn.Dropout(args.dropout)

    def forward(self, seq_h):
        position_ids = torch.arange(0, seq_h.size(1), dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).view(-1, seq_h.size(1))
        pe = self.pos_embed(position_ids).repeat(seq_h.size(0), 1, 1)
        return self.Drop(pe)


class SSAN(nn.Module):
    def __init__(self, args):
        super(SSAN, self).__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads

        self.W1 = nn.Linear(args.max_len, args.max_len * 2)
        self.W2 = nn.Linear(args.max_len * 2, args.max_len)
        self.PE = PositionEmbedding(args)
        self.Encoder = TrmEncoder(input_size=args.dim, n_heads=self.n_heads)
        self.drop = nn.Dropout(args.dropout)

    def StrSphere(self, dis_matrix):
        ex = torch.ones_like(dis_matrix) * 2.718  # e \approx 2.718
        Psi = 1 / torch.log(dis_matrix + ex)
        Psi = self.W2(torch.relu(self.W1(Psi)))
        return Psi

    def forward(self, seq_h, pad_mask, matrix_d):
        Psi = self.StrSphere(matrix_d)
        seq_h = seq_h + self.PE(seq_h)

        seq_h = self.Encoder(self.drop(seq_h), seq_h, seq_h, pad_mask, Psi)
        return self.drop(seq_h)


class TCAN(nn.Module):
    def __init__(self, args):
        super(TCAN, self).__init__()
        self.dim = args.dim
        self.ws = args.window_size

        self.Encoder = TrmEncoder(input_size=self.dim, n_heads=args.n_heads)
        self.Stage_Agg = Fusion(args)
        self.drop = nn.Dropout(args.dropout)

    def stageDivision(self, inputs, timestamp=None):
        if timestamp is None:
            cumsum_inputs = torch.cumsum(inputs, dim=1)
            i_clone = inputs.clone()
            i_clone[:, self.ws:, :] = cumsum_inputs[:, :-self.ws, :]
            current_stage = cumsum_inputs - i_clone
        # else:
        Sk = self.Stage_Agg(inputs, current_stage)
        return self.drop(Sk)

    def forward(self, inputs, mask):
        query = self.stageDivision(inputs)
        x = self.Encoder(query, inputs, inputs, mask)

        return self.drop(x)
    

class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()
        self.dim = args.dim
        self.W1 = nn.Linear(self.dim, self.dim)
        self.W2 = nn.Linear(self.dim, self.dim)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, h1, h2):
        a = self.drop(torch.sigmoid(self.W1(h1) + self.W2(h2)))
        h = (1-a) * h1 + a * h2
        return h
    

class TrmEncoder(nn.Module):
    def __init__(self, input_size, d_k=128, d_v=128, n_heads=4, is_layer_norm=True, attn_dropout=0.1):
        super(TrmEncoder, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v * n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()
    

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, attn_mask, attn_bias, epsilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + epsilon)

        pad_mask = attn_mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
        attn_mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().cuda()
        mask_ = attn_mask + pad_mask
        Q_K = Q_K.masked_fill(mask_, -2 ** 32 + 1)

        if attn_bias is not None:
            attn_bias = attn_bias.masked_fill(mask_, -2 ** 32 + 1)
            attn_weight = F.softmax(Q_K, dim=-1) + F.softmax(attn_bias, dim=-1)

        else:
            attn_weight = F.softmax(Q_K, dim=-1)

        attn_weight = self.dropout(attn_weight)
        return attn_weight @ V

    def multi_head_attention(self, Q, K, V, mask, bias):

        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)

        mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)
        mask = mask.reshape(-1, mask.size(-1))
        if bias is not None:
            bias = bias.unsqueeze(dim=1).expand(-1, self.n_heads, -1, -1)
            bias = bias.reshape(-1, mask.size(-1), mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask, bias)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output

    def forward(self, Q, K, V, mask, bias=None):
        V_att = self.multi_head_attention(Q, K, V, mask, bias)

        if self.is_layer_norm:
            X = self.layer_norm(Q + V_att)
            output = self.layer_norm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output
