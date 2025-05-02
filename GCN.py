import math
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import pywt
import ptwt

from data_utils import get_dct_matrix
from utils import bipartite_soft_matching


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DeepGraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.I = torch.eye(self.weight.shape[0])
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, H0=None, num_layers=0, lamda=1.5, alpha=0.2):
        if num_layers > 0:
            beta = math.log(lamda / num_layers + 1)
            support = (1 - alpha) * torch.matmul(self.att, input) + alpha * H0
            output = torch.matmul(support, beta * self.weight + (1 - beta) * self.I.to(self.weight.device))
        else:
            support = torch.matmul(input, self.weight)
            output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = DeepGraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = DeepGraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, H0=None, num_layers=0):
        y = self.gc1(x, H0, num_layers)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y, H0, num_layers + 1)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MS_GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(MS_GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1_s1 = DeepGraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1_s1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2_s1 = DeepGraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2_s1 = nn.BatchNorm1d(node_n * in_features)

        self.gc1_s2 = DeepGraphConvolution(in_features, in_features, node_n=33, bias=bias)
        self.bn1_s2 = nn.BatchNorm1d(33 * in_features)

        self.gc2_s2 = DeepGraphConvolution(in_features, in_features, node_n=33, bias=bias)
        self.bn2_s2 = nn.BatchNorm1d(33 * in_features)

        self.gc1_s3 = DeepGraphConvolution(in_features, in_features, node_n=18, bias=bias)
        self.bn1_s3 = nn.BatchNorm1d(18 * in_features)

        self.gc2_s3 = DeepGraphConvolution(in_features, in_features, node_n=18, bias=bias)
        self.bn2_s3 = nn.BatchNorm1d(18 * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x_s1, H0_s1=None, H0_s2=None, H0_s3=None, num_layers=0):

        merging_s1tos2, unmerging_s2tos1 = bipartite_soft_matching(x_s1, r=33)
        x_s2 = merging_s1tos2(x_s1)
        merging_s2tos3, unmerging_s3tos2 = bipartite_soft_matching(x_s2, r=15)
        x_s3 = merging_s2tos3(x_s2)

        y_s1 = self.gc1_s1(x_s1, H0_s1, num_layers)
        b, n, f = y_s1.shape
        y_s1 = self.bn1_s1(y_s1.view(b, -1)).view(b, n, f)
        y_s1 = self.act_f(y_s1)
        y_s1 = self.do(y_s1)

        y_s1 = self.gc2_s1(y_s1, H0_s1, num_layers + 1)
        b, n, f = y_s1.shape
        y_s1 = self.bn2_s1(y_s1.view(b, -1)).view(b, n, f)
        y_s1 = self.act_f(y_s1)
        y_s1 = self.do(y_s1)

        y_s2 = self.gc1_s2(x_s2, H0_s2, num_layers)
        b, n, f = y_s2.shape
        y_s2 = self.bn1_s2(y_s2.view(b, -1)).view(b, n, f)
        y_s2 = self.act_f(y_s2)
        y_s2 = self.do(y_s2)

        y_s2 = self.gc2_s2(y_s2, H0_s2, num_layers + 1)
        b, n, f = y_s2.shape
        y_s2 = self.bn2_s2(y_s2.view(b, -1)).view(b, n, f)
        y_s2 = self.act_f(y_s2)
        y_s2 = self.do(y_s2)

        y_s3 = self.gc1_s3(x_s3, H0_s3, num_layers)
        b, n, f = y_s3.shape
        y_s3 = self.bn1_s3(y_s3.view(b, -1)).view(b, n, f)
        y_s3 = self.act_f(y_s3)
        y_s3 = self.do(y_s3)

        y_s3 = self.gc2_s3(y_s3, H0_s3, num_layers + 1)
        b, n, f = y_s3.shape
        y_s3 = self.bn2_s3(y_s3.view(b, -1)).view(b, n, f)
        y_s3 = self.act_f(y_s3)
        y_s3 = self.do(y_s3)

        y_s2to1 = unmerging_s2tos1(y_s2)
        y_s3to1 = unmerging_s2tos1(unmerging_s3tos2(y_s3))
        y_s1 = y_s1 + 0.3 * y_s2to1 + 0.3 * y_s3to1

        return y_s1 + x_s1

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Embedding(nn.Module):
    """ Apply embedding, including Linear Embedding and GCN Embedding
    Args:
        in_dims (int): Feature dimension of the input
        embedding_dims (int): Feature dimension of the embedding
        num_nodes (int): Number of human joints (20 for h3.6m)
    """

    def __init__(self, in_dims, embedding_dims, num_nodes):
        super().__init__()
        self.embedding_gcn = GraphConvolution(in_dims, embedding_dims, node_n=num_nodes)
        self.norm_layer = nn.BatchNorm1d(num_nodes * embedding_dims)
        self.act_layer = nn.ReLU()

    def forward(self, x, return_init=True):
        """
        Args:
            x (Tensor): Input sequences [B, N, C]
            return_init (bool): Whether return the initial state
        """
        B, N, C = x.shape
        x = self.embedding_gcn(x)
        H0 = x if return_init is True else None
        x = self.norm_layer(x.view(B, -1)).view(B, N, -1)
        x = self.act_layer(x)
        return x, H0 if return_init is True else x


class Predictor(nn.Module):
    """ Head of model to generate the future motion sequences
    """

    def __init__(self, hid_dims, out_dims, num_nodes):
        super().__init__()
        self.head = GraphConvolution(hid_dims, out_dims, node_n=num_nodes)

    def forward(self, x):
        x = self.head(x)
        return x


class GCNModeling(nn.Module):
    """ Use GCN to extract the spatial features
    Args:
        num_blocks (int): Number of Computational blocks
        num_nodes (int): Number of human joints
    """

    def __init__(self, hid_dims, num_blocks, num_nodes, drop_rate):
        super().__init__()
        self.GCN_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.GCN_blocks.append(GC_Block(hid_dims, p_dropout=drop_rate, node_n=num_nodes))

    def forward(self, x, H0=None, return_layers=None):
        """
        Args:
            x (Tensor): Input sequences [B, N, C]
            H0 (Tensor): The initial state
            return_layers (list): Specify the layers for which features need be returned
        """
        for i in range(len(self.GCN_blocks)):
            x = self.GCN_blocks[i](x, H0=H0, num_layers=i * 2 + 1)
        return x


class MS_GCNModeling(nn.Module):
    """ Use GCN to extract the spatial features
    Args:
        num_blocks (int): Number of Computational blocks
        num_nodes (int): Number of human joints
    """

    def __init__(self, hid_dims, num_blocks, num_nodes, drop_rate):
        super().__init__()
        self.MS_GCN_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.MS_GCN_blocks.append(MS_GC_Block(hid_dims, p_dropout=drop_rate, node_n=num_nodes))

    def forward(self, x_s1, x_s2, x_s3, H0_s1=None, H0_s2=None, H0_s3=None):
        """
        Args:
            x_s1 (Tensor): Input sequences(scale1) [B, N, C]
            H0_s1 (Tensor): The initial state(scale1)
        """
        for i in range(len(self.MS_GCN_blocks)):
            x_s1 = self.MS_GCN_blocks[i](x_s1, H0_s1, H0_s2, H0_s3, num_layers=i * 2 + 1)
        return x_s1


class CMGC(nn.Module):
    """ Modeling the input sequences
    Args:
        in_dims (int): Feature Dimension of the input
        hid_dims (int): Feature Dimension of the hidden state
        out_dims (int): Feature Dimension of the output
        drop_rate (float): probability of drop out
        num_blocks (int): Total number of Computational Block
        num_nodes (int): Number of human joints (20 for h36m)
        opt (class): Option
    """

    def __init__(self, in_dims, hid_dims, out_dims, drop_rate=0.5, num_blocks=12, num_nodes=66, opt=None):
        super().__init__()

        len_seq = opt.len_input + opt.len_output
        self.dct_n = opt.dct_n
        self.ori_DCT, self.ori_IDCT = get_dct_matrix(len_seq, opt)
        self.low_DCT, self.low_IDCT = get_dct_matrix(12, opt)

        self.embedding = Embedding(opt.dct_n, hid_dims, num_nodes)
        self.embedding_s2 = Embedding(opt.dct_n, hid_dims, 33)
        self.embedding_s3 = Embedding(opt.dct_n, hid_dims, 18)
        self.low_embedding = Embedding(12, hid_dims, num_nodes)
        self.high_embedding = Embedding(12, hid_dims, num_nodes)

        self.ori_feature_extractor = MS_GCNModeling(hid_dims, num_blocks, num_nodes, drop_rate)
        self.low_feature_extractor = GCNModeling(hid_dims, num_blocks, num_nodes, drop_rate)
        self.high_feature_extractor = GCNModeling(hid_dims, num_blocks, num_nodes, drop_rate)

        self.ori_head = Predictor(hid_dims, opt.dct_n, num_nodes)
        self.low_head = Predictor(hid_dims, 12, num_nodes)
        self.high_head = Predictor(hid_dims, 12, num_nodes)

    def forward(self, x):
        """ x: [B, N, T] [B, 66, 20] for h36m """

        B, N, T = x.shape

        # Obtain wavelet component (only one level)
        wavelet = pywt.Wavelet('bior1.3')  # db1  sym2 bior1.1  gaus1
        x_low, x_high = ptwt.wavedec(x.reshape(-1, T), wavelet, mode='zero', level=1)
        x_low, x_high = x_low.view(B, N, -1), x_high.view(B, N, -1)

        # Transform the original seq and the high frequency component of DWT to DCT domain
        x_dct = torch.matmul(self.ori_DCT[:self.dct_n].unsqueeze(dim=0), x.transpose(1, 2)).transpose(1, 2)
        x_low_dct = torch.matmul(self.low_DCT[:12].unsqueeze(dim=0), x_low.transpose(1, 2)).transpose(1, 2)

        # Embedding by GCN
        merging_s1tos2, unmerging_s2tos1 = bipartite_soft_matching(x_dct, r=33)
        x_dct_s2 = merging_s1tos2(x_dct)
        merging_s2tos3, unmerging_s3tos2 = bipartite_soft_matching(x_dct_s2, r=15)
        x_dct_s3 = merging_s2tos3(x_dct_s2)
        y_dct_s2, h0_dct_s2 = self.embedding_s2(x_dct_s2)
        y_dct_s3, h0_dct_s3 = self.embedding_s3(x_dct_s3)
        y_dct, h0_dct = self.embedding(x_dct)
        y_low_dct, h0_low_dct = self.low_embedding(x_low_dct)
        y_high, h0_high = self.high_embedding(x_high)

        # Extract the features
        y_dct = self.ori_feature_extractor(
            y_dct, y_dct_s2, y_dct_s3, h0_dct, h0_dct_s2, h0_dct_s3
        )
        y_low_dct = self.low_feature_extractor(y_low_dct, h0_low_dct)
        y_high = self.high_feature_extractor(y_high, h0_high)

        # Generate the motion sequences
        y_dct = self.ori_head(y_dct)
        y_low_dct = self.low_head(y_low_dct)
        y_high = self.high_head(y_high)

        # Recover the sequences to the temporal domain
        y = torch.matmul(self.ori_IDCT[:, :self.dct_n].unsqueeze(0), y_dct.transpose(1, 2)).transpose(1, 2)
        y_low = torch.matmul(self.low_IDCT[:, :12].unsqueeze(0), y_low_dct.transpose(1, 2)).transpose(1, 2)

        # Inverse DWT
        y_low, y_high = y_low.reshape(-1, 12), y_high.reshape(-1, 12)
        y_idwt = ptwt.waverec([y_low, y_high], wavelet).view(B, N, -1).contiguous()

        y = y + y_idwt + x

        return [y.permute(0, 2, 1)]
