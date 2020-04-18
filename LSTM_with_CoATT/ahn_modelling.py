import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np


def to_onehot(id, batchsize, num_id):
    uid_onehot = torch.FloatTensor(batchsize, num_id).cuda()
    uid_onehot.zero_()
    uid_onehot.scatter_(1, id.unsqueeze(1), 1)

    return uid_onehot


class LSTMForUserItemPredictionHIRCOAA(nn.Module):
    def __init__(self, args):
        super(LSTMForUserItemPredictionHIRCOAA, self).__init__()

        self._word_embedding = nn.Embedding(args.embed_num,
                                            args.embed_dim)
        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        args.embed_dim,
                                        args.hidden_size // 2,
                                        bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)
        self.coatt_rev = Co_Attention_wATT(args)
        self.num_head = args.num_head
        for num in range(self.num_head):
            setattr(self, 'coatt_' + str(num), Co_Attention_wATT(args))
        self.user_mapping = nn.Linear(args.num_uid, args.hidden_size, bias=False)
        self.item_mapping = nn.Linear(args.num_iid, args.hidden_size, bias=False)
        self.FM = TorchFM(args.hidden_size * (1 * 2 + 2), 10)
        self.num_uid = args.num_uid
        self.num_iid = args.num_iid
        self.num_reviews = args.num_reviews
        self.seq_encoder = nn.LSTM(input_size=300, hidden_size=150,
                                   num_layers=1, batch_first=True,
                                   bidirectional=True)

        self.item_sent_att = Gated_Attention(args)
        self.item_review_att = Gated_Attention(args)

        self.s2rmaping = nn.Linear(args.hidden_size, args.hidden_size)
        self.s2rrmaping = nn.Linear(args.hidden_size * args.num_head, args.hidden_size)

    def forward(self, u_input_ids, input_ids, u_sent_mask, i_sent_mask, u_sent_len, i_sent_len,
                u_review_mask, i_review_mask, i_review_label, u_ids, i_ids):
        
        batch_size = u_input_ids.size(0)
        size = batch_size * u_input_ids.size(1) * u_input_ids.size(2)
        num_sent = u_input_ids.size(2)
        u_input_ids, input_ids, = u_input_ids.view(size, -1), input_ids.view(size, -1)

        u_embed = self._word_embedding(u_input_ids)
        i_embed = self._word_embedding(input_ids)
        
        u_lstm_output = self._encoding(u_embed, u_sent_len.view(-1))
        i_lstm_output = self._encoding(i_embed, i_sent_len.view(-1))
        u_pooled_output = F.max_pool1d(u_lstm_output.transpose(1, 2), u_lstm_output.size(1)).squeeze(2)
        pooled_output = F.max_pool1d(i_lstm_output.transpose(1, 2), i_lstm_output.size(1)).squeeze(2)

        u_rep_batch = u_pooled_output.view(batch_size * self.num_reviews, num_sent, -1)
        i_rep_batch = pooled_output.view(batch_size * self.num_reviews, num_sent, -1)

        u_pooled_output = u_rep_batch.contiguous().view(batch_size, self.num_reviews, num_sent, -1)
        pooled_output = i_rep_batch.contiguous().view(batch_size, self.num_reviews, num_sent, -1)

        u_pooled_output = u_pooled_output * u_sent_mask.unsqueeze(3).expand_as(u_pooled_output).float()
        pooled_output = pooled_output * i_sent_mask.unsqueeze(3).expand_as(pooled_output).float()

        u_rep_batch = u_pooled_output.view(batch_size * self.num_reviews, num_sent, -1)
        i_rep_batch = pooled_output.view(batch_size * self.num_reviews, num_sent, -1)

        u_rep_concat = u_pooled_output.view(batch_size, self.num_reviews * num_sent, -1) \
            .unsqueeze(1).expand(-1, self.num_reviews, -1, -1)
        i_rep_concat = pooled_output.view(batch_size, self.num_reviews * num_sent, -1) \
            .unsqueeze(1).expand(-1, self.num_reviews, -1, -1)

        i_rep_concat = i_rep_concat.contiguous().view(batch_size * self.num_reviews, self.num_reviews * num_sent, -1)
        i_sent_mask_concat = i_sent_mask.view(-1, self.num_reviews * num_sent).unsqueeze(1) \
            .expand(-1, self.num_reviews, -1).contiguous()

        i_sent_mask = i_sent_mask.view(i_sent_mask.size(0) * i_sent_mask.size(1), -1)
        weighted_output, all_i_sw, overall_i_sw = self.item_sent_att(i_rep_batch, i_sent_mask)

        i_coatt_output = torch.sum(weighted_output, 1)
        i_coatt_output = i_coatt_output.contiguous().view(batch_size, self.num_reviews, -1)
        i_coatt_output = F.relu(self.s2rmaping(i_coatt_output))

        i_sw = overall_i_sw.view(batch_size, self.num_reviews * num_sent).unsqueeze(1) \
            .expand(-1, self.num_reviews, -1).contiguous() \
            .view(batch_size * self.num_reviews, -1)
        att_output = []
        us_weight = []

        u_sent_mask = u_sent_mask.contiguous().view(u_sent_mask.size(0) * u_sent_mask.size(1), -1)
        i_sent_mask = i_sent_mask.contiguous().view(i_sent_mask.size(0) * i_sent_mask.size(1), -1)

        for num in range(self.num_head):
            coatt_output, u_sw = getattr(self, 'coatt_' + str(num)) \
                (u_rep_batch, i_rep_concat, u_sent_mask, i_sent_mask_concat, i_sw, review_level=False)
            att_output.append(coatt_output)
            us_weight.append(u_sw)
        u_coatt_output = torch.cat(att_output, 1)
        all_u_sw = torch.cat(us_weight, 1)

        ir_weighted, i_rw, overall_i_rw = self.item_review_att(i_coatt_output, i_review_mask)

        ir_pooled = torch.sum(ir_weighted, 1)
        u_coatt_output = u_coatt_output.contiguous().view(batch_size, self.num_reviews, -1)
        u_coatt_output = F.relu(self.s2rrmaping(u_coatt_output))
        ur_pooled, u_rw = self.coatt_rev(u_coatt_output, i_coatt_output, u_review_mask,
                                         i_review_mask, overall_i_rw, review_level=False)

        uid_onehot = to_onehot(u_ids, batch_size, self.num_uid)
        iid_onehot = to_onehot(i_ids, batch_size, self.num_iid)

        uid_emb = self.dropout(self.user_mapping(uid_onehot))
        iid_emb = self.dropout(self.item_mapping(iid_onehot))
        
        ur = torch.cat((ur_pooled, uid_emb), 1)
        ir = torch.cat((ir_pooled, iid_emb), 1)
        
        concat_pooled_output = torch.cat((ur, ir), 1)
        logits = self.FM(concat_pooled_output)

        return logits, all_u_sw, all_i_sw, u_rw, i_rw


class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.
    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase), \
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.
        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        sorted_batch, sorted_lengths, _, restoration_idx = \
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


class Gated_Attention(nn.Module):
    def __init__(self, config):
        super(Gated_Attention, self).__init__()
        self.proj_matrix = nn.Linear(300, 300, bias=False)
        self.gated_matrix = nn.Linear(300, 300, bias=False)
        self.item_att_v = nn.Parameter(torch.randn(config.hidden_size, 1), requires_grad=True)

    #     config.coatt_hidden_size
    def forward(self, item_input, i_mask):
        item_att_v = self.item_att_v.unsqueeze(0).expand(item_input.size(0), -1, -1)
        # bsz * L_u
        weighted_ir = torch.bmm(torch.tanh(self.proj_matrix(item_input)) * torch.sigmoid(self.gated_matrix(item_input)),
                                item_att_v).squeeze(2)  # bsz * 1 * L_u
        weighted_ir = weighted_ir / torch.sqrt(torch.tensor(300.0).cuda())
        i_att_weight = masked_softmax(weighted_ir, i_mask).unsqueeze(1)  # bsz * 1* L_i
        overall_w = masked_softmax(weighted_ir.view(-1), i_mask.view(-1)).view(weighted_ir.size(0), -1)

        item_rep = i_att_weight.transpose(1, 2).expand_as(item_input) * item_input
        return item_rep, i_att_weight, overall_w


class Co_Attention_wATT(nn.Module):
    def __init__(self, config):
        super(Co_Attention_wATT, self).__init__()
        self.proj_matrix = nn.Linear(300, 300, bias=False)
        self.V = nn.Parameter(glorot([300, 300]), requires_grad=True)

    #     config.coatt_hidden_size
    def forward(self, user_input, item_input, u_sent_mask, i_sent_mask, i_sw, review_level=False):

        if review_level:
            item_input = item_input
        else:
            norm_i_sw = i_sw.div(torch.norm(i_sw, dim=-1).unsqueeze(-1).expand_as(i_sw) + 1e-13)
            norm_i_sw = norm_i_sw.unsqueeze(-1).expand_as(item_input)
            item_input = item_input * norm_i_sw.detach()

        project_user = torch.bmm(user_input, self.V.unsqueeze(0).expand(user_input.size(0), -1, -1))
        G = torch.bmm(project_user, item_input.transpose(1, 2))  # bsz * L_u * L_i
        G[G == 0] = -1000

        user_coatt = F.max_pool1d(G, G.size(2)).squeeze(2)
        user_coatt = user_coatt / torch.sqrt(torch.tensor(300.0).cuda())
        u_att_weight = masked_softmax(user_coatt, u_sent_mask).unsqueeze(1)  # bsz * 1 * L_u

        user_rep = torch.bmm(u_att_weight, user_input).squeeze(1)
        return user_rep, u_att_weight


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range
    return init


class TorchFM(nn.Module):
    def __init__(self, n=None, k=None):
        super(TorchFM, self).__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(glorot([n, k]), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)  # S_2

        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        return out



def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    """
    Sort a batch of padded variable length sequences by their length.
    Args:
        batch: A batch of padded variable length sequences. The batch should
            have the dimensions (batch_size x max_sequence_length x *).
        sequences_lengths: A tensor containing the lengths of the sequences in the
            input batch. The tensor should be of size (batch_size).
        descending: A boolean value indicating whether to sort the sequences
            by their lengths in descending order. Defaults to True.
    Returns:
        sorted_batch: A tensor containing the input batch reordered by
            sequences lengths.
        sorted_seq_lens: A tensor containing the sorted lengths of the
            sequences in the input batch.
        sorting_idx: A tensor containing the indices used to permute the input
            batch in order to get 'sorted_batch'.
        restoration_idx: A tensor containing the indices that can be used to
            restore the order of the sequences in 'sorted_batch' so that it
            matches the input batch.
    """
    sorted_seq_lens, sorting_index = \
        sequences_lengths.sort(0, descending=descending)

    sorted_batch = batch.index_select(0, sorting_index)

    idx_range = \
        sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


