from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from bert.modeling import BertConfig, BERTLayerNorm, BERTLayer, BERTEmbeddings, BERTPooler

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()

def reconstruct(x, ref):
    if len(x.size()) == 1:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        return x.view([batch_size, turn_num])
    elif len(x.size()) == 2:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        sequence_length = x.size()[1]
        return x.view([batch_size, turn_num, sequence_length])
    else:
        raise Exception()

def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.view([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

def get_span_representation(span_starts, span_ends, input, input_mask):
    '''
    :param span_starts: [N, M]
    :param span_ends: [N, M]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N*M, JR, D], [N*M, JR]
    '''
    input_mask = input_mask.to(dtype=span_starts.dtype)  # fp16 compatibility
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.view([-1])  # [N*M]
    span_ends_offset = span_ends_offset.view([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)

    context_outputs = flatten_emb_by_sentence(input, input_mask)  # [<N*L, D]
    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]    # [N*M, JR, D]

    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask

def get_self_att_representation(input, input_score, input_mask):
    '''
    :param input: [N, L, D]
    :param input_score: [N, L]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=1)
    return output

def distant_cross_entropy(logits, positions):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                           torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

def get_weighted_att_representation(query, input, input_mask):
    '''
    :param query: [N, D]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    attention_score = torch.matmul(query.unsqueeze(1), input.transpose(-1, -2)) # [N, 1, L]
    attention_score = attention_score.squeeze(1)
    input_mask = input_mask.to(dtype=attention_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    attention_score = attention_score + input_mask
    attention_prob = nn.Softmax(dim=-1)(attention_score)
    attention_prob = attention_prob.unsqueeze(-1)
    output = torch.sum(attention_prob * input, dim=1)
    return output

class EarlyStopBERTEncoder(nn.Module):
    def __init__(self, config):
        super(EarlyStopBERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, num_hidden_stop, hidden_states, attention_mask):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
            if i == num_hidden_stop - 1:
                break
        return all_encoder_layers

class EarlyStopBertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super(EarlyStopBertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = EarlyStopBERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, num_hidden_stop, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(num_hidden_stop, embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output


class BertForRankingAndReadingAndReranking(nn.Module):
    def __init__(self, config, num_hidden_rank):
        super(BertForRankingAndReadingAndReranking, self).__init__()
        self.num_hidden_rank = num_hidden_rank
        self.num_hidden_read = config.num_hidden_layers
        self.bert = EarlyStopBertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.Tanh()
        self.rank_affine = nn.Linear(config.hidden_size, 1)
        self.rank_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.rank_classifier = nn.Linear(config.hidden_size, 2)
        self.read_affine = nn.Linear(config.hidden_size, 2)
        self.rerank_affine = nn.Linear(config.hidden_size, 1)
        self.rerank_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.rerank_classifier = nn.Linear(config.hidden_size, 1)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, rank_labels=None, start_positions=None,
                end_positions=None, span_starts=None, span_ends=None, hard_labels=None, soft_labels=None,
                sequence_input=None):
        if mode == 'rank':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(self.num_hidden_rank, input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            sequence_weights = self.rank_affine(sequence_output).squeeze(-1)
            pooled_output = get_self_att_representation(sequence_output, sequence_weights, attention_mask)

            pooled_output = self.rank_dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
            rank_logits = self.rank_classifier(pooled_output)

            if rank_labels is not None:
                rank_loss_fct = CrossEntropyLoss()
                rank_loss = rank_loss_fct(rank_logits, rank_labels)
                return rank_loss
            else:
                return rank_logits

        elif mode == 'read_inference':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(self.num_hidden_read, input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            logits = self.read_affine(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            return start_logits, end_logits, sequence_output

        elif mode == 'rerank_inference':
            assert span_starts is not None and span_ends is not None and sequence_input is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_input,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]

            span_weights = self.rerank_affine(span_output).squeeze(-1)
            span_pooled_output = get_self_att_representation(span_output, span_weights, span_mask)    # [N*M, D]

            span_pooled_output = self.rerank_dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            rerank_logits = self.rerank_classifier(span_pooled_output).squeeze(-1)
            rerank_logits = reconstruct(rerank_logits, span_starts)
            return rerank_logits

        elif mode == 'read_rerank_train':
            assert input_ids is not None and token_type_ids is not None
            assert start_positions is not None and end_positions is not None
            all_encoder_layers, _ = self.bert(self.num_hidden_read, input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            logits = self.read_affine(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            read_loss = (start_loss + end_loss) / 2

            assert span_starts is not None and span_ends is not None and hard_labels is not None and soft_labels is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.rerank_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.rerank_dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            rerank_logits = self.rerank_classifier(span_pooled_output).squeeze(-1)
            rerank_logits = reconstruct(rerank_logits, span_starts)
            norm_rerank_logits = rerank_logits / torch.sum(rerank_logits, -1, True)

            hard_loss = distant_cross_entropy(rerank_logits, hard_labels)
            soft_loss_fct = MSELoss()
            soft_loss = soft_loss_fct(norm_rerank_logits, soft_labels.to(dtype=rerank_logits.dtype))
            rerank_loss = hard_loss + soft_loss
            return read_loss + rerank_loss

        else:
            raise Exception

class BertForRankingAndDistantReadingAndReranking(nn.Module):
    def __init__(self, config, num_hidden_rank):
        super(BertForRankingAndDistantReadingAndReranking, self).__init__()
        super(BertForRankingAndDistantReadingAndReranking, self).__init__()
        self.num_hidden_rank = num_hidden_rank
        self.num_hidden_read = config.num_hidden_layers
        self.bert = EarlyStopBertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.Tanh()
        self.rank_affine = nn.Linear(config.hidden_size, 1)
        self.rank_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.rank_classifier = nn.Linear(config.hidden_size, 2)
        self.read_affine = nn.Linear(config.hidden_size, 2)
        self.rerank_affine = nn.Linear(config.hidden_size, 1)
        self.rerank_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.rerank_classifier = nn.Linear(config.hidden_size, 1)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, rank_labels=None, start_positions=None,
                end_positions=None, span_starts=None, span_ends=None, hard_labels=None, soft_labels=None,
                sequence_input=None):
        if mode == 'rank':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(self.num_hidden_rank, input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            sequence_weights = self.rank_affine(sequence_output).squeeze(-1)
            pooled_output = get_self_att_representation(sequence_output, sequence_weights, attention_mask)

            pooled_output = self.rank_dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
            rank_logits = self.rank_classifier(pooled_output)

            if rank_labels is not None:
                rank_loss_fct = CrossEntropyLoss()
                rank_loss = rank_loss_fct(rank_logits, rank_labels)
                return rank_loss
            else:
                return rank_logits

        elif mode == 'read_inference':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(self.num_hidden_read, input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            logits = self.read_affine(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            return start_logits, end_logits, sequence_output

        elif mode == 'rerank_inference':
            assert span_starts is not None and span_ends is not None and sequence_input is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_input,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]

            span_weights = self.rerank_affine(span_output).squeeze(-1)
            span_pooled_output = get_self_att_representation(span_output, span_weights, span_mask)    # [N*M, D]

            span_pooled_output = self.rerank_dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            rerank_logits = self.rerank_classifier(span_pooled_output).squeeze(-1)
            rerank_logits = reconstruct(rerank_logits, span_starts)
            return rerank_logits

        elif mode == 'read_rerank_train':
            assert input_ids is not None and token_type_ids is not None
            assert start_positions is not None and end_positions is not None
            all_encoder_layers, _ = self.bert(self.num_hidden_read, input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

            logits = self.read_affine(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            start_loss = distant_cross_entropy(start_logits, start_positions)
            end_loss = distant_cross_entropy(end_logits, end_positions)
            read_loss = (start_loss + end_loss) / 2

            assert span_starts is not None and span_ends is not None and hard_labels is not None and soft_labels is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.rerank_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.rerank_dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            rerank_logits = self.rerank_classifier(span_pooled_output).squeeze(-1)
            rerank_logits = reconstruct(rerank_logits, span_starts)
            norm_rerank_logits = rerank_logits / torch.sum(rerank_logits, -1, True)

            hard_loss = distant_cross_entropy(rerank_logits, hard_labels)
            soft_loss_fct = MSELoss()
            soft_loss = soft_loss_fct(norm_rerank_logits, soft_labels.to(dtype=rerank_logits.dtype))
            rerank_loss = hard_loss + soft_loss
            return read_loss + rerank_loss

        else:
            raise Exception
