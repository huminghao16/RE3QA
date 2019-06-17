import json
import random
import math
import six
import collections
import numpy as np
from typing import List
import bert.tokenization as tokenization
from squad.squad_utils import _improve_answer_span, _get_best_indexes, _compute_softmax, get_final_text
from squad.squad_evaluate import exact_match_score, f1_score, span_f1


class ExtractedParagraphWithAnswers(object):
    __slots__ = ["text", "start", "end", "answer_spans"]

    def __init__(self, text: List[str], start: int, end: int, answer_spans: np.ndarray):
        """
        :param text: List of source paragraphs that have been merged to form `self`
        :param start: start token of this text in the source document
        :param end: end token of this text in the source document
        """
        self.text = text
        self.start = start
        self.end = end
        self.answer_spans = answer_spans

    @property
    def n_context_words(self):
        return len(self.text)

    def __repr__(self):
        s = ""
        s += "text: %s ..." % (" ".join(self.text[:10]))
        s += ", start: %d" % (self.start)
        s += ", end: %d" % (self.end)
        s += ", answer_spans: {}".format(self.answer_spans)
        return s


class DocParagraphWithAnswers(ExtractedParagraphWithAnswers):
    __slots__ = ["doc_id"]

    def __init__(self, text: List[str], start: int, end: int, answer_spans: np.ndarray,
                 doc_id):
        super().__init__(text, start, end, answer_spans)
        self.doc_id = doc_id


class DocumentAndQuestion(object):
    def __init__(self,
                 document_id,
                 qas_id,
                 question_text, # str
                 doc_tokens,
                 orig_answer_texts=None,
                 start_positions=None,
                 end_positions=None):
        self.document_id = document_id
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_texts = orig_answer_texts
        self.start_positions = start_positions
        self.end_positions = end_positions

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "document_id: %s" % (self.document_id)
        s += ", qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: %s ..." % (" ".join(self.doc_tokens[:20]))
        s += ", length of doc_tokens: %d" % (len(self.doc_tokens))
        if self.orig_answer_texts:
            s += ", orig_answer_texts: {}".format(self.orig_answer_texts)
        if self.start_positions and self.end_positions:
            s += ", start_positions: {}".format(self.start_positions)
            s += ", end_positions: {}".format(self.end_positions)
            s += ", token_answer: "
            for start, end in zip(self.start_positions, self.end_positions):
                s += "{}, ".format(" ".join(self.doc_tokens[start:(end+1)]))
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_positions=None,
                 end_positions=None,
                 start_indexes=None,
                 end_indexes=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.start_indexes = start_indexes
        self.end_indexes = end_indexes
        self.is_impossible = is_impossible


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride,
                                 max_query_length, verbose_logging=False, logger=None):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_positions = []
        tok_end_positions = []
        for start_position, end_position in \
                zip(example.start_positions, example.end_positions):
            tok_start_position = orig_to_tok_index[start_position]
            if end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            tok_start_positions.append(tok_start_position)
            tok_end_positions.append(tok_end_position)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            # For distant supervision, we annotate the positions of all answer spans
            start_positions = [0] * len(input_ids)
            end_positions = [0] * len(input_ids)
            start_indexes, end_indexes = [], []
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            is_impossible = True
            for tok_start_position, tok_end_position in zip(tok_start_positions, tok_end_positions):
                if (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                    start_positions[start_position] = 1
                    end_positions[end_position] = 1
                    start_indexes.append(start_position)
                    end_indexes.append(end_position)
                    is_impossible = False

            if is_impossible:
                start_positions[0] = 1
                end_positions[0] = 1
                start_indexes.append(0)
                end_indexes.append(0)

            if example_index < 2 and verbose_logging:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("doc_span_start: %s" % (doc_span.start))
                if is_impossible:
                    logger.info("impossible example")
                else:
                    logger.info("start_indexes: {}".format(start_indexes))
                    logger.info("end_indexes: {}".format(end_indexes))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    start_indexes=start_indexes,
                    end_indexes=end_indexes,
                    is_impossible=is_impossible))
            unique_id += 1

            if len(features) % 5000 == 0:
                logger.info("Processing features: %d" % (len(features)))

    return features

def annotate_candidates(all_examples, batch_features, batch_results, filter_type, is_training, n_best_size,
                        max_answer_length, do_lower_case, verbose_logging, logger):
    """Annotate top-k candidate answers into features."""
    unique_id_to_result = {}
    for result in batch_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "rank_logit"])

    batch_span_starts, batch_span_ends, batch_hard_labels, batch_soft_labels = [], [], [], []
    for (feature_index, feature) in enumerate(batch_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        prelim_predictions_per_feature = []
        start_indexes = _get_best_indexes(result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)
        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue

                prelim_predictions_per_feature.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index],
                        rank_logit=result.rank_logit))

        prelim_predictions_per_feature = sorted(
            prelim_predictions_per_feature,
            key=lambda x: (x.start_logit + x.end_logit + x.rank_logit),
            reverse=True)

        seen_predictions = {}
        span_starts, span_ends, hard_labels, soft_labels = [], [], [], []

        if is_training:
            # add no-answer option into candidate answers
            span_starts.append(0)
            span_ends.append(0)
            if feature.is_impossible:
                hard_labels.append(1)
                soft_labels.append(1.)
            else:
                hard_labels.append(0)
                soft_labels.append(0.)

        for i, pred_i in enumerate(prelim_predictions_per_feature):
            if len(span_starts) >= int(n_best_size/4):
                break
            tok_tokens = feature.tokens[pred_i.start_index:(pred_i.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred_i.start_index]
            orig_doc_end = feature.token_to_orig_map[pred_i.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging, logger)
            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True

            if is_training:
                if pred_i.start_index != 0 and pred_i.end_index != 0:
                    span_starts.append(pred_i.start_index)
                    span_ends.append(pred_i.end_index)
                    if feature.is_impossible:
                        hard_labels.append(0)
                        soft_labels.append(0.)
                    else:
                        max_em, max_f1 = 0, 0
                        for orig_answer_text in example.orig_answer_texts:
                            em = int(exact_match_score(final_text, orig_answer_text))
                            f1 = float(f1_score(final_text, orig_answer_text))
                            if em > max_em:
                                max_em = em
                            if f1 > max_f1:
                                max_f1 = f1
                        hard_labels.append(max_em)
                        soft_labels.append(max_f1)
            else:
                span_starts.append(pred_i.start_index)
                span_ends.append(pred_i.end_index)

            # filter out redundant candidates
            if (i+1) < len(prelim_predictions_per_feature):
                indexes = []
                for j, pred_j in enumerate(prelim_predictions_per_feature[(i+1):]):
                    if filter_type == 'em':
                        if pred_i.start_index == pred_j.start_index or pred_i.end_index == pred_j.end_index:
                            indexes.append(i + j + 1)
                    elif filter_type == 'f1':
                        if span_f1([pred_i.start_index, pred_i.end_index], [pred_j.start_index, pred_j.end_index]) > 0:
                            indexes.append(i + j + 1)
                    elif filter_type == 'none':
                        indexes = []
                    else:
                        raise Exception
                [prelim_predictions_per_feature.pop(index - k) for k, index in enumerate(indexes)]

        # Pad to fixed length
        while len(span_starts) < int(n_best_size/4):
            span_starts.append(0)
            span_ends.append(0)
            if is_training:
                if feature.is_impossible:
                    hard_labels.append(1)
                    soft_labels.append(1.)
                else:
                    hard_labels.append(0)
                    soft_labels.append(0.)
        assert len(span_starts) == int(n_best_size/4)
        if is_training:
            assert len(hard_labels) == int(n_best_size/4)

        # Add ground truth answer spans if there is no positive label
        if is_training:
            if max(hard_labels) == 0:
                sample_start = random.sample(feature.start_indexes, 1)
                sample_end = random.sample(feature.end_indexes, 1)
                span_starts[-1] = sample_start[0]
                span_ends[-1] = sample_end[0]
                hard_labels[-1] = 1
                soft_labels[-1] = 1.

        batch_span_starts.append(span_starts)
        batch_span_ends.append(span_ends)
        batch_hard_labels.append(hard_labels)
        batch_soft_labels.append(soft_labels)
    return batch_span_starts, batch_span_ends, batch_hard_labels, batch_soft_labels








