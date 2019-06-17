import json
import random
import math
import collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from typing import List, TypeVar, Iterable

import bert.tokenization as tokenization
from squad.squad_utils import _improve_answer_span, _get_best_indexes, get_final_text
from squad.squad_evaluate import exact_match_score, f1_score, span_f1

T = TypeVar('T')

stop_words = {'t', '–', 'there', 'but', 'needn', 'themselves', '’', '~', '$', 'few', '^', '₹', ']', 'we', 're',
              'again', '?', 'they', 'ain', 'o', 'you', '+', 'has', 'by', 'than', 'whom', 'same', 'don', 'her',
              'are', '(', 'an', 'so', 'the', 'been', 'wouldn', 'a', 'many', 'she', 'how', 'your', '°', 'do',
              'shan', 'himself', 'between', 'ours', 'at', 'should', 'doesn', 'hasn', 'he', 'have', 'over',
              'hadn', 'was', 'weren', 'down', 'above', '_', 'those', 'not', 'having', 'its', 'ourselves',
              'for', 'when', 'if', ',', ';', 'about', 'theirs', 'him', '}', 'here', 'any', 'own', 'itself',
              'very', 'on', 'myself', 'mustn', ')', 'because', 'now', '/', 'isn', 'to', 'just', 'these',
              'i', 'further', 'mightn', 'll', '@', 'am', '”', 'below', 'shouldn', 'my', 'who', 'yours', 'why',
              'such', '"', 'does', 'did', 'before', 'being', 'and', 'had', 'aren', '£', 'with', 'more', 'into',
              '<', 'herself', 'which', '[', "'", 'of', 'haven', 'that', 'will', 'yourself', 'in', 'doing', '−',
              'them', '‘', 'some', '`', 'while', 'each', 'it', 'through', 'all', 'their', ':', '\\', 'where',
              'both', 'hers', '¢', '—', 'm', '.', 'from', 'or', 'other', 'too', 'couldn', 'as', 'our', 'off',
              '%', '&', '-', '{', '=', 'didn', 'yourselves', 'under', 'y', 'ma', 'won', '!', '|', 'against',
              '#', '¥', 'is', 'nor', 'up', 'most', 's', 'no', 'can', '>', '*', 'during', 'once', 'what', 'me',
              'then', 'd', 'only', 'de', 've', 'were', '€', 'until', 'his', 'out', 'wasn', 'this', 'after',
              'be'}


def flatten_iterable(listoflists: Iterable[Iterable[T]]) -> List[T]:
    return [item for sublist in listoflists for item in sublist]


class Question(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class Paragraph(object):
    def __init__(self,
                 paragraph_id,
                 paragraph_text,
                 doc_tokens,
                 questions: List[Question]):
        self.paragraph_id = paragraph_id
        self.paragraph_text = paragraph_text
        self.doc_tokens = doc_tokens
        self.questions = questions

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "paragraph_id: %s" % (self.paragraph_id)
        s += ", qas_num: %s" % (len(self.questions))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens[:20]))
        return s


class Document(object):
    def __init__(self, document_id: str, paragraphs: List[Paragraph]):
        self.document_id = document_id
        self.paragraphs = paragraphs

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "document_id: %s" % (self.document_id)
        s += ", paragraph_num: %s" % (len(self.paragraphs))
        return s


class DocumentAndQuestion(object):
    def __init__(self,
                 document_id,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.document_id = document_id
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "document_id: %s" % (self.document_id)
        s += ", qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens[:20]))
        s += ", length of doc_tokens: [%d]" % (len(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
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
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "unique_id: %s" % (self.unique_id)
        s += ", \nexample_index: {}".format(self.example_index)
        s += ", \ndoc_span_index: {}".format(self.doc_span_index)
        s += ", \ntokens: {}".format(self.tokens)
        if self.start_position:
            s += ", \nstart_position: {}".format(self.start_position)
        if self.end_position:
            s += ", \nend_position: {}".format(self.end_position)
        if self.is_impossible:
            s += ", \nis_impossible: {}".format(self.is_impossible)
        return s


def read_squad_doc_examples(input_file, logger, debug=False):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    documents = []
    for article_ix, article in enumerate(input_data):
        document_id = "%s-%d" % (article['title'], article_ix)
        paragraphs = []
        for paragraph_ix, paragraph in enumerate(article["paragraphs"]):
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)  # add a new word
                    else:
                        doc_tokens[-1] += c  # add a new character
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            questions = []
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                if len(qa["answers"]) == 0:
                    continue
                answer = qa["answers"][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length - 1]
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(
                    tokenization.whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    logger.warning("Could not find answer: '%s' vs. '%s'",
                                   actual_text, cleaned_answer_text)
                    continue

                question = Question(
                    qas_id=qas_id,
                    question_text=question_text,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                questions.append(question)
            paragraphs.append(Paragraph(paragraph_ix, paragraph_text, doc_tokens, questions))
        documents.append(Document(document_id, paragraphs))
        if debug:
            break
    return documents


class SquadTfIdfRanker(object):
    """
    TF-IDF ranking for SQuAD, this does the same thing as `TopTfIdf`, but its supports efficient usage
    when have many many questions per document
    """

    def __init__(self, tokenizer, n_to_select: int, is_training: bool, sort_passage: bool):
        self.tokenizer = tokenizer
        self.n_to_select = n_to_select
        self.is_training = is_training
        self.sort_passage = sort_passage
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=stop_words)

    def rank(self, questions: List[str], paragraphs: List[str]):
        tfidf = self._tfidf
        para_features = tfidf.fit_transform(paragraphs)
        q_features = tfidf.transform(questions)
        scores = pairwise_distances(q_features, para_features, "cosine")
        return scores

    def ranked_questions(self, docs: List[Document]):
        ir_count, total_doc_length, pruned_doc_length = 0, 0, 0
        out = []
        for doc in docs:  # 48
            scores = self.rank(flatten_iterable([q.question_text for q in x.questions] for x in doc.paragraphs),
                               [x.paragraph_text for x in doc.paragraphs])  # [269, 55]
            total_doc_length += sum(len(x.doc_tokens) for x in doc.paragraphs)
            q_ix = 0
            for para_ix, para in enumerate(doc.paragraphs):
                for q in para.questions:
                    para_scores = scores[q_ix]  # 55
                    para_ranks = np.argsort(para_scores)
                    selection = [i for i in para_ranks[:self.n_to_select]]
                    rank = [i for i in np.arange(self.n_to_select)]

                    if para_ix in selection:
                        ir_count += 1

                    # force to add ground truth passage into the document
                    if self.is_training and para_ix not in selection:
                        selection[-1] = para_ix

                    if self.sort_passage:
                        dictionary = dict(zip(selection, rank))
                        selection = np.sort(selection)
                        rank = [dictionary[key] for key in selection]

                    doc_tokens = []
                    ans_exist = False
                    start_position, end_position = 0, 0
                    for ix, (selection_ix, rank_ix) in enumerate(zip(selection, rank)):
                        offset_doc = len(doc_tokens)

                        current_para = doc.paragraphs[selection_ix]
                        doc_tokens += current_para.doc_tokens

                        if selection_ix == para_ix:
                            assert q.orig_answer_text != None and q.start_position != None and q.end_position != None
                            ans_exist = True
                            start_position = q.start_position + offset_doc
                            end_position = q.end_position + offset_doc

                    pruned_doc_length += len(doc_tokens)

                    if ans_exist:
                        out.append(DocumentAndQuestion(doc.document_id, q.qas_id, q.question_text, doc_tokens,
                                                       q.orig_answer_text, start_position, end_position, False))
                    else:
                        out.append(DocumentAndQuestion(doc.document_id, q.qas_id, q.question_text, doc_tokens,
                                                       '', 0, 0, True))
                    q_ix += 1
        print("Recall of answer existence in documents: {:.3f}".format(ir_count / len(out)))
        print("Average length of documents: {:.3f}".format(total_doc_length / len(docs)))
        print("Average pruned length of documents: {:.3f}".format(pruned_doc_length / len(out)))
        return out


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

        if example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        else:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

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

            if example.is_impossible:
                out_of_span = True
                start_position = 0
                end_position = 0
            else:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if example_index < 2 and verbose_logging:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                # logger.info("tokens: %s" % " ".join(
                #     [tokenization.printable_text(x) for x in tokens]))
                # logger.info("token_to_orig_map: %s" % " ".join(
                #     ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                # ]))
                # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                # logger.info(
                #     "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                # logger.info(
                #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if example.is_impossible:
                    logger.info("impossible example")
                else:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info("answer: %s" % (tokenization.printable_text(answer_text)))

            # For classification
            is_impossible = 1 if start_position == 0 and end_position == 0 else 0

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
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible))
            unique_id += 1

            if len(features) % 5000 == 0:
                logger.info("Processing features: %d" % (len(features)))

    return features

RawRankResult = collections.namedtuple("RawRankResult", ["unique_id", "rank_logit"])

RawReadResult = collections.namedtuple("RawReadResult",
                                       ["unique_id", "start_logits", "end_logits", "rank_logit"])

RawFinalResult = collections.namedtuple("RawFinalResult",
                                        ["unique_id", "start_logits", "end_logits", "rank_logit",
                                         "rerank_logits", "start_indexes", "end_indexes"])

class PrelimRankPrediction(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 is_impossible,
                 rank_logit):
        self.unique_id = unique_id
        self.is_impossible = is_impossible
        self.rank_logit = rank_logit

def eval_ranking(force_answer, n_best_size, all_examples, all_features, all_results):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    map, mrr, top_1, top_3, top_5, top_7 = 0., 0., 0., 0., 0., 0.
    num_valid, total_features = 0, 0
    prelim_predictions = []
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        f_id, y_pred, y_true = [], [], []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            f_id.append(feature.unique_id)
            y_pred.append(result.rank_logit)
            y_true.append(not feature.is_impossible)

        map += eval_map(y_true=y_true, y_pred=y_pred)
        mrr += eval_mrr(y_true=y_true, y_pred=y_pred)
        top_1 += eval_recall(y_true=y_true, y_pred=y_pred, k=1)
        top_3 += eval_recall(y_true=y_true, y_pred=y_pred, k=3)
        top_5 += eval_recall(y_true=y_true, y_pred=y_pred, k=5)
        top_7 += eval_recall(y_true=y_true, y_pred=y_pred, k=7)
        num_valid += 1

        n_best_indexes = _get_best_indexes_with_force(force_answer, n_best_size, y_pred, y_true)
        for n_best_index in n_best_indexes:
            prelim_predictions.append(
                PrelimRankPrediction(
                    unique_id=f_id[n_best_index],
                    is_impossible=not y_true[n_best_index],
                    rank_logit=y_pred[n_best_index]))

        total_features += len(features)

    metrics = {
        "map": map / num_valid,
        "mrr": mrr / num_valid,
        "top_1": top_1 / num_valid,
        "top_3": top_3 / num_valid,
        "top_5": top_5 / num_valid,
        "top_7": top_7 / num_valid,
        "retrieval_rate": len(prelim_predictions) / total_features
    }
    return metrics, prelim_predictions


def _get_best_indexes_with_force(force_answer, n_best_size, logits, labels):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])

    if force_answer:
        pos_num = min(sum(labels), math.ceil(n_best_size / 4))
        if pos_num == 0:
            pos_num = 1
        # if there is no positive example in top-k examples, then force to add
        if not max([labels[index] for index in best_indexes]):
            j = 0
            for i in range(len(index_and_score)):
                if j >= pos_num:
                    break
                index = index_and_score[i][0]
                if labels[index]:
                    best_indexes[-1-j] = index
                    j += 1

    return best_indexes


def eval_map(y_true, y_pred, rel_threshold=0):
    assert y_true != [] and y_pred != []
    s = 0.
    if len(y_true) == 1 and len(y_pred) == 1:
        if y_true[0] > rel_threshold:
            s = 1.
    else:
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        ipos = 0
        for j, (g, p) in enumerate(c):
            if g > rel_threshold:
                ipos += 1.
                s += ipos / (j + 1.)
        if ipos == 0:
            s = 0.
        else:
            s /= ipos
    return s

def eval_mrr(y_true, y_pred, rel_threshold=0):
    assert y_true != [] and y_pred != []
    s = 0.
    if len(y_true) == 1 and len(y_pred) == 1:
        if y_true[0] > rel_threshold:
            s = 1.
    else:
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        for j, (g, p) in enumerate(c):
            if g > rel_threshold:
                s = 1. / ( j + 1.)
                break
    return s

def eval_recall(y_true, y_pred, k=10, rel_threshold=0.):
    assert y_true != [] and y_pred != []
    assert k > 0
    recall = 0.
    if len(y_true) == 1 and len(y_pred) == 1:
        if y_true[0] > rel_threshold:
            recall = 1.
    else:
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                recall = 1.
                break
    return recall

def random_filter_features(all_examples, all_features, n_best_size, is_training):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    filtered_features = []
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        if is_training:
            positive_feature_indexes, negative_feature_indexes = [], []
            for (feature_index, feature) in enumerate(features):
                if feature.is_impossible:
                    negative_feature_indexes.append(feature_index)
                else:
                    positive_feature_indexes.append(feature_index)

            negative_sample_size = min(len(negative_feature_indexes), n_best_size - 1)
            negative_feature_indexes = random.sample(negative_feature_indexes, negative_sample_size)
            positive_feature_indexes = random.sample(positive_feature_indexes, 1)

            for feature_index in negative_feature_indexes + positive_feature_indexes:
                feature = features[feature_index]
                filtered_features.append(feature)
        else:
            sample_size = min(len(features), n_best_size)
            sampled_features = random.sample(features, sample_size)
            filtered_features += sampled_features
    return filtered_features

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, verbose_logging, logger):
    """Write final predictions to the json file."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "rank_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

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

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            rank_logit=result.rank_logit))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit + x.rank_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "rank_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
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
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    rank_logit=pred.rank_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, rank_logit=0.))

        assert len(nbest) >= 1

        span_scores = []
        rank_scores = []
        for entry in nbest:
            span_scores.append(entry.start_logit + entry.end_logit)
            rank_scores.append(entry.rank_logit)

        final_scores = [span_score + rank_score for span_score, rank_score
                        in zip(span_scores, rank_scores)]
        nbest_indexes = np.argsort(final_scores)[::-1]

        nbest_json = []
        for index in nbest_indexes:
            entry = nbest[index]
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["final_score"] = final_scores[index]
            output["span_score"] = span_scores[index]
            output["rank_score"] = rank_scores[index]
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json

def write_rerank_predictions(all_examples, all_features, all_results, length_heuristic, rank_weight, rerank_weight, ablate_type,
                             n_best_size, max_answer_length, do_lower_case, verbose_logging, logger):
    """Write final predictions to the json file."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "rank_logit", "rerank_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

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

                    rerank_logit = 0.
                    for i, (candidate_start, candidate_end) in enumerate(zip(result.start_indexes, result.end_indexes)):
                        if start_index == candidate_start and end_index == candidate_end:
                            rerank_logit = float(result.rerank_logits[i])
                            break

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            rank_logit=result.rank_logit,
                            rerank_logit=rerank_logit))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit + x.rank_logit + x.rerank_logit - length_heuristic*(x.end_index - x.start_index + 1)),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_index", "end_index", "start_logit", "end_logit", "rank_logit", "rerank_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
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
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_index=pred.start_index,
                    end_index=pred.end_index,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    rank_logit=pred.rank_logit,
                    rerank_logit=pred.rerank_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_index=0.0, end_index=0.0, start_logit=0.0, end_logit=0.0, rank_logit=0., rerank_logit=0.))

        assert len(nbest) >= 1

        span_scores = []
        rank_scores = []
        rerank_scores = []
        for entry in nbest:
            span_scores.append(entry.start_logit + entry.end_logit - length_heuristic*(entry.end_index - entry.start_index + 1))
            rank_scores.append(entry.rank_logit)
            rerank_scores.append(entry.rerank_logit)

        if ablate_type == 'rank':
            final_scores = [span_score + rerank_weight * rerank_score for span_score, _, rerank_score
                            in zip(span_scores, rank_scores, rerank_scores)]
        elif ablate_type == 'rerank':
            final_scores = [span_score + rank_weight * rank_score for span_score, rank_score, _
                            in zip(span_scores, rank_scores, rerank_scores)]
        elif ablate_type == 'both':
            final_scores = [span_score for span_score, _, _
                            in zip(span_scores, rank_scores, rerank_scores)]
        elif ablate_type == 'none':
            final_scores = [span_score + rank_weight * rank_score + rerank_weight * rerank_score
                            for span_score, rank_score, rerank_score in zip(span_scores, rank_scores, rerank_scores)]
        else:
            raise Exception

        nbest_indexes = np.argsort(final_scores)[::-1]

        nbest_json = []
        for index in nbest_indexes:
            entry = nbest[index]
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["final_score"] = final_scores[index]
            output["span_score"] = span_scores[index]
            output["rank_score"] = rank_scores[index]
            output["rerank_score"] = rerank_scores[index]
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json

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
                        hard_labels.append(int(exact_match_score(final_text, example.orig_answer_text)))
                        soft_labels.append(float(f1_score(final_text, example.orig_answer_text)))
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
                span_starts[-1] = feature.start_position
                span_ends[-1] = feature.end_position
                hard_labels[-1] = 1
                soft_labels[-1] = 1.

        batch_span_starts.append(span_starts)
        batch_span_ends.append(span_ends)
        batch_hard_labels.append(hard_labels)
        batch_soft_labels.append(soft_labels)
    return batch_span_starts, batch_span_ends, batch_hard_labels, batch_soft_labels