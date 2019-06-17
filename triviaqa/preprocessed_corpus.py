import sys
import gzip
import random
import pickle
from collections import Counter
from threading import Lock
from typing import List, Iterable, Optional

import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from triviaqa.utils import split, flatten_iterable, group
from triviaqa.configurable import Configurable
from triviaqa.read_data import TriviaQaQuestion
from triviaqa.triviaqa_document_utils import ExtractedParagraphWithAnswers, DocParagraphWithAnswers, DocumentAndQuestion

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

class ParagraphsSet(object):
    def __init__(self, paragraphs: List[ExtractedParagraphWithAnswers], ir_hit: bool):
        self.paragraphs = paragraphs
        self.ir_hit = ir_hit

class ParagraphFilter(Configurable):
    """ Selects and ranks paragraphs """

    def prune(self, question, paragraphs: List[ExtractedParagraphWithAnswers]):
        raise NotImplementedError()

class TopTfIdf(ParagraphFilter):
    def __init__(self, n_to_select: int, is_training: bool=False, sort_passage: bool=True):
        self.n_to_select = n_to_select
        self.is_training = is_training
        self.sort_passage = sort_passage

    def prune(self, question: List[str], paragraphs: List[ExtractedParagraphWithAnswers]):
        tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=stop_words)
        text = []
        for para in paragraphs:
            text.append(" ".join(para.text))
        try:
            para_features = tfidf.fit_transform(text)
            q_features = tfidf.transform([" ".join(question)])
        except ValueError:
            return []

        dists = pairwise_distances(q_features, para_features, "cosine").ravel() # [N]
        sorted_ix = np.lexsort(([x.start for x in paragraphs], dists))  # in case of ties, use the earlier paragraph [N]

        selection = [i for i in sorted_ix[:self.n_to_select]]
        selected_paras = [paragraphs[i] for i in selection]
        ir_hit = 0. if all(len(x.answer_spans) == 0 for x in selected_paras) else 1.

        if self.is_training and not ir_hit:
            gold_indexes = [i for i, x in enumerate(paragraphs) if len(x.answer_spans) != 0]
            gold_index = random.choice(gold_indexes)
            selection[-1] = gold_index

        if self.sort_passage:
            selection = np.sort(selection)

        return [paragraphs[i] for i in selection], ir_hit

class ShallowOpenWebRanker(ParagraphFilter):
    # Hard coded weight learned from a logistic regression classifier
    TFIDF_W = 5.13365065
    LOG_WORD_START_W = 0.46022765
    FIRST_W = -0.08611607
    LOWER_WORD_W = 0.0499123
    WORD_W = -0.15537181

    def __init__(self, n_to_select: int, is_training: bool=False, sort_passage: bool=True):
        self.n_to_select = n_to_select
        self.is_training = is_training
        self.sort_passage = sort_passage
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=stop_words)

    def score_paragraphs(self, question, paragraphs: List[ExtractedParagraphWithAnswers]):
        tfidf = self._tfidf
        text = []
        for para in paragraphs:
            text.append(" ".join(para.text))
        try:
            para_features = tfidf.fit_transform(text)
            q_features = tfidf.transform([" ".join(question)])
        except ValueError:
            return []

        q_words = {x for x in question if x.lower() not in stop_words}
        q_words_lower = {x.lower() for x in q_words}
        word_matches_features = np.zeros((len(paragraphs), 2))
        for para_ix, para in enumerate(paragraphs):
            found = set()
            found_lower = set()
            for word in para.text:
                if word in q_words:
                    found.add(word)
                elif word.lower() in q_words_lower:
                    found_lower.add(word.lower())
            word_matches_features[para_ix, 0] = len(found)
            word_matches_features[para_ix, 1] = len(found_lower)

        tfidf = pairwise_distances(q_features, para_features, "cosine").ravel()
        starts = np.array([p.start for p in paragraphs])
        log_word_start = np.log(starts / 200.0 + 1)
        first = starts == 0
        scores = tfidf * self.TFIDF_W + self.LOG_WORD_START_W * log_word_start + self.FIRST_W * first + \
                 self.LOWER_WORD_W * word_matches_features[:, 1] + self.WORD_W * word_matches_features[:, 0]
        return scores

    def prune(self, question: List[str], paragraphs: List[ExtractedParagraphWithAnswers]):
        scores = self.score_paragraphs(question, paragraphs)
        sorted_ix = np.argsort(scores)

        selection = [i for i in sorted_ix[:self.n_to_select]]
        selected_paras = [paragraphs[i] for i in selection]
        ir_hit = 0. if all(len(x.answer_spans) == 0 for x in selected_paras) else 1.

        if self.is_training and not ir_hit:
            gold_indexes = [i for i, x in enumerate(paragraphs) if len(x.answer_spans) != 0]
            gold_index = random.choice(gold_indexes)
            selection[-1] = gold_index

        if self.sort_passage:
            selection = np.sort(selection)

        return [paragraphs[i] for i in selection], ir_hit


class Preprocessor(Configurable):

    def preprocess(self, question: Iterable, evidence) -> object:
        """ Map elements to an unspecified intermediate format """
        raise NotImplementedError()

    def finalize_chunk(self, x):
        """ Finalize the output from `preprocess`, in multi-processing senarios this will still be run on
         the main thread so it can be used for things like interning """
        pass

def _preprocess_and_count(questions: List, evidence, preprocessor: Preprocessor):
    count = len(questions)
    output = preprocessor.preprocess(questions, evidence)
    return output, count

def preprocess_par(questions: List, evidence, preprocessor,
                   n_processes=2, chunk_size=200, name=None):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be >= 0, but got %s" % chunk_size)
    if n_processes is not None and n_processes <= 0:
        raise ValueError("n_processes must be >= 1 or None, but got %s" % n_processes)
    n_processes = min(len(questions), n_processes)

    if n_processes == 1:
        out = preprocessor.preprocess(tqdm(questions, desc=name, ncols=80), evidence)
        preprocessor.finalize_chunk(out)
        return out
    else:
        from multiprocessing import Pool
        chunks = split(questions, n_processes)
        chunks = flatten_iterable([group(c, chunk_size) for c in chunks])
        print("Processing %d chunks with %d processes" % (len(chunks), n_processes))
        pbar = tqdm(total=len(questions), desc=name, ncols=80)
        lock = Lock()

        def call_back(results):
            preprocessor.finalize_chunk(results[0])
            with lock:  # FIXME Even with the lock, the progress bar still is jumping around
                pbar.update(results[1])

        with Pool(n_processes) as pool:
            results = [pool.apply_async(_preprocess_and_count, [c, evidence, preprocessor], callback=call_back)
                       for c in chunks]
            results = [r.get()[0] for r in results]

        pbar.close()
        output = results[0]
        for r in results[1:]:
            output += r
        return output

class FilteredData(object):
    def __init__(self, data: List, true_len: int, ir_count: int,
                 total_doc_num: int, total_doc_length: int, pruned_doc_length: int):
        self.data = data
        self.true_len = true_len
        self.ir_count = ir_count
        self.total_doc_num = total_doc_num
        self.total_doc_length = total_doc_length
        self.pruned_doc_length = pruned_doc_length

    def __add__(self, other):
        return FilteredData(self.data + other.data, self.true_len + other.true_len, self.ir_count + other.ir_count,
                            self.total_doc_num + other.total_doc_num, self.total_doc_length + other.total_doc_length,
                            self.pruned_doc_length + other.pruned_doc_length)

def split_annotated(doc: List[List[str]], spans: np.ndarray):
    out = []
    offset = 0
    for para in doc:
        para_start = offset
        para_end = para_start + len(para)
        para_spans = spans[np.logical_and(spans[:, 0] >= para_start, spans[:, 1] < para_end)] - para_start
        out.append(ExtractedParagraphWithAnswers(para, para_start, para_end, para_spans))
        offset += len(para)
    return out

class ExtractMultiParagraphsPerQuestion(Preprocessor):
    def __init__(self, ranker: ParagraphFilter, intern: bool=False, is_training=False):
        self.ranker = ranker
        self.intern = intern
        self.is_training = is_training

    def preprocess(self, questions: List[TriviaQaQuestion], evidence):  # TriviaQaEvidenceCorpusTxt evidence_corpus.py
        ir_count, total_doc_num, total_doc_length, pruned_doc_length = 0, 0, 0, 0

        instances = []
        for q in questions:
            doc_paras = []
            doc_count, doc_length = 0, 0
            for doc in q.all_docs:
                if self.is_training and len(doc.answer_spans) == 0:
                    continue
                text = evidence.get_document(doc.doc_id)  # List[List[str]]
                if text is None:
                    raise ValueError("No evidence text found document: " + doc.doc_id)
                if doc.answer_spans is not None:
                    paras = split_annotated(text, doc.answer_spans)
                else:
                    # this is kind of a hack to make the rest of the pipeline work, only
                    # needed for test cases
                    paras = split_annotated(text, np.zeros((0, 2), dtype=np.int32))
                doc_paras.extend([DocParagraphWithAnswers(x.text, x.start, x.end, x.answer_spans, doc.doc_id)
                                 for x in paras])  # List[DocParagraphWithAnswers]
                doc_length += sum(len(para) for para in text)
                doc_count += 1

            if len(doc_paras) == 0:
                continue

            doc_paras, ir_hit = self.ranker.prune(q.question, doc_paras)    # List[ExtractedParagraphWithAnswers] len=4
            total_doc_num += doc_count
            total_doc_length += doc_length
            ir_count += ir_hit

            # merge into documentandquestion
            doc_tokens, start_positions, end_positions = [], [], []
            for x in doc_paras:
                offset_doc = len(doc_tokens)
                doc_tokens += x.text
                if len(x.answer_spans) != 0:
                    start_position = x.answer_spans[:, 0] + offset_doc
                    end_position = x.answer_spans[:, 1] + offset_doc
                    start_positions.extend(start_position)
                    end_positions.extend(end_position)
            instance = DocumentAndQuestion(q.all_docs[0].doc_id, q.question_id, " ".join(q.question), doc_tokens,
                                           None if q.answer is None else q.answer.all_answers, start_positions,
                                           end_positions)
            pruned_doc_length += len(doc_tokens)

            instances.append(instance)
        return FilteredData(instances, len(questions), ir_count, total_doc_num, total_doc_length, pruned_doc_length)

    def finalize_chunk(self, f: FilteredData):
        if self.intern:
            for ins in f.data:
                ins.document_id = sys.intern(ins.document_id)
                ins.qas_id = sys.intern(ins.qas_id)
                ins.question_text = sys.intern(ins.question_text)


# class ExtractMultiParagraphs(Preprocessor):
#     def __init__(self, ranker: ParagraphFilter, intern: bool=False, is_training=False):
#         self.ranker = ranker
#         self.intern = intern
#         self.is_training = is_training
#
#     def preprocess(self, questions: List[TriviaQaQuestion], evidence):  # TriviaQaEvidenceCorpusTxt evidence_corpus.py
#         true_len = 0
#         ir_count, ir_total, pruned_doc_length = 0, 0, 0
#
#         instances = []
#         for q in questions:
#             true_len += len(q.all_docs)
#             for doc in q.all_docs:
#                 if self.is_training and len(doc.answer_spans) == 0:
#                     continue
#                 text = evidence.get_document(doc.doc_id)  # List[List[str]]
#                 if text is None:
#                     raise ValueError("No evidence text found document: " + doc.doc_id)
#                 if doc.answer_spans is not None:
#                     paras = split_annotated(text, doc.answer_spans)
#                 else:
#                     # this is kind of a hack to make the rest of the pipeline work, only
#                     # needed for test cases
#                     paras = split_annotated(text, np.zeros((0, 2), dtype=np.int32))
#
#                 if len(paras) == 0:
#                     continue
#
#                 paras, ir_hit = self.ranker.prune(q.question, paras)    # List[ExtractedParagraphWithAnswers] len=4
#                 ir_count += ir_hit
#                 ir_total += 1
#
#                 # merge into documentandquestion
#                 doc_tokens, start_positions, end_positions = [], [], []
#                 for x in paras:
#                     offset_doc = len(doc_tokens)
#                     doc_tokens += x.text
#                     if len(x.answer_spans) != 0:
#                         start_position = x.answer_spans[:, 0] + offset_doc
#                         end_position = x.answer_spans[:, 1] + offset_doc
#                         start_positions.extend(start_position)
#                         end_positions.extend(end_position)
#                 instance = DocumentAndQuestion(doc.doc_id, q.question_id, " ".join(q.question), doc_tokens,
#                                                q.answer.all_answers, start_positions, end_positions)
#                 pruned_doc_length += len(doc_tokens)
#
#                 instances.append(instance)
#         return FilteredData(instances, true_len, ir_count, ir_total, pruned_doc_length)
#
#     def finalize_chunk(self, f: FilteredData):
#         if self.intern:
#             for ins in f.data:
#                 ins.document_id = sys.intern(ins.document_id)
#                 ins.qas_id = sys.intern(ins.qas_id)
#                 ins.question_text = sys.intern(ins.question_text)