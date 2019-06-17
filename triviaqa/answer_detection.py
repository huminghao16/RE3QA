import re
import string

import numpy as np
from tqdm import tqdm
from typing import List

from triviaqa.read_data import TriviaQaQuestion
from triviaqa.triviaqa_eval import normalize_answer, f1_score
from triviaqa.utils import flatten_iterable, split


class FastNormalizedAnswerDetector(object):
    """ almost twice as fast and very,very close to NormalizedAnswerDetector's output """

    def __init__(self):
        # These come from the TrivaQA official evaluation script
        self.skip = {"a", "an", "the", ""}
        self.strip = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])

        self.answer_tokens = None

    def set_question(self, normalized_aliases):
        self.answer_tokens = normalized_aliases

    def any_found(self, para):  # List[str]
        # Normalize the paragraph
        words = [w.lower().strip(self.strip) for w in para]
        occurances = []
        for answer_ix, answer in enumerate(self.answer_tokens):
            # Locations where the first word occurs
            if len(answer) == 0:
                continue
            word_starts = [i for i, w in enumerate(words) if answer[0] == w]    # [12, 50, 63 ...]
            n_tokens = len(answer)  # 2

            # Advance forward until we find all the words, skipping over articles
            for start in word_starts:
                end = start + 1
                ans_token = 1
                while ans_token < n_tokens and end < len(words):
                    next = words[end]
                    if answer[ans_token] == next:
                        ans_token += 1
                        end += 1
                    elif next in self.skip:
                        end += 1
                    else:
                        break
                if n_tokens == ans_token:
                    occurances.append((start, end))
        return list(set(occurances))


def compute_answer_spans(questions: List[TriviaQaQuestion], corpus, tokenizer,
                         detector):

    for i, q in enumerate(questions):
        if i % 500 == 0:
            print("Completed question %d of %d (%.3f)" % (i, len(questions), i/len(questions)))
        q.question = tokenizer.tokenize(q.question)
        if q.answer is None:
            continue
        tokenized_aliases = [tokenizer.tokenize(x) for x in q.answer.all_answers]
        if len(tokenized_aliases) == 0:
            raise ValueError()
        detector.set_question(tokenized_aliases)
        for doc in q.all_docs:
            text = corpus.get_document(doc.doc_id)  # List[List[str]]
            if text is None:
                raise ValueError()
            spans = []
            offset = 0
            for para_ix, para in enumerate(text):
                for s, e in detector.any_found(para):
                    spans.append((s+offset, e+offset-1))  # turn into inclusive span
                offset += len(para)
            if len(spans) == 0:
                spans = np.zeros((0, 2), dtype=np.int32)
            else:
                spans = np.array(spans, dtype=np.int32)
            doc.answer_spans = spans


def _compute_answer_spans_chunk(questions, corpus, tokenizer, detector):
    compute_answer_spans(questions, corpus, tokenizer, detector)
    return questions


def compute_answer_spans_par(questions: List[TriviaQaQuestion], corpus,
                             tokenizer, detector, n_processes: int):
    if n_processes == 1:
        compute_answer_spans(questions, corpus, tokenizer, detector)
        return questions
    from multiprocessing import Pool
    with Pool(n_processes) as p:
        chunks = split(questions, n_processes)
        questions = flatten_iterable(p.starmap(_compute_answer_spans_chunk,
                                               [[c, corpus, tokenizer, detector] for c in chunks]))
        return questions