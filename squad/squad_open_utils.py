import json
import random
import math
import collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from typing import List, TypeVar, Iterable

import bert.tokenization as tokenization

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
                 doc_index,
                 para_index,
                 question_text,
                 answer_texts=None):
        self.qas_id = qas_id
        self.doc_index = doc_index
        self.para_index = para_index
        self.question_text = question_text
        self.answer_texts = answer_texts

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += "doc_index: %d" % (self.doc_index)
        s += "para_index: %d" % (self.para_index)
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        if self.answer_texts is not None:
            s += ", answer_texts: ".format(self.answer_texts)
        return s


class Paragraph(object):
    def __init__(self,
                 paragraph_id,
                 paragraph_text):
        self.paragraph_id = paragraph_id
        self.paragraph_text = paragraph_text

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "paragraph_id: %s" % (self.paragraph_id)
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

    def get_doc_text(self):
        all_doc_text = ''
        for idx, para in enumerate(self.paragraphs):
            if idx == 0:
                all_doc_text += para.paragraph_text
            else:
                all_doc_text += ' '
                all_doc_text += para.paragraph_text
        return all_doc_text

def tfidf_rank(questions: List[str], documents: List[str]):
    tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=stop_words)
    doc_features = tfidf.fit_transform(documents)
    q_features = tfidf.transform(questions)
    scores = pairwise_distances(q_features, doc_features, "cosine")
    return scores

def read_squad_open_examples(input_file, n_to_select, is_training, debug=False):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    documents = []
    questions = []
    for article_ix, article in enumerate(input_data):
        document_id = "%s-%d" % (article['title'], article_ix)
        paragraphs = []
        for paragraph_ix, paragraph in enumerate(article["paragraphs"]):
            paragraph_text = paragraph["context"]
            paragraphs.append(Paragraph(paragraph_ix, paragraph_text))

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                answer_texts = []
                for answer in qa["answers"]:
                    answer_texts.append(answer["text"])
                questions.append(Question(qas_id, article_ix, paragraph_ix, question_text, answer_texts))

        documents.append(Document(document_id, paragraphs))
        if (article_ix+1) == 10 and debug:
            break

    scores = tfidf_rank([x.question_text for x in questions], [x.get_doc_text() for x in documents])    # [1177, 3]

    ir_count = 0
    for que_ix, question in enumerate(questions):
        doc_scores = scores[que_ix]
        doc_ranks = np.argsort(doc_scores)
        selection = [i for i in doc_ranks[:n_to_select]]
        rank = [i for i in np.arange(n_to_select)]

        if question.doc_index in selection:
            ir_count += 1

        if is_training and question.doc_index not in selection:
            selection[-1] = question.doc_index

    print("Retrieve {} questions from {} documents".format(len(questions), len(documents)))
    print("Recall of answer existence in documents: {:.3f}".format(ir_count / len(questions)))

read_squad_open_examples("../data/squad/dev-v1.1.json", 5, False, False)