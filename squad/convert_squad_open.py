import pickle
import argparse
import numpy as np
from os.path import relpath, join, exists, expanduser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from typing import List, TypeVar, Iterable
from tqdm import tqdm

import bert.tokenization as tokenization
from triviaqa.evidence_corpus import MergeParagraphs
from triviaqa.build_span_corpus import FastNormalizedAnswerDetector
from squad.squad_document_utils import DocumentAndQuestion

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

class SquadOpenExample(object):
    def __init__(self, qas_id, question_text, answer_texts, doc_text):
        self.qas_id = qas_id
        self.question_text = question_text
        self.answer_texts = answer_texts
        self.doc_text = doc_text

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % self.question_text
        s += ", answer_texts: {}".format(self.answer_texts)
        s += ", doc_text: %s" % self.doc_text[:1000]
        return s

def rank(tfidf, questions: List[str], paragraphs: List[str]):
    para_features = tfidf.fit_transform(paragraphs)
    q_features = tfidf.transform(questions)
    scores = pairwise_distances(q_features, para_features, "cosine")
    return scores

def main():
    parse = argparse.ArgumentParser("Pre-tokenize the SQuAD open dev file")
    parse.add_argument("--input_file", type=str, default=join("data", "squad", "squad_dev_open.pkl"))
    # This is slow, using more processes is recommended
    parse.add_argument("--max_tokens", type=int, default=200, help="Number of maximal tokens in each merged paragraph")
    parse.add_argument("--n_to_select", type=int, default=30, help="Number of paragraphs to retrieve")
    parse.add_argument("--sort_passage", type=bool, default=True, help="Sort passage according to order")
    parse.add_argument("--debug", type=bool, default=False, help="Whether to run in debug mode")
    args = parse.parse_args()

    dev_examples = pickle.load(open(args.input_file, 'rb'))

    tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
    splitter = MergeParagraphs(args.max_tokens)
    tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=stop_words)
    detector = FastNormalizedAnswerDetector()

    ir_count, total_doc_length, pruned_doc_length = 0, 0, 0
    out = []
    for example_ix, example in tqdm(enumerate(dev_examples), total=len(dev_examples)):
        paras = [x for x in example.doc_text.split("\n") if len(x) > 0]
        paragraphs = [tokenizer.tokenize(x) for x in paras]
        merged_paragraphs = splitter.merge(paragraphs)

        scores = rank(tfidf, [example.question_text], [" ".join(x) for x in merged_paragraphs])
        para_scores = scores[0]
        para_ranks = np.argsort(para_scores)
        selection = [i for i in para_ranks[:args.n_to_select]]

        if args.sort_passage:
            selection = np.sort(selection)

        doc_tokens = []
        for idx in selection:
            current_para = merged_paragraphs[idx]
            doc_tokens += current_para

        tokenized_answers = [tokenizer.tokenize(x) for x in example.answer_texts]
        detector.set_question(tokenized_answers)
        if len(detector.any_found(doc_tokens)) > 0:
            ir_count += 1

        total_doc_length += sum(len(para) for para in merged_paragraphs)
        pruned_doc_length += len(doc_tokens)

        out.append(DocumentAndQuestion(example_ix, example.qas_id, example.question_text, doc_tokens,
                                       '', 0, 0, True))
        if args.debug and example_ix > 5:
            break
    print("Recall of answer existence in documents: {:.3f}".format(ir_count / len(out)))
    print("Average length of documents: {:.3f}".format(total_doc_length / len(out)))
    print("Average pruned length of documents: {:.3f}".format(pruned_doc_length / len(out)))
    output_file = join("data", "squad", "eval_open_{}paras_examples.pkl".format(args.n_to_select))
    pickle.dump(out, open(output_file, 'wb'))

if __name__ == "__main__":
    main()