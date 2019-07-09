import argparse
import json
import pickle
import unicodedata
from itertools import islice
from typing import List, Optional, Dict
from os import mkdir
from os.path import join, exists
import bert.tokenization as tokenization
from triviaqa.configurable import Configurable
from triviaqa.read_data import iter_trivia_question, TriviaQaQuestion
from triviaqa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from triviaqa.answer_detection import compute_answer_spans_par, FastNormalizedAnswerDetector

TRIVIA_QA = join("data", "triviaqa")


def build_dataset(name: str, tokenizer, train_files: Dict[str, str],
                  answer_detector, n_process: int, prune_unmapped_docs=True,
                  sample=None):
    out_dir = join("data", "triviaqa", name)
    if not exists(out_dir):
        mkdir(out_dir)

    file_map = {}  # maps document_id -> filename

    for name, filename in train_files.items():
        print("Loading %s questions" % name)
        if sample is None:
            questions = list(iter_trivia_question(filename, file_map, False))
        else:
            if isinstance(sample,  int):
                questions = list(islice(iter_trivia_question(filename, file_map, False), sample))
            elif isinstance(sample, dict):
                questions = list(islice(iter_trivia_question(filename, file_map, False), sample[name]))
            else:
                raise ValueError()

        if prune_unmapped_docs:
            for q in questions:
                if q.web_docs is not None:
                    q.web_docs = [x for x in q.web_docs if x.doc_id in file_map]
                q.entity_docs = [x for x in q.entity_docs if x.doc_id in file_map]

        print("Adding answers for %s question" % name)
        corpus = TriviaQaEvidenceCorpusTxt(file_map)
        questions = compute_answer_spans_par(questions, corpus, tokenizer, answer_detector, n_process)
        for q in questions:  # Sanity check, we should have answers for everything (even if of size 0)
            if q.answer is None:
                continue
            for doc in q.all_docs:
                if doc.doc_id in file_map:
                    if doc.answer_spans is None:
                        raise RuntimeError()

        print("Saving %s question" % name)
        with open(join(out_dir, name + ".pkl"), "wb") as f:
            pickle.dump(questions, f)

    print("Dumping file mapping")
    with open(join(out_dir, "file_map.json"), "w") as f:
        json.dump(file_map, f)

    print("Complete")

class TriviaQaSpanCorpus(Configurable):
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name  # web-sample
        self.dir = join("data", "triviaqa", corpus_name)
        with open(join(self.dir, "file_map.json"), "r") as f:
            file_map = json.load(f)
        for k, v in file_map.items():
            file_map[k] = unicodedata.normalize("NFD", v)
        self.evidence = TriviaQaEvidenceCorpusTxt(file_map) # evidence_corpus.py

    def get_train(self) -> List[TriviaQaQuestion]:
        with open(join(self.dir, "train.pkl"), "rb") as f:
            return pickle.load(f)

    def get_dev(self) -> List[TriviaQaQuestion]:
        with open(join(self.dir, "dev.pkl"), "rb") as f:
            return pickle.load(f)

    def get_test(self) -> List[TriviaQaQuestion]:
        with open(join(self.dir, "test.pkl"), "rb") as f:
            return pickle.load(f)

    def get_verified(self) -> Optional[List[TriviaQaQuestion]]:
        verified_dir = join(self.dir, "verified.pkl")
        if not exists(verified_dir):
            return None
        with open(verified_dir, "rb") as f:
            return pickle.load(f)

    @property
    def name(self):
        return self.corpus_name

class TriviaQaWebDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("web")

class TriviaQaWikiDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("wiki")

class TriviaQaUnfilteredDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("unfiltered")

class TriviaQaSampleWebDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("web-sample")

class TriviaQaSampleWikiDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("wiki-sample")

class TriviaQaSampleUnfilteredDataset(TriviaQaSpanCorpus):
    def __init__(self):
        super().__init__("unfiltered-sample")

def build_wiki_corpus(n_processes):
    build_dataset("wiki", tokenization.BasicTokenizer(do_lower_case=True),
                  dict(
                      verified=join(TRIVIA_QA, "wiki", "verified-wikipedia-dev.json"),
                      dev=join(TRIVIA_QA, "wiki", "wikipedia-dev.json"),
                      train=join(TRIVIA_QA, "wiki", "wikipedia-train.json"),
                      test=join(TRIVIA_QA, "wiki", "wikipedia-test-without-answers.json")
                  ),
                  FastNormalizedAnswerDetector(), n_processes)

def build_web_corpus(n_processes):
    build_dataset("web", tokenization.BasicTokenizer(do_lower_case=True),
                  dict(
                      verified=join(TRIVIA_QA, "web", "verified-web-dev.json"),
                      dev=join(TRIVIA_QA, "web", "web-dev.json"),
                      train=join(TRIVIA_QA, "web", "web-train.json"),
                      test=join(TRIVIA_QA, "web", "web-test-without-answers.json")
                  ),
                  FastNormalizedAnswerDetector(), n_processes)

def build_unfiltered_corpus(n_processes):
    build_dataset("unfiltered", tokenization.BasicTokenizer(do_lower_case=True),
                  dict(
                      dev=join(TRIVIA_QA, "unfiltered", "unfiltered-web-dev.json"),
                      train=join(TRIVIA_QA, "unfiltered", "unfiltered-web-train.json"),
                      test=join(TRIVIA_QA, "unfiltered", "unfiltered-web-test-without-answers.json")
                  ),
                  FastNormalizedAnswerDetector(), n_processes)

def build_wiki_sample_corpus(n_processes):
    build_dataset("wiki-sample", tokenization.BasicTokenizer(do_lower_case=True),
                  dict(
                      verified=join(TRIVIA_QA, "wiki-sample", "verified-wikipedia-dev.json"),
                      dev=join(TRIVIA_QA, "wiki-sample", "wikipedia-dev.json"),
                      train=join(TRIVIA_QA, "wiki-sample", "wikipedia-train.json"),
                      test=join(TRIVIA_QA, "wiki-sample", "wikipedia-test-without-answers.json")
                  ),
                  FastNormalizedAnswerDetector(), n_processes, sample=20)

def build_web_sample_corpus(n_processes):
    build_dataset("web-sample", tokenization.BasicTokenizer(do_lower_case=True),
                  dict(
                      verified=join(TRIVIA_QA, "web-sample", "verified-web-dev.json"),
                      dev=join(TRIVIA_QA, "web-sample", "web-dev.json"),
                      train=join(TRIVIA_QA, "web-sample", "web-train.json"),
                      test=join(TRIVIA_QA, "web-sample", "web-test-without-answers.json")
                  ),
                  FastNormalizedAnswerDetector(), n_processes, sample=20)

def build_unfiltered_sample_corpus(n_processes):
    build_dataset("unfiltered-sample", tokenization.BasicTokenizer(do_lower_case=True),
                  dict(
                      dev=join(TRIVIA_QA, "unfiltered-sample", "unfiltered-web-dev.json"),
                      train=join(TRIVIA_QA, "unfiltered-sample", "unfiltered-web-train.json"),
                      test=join(TRIVIA_QA, "unfiltered-sample", "unfiltered-web-test-without-answers.json")
                  ),
                  FastNormalizedAnswerDetector(), n_processes, sample=20)

def main():
    parser = argparse.ArgumentParser("Pre-procsess TriviaQA data")
    parser.add_argument("corpus", choices=["web", "wiki", "unfiltered", "web-sample", "wiki-sample", "unfiltered-sample"])
    parser.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use")
    args = parser.parse_args()
    if args.corpus == "web":
        build_web_corpus(args.n_processes)
    elif args.corpus == "wiki":
        build_wiki_corpus(args.n_processes)
    elif args.corpus == "unfiltered":
        build_unfiltered_corpus(args.n_processes)
    elif args.corpus == "web-sample":
        build_web_sample_corpus(args.n_processes)
    elif args.corpus == "wiki-sample":
        build_wiki_sample_corpus(args.n_processes)
    elif args.corpus == "unfiltered-sample":
        build_unfiltered_sample_corpus(args.n_processes)
    else:
        raise RuntimeError()


if __name__ == "__main__":
    main()