import argparse
import re
from os import walk, mkdir, makedirs
from os.path import relpath, join, exists, expanduser
from typing import Set
from tqdm import tqdm
from typing import List

import bert.tokenization as tokenization
from triviaqa.utils import split, flatten_iterable, group
from triviaqa.read_data import normalize_wiki_filename

TRIVIA_QA = join(expanduser("~"), "data", "triviaqa")

class MergeParagraphs(object):
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def merge(self, paragraphs: List[List[str]]):
        all_paragraphs = []

        on_paragraph = []  # text we have collect for the current paragraph
        cur_tokens = 0  # number of tokens in the current paragraph

        word_ix = 0
        for para in paragraphs:
            n_words = len(para)
            start_token = word_ix
            end_token = start_token + n_words
            word_ix = end_token

            if cur_tokens + n_words > self.max_tokens:
                if cur_tokens != 0: # end the current paragraph
                    all_paragraphs.append(on_paragraph)
                    on_paragraph = []
                    cur_tokens = 0

                if n_words >= self.max_tokens:  # either add current paragraph, or begin a new paragraph
                    all_paragraphs.append(para)
                else:
                    on_paragraph += para
                    cur_tokens = n_words
            else:
                on_paragraph += para
                cur_tokens += n_words

        if on_paragraph != []:
            all_paragraphs.append(on_paragraph)
        return all_paragraphs

def _gather_files(input_root, output_dir, skip_dirs, wiki_only):
    if not exists(output_dir):
        mkdir(output_dir)

    all_files = []
    for root, dirs, filenames in walk(input_root):
        if skip_dirs:   # False
            output = join(output_dir, relpath(root, input_root))
            if exists(output):
                continue
        path = relpath(root, input_root)
        normalized_path = normalize_wiki_filename(path)
        if not exists(join(output_dir, normalized_path)):
            mkdir(join(output_dir, normalized_path))
        all_files += [join(path, x) for x in filenames]
    if wiki_only:
        all_files = [x for x in all_files if "wikipedia/" in x]
    return all_files

def build_tokenized_files(filenames, input_root, output_root, tokenizer, splitter, override=True) -> Set[str]:
    """
    For each file in `filenames` loads the text, tokenizes it with `tokenizer, and
    saves the output to the same relative location in `output_root`.
    @:return a set of all the individual words seen
    """
    voc = set()
    for filename in filenames:
        out_file = normalize_wiki_filename(filename[:filename.rfind(".")]) + ".txt"
        out_file = join(output_root, out_file)
        if not override and exists(out_file):
            continue
        with open(join(input_root, filename), "r") as in_file:
            text = in_file.read().strip()
        paras = [x for x in text.split("\n") if len(x) > 0]
        paragraphs = [tokenizer.tokenize(x) for x in paras]
        merged_paragraphs = splitter.merge(paragraphs)

        for para in merged_paragraphs:
            for i, word in enumerate(para):
                voc.update(word)

        with open(out_file, "w") as in_file:
            in_file.write("\n\n".join(" ".join(para) for para in merged_paragraphs))
    return voc

def build_tokenized_corpus(input_root, tokenizer, splitter, output_dir, skip_dirs=False,
                           n_processes=1, wiki_only=False):
    if not exists(output_dir):
        makedirs(output_dir)

    all_files = _gather_files(input_root, output_dir, skip_dirs, wiki_only)

    if n_processes == 1:
        voc = build_tokenized_files(tqdm(all_files, ncols=80), input_root, output_dir, tokenizer, splitter)
    else:
        voc = set()
        from multiprocessing import Pool
        with Pool(n_processes) as pool:
            chunks = split(all_files, n_processes)
            chunks = flatten_iterable(group(c, 500) for c in chunks)
            pbar = tqdm(total=len(chunks), ncols=80)
            for v in pool.imap_unordered(_build_tokenized_files_t,
                                         [[c, input_root, output_dir, tokenizer, splitter] for c in chunks]):
                voc.update(v)
                pbar.update(1)
            pbar.close()

def _build_tokenized_files_t(arg):
    return build_tokenized_files(*arg)

class TriviaQaEvidenceCorpusTxt(object):
    """
    Corpus of the tokenized text from the given TriviaQa evidence documents.
    Allows the text to be retrieved by document id
    """

    _split_para = re.compile("\n\n+")

    def __init__(self, file_id_map=None):
        self.directory = join("data", "triviaqa/evidence")
        self.file_id_map = file_id_map

    def get_document(self, doc_id):
        if self.file_id_map is None:
            file_id = doc_id
        else:
            file_id = self.file_id_map.get(doc_id)

        if file_id is None:
            return None

        file_id = join(self.directory, file_id + ".txt")
        if not exists(file_id):
            return None

        with open(file_id, "r") as f:
            text = f.read()
            paragraphs = []
            for para in self._split_para.split(text):
                paragraphs.append(para.split(" "))
            return paragraphs   # List[List[str]]

def main():
    parse = argparse.ArgumentParser("Pre-tokenize the TriviaQA evidence corpus")
    parse.add_argument("-o", "--output_dir", type=str, default=join("data", "triviaqa", "evidence"))
    parse.add_argument("-s", "--source", type=str, default=join(TRIVIA_QA, "evidence"))
    # This is slow, using more processes is recommended
    parse.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use")
    parse.add_argument("--max_tokens", type=int, default=200, help="Number of maximal tokens in each merged paragraph")
    parse.add_argument("--wiki_only", action="store_true")
    args = parse.parse_args()

    tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
    splitter = MergeParagraphs(args.max_tokens)
    build_tokenized_corpus(args.source, tokenizer, splitter, args.output_dir,
                           n_processes=args.n_processes, wiki_only=args.wiki_only)

if __name__ == "__main__":
    main()