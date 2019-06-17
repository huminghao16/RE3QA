from typing import List, TypeVar, Iterable
import collections
import re
import string
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
              'be', "' s", "' t"}

def flatten_iterable(listoflists: Iterable[Iterable[T]]) -> List[T]:
    return [item for sublist in listoflists for item in sublist]


def split(lst: List[T], n_groups) -> List[List[T]]:
    """ partition `lst` into `n_groups` that are as evenly sized as possible  """
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups

def group(lst: List[T], max_group_size) -> List[List[T]]:
    """ partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    n_groups = (len(lst)+max_group_size-1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


def simple_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def get_max_f1_span(words, answer, window_size):
    max_f1 = 0
    max_span = (0, 0)

    for idx1, word1 in enumerate(words):
        for idx2, word2 in enumerate(words[idx1: idx1 + window_size + 1]):
            candidate_answer = words[idx1: idx1 + idx2 + 1]
            f1 = compute_f1(' '.join(answer), ' '.join(candidate_answer))
            if f1 > max_f1:
                max_f1 = f1
                max_span = (idx1, idx1 + idx2)
    return max_span, max_f1