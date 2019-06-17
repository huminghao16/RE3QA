import argparse
import os
import pickle

from triviaqa.build_span_corpus import TriviaQaWikiDataset, TriviaQaSampleWikiDataset
from triviaqa.preprocessed_corpus import preprocess_par, ExtractMultiParagraphsPerQuestion, TopTfIdf


def main():
    parser = argparse.ArgumentParser(description='Train a model on TriviaQA web')
    parser.add_argument("--debug", default=False, action='store_true', help="Whether to run in debug mode.")
    parser.add_argument("--data_dir", default="data/triviaqa/wiki", type=str, help="Triviaqa wiki data dir")
    parser.add_argument('--n_processes', type=int, default=1,
                        help="Number of processes (i.e., select which paragraphs to train on) "
                             "the data with")
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help="Size of one chunk")
    parser.add_argument('--n_para_train', type=int, default=2,
                        help="Num of selected paragraphs during training")
    parser.add_argument('--n_para_dev', type=int, default=4,
                        help="Num of selected paragraphs during evaluation")
    parser.add_argument('--n_para_verified', type=int, default=4,
                        help="Num of selected paragraphs during evaluation")
    parser.add_argument('--n_para_test', type=int, default=4,
                        help="Num of selected paragraphs during testing")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to process train set.")
    parser.add_argument("--do_dev", default=False, action='store_true', help="Whether to process dev set.")
    parser.add_argument("--do_verified", default=False, action='store_true', help="Whether to process verified set.")
    parser.add_argument("--do_test", default=False, action='store_true', help="Whether to process test set.")
    args = parser.parse_args()

    if args.debug:
        corpus = TriviaQaSampleWikiDataset()
    else:
        corpus = TriviaQaWikiDataset()

    if args.do_train:
        train_questions = corpus.get_train()  # List[TriviaQaQuestion]
        train_preprocesser = ExtractMultiParagraphsPerQuestion(TopTfIdf(n_to_select=args.n_para_train, is_training=True),
                                                               intern=True, is_training=True)
        _train = preprocess_par(train_questions, corpus.evidence, train_preprocesser, args.n_processes, args.chunk_size,
                                "train")
        print("Recall of answer existence in {} set: {:.3f}".format("train", _train.ir_count / len(_train.data)))
        print("Average number of documents in {} set: {:.3f}".format("train", _train.total_doc_num / len(_train.data)))
        print("Average length of documents in {} set: {:.3f}".format("train", _train.total_doc_length / len(_train.data)))
        print("Average pruned length of documents in {} set: {:.3f}".format("train", _train.pruned_doc_length / len(_train.data)))
        print("Number of examples: {}".format(len(_train.data)))

        train_examples_path = os.path.join(args.data_dir, "train_{}paras_examples.pkl".format(args.n_para_train))
        pickle.dump(_train.data, open(train_examples_path, 'wb'))

    if args.do_dev:
        dev_questions = corpus.get_dev()
        dev_preprocesser = ExtractMultiParagraphsPerQuestion(TopTfIdf(n_to_select=args.n_para_dev, is_training=False),
                                                             intern=True, is_training=False)
        _dev = preprocess_par(dev_questions, corpus.evidence, dev_preprocesser, args.n_processes, args.chunk_size, "dev")
        print("Recall of answer existence in {} set: {:.3f}".format("dev", _dev.ir_count / len(_dev.data)))
        print("Average number of documents in {} set: {:.3f}".format("dev", _dev.total_doc_num / len(_dev.data)))
        print("Average length of documents in {} set: {:.3f}".format("dev", _dev.total_doc_length / len(_dev.data)))
        print("Average pruned length of documents in {} set: {:.3f}".format("dev", _dev.pruned_doc_length / len(_dev.data)))
        print("Number of examples: {}".format(len(_dev.data)))

        dev_examples_path = os.path.join(args.data_dir, "dev_{}paras_examples.pkl".format(args.n_para_dev))
        pickle.dump(_dev.data, open(dev_examples_path, 'wb'))

    if args.do_verified:
        verified_questions = corpus.get_verified()
        verified_preprocesser = ExtractMultiParagraphsPerQuestion(TopTfIdf(n_to_select=args.n_para_verified, is_training=False),
                                                                  intern=True, is_training=False)
        _verified = preprocess_par(verified_questions, corpus.evidence, verified_preprocesser, args.n_processes,
                                   args.chunk_size, "verified")
        print("Recall of answer existence in {} set: {:.3f}".format("verified", _verified.ir_count / len(_verified.data)))
        print("Average number of documents in {} set: {:.3f}".format("verified", _verified.total_doc_num / len(_verified.data)))
        print("Average length of documents in {} set: {:.3f}".format("verified", _verified.total_doc_length / len(_verified.data)))
        print("Average pruned length of documents in {} set: {:.3f}".format("verified", _verified.pruned_doc_length / len(_verified.data)))
        print("Number of examples: {}".format(len(_verified.data)))

        verified_examples_path = os.path.join(args.data_dir, "verified_{}paras_examples.pkl".format(args.n_para_verified))
        pickle.dump(_verified.data, open(verified_examples_path, 'wb'))

    if args.do_test:
        test_questions = corpus.get_test()
        test_preprocesser = ExtractMultiParagraphsPerQuestion(TopTfIdf(n_to_select=args.n_para_test, is_training=False),
                                                              intern=True, is_training=False)
        _test = preprocess_par(test_questions, corpus.evidence, test_preprocesser, args.n_processes,
                                   args.chunk_size, "test")
        print("Recall of answer existence in {} set: {:.3f}".format("test", _test.ir_count / len(_test.data)))
        print("Average number of documents in {} set: {:.3f}".format("test", _test.total_doc_num / len(_test.data)))
        print("Average length of documents in {} set: {:.3f}".format("test", _test.total_doc_length / len(_test.data)))
        print("Average pruned length of documents in {} set: {:.3f}".format("test", _test.pruned_doc_length / len(_test.data)))
        print("Number of examples: {}".format(len(_test.data)))

        test_examples_path = os.path.join(args.data_dir, "test_{}paras_examples.pkl".format(args.n_para_test))
        pickle.dump(_test.data, open(test_examples_path, 'wb'))


if __name__ == "__main__":
    main()