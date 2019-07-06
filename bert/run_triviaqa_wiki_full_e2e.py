# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on Triviaqa."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import json
import pickle
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import bert.tokenization as tokenization
from bert.modeling import BertConfig
from bert.custom_modeling import BertForRankingAndDistantReadingAndReranking
from bert.optimization import BERTAdam
from squad.squad_document_utils import random_filter_features, RawRankResult, RawReadResult, RawFinalResult, \
    write_rerank_predictions, eval_ranking

from triviaqa.triviaqa_document_utils import convert_examples_to_features, annotate_candidates
from triviaqa.triviaqa_eval import read_triviaqa_data, get_key_to_ground_truth_per_question, evaluate_triviaqa

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan

def bert_load_state_dict(model, state_dict):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    return model

def prepare_optimizer(args, model, num_train_steps):
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    ]
    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    return optimizer, param_optimizer

def post_process_loss(args, n_gpu, loss):
    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if args.fp16 and args.loss_scale != 1.0:
        # rescale loss for fp16 training
        # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
        loss = loss * args.loss_scale
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
    return loss

def read_train_data(args, tokenizer, logger):
    train_examples_path = os.path.join(args.data_dir, "train_{}paras_examples.pkl".format(args.n_para_train))
    assert os.path.exists(train_examples_path)
    train_examples = pickle.load(open(train_examples_path, 'rb'))
    logger.info("Loading examples from: %s" % (train_examples_path))
    
    train_features_path = os.path.join(args.data_dir, "train_{}paras_{}max_{}stride_features.pkl"
                                       .format(args.n_para_train, args.max_seq_length, args.doc_stride))
    if os.path.exists(train_features_path):
        train_features = pickle.load(open(train_features_path, 'rb'))
        logger.info("Loading features from: %s" % (train_features_path))
    else:
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            verbose_logging=args.verbose_logging,
            logger=logger)
        pickle.dump(train_features, open(train_features_path, 'wb'))

    # filter features
    logger.info("Filtering features randomly")
    filtered_train_features = random_filter_features(train_examples, train_features, args.n_best_size_rank,
                                                     is_training=True)
    return build_train_data(args, train_examples, train_features, filtered_train_features, logger)

def reconstruct_train_data(args, train_examples, train_features, logger):
    assert args.rank_train_file is not None
    rank_train_path = os.path.join(args.output_dir, args.rank_train_file)
    logger.info("Filtering features based on: %s", rank_train_path)
    rank_predictions = pickle.load(open(rank_train_path, 'rb'))
    rank_list = []
    filtered_train_features = []
    for pred in rank_predictions:
        rank_list.append(pred.unique_id)
    for train_feature in train_features:
        if train_feature.unique_id in rank_list:
            filtered_train_features.append(train_feature)
    return build_train_data(args, train_examples, train_features, filtered_train_features, logger)

def build_train_data(args, train_examples, train_features, filtered_train_features, logger):
    num_train_steps = int(
        len(filtered_train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    train_batch_size_for_rank = int(len(train_features) /
                                    (num_train_steps / args.num_train_epochs * args.gradient_accumulation_steps))

    logger.info("Num orig examples = %d", len(train_examples))
    logger.info("Num split features = %d", len(train_features))
    logger.info("Num split filtered features = %d", len(filtered_train_features))
    logger.info("Batch size for ranker = %d", train_batch_size_for_rank)
    logger.info("Batch size for reader = %d", args.train_batch_size)
    logger.info("Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_rank_labels = torch.tensor([not f.is_impossible for f in train_features], dtype=torch.long)
    train_rank_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_rank_labels)
    if args.local_rank == -1:
        train_rank_sampler = RandomSampler(train_rank_data)
    else:
        train_rank_sampler = DistributedSampler(train_rank_data)
    train_rank_dataloader = DataLoader(train_rank_data, sampler=train_rank_sampler,
                                       batch_size=train_batch_size_for_rank)

    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    train_distill_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    if args.local_rank == -1:
        train_distill_sampler = SequentialSampler(train_distill_data)
    else:
        train_distill_sampler = DistributedSampler(train_distill_data)
    train_distill_dataloader = DataLoader(train_distill_data, sampler=train_distill_sampler,
                                          batch_size=train_batch_size_for_rank)

    all_input_ids = torch.tensor([f.input_ids for f in filtered_train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in filtered_train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in filtered_train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_positions for f in filtered_train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_positions for f in filtered_train_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    train_read_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions, all_example_index)
    if args.local_rank == -1:
        train_read_sampler = RandomSampler(train_read_data)
    else:
        train_read_sampler = DistributedSampler(train_read_data)
    train_read_dataloader = DataLoader(train_read_data, sampler=train_read_sampler, batch_size=args.train_batch_size)
    return train_examples, train_features, filtered_train_features, train_rank_dataloader, train_read_dataloader, \
           train_distill_dataloader, num_train_steps

def read_dev_data(args, tokenizer, logger):
    dev_examples_path = os.path.join(args.data_dir, "dev_{}paras_examples.pkl".format(args.n_para_dev))
    assert os.path.exists(dev_examples_path)
    dev_examples = pickle.load(open(dev_examples_path, 'rb'))
    logger.info("Loading examples from: %s" % (dev_examples_path))

    dev_features_path = os.path.join(args.data_dir, "dev_{}paras_{}max_{}stride_features.pkl"
                                     .format(args.n_para_dev, args.max_seq_length, args.doc_stride))
    if os.path.exists(dev_features_path):
        dev_features = pickle.load(open(dev_features_path, 'rb'))
        logger.info("Loading features from: %s" % (dev_features_path))
    else:
        dev_features = convert_examples_to_features(
            examples=dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            verbose_logging=args.verbose_logging,
            logger=logger)
        pickle.dump(dev_features, open(dev_features_path, 'wb'))

    logger.info("Filtering features randomly")
    filtered_dev_features = random_filter_features(dev_examples, dev_features, args.n_best_size_rank,
                                                   is_training=False)
    filtered_rank_logits = [0.] * len(filtered_dev_features)
    return build_eval_data(args, dev_examples, dev_features, filtered_dev_features, filtered_rank_logits, logger)

def read_test_data(args, tokenizer, logger):
    test_examples_path = os.path.join(args.data_dir, "test_{}paras_examples.pkl".format(args.n_para_test))
    assert os.path.exists(test_examples_path)
    test_examples = pickle.load(open(test_examples_path, 'rb'))
    logger.info("Loading examples from: %s" % (test_examples_path))

    test_features_path = os.path.join(args.data_dir, "test_{}paras_{}max_{}stride_features.pkl"
                                      .format(args.n_para_test, args.max_seq_length, args.doc_stride))
    if os.path.exists(test_features_path):
        test_features = pickle.load(open(test_features_path, 'rb'))
        logger.info("Loading features from: %s" % (test_features_path))
    else:
        test_features = convert_examples_to_features(
            examples=test_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            verbose_logging=args.verbose_logging,
            logger=logger)
        pickle.dump(test_features, open(test_features_path, 'wb'))

    logger.info("Filtering features randomly")
    filtered_test_features = random_filter_features(test_examples, test_features, args.n_best_size_rank,
                                                    is_training=False)
    filtered_rank_logits = [0.] * len(filtered_test_features)
    return build_eval_data(args, test_examples, test_features, filtered_test_features, filtered_rank_logits, logger)

def reconstruct_eval_data(args, eval_examples, eval_features, logger):
    assert args.rank_pred_file is not None
    rank_pred_path = os.path.join(args.output_dir, args.rank_pred_file)
    logger.info("Filtering features based on: %s", rank_pred_path)
    rank_predictions = pickle.load(open(rank_pred_path, 'rb'))
    rank_dict = {}
    filtered_eval_features, filtered_rank_logits = [], []
    for pred in rank_predictions:
        rank_dict[pred.unique_id] = pred.rank_logit
    for eval_feature in eval_features:
        if eval_feature.unique_id in rank_dict.keys():
            filtered_eval_features.append(eval_feature)
            filtered_rank_logits.append(rank_dict[eval_feature.unique_id])
    return build_eval_data(args, eval_examples, eval_features, filtered_eval_features, filtered_rank_logits, logger)

def build_eval_data(args, eval_examples, eval_features, filtered_eval_features, filtered_rank_logits, logger):
    predict_batch_size_for_rank = 2 * args.predict_batch_size

    logger.info("Num orig examples = %d", len(eval_examples))
    logger.info("Num split features = %d", len(eval_features))
    logger.info("Num split filtered features = %d", len(filtered_eval_features))
    logger.info("Batch size for ranker = %d", predict_batch_size_for_rank)
    logger.info("Batch size for reader = %d", args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_rank_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    if args.local_rank == -1:
        eval_rank_sampler = SequentialSampler(eval_rank_data)
    else:
        eval_rank_sampler = DistributedSampler(eval_rank_data)
    eval_rank_dataloader = DataLoader(eval_rank_data, sampler=eval_rank_sampler, batch_size=predict_batch_size_for_rank)

    all_input_ids = torch.tensor([f.input_ids for f in filtered_eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in filtered_eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in filtered_eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_read_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    if args.local_rank == -1:
        eval_read_sampler = SequentialSampler(eval_read_data)
    else:
        eval_read_sampler = DistributedSampler(eval_read_data)
    eval_read_dataloader = DataLoader(eval_read_data, sampler=eval_read_sampler, batch_size=args.predict_batch_size)
    return eval_examples, eval_features, filtered_eval_features, filtered_rank_logits, eval_rank_dataloader, \
           eval_read_dataloader

def run_train_epoch(args, global_step, model, param_optimizer,
                    train_examples, train_read_features, train_rank_dataloader, train_read_dataloader,
                    optimizer, n_gpu, device, eval_examples, eval_rank_features, eval_read_features,
                    eval_rank_logits, eval_rank_dataloader, eval_read_dataloader,
                    logger, log_path, save_path, best_f1, epoch):
    running_loss, count = 0.0, 0
    model.train()
    for step, (rank_batch, read_batch) in enumerate(zip(train_rank_dataloader, train_read_dataloader)):
        if n_gpu == 1:
            rank_batch = tuple(t.to(device) for t in rank_batch)  # multi-gpu does scattering it-self
            read_batch = tuple(t.to(device) for t in read_batch)  # multi-gpu does scattering it-self

        input_ids, input_mask, segment_ids, rank_labels = rank_batch
        rank_loss = model('rank', input_mask, input_ids=input_ids, token_type_ids=segment_ids, rank_labels=rank_labels)
        rank_loss = post_process_loss(args, n_gpu, rank_loss)

        input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices = read_batch
        batch_start_logits, batch_end_logits, _ = model('read_inference', input_mask,
                                                        input_ids=input_ids, token_type_ids=segment_ids)
        batch_features, batch_results = [], []
        for j, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[j].detach().cpu().tolist()
            end_logits = batch_end_logits[j].detach().cpu().tolist()
            train_feature = train_read_features[example_index.item()]
            unique_id = int(train_feature.unique_id)
            batch_features.append(train_feature)
            batch_results.append(RawReadResult(unique_id=unique_id,
                                               start_logits=start_logits,
                                               end_logits=end_logits,
                                               rank_logit=0.))

        span_starts, span_ends, hard_labels, soft_labels = annotate_candidates(train_examples, batch_features,
                                                                               batch_results, args.filter_type,
                                                                               True, args.n_best_size_read,
                                                                               args.max_answer_length,
                                                                               args.do_lower_case,
                                                                               args.verbose_logging, logger)

        span_starts = torch.tensor(span_starts, dtype=torch.long)
        span_ends = torch.tensor(span_ends, dtype=torch.long)
        hard_labels = torch.tensor(hard_labels, dtype=torch.long)
        soft_labels = torch.tensor(soft_labels, dtype=torch.long)
        span_starts = span_starts.to(device)
        span_ends = span_ends.to(device)
        hard_labels = hard_labels.to(device)
        soft_labels = soft_labels.to(device)

        read_rerank_loss = model('read_rerank_train', input_mask, input_ids=input_ids, token_type_ids=segment_ids,
                                 start_positions=start_positions, end_positions=end_positions, span_starts=span_starts,
                                 span_ends=span_ends, hard_labels=hard_labels, soft_labels=soft_labels)
        read_rerank_loss = post_process_loss(args, n_gpu, read_rerank_loss)

        loss = rank_loss + read_rerank_loss
        loss.backward()
        running_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16 or args.optimize_on_cpu:
                if args.fp16 and args.loss_scale != 1.0:
                    # scale down gradients for fp16 training
                    for param in model.parameters():
                        param.grad.data = param.grad.data / args.loss_scale
                is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                if is_nan:
                    logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                    args.loss_scale = args.loss_scale / 2
                    model.zero_grad()
                    continue
                optimizer.step()
                copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
            else:
                optimizer.step()
            model.zero_grad()
            global_step += 1
            count += 1

            if global_step % 1000 == 0 and count != 0:
                logger.info("step: {}, loss: {:.3f}".format(global_step, running_loss / count))
                running_loss, count = 0.0, 0
            if args.debug:
                break

    model.eval()
    logger.info("***** Running ranking evaluation *****")
    rank_metrics = evaluate_rank(args, model, device, eval_examples, eval_rank_features, eval_rank_dataloader,
                                 logger, 'dev', args.n_para_dev, force_answer=False, write_pred=False)
    logger.info("***** Running reading evaluation *****")
    read_metrics = evaluate_read(args, model, device, eval_examples, eval_read_features, eval_rank_logits,
                                 eval_read_dataloader, args.dev_file, logger, write_pred=False)
    f = open(log_path, "a")
    print("Ranker, step: {}, loss: {:.3f}, map: {:.3f}, mrr: {:.3f}, "
          "top_1: {:.3f}, top_3: {:.3f}, top_5: {:.3f}, top_7: {:.3f}"
          .format(global_step, running_loss / count, rank_metrics['map'], rank_metrics['mrr'],
                  rank_metrics['top_1'], rank_metrics['top_3'], rank_metrics['top_5'], rank_metrics['top_7']),
          file=f)
    print("Reader, step: {}, loss: {:.3f}, em: {:.3f}, f1: {:.3f}"
          .format(global_step, running_loss / count, read_metrics['exact_match'],
                  read_metrics['f1']), file=f)
    print(" ", file=f)
    f.close()
    if read_metrics['f1'] > best_f1:
        best_f1 = read_metrics['f1']
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': global_step,
        }, save_path)
    return global_step, model, best_f1

def evaluate_rank(args, model, device, eval_examples, eval_features, eval_dataloader, logger, type, n_para,
                  force_answer=False, write_pred=False, verbose_logging=False):
    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in eval_dataloader:
        if len(all_results) % 5000 == 0 and verbose_logging:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_rank_logits = model('rank', input_mask, input_ids=input_ids, token_type_ids=segment_ids)
        for i, example_index in enumerate(example_indices):
            rank_logits = batch_rank_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawRankResult(unique_id=unique_id, rank_logit=float(rank_logits[1])))
    metrics, rank_predictions = eval_ranking(force_answer, args.n_best_size_rank, eval_examples, eval_features, all_results)

    if write_pred:
        rank_pred_file = "{}_{}paras_{}best.pkl".format(type, n_para, args.n_best_size_rank)
        rank_pred_path = os.path.join(args.output_dir, rank_pred_file)
        pickle.dump(rank_predictions, open(rank_pred_path, 'wb'))
        if type == 'distill':
            args.rank_train_file = rank_pred_file
        else:
            args.rank_pred_file = rank_pred_file
    return metrics

def evaluate_read(args, model, device, eval_examples, eval_features, eval_rank_logits, eval_dataloader, predict_file,
                  logger, type=None, write_pred=False, verbose_logging=False, forward=False):
    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in eval_dataloader:
        if len(all_results) % 5000 == 0 and verbose_logging:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, sequence_output = model('read_inference', input_mask,
                                                                          input_ids=input_ids,
                                                                          token_type_ids=segment_ids)
        batch_features, batch_results = [], []
        for j, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[j].detach().cpu().tolist()
            end_logits = batch_end_logits[j].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            eval_rank_logit = eval_rank_logits[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            batch_features.append(eval_feature)
            batch_results.append(RawReadResult(unique_id=unique_id,
                                               start_logits=start_logits,
                                               end_logits=end_logits,
                                               rank_logit=eval_rank_logit))

        span_starts, span_ends, _, _ = annotate_candidates(eval_examples, batch_features, batch_results,
                                                           args.filter_type, False, args.n_best_size_read,
                                                           args.max_answer_length, args.do_lower_case,
                                                           args.verbose_logging, logger)

        span_starts = torch.tensor(span_starts, dtype=torch.long)
        span_ends = torch.tensor(span_ends, dtype=torch.long)
        span_starts = span_starts.to(device)
        span_ends = span_ends.to(device)
        sequence_output = sequence_output.to(device)
        with torch.no_grad():
            batch_rerank_logits = model('rerank_inference', input_mask, span_starts=span_starts,
                                        span_ends=span_ends, sequence_input=sequence_output)
        for j, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[j].detach().cpu().tolist()
            end_logits = batch_end_logits[j].detach().cpu().tolist()
            rerank_logits = batch_rerank_logits[j].detach().cpu().numpy()
            start_indexes = span_starts[j].detach().cpu().tolist()
            end_indexes = span_ends[j].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            eval_rank_logit = eval_rank_logits[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawFinalResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits,
                                              rank_logit=eval_rank_logit, rerank_logits=rerank_logits,
                                              start_indexes=start_indexes, end_indexes=end_indexes))

    all_predictions, all_nbest_json = write_rerank_predictions(eval_examples, eval_features, all_results, args.length_heuristic,
                                                               args.pred_rank_weight, args.pred_rerank_weight,
                                                               args.ablate_type, args.n_best_size_read,
                                                               args.max_answer_length, args.do_lower_case,
                                                               args.verbose_logging, logger)

    if write_pred:
        output_prediction_file = os.path.join(args.output_dir, "{}_predictions.json".format(type))
        output_nbest_file = os.path.join(args.output_dir, "{}_nbest_predictions.json".format(type))
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        logger.info("Writing predictions to: %s" % (output_prediction_file))
        logger.info("Writing nbest to: %s" % (output_nbest_file))

    metrics = None
    if not forward:
        predict_path = os.path.join(args.data_dir, predict_file)
        dataset_json = read_triviaqa_data(predict_path)
        key_to_ground_truth = get_key_to_ground_truth_per_question(dataset_json)
        metrics = evaluate_triviaqa(key_to_ground_truth, all_predictions)
    return metrics

def run_rank_eval(args, global_step, model, device, eval_examples, eval_features, eval_dataloader,
                  logger, log_path, save_path, type, n_para, force_answer=False, write_pred=False, forward=False):
    # restore from best checkpoint
    if save_path and os.path.isfile(save_path) and args.do_train:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'])
        step = checkpoint['step']
        logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                    .format(save_path, step))
    model.eval()
    metrics = evaluate_rank(args, model, device, eval_examples, eval_features, eval_dataloader, logger, type, n_para,
                            force_answer=force_answer, write_pred=write_pred, verbose_logging=True)
    if not forward:
        f = open(log_path, "a")
        print("Ranker, type: {}, step: {}, map: {:.3f}, mrr: {:.3f}, "
              "top_1: {:.3f}, top_3: {:.3f}, top_5: {:.3f}, top_7: {:.3f}, retrieval_rate: {:.3f}"
              .format(type, global_step, metrics['map'], metrics['mrr'], metrics['top_1'], metrics['top_3'], metrics['top_5'],
                      metrics['top_7'], metrics['retrieval_rate']), file=f)
        print(" ", file=f)
        f.close()

def run_read_eval(args, global_step, model, device, eval_examples, eval_features, eval_rank_logits, eval_dataloader,
                  predict_file, logger, log_path, save_path, type, write_pred=False, forward=False):
    # restore from best checkpoint
    if save_path and os.path.isfile(save_path) and args.do_train:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'])
        step = checkpoint['step']
        logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                    .format(save_path, step))
    model.eval()
    metrics = evaluate_read(args, model, device, eval_examples, eval_features, eval_rank_logits, eval_dataloader,
                            predict_file, logger, type, write_pred=write_pred, verbose_logging=True, forward=forward)
    if not forward:
        f = open(log_path, "a")
        print("Reader, type: {}, step: {}, em: {:.3f}, f1: {:.3f}"
              .format(type, global_step, metrics['exact_match'], metrics['f1']), file=f)
        print(" ", file=f)
        f.close()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the reader's checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--debug", default=False, action='store_true', help="Whether to run in debug mode.")
    parser.add_argument("--data_dir", default="data/triviaqa/wiki", type=str, help="Triviaqa wiki data dir")
    parser.add_argument("--dev_file", default="wikipedia-dev.json", type=str, help="Triviaqa wiki dev file")
    parser.add_argument("--rank_train_file", default=None, type=str,
                        help="ranking train file contains predictions on train features")
    parser.add_argument("--rank_pred_file", default=None, type=str,
                        help="ranking prediction file contains predictions on eval features")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_dev", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=False, action='store_true', help="Whether to run forward on the test set.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.05, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--length_heuristic", default=0.05, type=float,
                        help="Weight on length heuristic.")
    parser.add_argument("--n_best_size_rank", default=6, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--n_best_size_read", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--num_hidden_rank", default=3, type=int,
                        help="The total number of hidden layers for ranker.")
    parser.add_argument("--n_para_train", default=12, type=int,
                        help="The total number of paragraph used in training.")
    parser.add_argument("--n_para_dev", default=14, type=int,
                        help="The total number of paragraph used in prediction.")
    parser.add_argument("--n_para_test", default=14, type=int,
                        help="The total number of paragraph used in testing.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--pred_rank_weight", default=1.4, type=float, help="Pred Rank weight")
    parser.add_argument("--pred_rerank_weight", default=1.4, type=float, help="Pred Rerank weight")
    parser.add_argument("--filter_type", default="em", type=str, help="Which filter type to use")
    parser.add_argument("--ablate_type", default="none", type=str, help="Which ablation type to use")
    parser.add_argument("--data_parallel", default=False, action='store_true',
                        help="Whether to use data_parallel during prediction")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info('output_dir: {}'.format(args.output_dir))
    save_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    log_path = os.path.join(args.output_dir, 'performance.txt')
    network_path = os.path.join(args.output_dir, 'network.txt')
    parameter_path = os.path.join(args.output_dir, 'parameter.txt')

    f = open(parameter_path, "w")
    for arg in sorted(vars(args)):
        print("{}: {}".format(arg, getattr(args, arg)), file=f)
    f.close()

    if not args.do_train and not args.do_dev and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_dev` or `do_test` must be True.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("torch_version: {} device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        torch.__version__, device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    logger.info("***** Preparing model *****")
    model = BertForRankingAndDistantReadingAndReranking(bert_config, args.num_hidden_rank)
    if args.init_checkpoint is not None:
        logger.info("Loading model from pretrained checkpoint: {}".format(args.init_checkpoint))
        model = bert_load_state_dict(model, torch.load(args.init_checkpoint, map_location='cpu'))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1 or args.data_parallel:
        model = torch.nn.DataParallel(model)

    global_step, best_f1 = 0, 0
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'])
        global_step = checkpoint['step']
        logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                    .format(save_path, checkpoint['step']))

    f = open(network_path, "w")
    for n, param in model.named_parameters():
        print("name: {}, size: {}, dtype: {}, requires_grad: {}"
              .format(n, param.size(), param.dtype, param.requires_grad), file=f)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters: {}".format(total_trainable_params), file=f)
    print("Total parameters: {}".format(total_params), file=f)
    f.close()

    if args.do_train:
        logger.info("***** Preparing training *****")
        args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
        train_examples, train_rank_features, train_read_features, train_rank_dataloader, train_read_dataloader, \
        train_distill_dataloader, _ = read_train_data(args, tokenizer, logger)
        logger.info("***** Preparing evaluation *****")
        eval_examples, eval_rank_features, eval_read_features, eval_rank_logits, eval_rank_dataloader, \
        eval_read_dataloader = read_dev_data(args, tokenizer, logger)

        logger.info("***** Running training distillation *****")
        run_rank_eval(args, global_step, model, device, train_examples, train_rank_features, train_distill_dataloader,
                      logger, log_path, save_path=None, type='distill', n_para=args.n_para_train,
                      force_answer=True, write_pred=True)
        logger.info("***** Reconstruct training data at {} *****".format(args.rank_train_file))
        train_examples, train_rank_features, train_read_features, train_rank_dataloader, train_read_dataloader, \
        train_distill_dataloader, num_train_steps = reconstruct_train_data(args, train_examples, train_rank_features, logger)

        logger.info("***** Running dev distillation *****")
        run_rank_eval(args, global_step, model, device, eval_examples, eval_rank_features, eval_rank_dataloader,
                      logger, log_path, save_path=None, type='dev', n_para=args.n_para_dev,
                      force_answer=False, write_pred=True)
        logger.info("***** Reconstruct eval data at {} *****".format(args.rank_pred_file))
        eval_examples, eval_rank_features, eval_read_features, eval_rank_logits, eval_rank_dataloader, \
        eval_read_dataloader = reconstruct_eval_data(args, eval_examples, eval_rank_features, logger)

        logger.info("***** Preparing optimizer *****")
        optimizer, param_optimizer = prepare_optimizer(args, model, num_train_steps)
        if os.path.isfile(save_path):
            checkpoint = torch.load(save_path)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Load optimizer from finetuned checkpoint: '{}' (step {}, epoch {})"
                        .format(save_path, checkpoint['step'], checkpoint['epoch']))

        logger.info("***** Running training *****")
        for epoch in range(int(args.num_train_epochs)):
            logger.info("***** Epoch: {} *****".format(epoch+1))
            global_step, model, best_f1 = run_train_epoch(args, global_step, model, param_optimizer, train_examples,
                                                          train_read_features, train_rank_dataloader, train_read_dataloader,
                                                          optimizer, n_gpu, device, eval_examples,
                                                          eval_rank_features, eval_read_features, eval_rank_logits,
                                                          eval_rank_dataloader, eval_read_dataloader, logger, log_path,
                                                          save_path, best_f1, epoch)

            if epoch + 1 < args.num_train_epochs:
                logger.info("***** Running training distillation *****")
                run_rank_eval(args, global_step, model, device, train_examples, train_rank_features, train_distill_dataloader,
                              logger, log_path, save_path=None, type='distill', n_para=args.n_para_train,
                              force_answer=True, write_pred=True)
                logger.info("***** Reconstruct training data at {} *****".format(args.rank_train_file))
                train_examples, train_rank_features, train_read_features, train_rank_dataloader, train_read_dataloader, \
                train_distill_dataloader, _ = reconstruct_train_data(args, train_examples, train_rank_features, logger)

                logger.info("***** Running dev distillation *****")
                run_rank_eval(args, global_step, model, device, eval_examples, eval_rank_features, eval_rank_dataloader,
                              logger, log_path, save_path=None, type='dev', n_para=args.n_para_dev,
                              force_answer=False, write_pred=True)
                logger.info("***** Reconstruct eval data at {} *****".format(args.rank_pred_file))
                eval_examples, eval_rank_features, eval_read_features, eval_rank_logits, eval_rank_dataloader, \
                eval_read_dataloader = reconstruct_eval_data(args, eval_examples, eval_rank_features, logger)

    if args.do_dev:
        logger.info("***** Preparing prediction on dev data *****")
        eval_examples, eval_rank_features, _, _, eval_rank_dataloader, _ = read_dev_data(args, tokenizer, logger)
        logger.info("***** Running ranking prediction *****")
        run_rank_eval(args, global_step, model, device, eval_examples, eval_rank_features, eval_rank_dataloader, logger,
                      log_path, save_path, type='dev', n_para=args.n_para_dev, force_answer=False, write_pred=True)
        logger.info("***** Reconstruct dev data at {} *****".format(args.rank_pred_file))
        eval_examples, _, eval_read_features, eval_rank_logits, _, eval_read_dataloader \
            = reconstruct_eval_data(args, eval_examples, eval_rank_features, logger)
        logger.info("***** Running reading prediction *****")
        run_read_eval(args, global_step, model, device, eval_examples, eval_read_features, eval_rank_logits,
                      eval_read_dataloader, args.dev_file, logger, log_path, save_path, type='dev', write_pred=True)

    if args.do_test:
        logger.info("***** Preparing prediction on test data *****")
        eval_examples, eval_rank_features, _, _, eval_rank_dataloader, _ = read_test_data(args, tokenizer, logger)
        logger.info("***** Running ranking prediction *****")
        run_rank_eval(args, global_step, model, device, eval_examples, eval_rank_features, eval_rank_dataloader, logger,
                      log_path, save_path, type='test', n_para=args.n_para_test,
                      force_answer=False, write_pred=True, forward=True)
        logger.info("***** Reconstruct test data at {} *****".format(args.rank_pred_file))
        eval_examples, _, eval_read_features, eval_rank_logits, _, eval_read_dataloader \
            = reconstruct_eval_data(args, eval_examples, eval_rank_features, logger)
        logger.info("***** Running reading prediction *****")
        run_read_eval(args, global_step, model, device, eval_examples, eval_read_features, eval_rank_logits,
                      eval_read_dataloader, None, logger, log_path, save_path,
                      type='test', write_pred=True, forward=True)


if __name__ == "__main__":
    main()
