# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import pickle
import random
import sys
from collections import defaultdict

import ipdb
from torch.nn import MSELoss
from tqdm import tqdm, trange

from sklearn.metrics import f1_score
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from nltk.tokenize import sent_tokenize
from torch.utils.data.distributed import DistributedSampler

from LSTM_with_CoATT.ahn_modelling import LSTMForUserItemPredictionHIRCOAA
from LSTM_with_CoATT.tokenizer import Tokenizer

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, id=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.id = id
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, sent_mask, review_mask, sent_len, review_label, id, label_id):
        self.input_ids = input_ids
        self.sent_mask = sent_mask
        self.review_mask = review_mask
        self.sent_len = sent_len
        self.review_label = review_label
        self.id = id
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, num_review):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, num_review):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, num_review):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_pkl(cls, input_file, data_dir, num_reviews, t2t, tid2label):
        """Reads a tab separated value file."""

        def coarse_process(id, t2t, length, tid2label):
            sent = t2t[id[1]]
            label = tid2label[id[1]]
            return sent, label

        def text_concate_notime(tid_list, u2tid, t2t, tid2label):
            len2keep = num_reviews
            ntid_list = []
            review_list = []
            tid_list.reverse()
            for tid in tid_list:
                # if u2tid == tid:
                #     continue
                ntid_list.append(tid)
                if len(ntid_list) == len2keep:
                    break
            if len(ntid_list) == 0:
                ntid_list.append((-1, -1))
            for date_id in ntid_list:
                if date_id == (-1, -1):
                    review_list.append(([[0] * 10], 1.0))
                else:
                    review_list.append(coarse_process(date_id, t2t, 0, tid2label))
            return review_list

        with open(data_dir + 'bid2tid.pkl', "rb") as f:
            bid2tid = pickle.load(f)

        with open(data_dir + 'uid2tid.pkl', "rb") as f:
            uid2tid = pickle.load(f)

        with open(input_file, "rb") as f:
            u2is = pickle.load(f)

        user_examples = []
        item_examples = []
        for i, u2i in enumerate(u2is):
            guid = "ins-%d" % (i)
            uid = u2i[0]
            bid = u2i[1]
            u2tid = u2i[2]
            label = u2i[3]
            utid_list = uid2tid[uid]
            btid_list = bid2tid[bid]
            #  extract 4 neighbour reviews for user and item
            text_a = text_concate_notime(utid_list, u2tid, t2t, tid2label)
            text_b = text_concate_notime(btid_list, u2tid, t2t, tid2label)
            user_examples.append(
                InputExample(guid=guid, text_a=text_a, id=uid, label=label))
            item_examples.append(
                InputExample(guid=guid, text_a=text_b, id=bid, label=label))
        return user_examples, item_examples

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class YelpProcessor(DataProcessor):
    """Processor for the Yelp data set."""

    def __init__(self, args, tokenizer):
        super(YelpProcessor, self).__init__()
        if os.path.exists(args.cache_dir + "t2t_glove_sent_token.pkl") is False:
            t2t = {}
            with open(args.data_dir + 't2t.pkl', "rb") as f:
                t2t_nontok = pickle.load(f)
            for id, review in t2t_nontok.items():
                id_list = []
                review_sents = sent_tokenize(review)
                for review_sent in review_sents:
                    tokens_a = tokenizer.tokenize(review_sent)
                    tokens = tokens_a + ["[SEP]"]
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    id_list.append(input_ids)
                t2t[id] = id_list
            with open(args.cache_dir + "t2t_glove_sent_token.pkl", "wb") as f:
                pickle.dump(t2t, f)
        else:
            with open(args.cache_dir + "t2t_glove_sent_token.pkl", "rb") as f:
                t2t = pickle.load(f)


        self.t2t = t2t
        tid2label = {}
        with open(args.data_dir + 'train.pkl', "rb") as f:
            u2is = pickle.load(f)
        for i, u2i in enumerate(u2is):
            rid = u2i[2][1]
            label = u2i[3]
            tid2label[rid] = label
        self.tid2label = tid2label

    def get_train_examples(self, data_dir, num_review):
        """See base class."""
        examples = self._read_pkl(os.path.join(data_dir, "train.pkl"), data_dir,
                                  num_review, self.t2t, self.tid2label)
        return examples

    def get_dev_examples(self, data_dir, num_review):
        """See base class."""

        examples = self._read_pkl(os.path.join(data_dir, "dev.pkl"), data_dir,
                                  num_review, self.t2t, self.tid2label)
        return examples

    def get_test_examples(self, data_dir, num_review):
        """See base class."""

        examples = self._read_pkl(os.path.join(data_dir, "test.pkl"), data_dir,
                                  num_review, self.t2t, self.tid2label)
        return examples

    def get_labels(self):
        """See base class."""
        return [1.0, 2.0, 3.0, 4.0, 5.0]


def convert_examples_to_features(u2i_examples, label_list, id_list, max_seq_length, tokenizer, num_reviews,
                                 max_sent_num):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    id_map = {id: i for i, id in enumerate(id_list)}

    features = []
    for id, u2i in enumerate(u2i_examples):
        examples = u2i.text_a
        input_ids_list = []
        label_id = label_map[u2i.label]
        sent_mask_list = []
        sen_len_list = []
        review_label_list = []
        for (ex_index, example) in enumerate(examples):
            sents = example[0]
            review_label = example[1]
            sent_list = []
            len_list = []
            for tokens_a in sents[:max_sent_num]:

                if len(tokens_a) > max_seq_length:
                    tokens = tokens_a[:(max_seq_length - 1)] + [tokens_a[-1]]
                else:
                    tokens = tokens_a

                input_ids = tokens
                len_list.append(len(input_ids))
                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding

                assert len(input_ids) == max_seq_length

                sent_list.append(input_ids)
            sent_mask = [1] * len(sent_list)
            if len(sent_list) < max_sent_num:
                padding_sents = max_sent_num - len(sents)
                for i in range(padding_sents):
                    sent_list.append([0] * max_seq_length)
                    sent_mask.append(0)
                    len_list.append(1)
            input_ids_list.append(sent_list)
            sent_mask_list.append(sent_mask)
            sen_len_list.append(len_list)
            review_label_list.append(review_label)
        review_mask = [1] * len(examples)
        if len(examples) < num_reviews:
            padding_reviews = num_reviews - len(examples)
            for i in range(padding_reviews):
                # Zero-pad up to the sequence length.
                input_ids = [0] * max_seq_length
                assert len(input_ids) == max_seq_length

                input_ids_list.append([input_ids] * max_sent_num)
                sent_mask_list.append([0] * max_sent_num)
                review_mask.append(0)
                sen_len_list.append([1] * max_sent_num)
                review_label_list.append(0)

        assert len(input_ids_list) == num_reviews
        assert len(review_mask) == num_reviews
        uid = id_map[u2i.id]
        features.append(
            InputFeatures(input_ids=input_ids_list,
                          sent_mask=sent_mask_list,
                          review_mask=review_mask,
                          sent_len=sen_len_list,
                          review_label=review_label_list,
                          id=uid,
                          label_id=label_id))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def eval(eval_examples, model, label_list, id_list, tokenizer, device, args):
    eval_u_features = convert_examples_to_features(
        eval_examples[0], label_list, id_list[0], args.max_seq_length, tokenizer, args.num_reviews, args.num_sentence)
    eval_i_features = convert_examples_to_features(
        eval_examples[1], label_list, id_list[1], args.max_seq_length, tokenizer, args.num_reviews, args.num_sentence)
    logger.info("***** Running evaluation *****")

    logger.info("  Num examples = %d", len(eval_examples[0]))
    logger.info("  Batch size = %d", args.eval_batch_size)
    u_input_ids = torch.stack(
        [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.input_ids], 0) for f in
         eval_u_features], 0)
    u_sent_mask = torch.stack(
        [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.sent_mask], 0) for f in
         eval_u_features], 0)
    u_review_mask = torch.stack([torch.tensor(f.review_mask, dtype=torch.long) for f in
                                 eval_u_features], 0)
    u_sent_len = torch.stack(
        [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.sent_len], 0) for f in
         eval_u_features], 0)
    i_input_ids = torch.stack(
        [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.input_ids], 0) for f in
         eval_i_features], 0)
    i_sent_mask = torch.stack(
        [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.sent_mask], 0) for f in
         eval_i_features], 0)
    i_sent_len = torch.stack(
        [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.sent_len], 0) for f in
         eval_i_features], 0)
    i_review_mask = torch.stack([torch.tensor(f.review_mask, dtype=torch.long) for f in
                                 eval_i_features], 0)
    i_review_label = torch.stack(
        [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.review_label], 0) for f in
         eval_i_features], 0)
    u_ids = torch.tensor([f.id for f in eval_u_features], dtype=torch.long)
    i_ids = torch.tensor([f.id for f in eval_i_features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in eval_i_features], dtype=torch.float)

    if args.fp16:
        label_ids = label_ids.half()
    eval_data = TensorDataset(u_input_ids, i_input_ids, u_sent_mask, i_sent_mask, u_sent_len, i_sent_len,
                              u_review_mask, i_review_mask, i_review_label, u_ids, i_ids, label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for u_input_ids, input_ids, u_sent_mask, i_sent_mask, u_sent_len, i_sent_len, \
        u_review_mask, i_review_mask, i_review_label, u_ids, i_ids, label_ids in eval_dataloader:
        u_input_ids = u_input_ids.to(device)
        input_ids = input_ids.to(device)
        u_sent_mask, i_sent_mask, u_review_mask, i_review_mask, u_sent_len, i_sent_len, i_review_label\
            = u_sent_mask.to(device), i_sent_mask.to(device), u_review_mask.to(device), i_review_mask.to(device), \
              u_sent_len.to(device), i_sent_len.to(device), i_review_label.to(device)
        label_ids = label_ids.to(device)
        u_ids = u_ids.to(device)
        i_ids = i_ids.to(device)

        with torch.no_grad():
            logits, u_sw, i_sw, u_rw, i_rw = model(u_input_ids, input_ids, u_sent_mask, i_sent_mask, u_sent_len,
                                                   i_sent_len,
                                                   u_review_mask, i_review_mask, i_review_label, u_ids, i_ids)
        loss_fct = MSELoss(reduction='sum')
        tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_examples
    return eval_loss


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where the data cache  is be put.")

    parser.add_argument("--vocab_file",
                        default='vocab-amkin.txt',
                        type=str,
                        help="The input vocab_file")

    parser.add_argument("--num_reviews",
                        default=8,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--num_head",
                        default=4,
                        type=int,
                        help="num of co-attention module")

    parser.add_argument("--num_sentence",
                        default=8,
                        type=int,
                        help="num of sentence in one review")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')

    parser.add_argument('-hidden-size', type=int, default=300, help='number of each kind of kernel')

    args = parser.parse_args()

    with open(args.data_dir + 'id_list.pkl', "rb") as f:
        id_list = pickle.load(f)
    args.num_uid = len(id_list[0])
    args.num_iid = len(id_list[1])
        
    args.embed_num = 25089

    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

    processors = {
        "yelp": YelpProcessor
    }

    num_labels_task = {
        "yelp": 5
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.output_dir = './output/' + args.output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        print("load model from directory ({})".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    tokenizer = Tokenizer(args.cache_dir + args.vocab_file, do_lower_case=args.do_lower_case)

    processor = processors[task_name](args, tokenizer)
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    train_examples = None
    num_train_steps = None

    # if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir, args.num_reviews)
    num_train_steps = int(
        len(train_examples[0]) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    train_u_features = convert_examples_to_features(
        train_examples[0], label_list, id_list[0], args.max_seq_length, tokenizer, args.num_reviews,
        args.num_sentence)
    train_i_features = convert_examples_to_features(
        train_examples[1], label_list, id_list[1], args.max_seq_length, tokenizer, args.num_reviews,
        args.num_sentence)
    train_ids = torch.tensor([f.label_id for f in train_i_features], dtype=torch.float)
    args.avg_rating = torch.mean(train_ids)


    model = LSTMForUserItemPredictionHIRCOAA(args)


    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    global_step = 0
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    tr_loss = 0
    nb_tr_steps = 1
    best_loss = sys.maxsize
    if args.do_train:

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples[0]))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        u_input_ids = torch.stack(
            [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.input_ids], 0) for f in
             train_u_features], 0)
        u_sent_mask = torch.stack(
            [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.sent_mask], 0) for f in
             train_u_features], 0)
        u_review_mask = torch.stack([torch.tensor(f.review_mask, dtype=torch.long) for f in
                                     train_u_features], 0)
        u_sent_len = torch.stack(
            [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.sent_len], 0) for f in
             train_u_features], 0)
        i_input_ids = torch.stack(
            [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.input_ids], 0) for f in
             train_i_features], 0)
        i_sent_mask = torch.stack(
            [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.sent_mask], 0) for f in
             train_i_features], 0)
        i_sent_len = torch.stack(
            [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.sent_len], 0) for f in
             train_i_features], 0)
        i_review_mask = torch.stack([torch.tensor(f.review_mask, dtype=torch.long) for f in
                                     train_i_features], 0)
        i_review_label = torch.stack(
            [torch.stack([torch.tensor((ids), dtype=torch.long) for ids in f.review_label], 0) for f in
             train_i_features], 0)
        u_ids = torch.tensor([f.id for f in train_u_features], dtype=torch.long)
        i_ids = torch.tensor([f.id for f in train_i_features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in train_i_features], dtype=torch.float)


        if args.fp16:
            label_ids = label_ids.half()

        train_data = TensorDataset(u_input_ids, i_input_ids, u_sent_mask, i_sent_mask, u_sent_len, i_sent_len,
                                   u_review_mask, i_review_mask, i_review_label, u_ids, i_ids, label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        eval_examples = processor.get_dev_examples(args.data_dir, args.num_reviews)

        model.train()

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                u_input_ids, input_ids, u_sent_mask, i_sent_mask, u_sent_len, i_sent_len, \
                u_review_mask, i_review_mask, i_review_label, u_ids, i_ids, label_ids = batch
                
                logits, u_sw, i_sw, u_rw, i_rw = model(u_input_ids, input_ids, u_sent_mask, i_sent_mask, u_sent_len,
                                                       i_sent_len,
                                                       u_review_mask, i_review_mask, i_review_label, u_ids, i_ids)

                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))
                loss += 1 * model.module.FM.V.pow(2).sum()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learniCng rate with special warm up BERT uses
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                sys.stdout.write(
                    '\rMSE loss[{}]'.format(loss.item()))

            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                val_model = LSTMForUserItemPredictionHIRCOAA(args)
                val_model.load_state_dict(model_to_save.state_dict())
                val_model.to(device)
                eval_loss = eval(eval_examples, val_model, label_list, id_list, tokenizer, device, args)
                logger.info("  loss = %f", eval_loss)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    # Save a trained model
                    # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                model.train()
    #
    # # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    best_model = LSTMForUserItemPredictionHIRCOAA(args)
    best_model.load_state_dict(model_state_dict)
    best_model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir, args.num_reviews)
        eval_loss = eval(eval_examples, best_model, label_list, id_list, tokenizer, device, args)

        result = {'test_loss': eval_loss,
                  'global_step': global_step,
                  'best_loss': best_loss,
                  'loss': tr_loss / nb_tr_steps}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
