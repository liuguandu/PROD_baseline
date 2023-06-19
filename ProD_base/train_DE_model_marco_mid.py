from os.path import join
import sys


sys.path += ['../']
sys.path += ['../../']
import argparse
import glob
import json
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import six
import pickle
import faiss

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
#  
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from model.models import BiEncoderNllLoss, BiBertEncoder, ColBERT, ColBERTNllLoss
import random
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_processors as processors
from torch import nn
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import pandas as pd

logger = logging.getLogger(__name__)
from utils.util import (
    set_seed,
    is_first_worker,
    TraditionDataset
)
from utils.dpr_utils import (
    load_states_from_checkpoint,
    get_model_obj,
    CheckpointState,
    get_optimizer,
    all_gather_list
)
import collections
from torch.nn.utils.rnn import pad_sequence
from utils.marco_until import (
    Rocketqa_v2Dataset
)

retrieverBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "q_ids",
        "q_attn_mask",
        "c_ids",
        "c_attn_mask",
        "c_q_mapping",
        "is_positive",
    ],
)

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

class Question_dataset(Dataset):
    def __init__(self, questions, tokenizer,maxlength=32):
        self.questions = questions
        self.tokenizer = tokenizer
        self.maxlength = maxlength
    def __getitem__(self, index):
        example = self.questions[index]
        input_ids = self.tokenizer.encode(example[1], add_special_tokens=True,
                                        max_length=self.maxlength, truncation=True,
                                       padding='max_length',return_tensors='pt') 
        return example[0], input_ids

    def __len__(self):
        return len(self.questions)

    @classmethod
    def get_collate_fn(cls, args):
        def fn(features):
            id_list = [feature[0] for feature in features]
            q_tensor = torch.cat([feature[1] for feature in features])
            return np.array(id_list), q_tensor, (q_tensor != 0).long()
        return fn
class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,tokenizer,maxlength =128):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlength = maxlength
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        title = convert_to_unicode(example[2])
        text = convert_to_unicode(example[1])
        input_ids = self.tokenizer.encode(title, text_pair=text, add_special_tokens=True,
                                        max_length=self.maxlength, truncation=True,
                                       padding='max_length',return_tensors='pt') 
        return example[0],input_ids
    @classmethod
    def get_collate_fn(cls, args):
        def fn(features):
            id_list = [feature[0] for feature in features]
            input_ids = torch.cat([feature[1] for feature in features])
            return np.array(id_list), input_ids, (input_ids!= 0).long()
        return fn
def train(args, model, tokenizer):
    """ Train the model """
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # nll loss for query
    optimizer = get_optimizer(args, model, weight_decay=args.weight_decay)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        # from apex.parallel import DistributedDataParallel as DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=False,
        )
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    tr_loss = 0.0
    tr_distll_loss = 0.0
    tr_contr_loss = 0.0
    model.zero_grad()
    model.train()
    set_seed(args)  # Added here for reproductibility
    iter_count = 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )
    global_step = 0
    train_dataset = Rocketqa_v2Dataset(args.origin_data_dir,tokenizer,num_hard_negatives=args.number_neg,
                                        max_seq_length=args.max_seq_length, corpus_path=args.corpus_path)
    train_sample = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sample,
                        collate_fn=Rocketqa_v2Dataset.get_collate_fn(args),
                        batch_size=args.train_batch_size,num_workers=10)

    if args.output_dir is not None:
        checkpoint_files = []
        if os.path.exists(args.output_dir):
            for item in os.scandir(args.output_dir):
                if item.is_file():
                    if "checkpoint" in item.path:
                        checkpoint_files.append(item.path)
            if len(checkpoint_files) != 0:
                checkpoint_files.sort(key=lambda f: int(f.split('checkpoint-')[1]), reverse=True)
                logger.info("***** load " + checkpoint_files[0] + " *****")
                saved_state = load_states_from_checkpoint(checkpoint_files[0])
                global_step = _load_saved_state(model, optimizer, scheduler, saved_state)
            else:
                logger.info("***** there are no checkpoint in" + args.output_dir + " *****")

    # validate_rank = evaluate_dev(args, model, tokenizer)[0]
    # print(validate_rank)
    while global_step < args.max_steps:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        if args.num_epoch != 0 and iter_count > args.num_epoch:
            break
        # train_dataset = load_stream_dataset(args)

        for step, batch in enumerate(epoch_iterator):
            model.train()

            batch_retriever = batch['retriever']
            inputs_retriever = {"query_ids": batch_retriever[0].long().to(args.device),
                            "attention_mask_q": batch_retriever[1].long().to(args.device),
                            "input_ids_a": batch_retriever[2].long().to(args.device),
                            "attention_mask_a": batch_retriever[3].long().to(args.device)}
            local_positive_idxs = batch_retriever[4]
            model.train()
            if args.model_class == 'dual_encoder':
                # print("query_ids:", inputs_retriever['query_ids'].shape)
                # print("input_ids_a:", inputs_retriever['input_ids_a'].shape)
                local_q_vector, local_ctx_vectors = model(**inputs_retriever)
                loss, is_correct = caculate_cont_loss(args, local_q_vector, local_ctx_vectors, local_positive_idxs)
            elif args.model_class == 'ColBERT':
                _, _, local_q_hidden, local_ctx_hidden = model(**inputs_retriever)
                # loss_function = ColBERTNllLoss()
                # loss, is_correct = loss_function.calc(
                #     args,
                #     local_q_hidden,
                #     local_ctx_hidden,
                #     inputs_retriever['attention_mask_a'],
                #     local_positive_idxs,
                # )
                loss, is_correct = caculate_Col_loss(args, local_q_hidden, local_ctx_hidden, inputs_retriever['attention_mask_a'], local_positive_idxs)
            else:
                logger.info("no such type model" + args.model_class)
                exit(0)

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = tr_loss / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    tr_loss = 0
                    if is_first_worker():
                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        logger.info(json.dumps({**logs, **{"step": global_step}}))
                # if global_step % 10 == 0:
                    # evaluate_metrics(args, tokenizer, model)
                    # evaluate_dev(args, model, tokenizer)
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    # if global_step > 500000:
                    #     validate_rank = evaluate_dev(args, model, tokenizer)
                    # else:
                    #     validate_rank = evaluate_dev(args, model, tokenizer)[0]
                    if is_first_worker():
                        _save_checkpoint(args, model, optimizer, scheduler, global_step)
                        # tb_writer.add_scalar("dev_nll_loss/dev_avg_rank", validate_rank, global_step)
                if global_step >= args.max_steps:
                    break
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()
    return global_step


def caculate_cont_loss(args, local_q_vector, local_ctx_vectors, local_positive_idxs):
    if torch.distributed.get_world_size() > 1:
        q_vector_to_send = (
            torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        )
        ctx_vector_to_send = (
            torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()
        )

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
            ],
            max_size=640000000,
        )

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        # hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx = item

            if i != args.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend(
                    [v + total_ctxs for v in local_positive_idxs]
                )
            total_ctxs += ctx_vectors.size(0)
        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)
    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs

    loss_function = BiEncoderNllLoss()
    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
    )
    return loss, is_correct

def caculate_Col_loss(args, local_q_hidden, local_ctx_hidden, local_ctx_mask, local_positive_idxs):
    if torch.distributed.get_world_size() > 1:
        q_hidden_to_send = (
            torch.empty_like(local_q_hidden).cpu().copy_(local_q_hidden).detach_()
        )
        ctx_hidden_to_send = (
            torch.empty_like(local_ctx_hidden).cpu().copy_(local_ctx_hidden).detach_()
        )
        ctx_mask_to_send = (
            torch.empty_like(local_ctx_mask).cpu().copy_(local_ctx_mask).detach_()
        )

        global_question_ctx_vectors = all_gather_list(
            [
                q_hidden_to_send,
                ctx_hidden_to_send,
                ctx_mask_to_send,
                local_positive_idxs,
            ],
            max_size=640000000,
        )

        global_q_hidden = []
        global_ctx_hidden = []
        global_ctx_mask = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        # hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_hidden, ctx_hidden, ctx_mask, positive_idx = item

            if i != args.local_rank:
                global_q_hidden.extend(q_hidden.to(local_q_hidden.device))
                global_ctx_hidden.extend(ctx_hidden.to(local_q_hidden.device))
                global_ctx_mask.extend(ctx_mask.to(local_q_hidden.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
            else:
                global_q_hidden.extend(local_q_hidden)
                global_ctx_hidden.extend(local_ctx_hidden)
                global_ctx_mask.extend(local_ctx_mask)
                positive_idx_per_question.extend(
                    [v + total_ctxs for v in local_positive_idxs]
                )
            total_ctxs += ctx_hidden.size(0)

        global_q_hidden = pad_sequence(global_q_hidden, batch_first=True)
        global_ctx_hidden = pad_sequence(global_ctx_hidden, batch_first=True)
        global_ctx_mask = pad_sequence(global_ctx_mask, batch_first=True)
        #global_q_hidden = torch.cat(global_q_hidden, dim=0)
        #global_ctx_hidden = torch.cat(global_ctx_hidden, dim=0)
    else:
        global_q_hidden = local_q_hidden
        global_ctx_hidden = local_ctx_hidden
        global_ctx_mask = local_ctx_mask
        positive_idx_per_question = local_positive_idxs

    loss_function = ColBERTNllLoss()
    loss, is_correct = loss_function.calc(
        args,
        global_q_hidden,
        global_ctx_hidden,
        global_ctx_mask,
        positive_idx_per_question,
    )
    return loss, is_correct

def sum_main(x, opt):
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x
def get_question_embeddings(args, questions, tokenizer, model):
    batch_size = 1024

    dataset = Question_dataset(questions,tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                             num_workers=20, collate_fn=Question_dataset.get_collate_fn(args))
    total = 0
    allids, allembeddings = [], []
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in enumerate(tqdm(dataloader)):
            inputs = {"input_ids": text_ids.long().to(args.device), "attention_mask": text_mask.long().to(args.device)}
            embs = model.module.query_emb(**inputs)
            embeddings = embs.detach().cpu()
            total += len(ids)

            allids.append(ids)
            allembeddings.append(embeddings)
            if k % 100 == 0:
                logger.info('Encoded question %d', total)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allids = np.array([x for idlist in allids for x in idlist])
    return allembeddings,allids 
def load_id_text(file_name):
    """load tsv files"""
    id_text = {}
    with open(file_name) as inp:
        for line in tqdm(inp):
            line = line.strip()
            id, text = line.split('\t')
            id_text[id] = text
    return id_text
def load_data(args):
    # passage_title_path = os.path.join('./marco',"para.title.txt")
    # passage_ctx_path = os.path.join('./marco',"para.txt")
    passage_title_path = "/data/liuguandu/SimXNS/PROD/marco/para.title.txt"
    passage_ctx_path = "/data/liuguandu/SimXNS/PROD/marco/para.txt"
    passage_title = load_id_text(passage_title_path)
    passages = []
    with open(passage_ctx_path) as inp:
        for line in tqdm(inp):
            line = line.strip()
            id, text = line.split('\t')
            # return id , passage , passage title
            passages.append((int(id), text, passage_title.get(id, '-')))
    return passages
def embed_passages(opt, passages, model, tokenizer):
    batch_size = 1024
    dataset = TextDataset(passages, tokenizer, 144)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=20, collate_fn=TextDataset.get_collate_fn(opt))
    total = 0
    allids, allembeddings = [], []
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in enumerate(tqdm(dataloader)):
            inputs = {"input_ids": text_ids.long().to(opt.device), "attention_mask": text_mask.long().to(opt.device)}
            embs = model.module.body_emb(**inputs)
            embeddings = embs.detach().cpu()
            # print(ids)
            # print(embeddings.size())
            total += len(ids)

            allids.append(ids)
            allembeddings.append(embeddings)
            # if k % 10000 == 0:
            #     break
            if k % 100 == 0:
                logger.info('Encoded passages %d', total)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allids = np.array([x for idlist in allids for x in idlist])
    return allids, allembeddings
def get_passage_embedding(args, passages, model, tokenizer):

    shard_size = len(passages) // 10000 // args.world_size
    print('shard_size:', shard_size)
    start_idx = args.local_rank * shard_size
    end_idx = start_idx + shard_size
    if args.local_rank == args.world_size - 1:
        end_idx = len(passages)
    passages_piece = passages[start_idx:end_idx]
    logger.info(f'Embedding generation for {len(passages_piece)} passages from idx {start_idx} to {end_idx}')
    allids, allembeddings = embed_passages(args, passages_piece, model, tokenizer)
    print("end_passage")
    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    dist.barrier()
    print("L1")
    pickle_path = os.path.join(args.output_dir,
                                "{1}_data_obj_{0}.pb".format(str(args.local_rank), 'passage_embedding'))
    with open(pickle_path, 'wb') as handle:
        pickle.dump(allembeddings, handle, protocol=4)
    print("L2")
    pickle_path = os.path.join(args.output_dir,
                                "{1}_data_obj_{0}.pb".format(str(args.local_rank), 'passage_embedding_id'))
    with open(pickle_path, 'wb') as handle:
        pickle.dump(allids, handle, protocol=4)
    print("L3")
    logger.info(f'Total passages processed {len(allids)}. Written to {pickle_path}.')
    dist.barrier()
    passage_embedding,passage_embedding_id = None,None
    
    if is_first_worker():
        passage_embedding_list = []
        passage_embedding_id_list = []
        for i in range(args.world_size):  # TODO: dynamically find the max instead of HardCode
            pickle_path = os.path.join(args.output_dir,"{1}_data_obj_{0}.pb".format(str(i),'passage_embedding'))
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                passage_embedding_list.append(b)
        for i in range(args.world_size):  # TODO: dynamically find the max instead of HardCode
            pickle_path = os.path.join(args.output_dir,"{1}_data_obj_{0}.pb".format(str(i),'passage_embedding_id'))
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                passage_embedding_id_list.append(b)
        passage_embedding = np.concatenate(passage_embedding_list, axis=0)
        passage_embedding_id = np.concatenate(passage_embedding_id_list, axis=0)
    dist.barrier()
    return passage_embedding, passage_embedding_id
def load_reference_from_stream(path_to_reference):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_to_relevant_passageids = {}
    with open(path_to_reference, 'r') as f:
        for l in f:
            try:
                l = l.strip().split('\t')
                qid = int(l[0])
                if qid in qids_to_relevant_passageids:
                    pass
                else:
                    qids_to_relevant_passageids[qid] = []
                qids_to_relevant_passageids[qid].append(int(l[1]))
            except:
                raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids
def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    MaxMRRRank = 10
    all_scores = {}
    MRR = 0
    qids_with_relevant_passages = 0
    ranking = []
    recall_q_top1 = set()
    recall_q_top5 = set()
    recall_q_top20 = set()
    recall_q_top50 = set()
    recall_q_top100 = set()
    recall_q_all = set()

    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    MRR += 1.0 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
            for i, pid in enumerate(candidate_pid):
                if pid in target_pid:
                    recall_q_all.add(qid)
                    if i < 100:
                        recall_q_top100.add(qid)
                    if i < 50:
                        recall_q_top50.add(qid)
                    if i < 20:
                        recall_q_top20.add(qid)
                    if i < 5:
                        recall_q_top5.add(qid)
                    if i == 0:
                        recall_q_top1.add(qid)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    MRR = MRR / len(qids_to_relevant_passageids)
    recall_top1 = len(recall_q_top1) * 1.0 / len(qids_to_relevant_passageids)
    recall_top5 = len(recall_q_top5) * 1.0 / len(qids_to_relevant_passageids)
    recall_top20 = len(recall_q_top20) * 1.0 / len(qids_to_relevant_passageids)
    recall_top50 = len(recall_q_top50) * 1.0 / len(qids_to_relevant_passageids)
    recall_top100 = len(recall_q_top100) * 1.0 / len(qids_to_relevant_passageids)
    recall_all = len(recall_q_all) * 1.0 / len(qids_to_relevant_passageids)
    all_scores['MRR @10'] = MRR
    all_scores["recall@1"] = recall_top1
    all_scores["recall@5"] = recall_top5
    all_scores["recall@20"] = recall_top20
    all_scores["recall@50"] = recall_top50
    all_scores["recall@100"] = recall_top100
    all_scores["recall@all"] = recall_all
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    all_scores['record'] = str(round(recall_top1*100, 2)) + " / " + str(round(recall_top5*100, 2)) +" / " + str(round(recall_top20*100, 2)) +" / " + str(round(recall_top50*100, 2)) +" / " + str(round(recall_top100*100, 2)) +" / " + str(round(recall_all*100, 2))
    return all_scores
def evaluate_metrics(args, tokenizer, model):
    print('=============begin eval==============')
    passages = load_data(args)
    print("***** inference of passages *****")
    passage_embedding, passage_embedding2id = get_passage_embedding(args, passages, model,tokenizer)
    test_qa_path = "/data/liuguandu/SimXNS/PROD/marco/dev.query.txt"

    if test_qa_path is not None and is_first_worker():
        test_questions = []
        with open(test_qa_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                id, text = line.split('\t')
                test_questions.append([int(id), text])
        test_question_embedding, test_question_embedding2id = get_question_embeddings(args, test_questions, 
                                                                                        tokenizer, model)
    if is_first_worker():
        passage_text = {}
        for passage in passages:
            passage_text[passage[0]] = (passage[1], passage[2])
        dim = passage_embedding.shape[1]
        print('passage embedding shape: ' + str(passage_embedding.shape))
        new_passage_embedding = passage_embedding.copy()
        for i in range(passage_embedding.shape[0]):
            new_passage_embedding[passage_embedding2id[i]] = passage_embedding[i]
        del (passage_embedding)
        passage_embedding = new_passage_embedding
        passage_embedding2id = np.arange(passage_embedding.shape[0])
        top_k = 1000
        # faiss.omp_set_num_threads(args.thread_num)
        cpu_index = faiss.IndexFlatIP(dim)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        # co.useFloat16 = True
        gpu_index_flat = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index,
            co=co
        )
        # gpu_index_flat = faiss.index_cpu_to_all_gpus(cpu_index)
        gpu_index_flat.add(passage_embedding.astype(np.float32))
        print("***** Done ANN Index *****")
        similar_scores, dev_I = gpu_index_flat.search(test_question_embedding.astype(np.float32),
                                                    top_k)
        qids_to_ranked_candidate_passages  = {}
        qids_to_ranked_candidate_scores = {}
        for index,ranked_candidate_passages in enumerate(dev_I):
            qids_to_ranked_candidate_passages[test_question_embedding2id[index]] = ranked_candidate_passages
            qids_to_ranked_candidate_scores[test_question_embedding2id[index]] = similar_scores[index]

        output_path = os.path.join(args.output_dir, 'test_result_dict_list.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump([qids_to_ranked_candidate_passages,qids_to_ranked_candidate_scores], f)
        ground_truth_path = "/data/liuguandu/SimXNS/PROD/marco/qrels.dev.tsv"
        qids_to_relevant_passageids = load_reference_from_stream(ground_truth_path)
        all_scores = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        ndcg_output_path = os.path.join(args.output_dir, "test_eval_result.json")
        print(all_scores)

def evaluate_dev(args, model, tokenizer):

    dev_dataset = Rocketqa_v2Dataset(args.origin_data_dir_dev,tokenizer,num_hard_negatives=args.number_neg,is_training=False,
                                        max_seq_length=args.max_seq_length, corpus_path=args.corpus_path)
    dev_sample = RandomSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sample,
                        collate_fn=Rocketqa_v2Dataset.get_collate_fn(args),
                        batch_size=args.train_batch_size,num_workers=0,shuffle=False)
    correct_predictions_count_all = 0
    example_num = 0
    total_loss = 0
    model.eval()
    print("dev_dataloader", len(dev_dataloader))
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            if i== 1000:
                break
            # print("LLLL")
            batch_retriever = batch['retriever']
            inputs_retriever = {"query_ids": batch_retriever[0].long().to(args.device),
                      "attention_mask_q": batch_retriever[1].long().to(args.device),
                      "input_ids_a": batch_retriever[2].long().to(args.device),
                      "attention_mask_a": batch_retriever[3].long().to(args.device)}

            if args.model_class == 'dual_encoder':
                local_q_vector, local_ctx_vectors = model(**inputs_retriever)
            elif args.model_class == 'ColBERT':
                local_q_vector, local_ctx_vectors, _, _ = model(**inputs_retriever)
            else:
                logger.info("no such type model" + args.model_class)
                exit(0)

            question_num = local_q_vector.size(0)
            retriever_local_ctx_vectors = local_ctx_vectors.reshape(question_num, local_ctx_vectors.size(0) // question_num, -1)

            relevance_logits = torch.einsum("bh,bdh->bd", [local_q_vector, retriever_local_ctx_vectors])
            # retriever_dist_p = F.softmax(retriever_simila, dim=1)
            # batch_reranker = tuple(t.to(args.device) for t in batch['reranker'])
            # inputs_reranker = {"input_ids": batch_reranker[0].long(), "attention_mask": batch_reranker[1].long()}
            # output_reranker = model(**inputs_reranker)
            # binary_logits,relevance_logits,_ =output_reranker
            relevance_target = torch.zeros(relevance_logits.size(0), dtype=torch.long).to(args.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            relative_loss = loss_fct(relevance_logits,relevance_target)
            total_loss+=relative_loss
            max_score, max_idxs = torch.max(relevance_logits, 1)
            correct_predictions_count = (max_idxs == 0).sum()
            correct_predictions_count_all+=correct_predictions_count
            example_num += batch['reranker'][1].size(0)
    example_num = torch.tensor(1).to(relevance_logits)*example_num
    total_loss = torch.tensor(1).to(relevance_logits)*total_loss
    correct_predictions_count_all = torch.tensor(1).to(relevance_logits)*correct_predictions_count_all
    correct_predictions_count_all = sum_main(correct_predictions_count_all,args)
    example_num = sum_main(example_num,args)
    total_loss = sum_main(total_loss,args)
    total_loss = total_loss / i
    correct_ratio = float(correct_predictions_count_all / example_num)
    logger.info('NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f', total_loss,
                correct_predictions_count_all.item(),
                example_num.item(),
                correct_ratio
                )

    model.train()
    return total_loss, correct_ratio


def do_biencoder_fwd_pass_eval(args, model, batch):
    batch = tuple(t.to(args.device) for t in batch)
    inputs = {"query_ids": batch[0][::2].long(), "attention_mask_q": batch[1][::2].long(),
              "input_ids_a": batch[3].long(), "attention_mask_a": batch[4].long()}

    local_q_vector, local_ctx_vectors = model(**inputs)

    q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
    ctx_vector_to_send = torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()

    global_question_ctx_vectors = all_gather_list(
        [q_vector_to_send, ctx_vector_to_send],
        max_size=640000000)

    global_q_vector = []
    global_ctxs_vector = []

    for i, item in enumerate(global_question_ctx_vectors):
        q_vector, ctx_vectors = item

        if i != args.rank:
            global_q_vector.append(q_vector.to(local_q_vector.device))
            global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
        else:
            global_q_vector.append(local_q_vector)
            global_ctxs_vector.append(local_ctx_vectors)

    global_q_vector = torch.cat(global_q_vector, dim=0)
    global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    scores = torch.matmul(global_q_vector, torch.transpose(global_ctxs_vector, 0, 1))
    if len(global_q_vector.size()) > 1:
        q_num = global_q_vector.size(0)
        scores = scores.view(q_num, -1)
    softmax_scores = F.log_softmax(scores, dim=1)
    positive_idx_per_question = [i * 2 for i in range(q_num)]
    loss = F.nll_loss(softmax_scores, torch.tensor(positive_idx_per_question).to(softmax_scores.device),
                      reduction='mean')
    max_score, max_idxs = torch.max(softmax_scores, 1)
    correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

    is_correct = correct_predictions_count.sum().item()

    if args.n_gpu > 1:
        loss = loss.mean()
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    return loss, is_correct

def triplet_fwd_pass(args, model, batch):
    batch = tuple(t.to(args.device) for t in batch)
    inputs = {"query_ids": batch[0].long(), "attention_mask_q": batch[1].long(),
              "input_ids_a": batch[3].long(), "attention_mask_a": batch[4].long(),
              "input_ids_b": batch[6].long(), "attention_mask_b": batch[7].long()}
    loss = model(**inputs)[0]

    if args.n_gpu > 1:
        loss = loss.mean()
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    return loss


def _save_checkpoint(args, model, optimizer, scheduler, step: int) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(args.output_dir, 'checkpoint-' + str(offset))

    meta_params = {}

    state = CheckpointState(model_to_save.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            offset,
                            epoch, meta_params
                            )
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp


def _load_saved_state(model, optimizer, scheduler, saved_state: CheckpointState):
    epoch = saved_state.epoch
    step = saved_state.offset
    logger.info('Loading checkpoint @ step=%s', step)

    model_to_load = get_model_obj(model)
    logger.info('Loading saved model state ...')
    model_to_load.load_state_dict(saved_state.model_dict)  # set strict=False if you use extra projection
    optimizer.load_state_dict(saved_state.optimizer_dict)
    scheduler.load_state_dict(saved_state.scheduler_dict)
    return step


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list:",
    )
    # cross encoder、ColBERT、dual-encoder
    parser.add_argument(
        "--model_class",
        default="dual_encoder",
        type=str,
        help="Text model selection",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path_ict",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--corpus_path",
        default=None,
        type=str,
        required=True,
        help="The directory where the marco title and text were written.",
    )

    parser.add_argument(
        "--num_epoch",
        default=0,
        type=int,
        help="Number of epoch to train, if specified will use training data instead of ann",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--num_hidden_layers",
        default=12,
        type=int,
        help="Number of floors of each tower in the two tower model",
    )

    parser.add_argument("--triplet", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--optimizer",
        default="adamW",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=300000,
        type=int,
        help="If > 0: set total number of training steps to perform",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
            "--gradient_checkpointing",
            default=False,
            action="store_true",
        )
    parser.add_argument(
            "--origin_data_dir",
            default=None,
            type=str,
        )
    parser.add_argument(
            "--origin_data_dir_dev",
            default=None,
            type=str,
        )
    # ----------------- ANN HyperParam ------------------

    parser.add_argument(
        "--load_optimizer_scheduler",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--single_warmup",
        default=True,
        action="store_true",
        help="use single or re-warmup",
    )

    parser.add_argument("--adv_data_path",
                        type=str,
                        default=None,
                        help="adv_data_path", )
                        
    parser.add_argument("--ann_data_path",
                        type=str,
                        default=None,
                        help="adv_data_path", )
    parser.add_argument(
            "--fix_embedding",
            default=False,
            action="store_true",
            help="use single or re-warmup",
        )
    parser.add_argument(
            "--continue_train",
            default=False,
            action="store_true",
            help="use single or re-warmup",
        )
    parser.add_argument(
            "--adv_loss_alpha",
            default=0.3,
            type=float,
            help="use single or re-warmup",
        )
    parser.add_argument(
            "--shuffle_positives",
            default=False,
            action="store_true",
            help="use single or re-warmup")
    parser.add_argument("--reranker_model_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--reranker_model_type", type=str, default="", help="For distant debugging.")
    parser.add_argument("--number_neg", type=int, default=20, help="For distant debugging.")
    parser.add_argument("--adv_max_norm", default=0., type=float)
    parser.add_argument("--adv_init_mag", default=0, type=float)
    parser.add_argument("--adv_lr", default=5e-2, type=float)
    parser.add_argument("--adv_steps", default=3, type=int)
    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--similarity_metric", type=str, default='cosine', help="For distant debugging.")
    args = parser.parse_args()

    return args


def set_env(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)


def load_states_from_checkpoint_ict(model_file: str) -> CheckpointState:
    from torch.serialization import default_restore_location
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    new_stae_dict = {}
    for key,value in state_dict['model']['query_model']['language_model'].items():
        new_stae_dict['question_model.'+key] = value
    for key,value in state_dict['model']['context_model']['language_model'].items():
        new_stae_dict['ctx_model.'+key] = value
    return new_stae_dict

# def load_model(args):
#     # Prepare GLUE task
#     args.output_mode = "classification"
#     label_list = ["0", "1"]
#     num_labels = len(label_list)

#     # store args
#     if args.local_rank != -1:
#         args.world_size = torch.distributed.get_world_size()
#         args.rank = dist.get_rank()

#     # Load pretrained model and tokenizer
#     if args.local_rank not in [-1, 0]:
#         torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

#     args.model_type = args.model_type
   
#     if is_first_worker():
#         # Create output directory if needed
#         if not os.path.exists(args.output_dir):
#             os.makedirs(args.output_dir)
#     tokenizer = BertTokenizer.from_pretrained(
#             "bert-base-uncased",
#             do_lower_case=True)
#     if args.model_class == "dual_encoder":
#         model = BiBertEncoder(args)
#     elif args.model_class == "ColBERT":
#         model = ColBERT(args)
#     else:
#         logger.info("no such type of model " + args.model_type)
#         exit(0)

#     if args.model_name_or_path_ict is not None:
#         saved_state = load_states_from_checkpoint_ict(args.model_name_or_path_ict)
#         model.load_state_dict(saved_state)
#     if args.model_name_or_path is not None:
#         saved_state = load_states_from_checkpoint(args.model_name_or_path)
#         model.load_state_dict(saved_state.model_dict,strict=False)

#     # if args.output_dir is not None:
#     #     checkpoint_files = []
#     #     if os.path.exists(args.output_dir):
#     #         for item in os.scandir(args.output_dir):
#     #             if item.is_file():
#     #                 if "checkpoint" in item.path:
#     #                     checkpoint_files.append(item.path)
#     #         if len(checkpoint_files) != 0:
#     #             checkpoint_files.sort(key=lambda f: int(f.split('checkpoint-')[1]),reverse=True)
#     #             logger.info("***** load " + checkpoint_files[0] + " *****")
#     #             saved_state = load_states_from_checkpoint(checkpoint_files[0])
#     #             model.load_state_dict(saved_state)
#     #         else:
#     #             logger.info("***** there are no checkpoint in" + args.output_dir + " *****")

        
#     # global_step = _load_saved_state(model, optimizer, scheduler, saved_state)
#     if args.fix_embedding:
#         word_embedding = model.ctx_model.get_input_embeddings()
#         word_embedding.requires_grad = False

#     if args.local_rank == 0:
#         torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

#     model.to(args.device)
#     return tokenizer, model


def load_model(args):
    # Prepare GLUE task
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

   
    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True)
    model = BiBertEncoder(args)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    return tokenizer, model

def main():
    args = get_arguments()
    set_env(args)
    tokenizer, model = load_model(args)
    # print("main_eval")
    # model = get_model_obj(model)
    # saved_state = load_states_from_checkpoint("../result/DE_6layer/checkpoint-40000")
    # model.load_state_dict(saved_state.model_dict)
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # evaluate_dev(args, model, tokenizer)

    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.output_dir, 'log.txt')
    # sh = logging.StreamHandler()
    handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(sh)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print(logger)

    global_step = train(args, model, tokenizer)
    logger.info(" global_step = %s", global_step)

    if args.local_rank != -1:
        dist.barrier()


if __name__ == "__main__":
    main()
