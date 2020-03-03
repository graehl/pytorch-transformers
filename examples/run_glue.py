# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
import timeit
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers import is_tf_available
if is_tf_available():
   from transformers import TFDistilBertForSequenceClassification

from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import InputExample as InputExample

from collections import Counter

mininterval = 10


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


import logging
class DisableLogger():
    #with DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
       logging.disable(logging.NOTSET)

logger = logging.getLogger(__name__)

loglevelstrs = {'debug': logging.DEBUG,
                'dbg': logging.DEBUG,
                'info': logging.INFO,
                'inf': logging.INFO,
                'warn': logging.WARNING,
                'warning': logging.WARNING,
                'error': logging.ERROR,
                'err': logging.ERROR,
                'critical': logging.CRITICAL,
                'crit': logging.CRITICAL}
def loglevelstr(x):
    x = x.lower()
    if x in loglevelstrs:
        return loglevelstrs[x]
    else:
        return int(x)


ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}

if is_tf_available():
   MODEL_CLASSES['tfdistilbert'] = (DistilBertConfig, TFDistilBertForSequenceClassification, DistilBertTokenizer)


verbosity = 1
verbose_outfile = None
stdout_verbose_every = 1


def rounded(x, digits=3):
    form = "{0:.%sf}".format(digits)
    def rx(x):
        return float(form.format(x)) if isinstance(x, float) else [rx(y) for y in x] if isinstance(x, list) else x
    return rx(x)


import sys
def outverbose(s, v=1, seq=0):
    s += '\n'
    if verbose_outfile is not None:
        verbose_outfile.write(s)
    if verbosity >= v:
        if seq % stdout_verbose_every == 0:
            sys.stdout.write('#%s: %s' % (seq, s))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def batch_inputs(batch, model_type):
    inputs = {'input_ids':      batch[0],
              'attention_mask': batch[1],
              'labels':         batch[3]}
    if model_type != 'distilbert':
        inputs['token_type_ids'] = batch[2] if model_type in ['bert', 'xlnet'] else None
        # XLM and RoBERTa and distilbert don't use segment (token_type_ids) ids
    return inputs


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = int(args.per_gpu_train_batch_size * max(1, args.n_gpu))
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], mininterval=mininterval)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

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
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def classify1(text, args, model, tokenizer):
    x = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', max_length=args.max_length)
    tensor = model(x['input_ids'], token_type_ids=x['token_type_ids'] if args.model_type in ['bert', 'xlnet'] else None)
    return tensor[0].tolist()[0]
    #.to_list


def classify(texts, args, model, tokenizer, verbose=1):
    if isinstance(texts, str):
        return classify([texts], args, model, tokenizer, verbose)[0]
    global verbosity
    verbosity = verbose
    global stdout_verbose_every
    stdout_verbose_every = args.verbose_every
    r = [classify1(x, args, model, tokenizer) for x in texts]
    return r
    #TODO: tokenizer.batch_encode_plus

import labeled_document_pb2 as ld

import regex
allpunc = regex.compile(r'^\p{P}*$')

def labeldoc(doc, args, model, tokenizer):
    stopwords = set()
    if not args.explain_stopwords:
        if not hasattr(args, 'stopwords'):
            from nltk.corpus import stopwords
            args.stopwords = set(stopwords.words('english'))
        stopwords = args.stopwords
    ldoc = ld.LabeledDocument()
    ldoc.document_id = doc.document_id
    sys.stderr.write("# got doc id=%s with %s segments\n" % (doc.document_id, len(doc.segments)))
    start_time = timeit.default_timer()
    labels = ldoc.labels
    import explanation
    for segment in doc.segments:
        label = ld.Label()
        if len(segment) > 0:
            # TODO: submit the various classify1 (incl args. explain) as a batch for much higher GPU perf
            segment = explanation.normalize_punctuation(segment)
            logits = classify1(segment, args, model, tokenizer)
            label.logits[:] = logits
            import confidence
            besti = confidence.arg_max(logits)
            confi = confidence.confidence_in(logits, besti)
            #log('confidence(%s)=%s for %s %s'%(besti, confi, logits, segment))
            confbelow = confi - args.explain_epsilon
            if args.explain:
                cwords = []
                groupbylc = confidence.group_by_lc(explanation.candidate_words(segment))
                for wordlc in groupbylc:
                    words = groupbylc[wordlc]
                    if wordlc in stopwords:
                        #log("skip stopword: '%s'" % word)
                        continue
                    punc = allpunc.match(wordlc)
                    #if punc: log("punc: '%s'" % wordlc)
                    if (not args.explain_punctuation) and punc:
                        #log("skip punc: '%s'" % wordlc)
                        continue
                    word = None
                    for w in words:
                        if w == wordlc:
                            word = wordlc
                        elif word != wordlc:
                            word = w
                    without = explanation.withoutwords(words, segment)
                    if without == segment:
                        log("skipped '%s' no change when removing from '%s'" % (word, segment))
                        continue
                    logits_no_w = classify1(without, args, model, tokenizer)
                    conf = confidence.confidence_in(logits_no_w, besti)
                    ouri = confidence.arg_max(logits_no_w)
                    #if besti != ouri: log("'%s' seems important - class changed from %s to %s %s" % (word, besti, ouri, rounded(logits_no_w)))
                    if conf < confi:
                        if conf < confbelow:
                            cwords.append(((word, words), conf))
                        else:
                            #log("'%s' was important but not by more than epsilon=%s (%s %s vs %s %s)" % (word, args.explain_epsilon, rounded(confi), rounded(logits), rounded(conf), rounded(logits_no_w)))
                            pass
                    else:
                        #log("'%s' wasn't important (confidence higher without - from %s to %s)" % (word, rounded(confi), rounded(conf)))
                        pass
                maxwords = min(int(.99 + len(groupbylc) * args.explain_maxwords_portion), args.explain_maxwords)
                for ww, conf in sorted(cwords, key=lambda x: x[1])[:maxwords]:
                    # keep the *lowest* confidence (i.e. most important)
                    word, words = ww
                    iw = ld.ImportantWords()
                    iw.word = word
                    iw.importance = confi - conf
                    for c in words:
                        if c != word:
                            iw.wordalt.append(c)
                    label.words.append(iw)
        labels.append(label)
    time = timeit.default_timer() - start_time
    #log("# writing proto result for doc %s (%s sec)\n" % (ldoc, time))
    return ldoc


def outserver(x):
    print(str(x))


def makelist(x):
    return x if isinstance(x, list) else [x]


def log(x):
    sys.stderr.write('#run_glue.py: %s\n' % str(x))


import labeled_document_pb2 as ld

def server(args, model, tokenizer, protobuf=False, verbose=1):
    import explanation
    batchsz = int(args.per_gpu_eval_batch_size * max(1, args.n_gpu))
    args.eval_batch_size = batchsz
    eof = False
    if args.kafka:
        from time import sleep
        from kafka import KafkaProducer, KafkaConsumer
        producer = KafkaProducer(bootstrap_servers=makelist(args.kafka_bootstrap), api_version=args.kafka_api_version)
        def parse_document(msg):
            doc = ld.Document()
            doc.ParseFromString(msg)
            return doc
        consumer = KafkaConsumer(args.kafka_in_topic, bootstrap_servers=makelist(args.kafka_bootstrap), api_version=args.kafka_api_version) # , value_deserializer=parse_document
        sys.stderr.write('kafka server running until Ctrl-C\n')
        try:
            for msg in consumer:
                # log('kafka msg: ' + str(msg))
                doc = ld.Document()
                doc.ParseFromString(msg.value)
                value = labeldoc(doc, args, model, tokenizer).SerializeToString()
                future = producer.send(args.kafka_out_topic, key=doc.document_id.encode('utf-8'), value=value)
                # Block for 'synchronous' sends
                try:
                    record_metadata = future.get(timeout=10)
                    log("wrote " + str(record_metadata) + " value: %s" % value)
                except Exception as e:
                    log("exception: %s" % e)
                    raise e
        except KeyboardInterrupt:
            if consumer is not None: consumer.close()
            if producer is not None: producer.close()
        return
    elif args.proto:
        import protostream
        stdout = os.fdopen(sys.stdout.fileno(), "wb", closefd=False) # or sys.stdout.buffer?
        stdin = os.fdopen(sys.stdin.fileno(), "rb", closefd=False) # or sys.stdin.buffer?
        with protostream.open(mode='wb', fileobj=stdout) as ostream:
            for doc in protostream.parse(stdin, ld.Document):
                ostream.write(labeldoc(doc, args, model, tokenizer))
        return
    else:
        from kafka_args import label_gap, label_str, logits_str
        import send_document
        for doc in send_document.stdin_docs(args):
            ldoc = labeldoc(doc, args, model, tokenizer)
            for i, l in enumerate(ldoc.labels):
                label, gap = label_gap(l.logits)
                labelstr = label_str(label)
                explained = explanation.with_explanation(l.words, doc.segments[i], labelstr, args)
                if args.brief_explanation:
                    out = '%s %s\t%s' % (label, logits_str(l.logits), explained)
                else:
                    out = '%s(+%s)[%s] %s' % (labelstr, rounded(gap), logits_str(l.logits), explained)
                outserver(out)
            outserver('')


def evaluate(args, model, tokenizer, prefix="", verbose=1):
    global verbosity
    verbosity = verbose
    global stdout_verbose_every
    stdout_verbose_every = args.verbose_every
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        global verbose_outfile
        vf = os.path.join(eval_output_dir, 'verbose.txt')
        verbose_outfile = open(vf, 'w', encoding='utf-8')
        logger.info('writing logits etc to %s' % vf)
        eval_dataset, eval_examples = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        logger.info(" #data=%s #examples=%s first=%s" % (len(eval_dataset), len(eval_examples), eval_examples[:1]))
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = int(args.per_gpu_eval_batch_size * max(1, args.n_gpu))
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        start_time = timeit.default_timer()

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        i = 0
        confs = None
        dups = Counter()
        nclasses = None
        for batch in tqdm(eval_dataloader, desc="Evaluating", mininterval=mininterval):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch) # t[0] pair with input
            inputs = batch_inputs(batch, args.model_type)
            with torch.no_grad():
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
                logs = logits.tolist()
                outverbose('%s\t%s' % (rounded(logs), inputs['labels'].tolist()), v=1, seq=nb_eval_steps)
                for l in logs:
                    if i >= len(eval_examples): break
                    ex = eval_examples[i]
                    minl = min(l)
                    if nclasses is None or len(l) > nclasses:
                        if nclasses is not None:
                            logger.warn("# of classes differed: %s vs %s in %s; dropping old data" % (nclasses, len(l), l))
                        nclasses = len(l)
                        confs = [[] for x in l]
                    for j in range(nclasses):
                        confj = l[j]
                        l[j] = minl
                        confmax = max(l)
                        l[j] = confj
                        conf = l[j] - confmax
                        if conf > 0:
                            t = ex.texts()
                            dups[t] += 1
                            if dups[t] == 1:
                                confs[j].append((conf, i, l, ex))
                                if conf > 8: outverbose('%s %s %s %s' % (rounded(conf), j, t, ex.label), v=1)
                    i += 1
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        outmax = 20
        docsentiment = [0 for x in range(nclasses)]
        nsents = i
        for j, c in enumerate(confs):
            s = sorted(c, reverse=True)
            for topi, x in enumerate(s):
                conf, i, logit, example = x
                desc = '%s %s [#%s gold:%s] %s %s' % (rounded(logit), j, i, example.label, rounded(conf), example.texts())
                docsentiment[j] += conf
                if topi < outmax:
                    sys.stdout.write(desc + '\n')
                outverbose(desc, v=2, seq=topi)
        scale = sum(docsentiment)
        scale = 1. / scale if scale > 0 else 0
        docsentimentraw = docsentiment
        docsentiment = [x * scale for x in docsentimentraw]
        docneg = docsentiment[0]
        docpos = docsentiment[1]
        docneu = docsentiment[2] if len(docsentiment) >= 3 else 0
        docposneg = (docpos - docneg) * (1. - docneu)
        logger.info('document sentiment (%s sentences): unnormalized: %s; normalized: %s; net positive/negative: %.3f; ' % (nsents, rounded(docsentimentraw), rounded(docsentiment), docposneg))
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        from transformers import glue_compute_metrics as compute_metrics
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(eval_dataset))
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        def fmtfloat(x):
            return str(x) if (not isinstance(x, float) or x.is_integer()) else '%.3f' % x

        with open(output_eval_file, "w") as writer:
            skeys = sorted(result.keys())
            for key in skeys:
                writer.write("%s = %s\n" % (key, str(result[key])))
                logger.info("%s = %s"%(key, fmtfloat(result[key])))
            logger.info("***** Eval results {} *****: {}".format(prefix, " ".join("%s = %s"%(key, fmtfloat(result[key])) for key in skeys)))

    return results


def load_eval_examples(args, task, tokenizer, text=None, evaluate=True):
    """as load_and_cache_examples but no cache"""
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    devname = 'dev'
    devtext = args.eval_text
    processor.__devtexturl = devtext
    logger.info("datadir=%s devtexturl=%s text=%s" % (args.data_dir, devtext, text))
    if devtext:
        devname = os.path.basename(devtext)
    else:
        devtext = args.data_dir
    if text:
        devtext = text
    examples = processor.get_dev_examples(devtext)
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=None, #TODO?
                                            max_length=args.max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return (dataset, examples)


import re
nonspaceblanks = re.compile(r'[\n\r\t]')


def justspaces(x):
    return nonspaceblanks.sub(' ', str(x))


def save_tsv(examples, f):
    if isinstance(f, str):
        f = open(f, 'w', encoding='utf-8')
    for x in examples:
        if isinstance(x, str):
            f.write(justspaces(x.rstrip('\n')))
        else:
            f.write('\t'.join(map(justspaces, x)))
        f.write('\n')
    f.close()


def load_tsv(f):
    if isinstance(f, str):
        f = open(f, 'r', encoding='utf-8')
    examples = []
    for line in f:
        examples.append(line.rstrip('\n').rstrip('\r').split('\t'))
    f.close()
    return examples

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    devname = 'dev'
    devtext = args.eval_text
    processor.__devtexturl = devtext
    logger.info("datadir=%s devtexturl=%s" % (args.data_dir, devtext))
    if devtext:
        devname = os.path.basename(devtext)
    else:
        devtext = args.data_dir
    cache = 'cached_{}_{}_{}_{}'.format(
        devname if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task))
    cached_features_file = os.path.join(args.data_dir, cache)
    cached_examples_file = os.path.join(args.data_dir, cache + '.examples.tsv')
    evalnocache = evaluate and args.no_cache
    if os.path.exists(cached_features_file) and os.path.exists(cached_examples_file) and not args.overwrite_cache and not evalnocache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        logger.info("Loading examples from cached file %s", cached_examples_file)
        examples = [InputExample(x[0], x[1], x[2], x[3]) for x in load_tsv(cached_examples_file)]
    else:
        logger.info("Creating features from dataset file at %s %s", args.data_dir, 'devtext=%s' % devtext if evaluate else '')
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(devtext) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
            )
        if args.local_rank in [-1, 0] and not evalnocache:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            logger.info("Saving examples into cached file %s", cached_examples_file)
            save_tsv(((x.guid, x.text_a, '' if x.text_b is None else x.text_b, x.label) for x in examples), cached_examples_file)


    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return (dataset, examples)


def main():
    parser = argparse.ArgumentParser()
    import server_args
    server_args.add_server_args(parser)

    parser.add_argument('--verbose', type=int, default=1, help="show eval logits => stdout (every n) and verbose.txt")
    parser.add_argument('--verbose_every', type=int, default=10, help="show every nth to stdout for verbose")
    parser.add_argument('--server', action='store_true', help='for each line on stdin, immediately output logit line [3.1, -2.1] - the argmax logit[i] is the class i, e.g. 0 negative 1 positive')
    parser.add_argument('--proto', action='store_true', help='stdin/stdout server: for each protobuf Document labeled_document.proto input, immediately output protobuf LabeledDocument (stdin/stdout); you may need to use python -u run_glue.py')

    parser.add_argument('--no_cache', action='store_true', help="Never cache evaluation set")
    parser.add_argument("--eval_text", default="", type=str, help="Eval lines of text from this file instead of normal dev.tsv; if no label, fake label 0 is used")

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
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
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=float, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=float, help="Batch size per GPU/CPU for evaluation.",
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
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--log_level", type=str, default="info", help="for python logging")
    args = parser.parse_args()

    # if loading an already fine-tuned model, no need to specify the path twice
    if args.output_dir is None and os.path.exists(args.model_name_or_path):
        args.output_dir = args.model_name_or_path


    if (args.do_train
        and not args.overwrite_output_dir
        and os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

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
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
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
        level=loglevelstr(args.log_level) if args.local_rank in [-1, 0] else logging.WARN,
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

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)[0]
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    doeval = args.do_eval
    if args.proto or args.kafka:
        args.server = True
    if (doeval or args.server) and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if doeval and args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for i,checkpoint in enumerate(checkpoints):
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            if doeval:
                result = evaluate(args, model, tokenizer, prefix=prefix)
                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)
            if args.server and i+1 == len(checkpoints):
                model.eval()
                server(args, model, tokenizer, verbose=args.verbose)

    return results


if __name__ == "__main__":
    main()
