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
    form = "{0:.%sf}" % digits
    def rx(x):
        return float(form.format(x)) if isinstance(x, float) else [rx(y) for y in x] if isinstance(x, list) else x
    return rx(x)


import sys

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def classify1(text, args, model, tokenizer):
    x = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', max_length=args.max_length)
    tensor = model(x['input_ids'])
    # token_type_ids=x['token_type_ids'] if args.model_type in ['bert', 'xlnet'] else None
    # token_type_ids for this fn only supported in https://github.com/graehl/transformers - not 'pip install transformers'
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


def nltk_stopwords(lang='english'):
    import nltk
    try:
        return nltk.corpus.stopwords.words(lang)
    except:
        nltk.download('stopwords', quiet=True)
        return nltk.corpus.stopwords.words(lang)


def labeldoc(doc, args, model, tokenizer):
    stopwords = set()
    if not args.explain_stopwords:
        if not hasattr(args, 'stopwords'):
            args.stopwords = set(nltk_stopwords())
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
                    if wordlc in stopwords:
                        #log("skip stopword: '%s'" % word)
                        continue
                    punc = allpunc.match(wordlc)
                    #if punc: log("punc: '%s'" % wordlc)
                    if (not args.explain_punctuation) and punc:
                        #log("skip punc: '%s'" % wordlc)
                        continue
                    words = groupbylc[wordlc]
                    word = None
                    for w in words:
                        if word != wordlc:
                            word = w
                    #if len(words) > 1: log("variants %s <= %s" % (word, words))
                    without = explanation.withoutwords(words, segment)
                    if without == segment:
                        log("skipped '%s' (punc=%s) no change when removing from '%s' words=%s" % (word, allpunc.match(word), segment, words))
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

def server(args, model, tokenizer):
    verbose = args.verbose
    model.to(args.device)
    model.eval()
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


def get_tokenizer_config(model_path):
    '''This looks like it ought to work in AutoConfig (see tokenization_utils.py), but doesn't'''
    f = '%s/tokenizer_config.json' % model_path
    if not os.path.exists(f):
        f = model_path
    try:
        import json
        with open(f, 'r') as reader:
            text = reader.read()
        return json.loads(text)
    except:
        return {}


def main():
    parser = argparse.ArgumentParser()
    import server_args
    server_args.add_server_args(parser)
    #import deprecated_args
    #deprecated_args.add_deprecated_args(parser)

    parser.add_argument('--verbose', type=int, default=1, help="logging verbosity")
    parser.add_argument('--proto', action='store_true', help='stdin/stdout server: for each protobuf Document labeled_document.proto input, immediately output protobuf LabeledDocument (stdin/stdout); you may need to use python -u run_glue.py')


    parser.add_argument(
        '--num_labels',
        default=None,
        type=int,
        help="number of labels (needed if task_name is unspecified and model type isn't auto")

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--model_type",
        default='auto',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        '--model',
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
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
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model (will use model/tokenizer_config.json if present - this is a fallback).",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Set this flag if you are using an uncased model (will use model/tokenizer_config.json if present - this is a fallback).",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=float, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--log_level", type=str, default="info", help="for python logging")
    args = parser.parse_args()


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=loglevelstr(args.log_level)
    )

    # Set seed
    set_seed(args)


    tokenizer_config_dict = get_tokenizer_config(args.model_name_or_path)
    do_lower_case = tokenizer_config_dict.get('do_lower_case', args.do_lower_case)
    max_length = tokenizer_config_dict.get('max_len', args.max_length)

    if args.model_type == 'auto' or (args.num_labels is None and args.task_name is None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=do_lower_case)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    else:
        num_labels = args.num_labels
        if num_labels is None:
             if args.task_name is None:
                 raise 'task_name or num_labels required'
             args.task_name = args.task_name.lower()
             if args.task_name not in processors:
                 raise ValueError("Task not found: %s" % (args.task_name))
             processor = processors[args.task_name]()
             args.output_mode = output_modes[args.task_name]
             label_list = processor.get_labels()
             num_labels = len(label_list)
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
            do_lower_case=do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )


    logger.info("parameters %s", args)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
    server(args, model, tokenizer)


if __name__ == "__main__":
    main()
