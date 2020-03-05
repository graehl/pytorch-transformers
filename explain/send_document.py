#!/usr/bin/env python -u

import labeled_document_pb2 as ld
import protostream
import argparse
import sys
import os
from time import sleep


def log(x):
    sys.stderr.write('#SEND: %s\n' % str(x))


def segment_fn(segment):
    if segment:
        import re
        blanksre = re.compile(r'\s+')
        from ftfy import fix_text
        try:
            import nltk.tokenize.punkt
        except ImportError:
            import nltk
            nltk.download("punkt", quiet=True)
        from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
        def text_sentences(text):
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            lines = []
            for line in text.splitlines(keepends=False) if isinstance(text, str) else text:
                line = fix_text(line.decode('utf-8') if isinstance(line, bytes) else line).strip()
                if len(line) <= 1: continue
                line = blanksre.sub(' ', line)
                lines.append(line)
            punkt_param = PunktParameters()
            punkt = PunktSentenceTokenizer(punkt_param)
            punkt.train('\n'.join(lines))
            r = []
            for line in lines:
                r.extend(punkt.tokenize(line))
            return r
        return text_sentences
    else:
        def noop_segmenter(text):
            return [text]
        return noop_segmenter


def all_segments(lines, segmenter):
    r = []
    for line in lines:
        r.extend(segmenter(line))
    return r


docid = 0
def segmented_document(lines, segmenter):
    doc = ld.Document()
    global docid
    docid += 1
    doc.document_id = 'doc#%s' % docid
    segs = all_segments(lines, segmenter)
    doc.segments[:] = segs
    return doc


def makelist(x):
    return x if isinstance(x, list) else [x]

def stdin_docs(args):
    segment = not args.segmented
    if segment:
        log("text (to be segmented) on STDIN; terminate documents by blank line or EOF")
    else:
        log("segments one per line on STDIN; terminate documents by blank line or EOF")
    segmenter = segment_fn(segment)
    eof = False
    savefile = None
    if args.segments_out is not None:
        savefile = open(args.segments_out, 'w', encoding='utf-8')
    while not eof:
        lines = []
        try:
            while True:
                line = input()
                assert line is not None
                line = line.strip()
                if len(line) == 0: break
                lines.append(line)
                if args.batch_num_segments is not None and len(lines) >= args.batch_num_segments: break
        except EOFError:
            eof = True
        if len(lines) > 0:
            log("sending doc of %s segments" % len(lines))
            doc = segmented_document(lines, segmenter)
            if savefile is not None:
                for i, segment in enumerate(doc.segments):
                    savefile.write('%s\t%s\t%s\n' % (doc.document_id, i+1, segment))
            yield doc
    if savefile is not None:
        log("saved unlabeled segments in %s" % args.segments_out)
        savefile.close()


def main(args):
    stdout = os.fdopen(sys.stdout.fileno(), "wb", closefd=False) # or sys.stdout.buffer?

    if hasattr(args, 'kafka') and args.kafka:
        from kafka import KafkaProducer
        producer = KafkaProducer(bootstrap_servers=makelist(args.kafka_bootstrap), api_version=args.kafka_api_version)
        ostream = None
    else:
        producer = None
        ostream = protostream.open(fileobj=stdout, mode='wb')

    for doc in stdin_docs(args):
        if producer is not None:
            value = doc.SerializeToString()
            future = producer.send(args.kafka_in_topic, key=doc.document_id.encode('utf-8'), value=value)
            try:
                record_metadata = future.get(timeout=10)
                log("wrote %s: %s" % (str(record_metadata), value))
            except Exception as e:
                log("exception: %s" % e)
                raise e
        if ostream is not None:
            ostream.write(doc)


def add_send_document_args(parser):
    parser.add_argument("--segmented", action='store_true', help='input already segmented one per line', default=False)
    parser.add_argument("--batch-num-segments", type=int, help='split input docs into at most this many segments')
    parser.add_argument("--segments-out", type=str, help='store all segments in this file', default='sent-docs.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_send_document_args(parser)
    from kafka_args import add_kafka_args
    add_kafka_args(parser)
    main(parser.parse_args())
