#!/usr/bin/env python -u

import labeled_document_pb2 as ld
import protostream
import sys
import os
import argparse
from kafka_args import doc_str


def log(x):
    sys.stderr.write('#RECV: ' + str(x) + '\n')


def makelist(x):
    return x if isinstance(x, list) else [x]


def main(args):
    if args.kafka:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(args.kafka_out_topic, bootstrap_servers=makelist(args.kafka_bootstrap), api_version=args.kafka_api_version)
        log('listening on kafka topic %s' % args.kafka_out_topic)
        for msg in consumer:
            #log('kafka msg: ' + str(msg))
            doc = ld.LabeledDocument()
            doc.ParseFromString(msg.value)
            print(doc_str(doc))
        return
    stdin = os.fdopen(sys.stdin.fileno(), "rb", closefd=False) # or sys.stdin.buffer?
    for doc in protostream.parse(stdin, ld.LabeledDocument):
        if doc is None:
            log('EOF')
            break
        log("got doc id=%s" % doc.document_id)
        print(doc_str(doc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from kafka_args import add_kafka_args
    add_kafka_args(parser)
    main(parser.parse_args())
