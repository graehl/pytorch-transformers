have_kafka = True
try:
    from kafka import KafkaProducer, KafkaConsumer
except ImportError:
    have_kafka = False


kafka_api_version = (0, 10)
kafka_in_topic_default = 'labelin'
kafka_out_topic_default = 'labelout'
kafka_bootstrap_default = 'localhost:9092'


def add_kafka_args(parser):
    if have_kafka:
        parser.add_argument('--kafka-api-version', type=tuple, help='kafka api version', default=kafka_api_version)
        parser.add_argument('--kafka-bootstrap', type=str, help='kafka bootstrap_servers', default=kafka_bootstrap_default)
        parser.add_argument('--kafka-in-topic', type=str, help='topic name for input Document protobuf requests', default=kafka_in_topic_default)
        parser.add_argument('--kafka-out-topic', type=str, help='topic name for output LabeledDocument protobuf responses', default=kafka_out_topic_default)
        parser.add_argument('--kafka', action='store_true', help='run kafka RPC-like service - read from kafka-in-topic and write to kafka-out-topic')


def rounded(x, digits=2):
    form = "{0:.%sf}" % digits
    def rx(x):
        return float(form.format(x)) if isinstance(x, float) else [rx(y) for y in x] if isinstance(x, list) else x
    return rx(x)


def label_str(x):
    ibest = 0
    best = None
    gap = 0
    for i, y in enumerate(x):
        if best is None or y > best:
            ibest = i
            gap = 9999 if best is None else y - best
            best = y
        elif best > y:
            gap = min(gap, best - y)
    if ibest == 0: ibest = 'NEG'
    elif ibest == 1: ibest = 'POS'
    elif ibest == 2: ibest = 'meh'
    return '%s(+%s)[%s]' % (ibest, rounded(gap), ' '.join(str(rounded(y)) for y in x))


def seg_str(docid, i, logits):
    return '%s\t%s\t%s' % (docid, i + 1, label_str(logits))


def doc_str(doc):
    return '\n'.join(seg_str(doc.document_id, i, x.logits) for i, x in enumerate(doc.labels))
