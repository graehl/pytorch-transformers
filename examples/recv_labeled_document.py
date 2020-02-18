#!/usr/bin/env python -u

import labeled_document_pb2 as ld
import protostream
import sys
import os


def log(x):
    sys.stderr.write('#RECV: ' + x + '\n')


def rounded(x, digits=3):
    form = "{0:.%sf}" % digits
    def rx(x):
        return float(form.format(x)) if isinstance(x, float) else [rx(y) for y in x] if isinstance(x, list) else x
    return rx(x)


def strlabel(x):
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


def main():
    stdin = os.fdopen(sys.stdin.fileno(), "rb", closefd=False) # or sys.stdin.buffer?
    for doc in protostream.parse(stdin, ld.LabeledDocument):
        if doc is None:
            log('EOF')
            break
        log("got doc id=%s" % doc.document_id)
        for i, x in enumerate(doc.labels):
            print('%s\t%s\t%s' % (doc.document_id, i + 1, strlabel(x.logits)))


if __name__ == "__main__":
    main()
