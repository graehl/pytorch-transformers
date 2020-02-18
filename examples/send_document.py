#!/usr/bin/env python -u

import labeled_document_pb2 as ld
import protostream
import sys
import os

def log(x):
    sys.stderr.write('#SEND: ' + x + '\n')


def segment_fn(segment):
    if segment:
        import re
        blanksre = re.compile(r'\s+')
        from ftfy import fix_text
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


def main(segment=True, maxbatchsz=None, savefilename=None):
    if segment:
        log("text (to be segmented) on STDIN; terminate documents by blank line or EOF")
    else:
        log("segments one per line on STDIN; terminate documents by blank line or EOF")
    segmenter = segment_fn(segment)
    stdout = os.fdopen(sys.stdout.fileno(), "wb", closefd=False) # or sys.stdout.buffer?
    ostream = protostream.open(fileobj=stdout, mode='wb')
    eof = False
    savefile = None
    if savefilename is not None:
        savefile = open(savefilename, 'w', encoding='utf-8')
    while not eof:
        lines = []
        try:
            while True:
                line = input()
                assert line is not None
                line = line.strip()
                if len(line) == 0: break
                lines.append(line)
                if maxbatchsz is not None and len(lines) >= maxbatchsz: break
        except EOFError:
            eof = True
        if len(lines) > 0:
            log("sending doc of %s segments" % len(lines))
            doc = segmented_document(lines, segmenter)
            for i, segment in enumerate(doc.segments):
                savefile.write('%s\t%s\t%s\n' % (doc.document_id, i+1, segment))
            ostream.write(doc)
            ostream.flush()
            lines = []
    if savefile is not None:
        log("saved unlabeled segments in %s" % savefilename)

if __name__ == "__main__":
    segments = len(sys.argv) == 1
    main(segments, savefilename='sent-docs.txt')
