#!/usr/bin/python

import os
import os.path
import sys
import subprocess
import glob


def pp(x):
    return x.replace('<br />', ' ').replace('\t', ' ').replace('  ', ' ')


def log(x):
    print(str(x), file=sys.stderr)


tarf = 'aclImdb_v1.tar.gz'
assert os.path.isfile(tarf) or subprocess.call(['wget', 'https://ai.stanford.edu/~amaas/data/sentiment/%s' % tarf]) == 0
ind = 'aclImdb'
assert os.path.isdir(ind) or subprocess.call(['tar', 'xzf', tarf]) == 0

outd = 'imdb'
try:
    os.makedirs(outd)
except:
    pass


for corpin, corpout in (('test', 'dev'), ('train', 'train')):
    outfn = '%s/%s.tsv' % (outd, corpout)
    outf = open(outfn, 'w', encoding='utf-8')
    for name, label in (('neg', '0'), ('pos', '1')):
        log((corpin, corpout, name, label, outfn))
        print('sentence\tlabel', file=outf)
        fs = '%s/%s/%s/*.txt' % (ind, corpin, name)
        for fn in glob.glob(fs):
            with open(fn, 'r', encoding='utf-8') as f:
                print('%s\t%s' % (pp(f.read().replace('\n', '')), label), file=outf)
