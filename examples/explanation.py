import regex
def wordre(word):
    return regex.compile(r'(?:\b|(?<=(?:\p{P}|\s)))%s(?:\b|(?=(?:\p{P}|\s)))' % regex.escape(word))


def normalize_punctuation(x):
    return x.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\u201c",'"').replace(u"\u201d", '"')
    # TODO: Â£ encoding error why? # .replace(u"Â", "")


from nltk import word_tokenize # for explain


def candidate_words(line):
    return word_tokenize(line)


#TODO: something exactly consistent with tokenizer proposing words
def replaceword(word, repl, line):
    return regex.sub(wordre(word), repl, line)


def withoutword(word, line):
    return replaceword(word, "", line)


def withoutwords(words, line):
    for word in words:
        line = withoutword(word, line)
    return line


def bytealpha(alpha):
    return min(255, int(256.0 * alpha))


def hexbyte(b):
    return format(int(b), '02x')


def html_color(rgb_float):
    return '#%s%s%s' % tuple(hexbyte(bytealpha(x)) for x in rgb_float)


def interpolate_tuple(bg, fg, alpha):
    return tuple(b + (f - b) * alpha for b, f in zip(bg, fg))


def with_highlighted_words(iw, line, color=None, black=(0,0,0), minalpha=.2, confluence=False):
    if len(iw) == 0: return line
    maximport = max(x.importance for x in iw)
    if maximport == 0: return line
    if color is None:
        if confluence:
            on = '*'
            off = '*'
        else:
            on = '<b>'
            off = '</b>'
    else:
        normimport = 1.0 / maximport
        restalpha = 1 - minalpha
        off = '*{color}' if confluence else '</font></b>'
    for x in iw:
        if color is not None:
            alpha = minalpha + restalpha * x.importance * normimport
            c = html_color(interpolate_tuple(black, color, alpha))
            on = ('{color:%s}*' if confluence else '<b><font color="%s">') % c
        w = x.word
        line = replaceword(w, on + w + off, line)
        for w in x.wordalt:
            line = replaceword(w, on + w + off, line)
    return line


def label_color(label):
    return (1,0,0) if label == 'NEG' else (0,1,0) if label == 'POS' else (0,0,1)


import cgi

def html_escape(x):
    # % html.escape(line)
    return cgi.escape(x)


hre = regex.compile(r'(</font></b>|<b><font color="[^"]+">)')


def html_escape_highlights(line):
    return ''.join(html_escape(x) if i % 2 == 0 else x for i, x in enumerate(regex.split(hre, line)))


def extended(r, xs):
    r.extend(xs)
    return r


def with_explanation(iw, line, label, args):
    if args.brief_explanation:
        return '\t'.join(' '.join(extended(['{0:.3f}'.format(x.importance), x.word], x.wordalt)) for x in iw)
    line = with_highlighted_words(iw, line, color=label_color(label), confluence=args.confluence_markup)
    if args.confluence_markup: return line
    return '%s<br/>' % html_escape_highlights(line)
    return "%s\t[%s because: %s]" % (line, label, ' '.join(x.word for x in iw))


def add_explanation_args(parser):
    parser.add_argument('--confluence-markup', action='store_true', help='output confluence wiki format, not HTML', default=False)
    parser.add_argument('--brief-explanation', action='store_true', help='output just list of important words and importance, not whole input', default=False)
