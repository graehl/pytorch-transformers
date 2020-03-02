def boolarg(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_explain_args(parser):
    parser.add_argument('--explain', action='store_true', help='explain which words matter most in determining output label; whatever when removed decreases the confidence by enough is important')
    parser.add_argument('--explain-epsilon', type=float, help='for explain, only care if confidence is decreased by at least this much', default=0.05)
    #parser.add_argument('--explain-delta', type=float, help='for explain, only indicate words whose absence drops certainty to below this margin', default=2)
    #parser.add_argument('--explain-delta-portion', type=float, help='for explain, only indicate words whose absence drops certainty to less than this times the decision margin', default=0.6)
    parser.add_argument('--explain-maxwords', type=int, help='for explain, choose at most this many words', default=10)
    parser.add_argument('--explain-maxwords-portion', type=float, help='for explain, choose at most this portion of words (portion <= 1)', default=0.2)
    parser.add_argument('--explain-stopwords', type=boolarg, help='for explain, allow stopwords', default=False)
    parser.add_argument('--explain-punctuation', type=boolarg, help='for explain, allow punctuation', default=False)
