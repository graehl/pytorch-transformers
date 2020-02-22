def add_explain_args(parser):
    parser.add_argument('--explain', action='store_true', help='explain which words matter most in determining output label')
    parser.add_argument('--explain-delta', type=float, help='for explain, only indicate words whose absence drops certainty to below this margin', default=2)
    parser.add_argument('--explain-delta-portion', type=float, help='for explain, only indicate words whose absence drops certainty to less than this times the decision margin', default=0.6)
    parser.add_argument('--explain-maxwords', type=int, help='for explain, choose at most this many words', default=100)
    parser.add_argument('--explain-maxwords-portion', type=float, help='for explain, choose at most this many words (portion <= 1)', default=0.25)
