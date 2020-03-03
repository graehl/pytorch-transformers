def add_server_args(parser):
    parser.add_argument('--max-length', '-m', type=int, default=512, help='maximum length handled by the model')

    from kafka_args import add_kafka_args
    add_kafka_args(parser)

    from explain_args import add_explain_args
    add_explain_args(parser)

    from send_document import add_send_document_args
    add_send_document_args(parser)

    import explanation
    explanation.add_explanation_args(parser)
