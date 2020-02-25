# Sentiment Microservice

For simplicity, we refer to sentiment analysis - labeling segments as
[negative, postitive, neutral] - when we really support general text
classification with fine-tuned LMs.

# installation

(not on github - `cp -a /home/graehl/pt/finmodel .`)
`pip install -r requirements.txt`

# text input/output server

`./sentiment.sh` (canned input - edit script if you want interactive stdin/stdout).
blank lines of input terminate batch and provide immediate response.

# protobuf input/output server

bytes in/out on stdin/stdout until EOF: (protobuf varint message size, protobuf message)*
(protobuf doesn't define a default 'stream of messages' format)

`./sentiment.sh -p`

# protobuf kafka server

read documents from 'labelin' topic, write labeled documents to 'labelout' topic.

`./sentiment.sh -k`

# protobuf message format

`cat examples/labeled_document.proto`

# Explanation

`--explain --explain-maxwords 4` gives a 50x slowdown (if 50 words in
a segment) to provide a cute bit of "which words in input were
important to the classification decision" feedback; namely, how much
did the decision change toward a different answer if a given word was
deleted. In plain text output you'll see HTML `<b>word</b>` bold tags.
