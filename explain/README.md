### Sentiment important words explanation/visualization

(note: this code is not sentiment specific - it works without
change for any segment classification models trained using
pytorch-transformers LM fine tuning)

### Installation/Usage

    export PATH=/usr/local/anaconda3/bin:$PATH

    pip install -r /home/graehl/explain/requirements.txt

    echo 'Perkins lifts dividend; earnings rise 15%' |
    python -u explain_server.py --model /build2/ca/sentiment/finmodel4 --explain --segmented

which has output highlighting (with intensity) the words relevant to the decision margin:

    POS(+6.409)[-3.5 4.15 -2.26] Perkins lifts dividend; <b><font color="#007400">earnings</font></b> <b><font color="#00ff00">rise</font></b> 15%<br/>

POS(+6.409)[-3.5 4.15 -2.26] Perkins lifts dividend; <b><font color="#007400">earnings</font></b> <b><font color="#00ff00">rise</font></b> 15%<br/>

(see <a href="http://htmlpreview.github.io/?https://github.com/graehl/pytorch-transformers/blob/master/explain/findev-highlights.html">findev-highlights.html</a> for example colorized important words)

(the `--segmented` flag means the input is already split into one sentence per line)

### Training

`https://github.com/graehl/pytorch-transformers.git` is a fork of
pytorch-transformers that supports a 3-class sentiment training
task. There's probably little need to install it for inference time at
this point (both work). An example colab/jupyter notebook by which
`finmodel4` was trained can be seen at
https://colab.research.google.com/drive/1LNTz3dQRZJ1wJUqLkm-QwJbGcn-YeH9R


### Protobuf microservice API

Two options: listen on a kafka queue and publish responses
(`explain_server.py --kafka`), or a binary stdin/stdout stream
(`python -u explain_server.py --proto`), where input and output
streams are (protobuf varint n, protobuf message encoded into n
bytes)*.

The request:

    message Document {
        required string document_id = 1;
        repeated string segments = 2; // sentences or lines without any terminating newline
    }


The response:

    message LabeledDocument {
        required string document_id = 1;
        repeated Label labels = 2; // one to one with segments in Document
    }

    message ImportantWords {
        required string word = 1;
        required float importance = 2;
        repeated string wordalt = 3;
        // different case variants not including word, if any (importance is of all taken as a group)
    }

    message Label {
        // favored class is argmax(logits)
        // for sentiment: confidences in negative, positive, neutral. the one that is higher than others = the classification. for empty input string, logits is empty
        repeated float logits = 1;
        // if explain option is enabled, these will be provided as being words most decisive in choosing the favored class:
        repeated ImportantWords words = 2;
    }

The explanation (important words) are only provided if you add the
`--explain` option to the server command line, since generating them
is an additional length of sentence times slowdown. `wordalt` in
`ImportantWords` is often empty, but for example if you had "Happy
hAppy birds are happy.", we would have "happy" for `word` and
['hAppy', 'Happy'] for `wordalt`, because the importance was assessed
not for each one in isolation, but by removing all together.

Logits are just the pre-softmax results for each of the N classes;
whichever is highest is the predicted class. For sentiment3 the
classes are [negative, positive, neutral].

### Protobuf messages to/from text examples

    cat $textfile | python -u ./explain/send_document.py |
    python -u explain_server.py --model /build2/ca/sentiment/finmodel4 --proto |
    python -u ./explain/recv_labeled_document.py

(the `python -u` is necessary if you want to see responses as soon as
inputs are provided without waiting for EOF). send_document has
options dictating whether the input is already segmented, etc:

    usage: send_document.py
      --segmented           input already segmented one per line
      --batch-num-segments BATCH_NUM_SEGMENTS
                            split input docs into at most this many segments
      --segments-out SEGMENTS_OUT
                            store all segments in this file
      --kafka-api-version KAFKA_API_VERSION
                            kafka api version
      --kafka-bootstrap KAFKA_BOOTSTRAP
                            kafka bootstrap_servers
      --kafka-in-topic KAFKA_IN_TOPIC
                            topic name for input Document protobuf requests
      --kafka-out-topic KAFKA_OUT_TOPIC
                            topic name for output LabeledDocument protobuf
                            responses
      --kafka               run kafka RPC-like service - read from kafka-in-topic
                            and write to kafka-out-topic

The option to save the segments to a file is to make it easier to
visualize the `LabeledDocument` output message which doesn't
recapitulate the input text (this could easily be changed).

Or start the kafka service in the background and then send/receive at leisure:

    python -u explain_server.py --model /build2/ca/sentiment/finmodel4 --explain \
      --kafka --kafka-bootstrap localhost:9092 --kafka-in-topic labelin --kafka-out-topic labelout &

    cat $textfile | python -u ./explain/send_document.py --kafka --kafka-bootstrap localhost:9092 --kafka-in-topic labelin --kafka-out-topic labelout

    python -u ./explain/recv_labeled_document.py --kafka --kafka-bootstrap localhost:9092 --kafka-in-topic labelin --kafka-out-topic labelout
