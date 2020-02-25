cd `dirname $0`
export PATH=/usr/local/anaconda3/bin:$PATH
if [[ $rebuild ]] ; then
  (cd examples; ./build_proto.sh)
  pip install .
fi
mkdir -p unusedin
touch unusedin/train.tsv
touch unusedin/dev.tsv
textfile=/tmp/texturl.txt
echo 'Long on JNJ!' > $textfile
echo 'The movie was dull.' >> $textfile
echo 'Stunning victory!' >> $textfile
echo >> $textfile
echo >> $textfile
echo 'Long on JNJ!' >> $textfile
echo 'The library comprises several example scripts.' >> $textfile
echo >> $textfile
textfile=tests/fixtures/sample_text.txt
python=${python:-python}
if [[ $debug = 1 ]] ; then
    pythonargs+=" -m pdb"
    pythonargs=""
fi
cmd="$python -u $pythonargs ./examples/run_glue.py --model_type distilbert --model_name_or_path finmodel --task_name sentiment3 --do_lower_case --overwrite_cache --no_cache --eval_text /dev/null --data_dir unusedin --max_seq_length 64 --per_gpu_eval_batch_size=32.0 --verbose_every 1 --server --verbose 0 --log_level warn --explain --explain-maxwords 4"
set -x
kafka_in_topic=labelin
kafka_out_topic=labelout
kafka_bootstrap=localhost:9092
if [[ $1 = -p ]] ; then
    cat $textfile | python -u ./examples/send_document.py | $cmd --proto  | python -u ./examples/recv_labeled_document.py
elif [[ $1 = -k ]] ; then
    kafkas="--kafka --kafka-bootstrap $kafka_bootstrap --kafka-in-topic $kafka_in_topic --kafka-out-topic $kafka_out_topic"
    $cmd $kafkas &
    sleep 15
    echo sending
    cat $textfile | python -u ./examples/send_document.py $kafkas
    echo listening
    python -u ./examples/recv_labeled_document.py $kafkas
    echo python -u ./examples/send_document.py $kafkas
else
    $cmd < $textfile
fi
#python -u ./examples/run_glue.py --model_type distilbert --model_name_or_path finmodel --task_name sentiment3 --do_lower_case --overwrite_cache --no_cache --eval_text /dev/null --data_dir unusedin --max_seq_length 64 --per_gpu_eval_batch_size=32.0 --output_dir finmodel  --verbose_every 1 --server --verbose 0 "$@"
