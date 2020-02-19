cd `dirname $0`
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
cmd="python -u ./examples/run_glue.py --model_type distilbert --model_name_or_path finmodel --task_name sentiment3 --do_lower_case --overwrite_cache --no_cache --eval_text /dev/null --data_dir unusedin --max_seq_length 64 --per_gpu_eval_batch_size=32.0 --output_dir finmodel  --verbose_every 1 --server --verbose 0"
set -x
kafka_in_topic=labelin
kafka_out_topic=labelout
kafka_bootstrap=localhost:9092
if [[ $1 = -p ]] ; then
    cat /tmp/texturl.txt | python -u ./examples/send_document.py | $cmd --proto  | python -u ./examples/recv_labeled_document.py
elif [[ $1 = -k ]] ; then
    kafkas="--kafka --kafka-bootstrap $kafka_bootstrap --kafka-in-topic $kafka_in_topic --kafka-out-topic $kafka_out_topic"
    $cmd $kafkas &
    sleep 15
    echo sending
    cat /tmp/texturl.txt | python -u ./examples/send_document.py $kafkas
    echo listening
    python -u ./examples/recv_labeled_document.py $kafkas
    echo python -u ./examples/send_document.py $kafkas
else
    $cmd < /tmp/texturl.txt
fi
#python -u ./examples/run_glue.py --model_type distilbert --model_name_or_path finmodel --task_name sentiment3 --do_lower_case --overwrite_cache --no_cache --eval_text /dev/null --data_dir unusedin --max_seq_length 64 --per_gpu_eval_batch_size=32.0 --output_dir finmodel  --verbose_every 1 --server --verbose 0 "$@"
