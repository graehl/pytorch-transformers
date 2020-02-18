cd `dirname $0`
mkdir unusedin
touch unusedin/train.tsv
touch unusedin/dev.tsv
echo 'Long on JNJ! The movie was dull. Stunning victory! Crushing defeat.' > /tmp/texturl.txt
echo >> /tmp/texturl.txt
echo >> /tmp/texturl.txt
echo 'Long on JNJ!' >> /tmp/texturl.txt
echo 'The library comprises several example scripts.' >> /tmp/texturl.txt
echo >> /tmp/texturl.txt
cmd="python -u ./examples/run_glue.py --model_type distilbert --model_name_or_path finmodel --task_name sentiment3 --do_lower_case --overwrite_cache --no_cache --eval_text /dev/null --data_dir unusedin --max_seq_length 64 --per_gpu_eval_batch_size=32.0 --output_dir finmodel  --verbose_every 1 --server --verbose 0"
set -x
if [[ $* ]] ; then
    cat /tmp/texturl.txt | python -u ./examples/send_document.py | $cmd --proto 2>/dev/null | python -u ./examples/recv_labeled_document.py
else
    $cmd
fi
#python -u ./examples/run_glue.py --model_type distilbert --model_name_or_path finmodel --task_name sentiment3 --do_lower_case --overwrite_cache --no_cache --eval_text /dev/null --data_dir unusedin --max_seq_length 64 --per_gpu_eval_batch_size=32.0 --output_dir finmodel  --verbose_every 1 --server --verbose 0 "$@"
