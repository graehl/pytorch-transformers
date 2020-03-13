cd `dirname $0`
export PATH=/usr/local/anaconda3/bin:$PATH
rebuild=${rebuild:-0}
pip install -r explain/requirements.txt
if [[ $rebuild = 1 ]] ; then
    (cd explain; ./build_proto.sh)
    #pip install .
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
#textfile=tests/fixtures/sample_text.txt
textfile2=tests/sdlfin.txt
devtextfile=tests/dev2.txt
devtextgold=$devtextfile.gold
dev=${dev:-0}
deveval=${deveval:-0}
if [[ $dev = 1 && -f $devtextfile ]] ; then
    textfile=$devtextfile
    if [[ $deveval = 1 ]] ; then
      brief=1
    fi
    echo
else
    deveval=0
    if [[ -f $textfile2 ]]; then
        textfile=$textfile2
    fi
fi
python=${python:-python}
if [[ $debug = 1 ]] ; then
    pythonargs+=" -m pdb"
    pythonargs=""
fi
confluence=${confluence:-0}
explainarg=""
explain=${explain:-1}
if [[ $explain = 1 ]] ; then
    explainarg="--explain"
fi
brief=${brief:-0}
if [[ $brief = 1 ]] ; then
    briefarg="--brief-explanation"
fi
#--task_name sentiment3
#--model_type distilbert
#--do_lower_case
#--server
#--verbose_every 1
#--per_gpu_eval_batch_size=32.0
#--overwrite_cache
#--do_lower_case --max_length 128
d=`dirname $0`
cmd="$python -u $pythonargs $d/explain/explain_server.py --model $d/finmodel4  --verbose 0 --log_level warn $explainarg $briefarg --explain-maxwords 7 --explain-punctuation False --segmented"
if [[ $confluence = 1 ]] ; then
    cmd+=" --confluence-markup"
fi
set -x
kafka_in_topic=labelin
kafka_out_topic=labelout
kafka_bootstrap=localhost:9092
header() {
    [[ $confluence = 1 ]] || echo '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />'
}
if [[ $1 = -p ]] ; then
    cat $textfile | python -u ./explain/send_document.py | $cmd --proto  | python -u ./explain/recv_labeled_document.py
elif [[ $1 = -k ]] ; then
    kafkas="--kafka --kafka-bootstrap $kafka_bootstrap --kafka-in-topic $kafka_in_topic --kafka-out-topic $kafka_out_topic"
    $cmd $kafkas &
    sleep 15
    echo sending
    cat $textfile | python -u ./explain/send_document.py $kafkas
    echo listening
    python -u ./explain/recv_labeled_document.py $kafkas
    echo python -u ./explain/send_document.py $kafkas
else
    outbase=/tmp/sentiment/
    mkdir -p $outbase
    outbase+=`basename $textfile`
    hf=$outbase.sentiment-importance.html
    out=$outbase.out
    outcls=$outbase.cls
    rm -f $hf $out
    wc -l $textfile
    set -o pipefail
    set -e
    ( header; $cmd < $textfile | tee $out ) | tee $hf
    echo $0/$hf
    echo $out
    cut -c1 < $out | head -n -1 > $outcls
    wc -l $out $outcls $textfile
    if [[ $deveval = 1 ]] ; then
        python tests/accuracy.py $devtextgold $outcls
    fi
    if [[ $open = 1 && -s $hf ]]; then
        open $hf
    fi
fi
