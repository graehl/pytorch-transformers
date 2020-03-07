Sentiment (or other fine-tuned LM text classifier) important words explanation/visualizaion

    export PATH=/usr/local/anaconda3/bin:$PATH

    pip install -r /home/graehl/explain/requirements.txt



    echo 'Perkins lifts dividend; earnings rise 15%' | python -u /home/graehl/explain/explain_server.py --model /home/graehl/pt/finmodel3 --verbose 0 --log_level warn --explain --explain-maxwords 7 --explain-punctuation False --segmented


which has output

    POS(+6.409)[-3.5 4.15 -2.26] Perkins lifts dividend; <b><font color="#007400">earnings</font></b> <b><font color="#00ff00">rise</font></b> 15%<br/>
