cat tests/dev.txt | python3 run_sent.py -i - -c -s | tee run_sent.out && cp run_sent.out run_sent.last
