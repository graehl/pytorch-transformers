import sys
gold, test = (open(x, 'r') for x in sys.argv[1:])
n = 0
nmatch = 0
golds = [line.strip() for line in gold]
for line in test:
    line = line.strip()
    if n == len(golds): break
    gold = golds[n]
    n += 1
    if gold == line:
        nmatch += 1
    else:
        print("mismatch on line #%s" % n, file=sys.stderr)
print ("%s/%s = %s" % (nmatch, n, float(nmatch)/n))
