DATA=${1:-.}
if ! test -d $DATA ; then
    mkdir -p $DATA; cd $DATA
    wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    tar xzf aclImdb_v1.tar.gz
    rm aclImdb_v1.tar.gz
fi
