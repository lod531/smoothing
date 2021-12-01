import os
from pathlib import Path
from collections import defaultdict as dd


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


Path("./synth_data").mkdir(parents=True, exist_ok=True) 

cc = {1:10000, 2:2, 3:3, 5:4, 6:5, 7:6}

t=0
res=[]
for n_x, r in cc.items():
    for _ in range(0, r):
        for _ in range(0, n_x):
            res.append(t)
        t+=1

# sanity check:
# calculate frequencies of tokens
fqs=dd(int)
for token in res:
    fqs[token]+=1
# calculate frequencies of frequencies
t_cc=dd(int)
for _, fq in fqs.items():
    t_cc[fq]+=1
# assert frequencies of frequencies match
for fq, fqfq in t_cc.items():
    assert(t_cc[fq] == cc[fq])

TOKENS_PER_SENTENCE = 10
dataset = list(chunks(res, TOKENS_PER_SENTENCE))


# creating train, test and valid since it's expected
# though the data is identical
files = ["train.tokens", "test.tokens", "valid.tokens"]
for fl in files:
    with open("./synth_data/" + fl, 'a') as the_file:
        for sentence in dataset:
            for token in sentence:
                the_file.write(str(token) + " ")
            the_file.write('\n')
