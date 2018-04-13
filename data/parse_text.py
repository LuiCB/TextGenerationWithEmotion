import os
import re
import io
import string
import json

data_dir = './aclimdb/'
output_dir = './aclimdb/'
files = os.listdir(data_dir + 'train/pos')
count = 1
dictionary = set()
review_f = open('pos_review.txt', 'w', encoding='utf-8')
rate_f = open('pos_rate.txt', 'w', encoding='utf-8')
with open(data_dir + 'imdb.vocab', 'r', encoding='utf-8') as d:
    for line in d:
        dictionary.add(line.strip())
for file in files:
    count += 1
    args = re.search('(.*)_(.*)\.txt', file)
    id = int(args.group(1))
    rate = int(args.group(2))

    f = open(data_dir + 'train/pos/' + file, encoding='utf8')
    iter_f = iter(f)
    s = ''
    for line in iter_f:
        s = s + line
    words = s.split()
    ns = ''
    for word in words:
        word.replace('<br />', ' ')
        word = word.strip("?:!.,;])\"\'").lower()
        if word in dictionary:
            ns = ns + word + ' '
    ns = ns.strip()
    review_f.write(ns + '\n')
    rate_f.write(str(id) + ' ' + str(rate) + '\n')
review_f.close()
rate_f.close()
