from collections import defaultdict as ddict
from io import open
import re
import random
import itertools as it


class DataManager:
    def __init__(self, vocabSize=10000):
        # vocabSize containes three tags: [START], [END] and [UNK]
        self.vocabSize = vocabSize

    def buildModel(self, trainData):
        self.wordMap = self._getWordMap(trainData)
        print "vocabulary built, with vocabulary size {}".format(min(self.vocabSize, len(self.wordMap)))
        return self

    def buildLookupTabel(self):
        self.wordIndex = {"SOS": 0,  "EOS": 1}
        counter = 2
        for key in self.wordMap:
            if (key == "SOS" or key == "EOS"):
                continue
            self.wordIndex[key] = counter
            counter += 1
        # build wordvec
        print "build initial lookup table"
        return self

    def data4NN(self, pos_review, neg_review, batch):
        # batch: is the size of batch
        # grams: is the input data, text file
        data = []
        if pos_review is None or neg_review is None:
            return None
        with open(pos_review, 'r', encoding="utf-8") as file:
            for line in file:
                line = [u"SOS"] + line.strip().split(" ") + [u"EOS"]
                for i in range(len(line)):
                    if line[i] not in self.wordMap:
                        line[i] = "UNK"
                data.append((line, 1))

        with open(neg_review, 'r', encoding="utf-8") as file:
            for line in file:
                line = [u"SOS"] + line.strip().split(" ") + [u"EOS"]
                for i in range(len(line)):
                    if line[i] not in self.wordMap:
                        line[i] = "UNK"
                data.append((line, 0))

        return WordManager(data, self.wordIndex, batch)

    def _getWordMap(self, Textaddr):
        tmpDict = ddict(int)
        endFreq = 0
        startFreq = 0
        with open(Textaddr, 'r', encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                for i in line.split(" "):
                    i = re.findall("\\w+", i)
                    for w in i:
                        tmpDict[w] += 1
        if (len(tmpDict) > self.vocabSize-3):
            tmpList = sorted([(key, tmpDict[key]) for key in tmpDict], key=lambda x:x[1], reverse=True)
            tmpDict = ddict(int)
            for idx in xrange(self.vocabSize-3):
                tmpDict[tmpList[idx][0]] = tmpList[idx][1]
        tmpDict[u"UNK"] = 1
        tmpDict[u"SOS"] = 1
        tmpDict[u"EOS"] = 1
        return tmpDict

    @staticmethod
    def topNGram(Ngrams, topN):
        if topN > len(Ngrams):
            print "error: top-N should be less than or equal to the size of Ngram model"
            return None
        else:
            rst = [(Ngrams[key], key) for key in Ngrams]
            return heapq.nlargest(topN, rst)


class WordManager:
    def __init__(self, data, lookupTable, batch):
        # data, a list[] of tuple (sen, tag)
        self.rawdata = data
        self.lookupTable = lookupTable
        self.batch = len(data) if batch is None else batch
        self.inverseIdx = None

    def getBatch(self, shuffled=True):
        # batch
        #print self.batch, "get batch"
        if shuffled:
            random.shuffle(self.rawdata)
        batches = it.izip(xrange(0, len(self.rawdata)+1, self.batch), xrange(self.batch, len(self.rawdata)+1, self.batch))
        for start, end in batches:
            #print start, end
            yield self._Helper(start, end)

    def getWordFromIdx(self, idxs):
        # idxs: [batch, ]
        if self.inverseIdx is None:
            self.inverseIdx = {self.lookupTable[key]:key for key in self.lookupTable}
        rst = [self.inverseIdx[i] for i in idxs]
        return " ".join(rst)

    def _Helper(self, start, end):
        tmp = self.rawdata[start:end]
        idx = []
        tag = []
        for tup in tmp:
            #print a==None, b==None, start, end, len(self.rawdata)
            sen = tup[0]
            t = tup[1]
            idx.append([self.lookupTable[w] for w in sen])
            tag.append(t)
        return idx, tag


def getReviewbyRateLen(rate, length, rate_p, review_p, out):
    # length: [lower, upper]
    # rate: [lower, upper]
    name = out + "/review_rate_{}_{}.txt".format(rate, length)
    line = 0
    with open(rate_p, 'r') as ratef, open(review_p, 'r') as reviewf, open(name, 'w') as of:
        for review in reviewf:
            _rate = int(ratef.next().strip().split()[-1])
            if _rate < rate[0] or _rate > rate[1]:
                continue
            l = len(review.strip().split())
            if (l >= length[0] and l <= length[1]):
                of.write(review)
                line += 1
    print "{} lines are selected.".format(line)


if __name__ == "__main__":
    pos_data = "./data/pos_review.txt"
    neg_data = "./data/neg_review.txt"
    DM = DataManager()
    DM.buildModel(pos_data).buildLookupTabel()
    mp = DM.wordMap
    words = sorted([(key, mp[key]) for key in mp], key=lambda p:p[1], reverse=True)
    print mp["SOS"], mp["EOS"], mp["UNK"]
    wm = DM.data4NN(pos_data, neg_data, 1)
    print wm.rawdata[0]
    _iter = wm.getBatch();
    print _iter.next()

    # rate = [8, 10]
    # rate_p = "./data/pos_rate.txt"
    # getReviewbyRateLen(rate, [50, 150], rate_p, traindata, "./data")
