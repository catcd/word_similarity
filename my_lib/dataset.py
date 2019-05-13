import numpy as np
from collections import Counter
from sklearn.utils import shuffle

from my_lib import utils

_, _, WORD_VEC = utils.load_w2v('data/w2v/word2vec.vec')
LABELS = {'SYN': 0, 'ANT': 1}

class Dataset:
    def __init__(self, word_1_s=None, word_2_s=None, labels=None):
        self.word_1_vecs = []
        self.word_2_vecs = []
        self.labels = []

        if word_1_s is not None and word_2_s is not None and labels is not None:
            cp, cn = 0, 0
            for w1, w2, l in zip(word_1_s, word_2_s, labels):
                if w1 not in WORD_VEC or w2 not in WORD_VEC:
                    cn += 1
                else:
                    cp += 1
                    self.word_1_vecs.append(WORD_VEC[w1])
                    self.word_2_vecs.append(WORD_VEC[w2])
                    self.labels.append(LABELS[l])

            print(cp, 'valid example.', cn, 'OOV examples')

    def shuffle(self):
        (
            self.labels,
            self.word_1_vecs,
            self.word_2_vecs
        ) = shuffle(
            self.labels,
            self.word_1_vecs,
            self.word_2_vecs
        )

    def one_vs_nine(self):
        c = Counter(self.labels)
        print('shape of data: {}'.format({k: c[k] for k in c}))
        num_of_example = len(self.labels)
        indicates = np.random.choice(num_of_example, num_of_example//10, replace=False)

        one_data = Dataset()
        one_data.word_1_vecs = [v for i, v in enumerate(self.word_1_vecs) if i in indicates]
        one_data.word_2_vecs = [v for i, v in enumerate(self.word_2_vecs) if i in indicates]
        one_data.labels = [v for i, v in enumerate(self.labels) if i in indicates]
        c = Counter(one_data.labels)
        print('shape of 10% data: {}'.format({k: c[k] for k in c}))

        nine_data = Dataset()
        nine_data.word_1_vecs = [v for i, v in enumerate(self.word_1_vecs) if i not in indicates]
        nine_data.word_2_vecs = [v for i, v in enumerate(self.word_2_vecs) if i not in indicates]
        nine_data.labels = [v for i, v in enumerate(self.labels) if i not in indicates]
        c = Counter(nine_data.labels)
        print('shape of 90% data: {}'.format({k: c[k] for k in c}))

        return one_data, nine_data
