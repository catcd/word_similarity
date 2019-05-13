from heapq import heappush, heappop

from my_lib import utils
from my_lib.metrics import cosine, dice, euclidean

_, _, WORD_VEC = utils.load_w2v('data/w2v/word2vec.vec')

def get_k_nearest_words(word, k, func):
    if word not in WORD_VEC:
        return ['WORD NOT IN VOCAB']
    else:
        h = []
        vec1 = WORD_VEC[word]
        for w in WORD_VEC:
            if w != word:
                vec2 = WORD_VEC[w]
                dist = func(vec1, vec2)

                heappush(h, (dist, w))

        result = []
        for i in range(k):
            result.append(heappop(h)[1])

        return result

if __name__ == '__main__':
    word = 'đồng_cỏ'
    k = 10

    k_cosine = get_k_nearest_words(word, k, cosine)
    k_dice = get_k_nearest_words(word, k, dice)
    k_euclidean = get_k_nearest_words(word, k, euclidean)

    print('word:', word)
    print(k, 'nearest word ({}):'.format('cosine'), k_cosine)
    print(k, 'nearest word ({}):'.format('dice'), k_dice)
    print(k, 'nearest word ({}):'.format('euclidean'), k_euclidean)