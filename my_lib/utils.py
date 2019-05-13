import codecs
import time

UNICODE = 'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ'
ASCII   = 'aaaaaaaaaaaaaaaaaoooooooooooooooooeeeeeeeeeeeuuuuuuuuuuuiiiiiyyyyydAAAAAAAAAAAAAAAAAOOOOOOOOOOOOOOOOOEEEEEEEEEEEUUUUUUUUUUUIIIIIYYYYYD'
TRANS = str.maketrans(UNICODE, ASCII)

def unikey(seq):
    return seq.translate(TRANS)

def load_w2v(input_fname):
    with codecs.open(input_fname, encoding='utf-8') as f:
        vocab_size, embedding_dim = f.readline().split()
        vocab_size = int(vocab_size)
        embedding_dim = int(embedding_dim)

        vocab = {}
        for l in f:
            w, vec = l.split(' ', maxsplit=1)
            vocab[w] = list(map(float, vec.split()))

        return vocab_size, embedding_dim, vocab


def get_vocab(input_fname):
    with codecs.open(input_fname, encoding='utf-8') as f:
        f.readline()

        vocab = []
        for l in f:
            w, _ = l.split(' ', maxsplit=1)
            vocab.append(w)

        vocab = list(set(vocab))
        vocab.sort(key=unikey)

        alphabet = list(set(''.join(vocab)))
        alphabet.sort(key=unikey)

        return vocab, alphabet

def read_vocab():
    v, a = [], []

    with open('data/alphabet.txt', 'w') as f:
        for c in f:
            a.append(c.strip())

    with open('data/vocab.txt', 'w') as f:
        for w in f:
            v.append(w.strip())

    return v, a


def write_vocab():
    v, a = get_vocab('data/w2v/word2vec.vec')

    with open('data/alphabet.txt', 'w') as f:
        f.write('PAD\n')
        for c in a:
            f.write('{}\n'.format(c))
        f.write('$UNK$')

    with open('data/vocab.txt', 'w') as f:
        f.write('PAD\n')
        for w in v:
            f.write('{}\n'.format(w))
        f.write('$UNK$')


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.job = None

    def start(self, job):
        if job is None:
            return None
        self.start_time = time.time()
        self.job = job
        print("[INFO] {job} started.".format(job=self.job))

    def stop(self):
        if self.job is None:
            return None
        elapsed_time = time.time() - self.start_time
        print("[INFO] {job} finished in {elapsed_time:0.3f} s."
              .format(job=self.job, elapsed_time=elapsed_time))
        self.job = None


if __name__ == '__main__':
    write_vocab()