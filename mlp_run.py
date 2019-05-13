import argparse
from sklearn.metrics import accuracy_score

from my_lib.deep_model import Model
from my_lib.dataset import Dataset

def main():
    parser = argparse.ArgumentParser(description='MLP for Synonym Antonym prediction')
    parser.add_argument('-i', help='Job identity', type=int, default=0)
    parser.add_argument('-bs', help='Batch size', type=int, default=32)
    parser.add_argument('-e', help='Number of epochs', type=int, default=20)
    parser.add_argument('-id', help='Input w2v dim', type=int, default=150)
    parser.add_argument('-sum', help='Use sum vector', type=int, default=1)
    parser.add_argument('-sub', help='Use subtract vector', type=int, default=1)
    parser.add_argument('-mul', help='Use multiplication vector', type=int, default=1)
    parser.add_argument('-hd', help='Hidden layer configurations', type=str, default='512')
    opt = parser.parse_args()
    print('Running opt: {}'.format(opt))

    print('load data')
    raw_w1 = []
    raw_w2 = []
    raw_label = []

    with open('data/ViCon-400/400_noun_pairs.txt') as f:
        f.readline()
        for l in f:
            w1, w2, label = l.strip().split()
            raw_w1.append(w1)
            raw_w2.append(w2)
            raw_label.append(label)

    with open('data/ViCon-400/400_verb_pairs.txt') as f:
        f.readline()
        for l in f:
            w1, w2, label = l.strip().split()
            raw_w1.append(w1)
            raw_w2.append(w2)
            raw_label.append(label)

    with open('data/ViCon-400/600_adj_pairs.txt') as f:
        f.readline()
        for l in f:
            w1, w2, label = l.strip().split()
            raw_w1.append(w1)
            raw_w2.append(w2)
            raw_label.append(label)

    data = Dataset(raw_w1, raw_w2, raw_label)
    test_data, train_data = data.one_vs_nine()

    print('build model')
    model = Model('syn_ant_{}'.format(opt.i), opt)
    model.build()

    model.run_train(train_data, epochs=opt.e, batch_size=opt.bs)

    y_pred = model.predict_on_test(test_data, predict_class=True)
    y_true = test_data.labels
    print('accuracy:', accuracy_score(y_true=y_true, y_pred=y_pred))

if __name__ == '__main__':
    main()

