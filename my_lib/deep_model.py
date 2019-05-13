import numpy as np
import tensorflow as tf

from my_lib.utils import Timer

seed = 13
np.random.seed(seed)


class Model:
    def __init__(self, model_name, opt):
        self._model_name = model_name

        self._input_w2v_dim = opt.id

        self._use_sum_vec = False if opt.sum == 0 else True
        self._use_subtract_vec = False if opt.sub == 0 else True
        self._use_product_vec = False if opt.mul == 0 else True

        self._hidden_layers = list(map(int, opt.hd.split(','))) if opt.hd != '0' else []

        self._num_of_class = 2

        self._trained_model = 'data/trained_weight/' + model_name

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self._label = tf.placeholder(name='label', shape=[None], dtype='int32')

        self._word_1 = tf.placeholder(name='word_1', dtype=tf.float32, shape=[None, 150])
        self._word_2 = tf.placeholder(name='word_2', dtype=tf.float32, shape=[None, 150])

        self._dropout = tf.placeholder(name='dropout', dtype=tf.float32, shape=[])

    def _add_input_representation(self):
        with tf.variable_scope('input_representation'):
            inputs = [
                tf.nn.dropout(self._word_1, keep_prob=self._dropout),
                tf.nn.dropout(self._word_2, keep_prob=self._dropout)
            ]

            if self._use_sum_vec:
                inputs.append(tf.add(self._word_1, self._word_2))

            if self._use_subtract_vec:
                inputs.append(tf.abs(tf.subtract(self._word_1, self._word_2)))

            if self._use_product_vec:
                inputs.append(tf.multiply(self._word_1, self._word_2))

            self._input = tf.concat(inputs, axis=-1)

    def _add_logit_op(self):
        with tf.variable_scope('logit'):
            output = self._input
            for i, v in enumerate(self._hidden_layers, start=1):
                output = tf.layers.dense(
                    inputs=output, units=v, name='hidden_layer_{}'.format(i),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                    activation=tf.nn.relu,
                )
                output = tf.nn.dropout(output, keep_prob=self._dropout)

            self._logit = tf.layers.dense(
                inputs=output, units=self._num_of_class, name='output_layer',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )
            self._predict = tf.nn.softmax(self._logit)

    def _add_loss_op(self):
        with tf.variable_scope('loss_layers'):
            log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._label, logits=self._logit)
            regularizer = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self._loss = tf.reduce_mean(log_likelihood)
            self._loss += tf.reduce_sum(regularizer)

    def _add_train_op(self):
        with tf.variable_scope('train_step'):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self._train_op = optimizer.minimize(self._loss)

    def build(self):
        timer = Timer()
        timer.start('Building model...')

        self._add_placeholders()
        self._add_input_representation()

        self._add_logit_op()

        self._add_loss_op()
        self._add_train_op()

        # f = tf.summary.FileWriter('tensorboard')
        # f.add_graph(tf.get_default_graph())
        # f.close()
        # exit(0)
        timer.stop()

    def _next_batch(self, data, batch_size):
        """

        :param int batch_size:
        :param my_lib.dataset.Dataset data:
        :return:
        """
        start = 0

        while start < len(data.labels):
            label = data.labels[start:start + batch_size]

            word_1 = data.word_1_vecs[start:start + batch_size]
            word_2 = data.word_2_vecs[start:start + batch_size]

            start += batch_size
            batch_data = {
                self._label: np.asarray(label),
                self._word_1: np.asarray(word_1),
                self._word_2: np.asarray(word_2)
            }

            yield batch_data

    def _train(self, data, epochs, batch_size, restart=True):
        """

        :param my_lib.dataset.Dataset data:
        :param int epochs:
        :param int batch_size:
        :return:
        """
        saver = tf.train.Saver(max_to_keep=10)

        with tf.Session() as sess:
            if restart:
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, self._trained_model)

            for e in range(epochs):
                data.shuffle()

                for idx, batch_data in enumerate(self._next_batch(data=data, batch_size=batch_size)):
                    feed_dict = {
                        **batch_data,
                        self._dropout: 1.0
                    }

                    _, loss_train = sess.run([self._train_op, self._loss], feed_dict=feed_dict)
                    if idx % 5 == 0:
                        print('Iter {}, Loss: {}'.format(idx, loss_train))

                print('End epochs {}'.format(e + 1))

            saver.save(sess, self._trained_model)

    def run_train(self, data, epochs, batch_size, restart=True):
        timer = Timer()
        timer.start('Training model...')
        self._train(data=data, epochs=epochs, batch_size=batch_size, restart=restart)
        timer.stop()

    def predict_on_test(self, test, predict_class=True):
        """

        :param predict_class:
        :param dataset.dataset.Dataset test:
        :return:
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('Testing model over test set')
            saver.restore(sess, self._trained_model)

            result = []

            for batch_data in self._next_batch(data=test, batch_size=128):
                feed_dict = {
                    **batch_data,
                    self._dropout: 1.0
                }
                preds = sess.run(self._predict, feed_dict=feed_dict)

                for pred in preds:
                    if predict_class:
                        decode_sequence = np.argmax(pred)
                        result.append(decode_sequence)
                    else:
                        result.append(pred)

        return result
