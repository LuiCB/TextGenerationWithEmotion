#-*- coding:utf-8 -*-

'''
Author: Luyi Ma
variational auto-encoder for Text Generation with Emotion

Design:
    input --> LSTM --> z = f(h) --> LSTM --> output
'''


import numpy as np 
import tensorflow as tf
import DataHub as DH
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense


'''
Hyper-parameters
'''

batch_size = 1
lstm_size = 400
SOS = 0
EOS = 1
maxiter = 100
pretrain_lr = 0.001
pretrain_decay = 0.5
pretrain_momentum = 0.7
device_cf = tf.ConfigProto(device_count={'CPU': 3},
                           inter_op_parallelism_threads = 4,
                           intra_op_parallelism_threads = 4,
                           log_device_placement = False)
vocabulary_size = 10000
embedding_size = 400
SHAPE = [vocabulary_size, embedding_size]
LENGTH = 152 # THE max length for text genreation
T = 1.0
clip_ratio = 2.5
lstm_layer = 1


'''
VAEmo: the class of the variational auto-encoder based model
       for controlled text generation.
'''
class VAEmo:
    def __init__(self, shape, t):
        # shape: [vocabulary_size, embedding_size]
        self.shape = shape
        self.embed = tf.Variable(tf.random_uniform(shape, -t, t),
                                 name="embedding")
   
    '''
    wrapper, to build multi-layer lstm model
    '''
    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in xrange(layer_size)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layers)

    '''
    wrapper, to get the variable under the specific scope name
    '''
    def _get_scope_variable(self, name):
        with tf.variable_scope(name):
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                    scope=tf.get_variable_scope().name)

    '''
    build the final model for the VAE-based model for controlled text generation.
    parameter:
        grad_clip: user defined norm for graident clipping, 
                   to adjust the gradient by the global norm 
                   t_list[i] * clip_norm / max(global_norm, clip_norm)
                   global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
        is_train: a flag to define the model structure for either training 
                  or evaluation.
    '''
    def build_model(self, grad_clip, is_train=1):
        data = tf.placeholder(tf.int32, shape=[1, None], name="input_id")
        train_data = tf.placeholder(tf.int32, shape=[1, None], name="train_id")
        train_label = tf.placeholder(tf.int32, shape=[1, None], name="trian_label")
        z_0 = tf.placeholder(tf.float32, shape=[1], name="prior_selection") # 1 or 0

        wrods = tf.nn.embedding_lookup(self.embed, data)
        decoder_input = tf.nn.embedding_lookup(self.embed, train_data)
        
        with tf.variable_scope("encoder"):
            encoder = self._get_simple_lstm(lstm_size, lstm_layer)
            words = tf.nn.embedding_lookup(self.embed, data)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, words, dtype=tf.float32)

        # define the variational approximation
        epsilon = tf.placeholder(tf.float32, shape=[1], name="epsilon")
        with tf.variable_scope("encoder_approx"):
            mean_encode_layer_1 = Dense(1) # 1
            #mean_encode_layer_2 = Dense(1) # -1
            var_encode_layer = Dense(1)
        mean_approx_1 = mean_encode_layer_1(encoder_state[lstm_layer-1][1])
        #mean_approx_2 = mean_encode_layer_2(encoder_state[0][1])
        var_approx = var_encode_layer(encoder_state[lstm_layer-1][1])
        # p(Z) = z_0 * N(1, 1) + (1-z_0) * N(-1, 1)
        self.Z = (2*z_0 - 1) * mean_approx_1 + epsilon * var_approx
        
        if is_train == 0:
            # do inference
            self.Z = tf.placeholder(tf.float32, shape=[1,1], name="Z_input")
            self.start_tokens = tf.placeholder(tf.int32, shape=[1], name='start_tokens')
            self.end_tokens = tf.placeholder(tf.int32, shape=(), name="end_tokens")
            #print self.end_tokens.shape
            helper = GreedyEmbeddingHelper(self.embed, self.start_tokens, self.end_tokens)
        elif is_train == 1:
            self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None],
                                                     name='decoder_seq_length')
            '''
            NOTICE: since it is an auto-encoder, the input of the traininghelper
                  is the first n-1 words and the output is the last n-1 words
                  Otherwise, it will be just an identity transformation
            '''
            # words' shape: [1, sen_length, vocab_dim]
            helper = TrainingHelper(decoder_input, self.decoder_seq_length)
                   
        with tf.variable_scope("decoder"):
            # decoder, use the latent variable to compute the new initial hidden state
            # and the cell state for the decoding lstm model.
            fc_rec = Dense(lstm_size)
            fc_rec2 = Dense(lstm_size)
            decoder_h = fc_rec(self.Z)
            decoder_c = fc_rec2(self.Z)
            fc_layer = Dense(self.shape[0])
            decoder_cell = self._get_simple_lstm(lstm_size, lstm_layer)
            d_i_s = tf.contrib.rnn.LSTMStateTuple(decoder_c, decoder_h)
            decoder = BasicDecoder(decoder_cell, helper, (d_i_s,), fc_layer)

        logits, final_state, final_sequence_lengths = dynamic_decode(decoder, maximum_iterations=LENGTH)

        if is_train == 0:
            loss = tf.reshape(tf.nn.softmax(logits.rnn_output), [-1, self.shape[0]]) # output shouldn't have SOS
            predict = tf.argmax(loss, axis=1)
            return predict, loss

        elif is_train == 1:
            # train
            targets = tf.reshape(train_label, [-1])
            logits_flatten = tf.reshape(logits.rnn_output, [-1, self.shape[0]])
            cross_ent = tf.losses.sparse_softmax_cross_entropy(targets, logits_flatten)
            #DL_loss = -0.5 * (2 * tf.log(var_approx) - z_0 * tf.square(mean_approx_1)
            #                - (1-z_0) * tf.square(mean_approx_2) + tf.square(var_approx)
            #                + z_0 * mean_approx_1 - (1-z_0) * mean_approx_2)
            DL_loss = -(0.5 * (tf.log(tf.square(var_approx)) - tf.square(mean_approx_1) - tf.square(var_approx))
                      + mean_approx_1)
            loss = DL_loss + cross_ent # negative ELOB
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
            optimizer = tf.train.AdamOptimizer(pretrain_lr)
            train_op = optimizer.apply_gradients(zip(grads, tvars)) # minimize the loss
            return train_op, loss, data, train_data, train_label, z_0, epsilon

    '''
    run the training process of the final model. Both epoch information and the averaged loss
    will be printed and stored in a log file.
    parameter:
        wm:  a WordManager instance
        out: output directory
        model: model path of the trained model
    '''
    def train(self, wm, epoch, out=".", model=None):
        # data: words ids, wm.getBatch() return this iterator
        train_op, loss, data, train_data, train_label, z_0, epsilon = self.build_model(clip_ratio)
        saver = tf.train.Saver()
        log = open(out + "/model_log.txt", 'w')
        with tf.Session(config=device_cf) as sess:
            tf.global_variables_initializer().run()
            if (model is not None):
                saver.restore(sess, model)
                print "load model"

            for i in range(epoch):
                ave_loss = 0
                sentences = wm.getBatch()
                count = 0.0
                for (sen, tag) in sentences:
                    length = [len(s)-1 for s in sen]
                    epsilon_val = np.random.normal(scale=0.1, size=(len(sen), ))
                    _, _loss = sess.run([train_op, loss], 
                                        feed_dict={data: sen, 
                                                  train_data: [sen[0][:-1]],
                                                  train_label: [sen[0][1:]],
                                                  self.decoder_seq_length:length,
                                                  z_0: tag, 
                                                  epsilon: epsilon_val})
                    count += 1
                    

                    ave_loss += (_loss - ave_loss) / count
                    if count % 100 == 0:
                        print "epoch {}, averaged loss: {}".format(i+1, ave_loss)
                        log.write("epoch {}, averaged loss: {}\n".format(i+1, ave_loss))
                print "epoch %d, ave_loss: %f"%(i, ave_loss)
                log.write("epoch %d, ave_loss: %f \n"%(i, ave_loss))
                saver.save(sess, out + '/' + "model2")
                print "*** model updated ***"
                log.write("*** model updated ***\n")
            file.close()


        saver.save(sess, out + '/' + "model2")
        print "*** model updated ***"

    '''
    run the evaluation process
    parameter:
        wm: is the instance of WordManager class
        _z_0: is a list of discriminator 
        _epsilon: normal noise
        model: the model instance
    '''
    def predict(self, wm,  _z_0, _epsilon, model):
        predict, loss = self.build_model(clip_ratio, is_train=0)
        saver = tf.train.Saver()
        print predict.shape
        with tf.Session(config=device_cf) as sess:
            tf.global_variables_initializer().run()
            if (model is not None):
                saver.restore(sess, model)
                print "load model"
            rst = {}
            for i in range(len(_z_0)):
                _ids, _loss = sess.run([predict, loss], 
                                       feed_dict={self.Z: [[_z_0[i] + _epsilon[i]]], 
                                                  self.start_tokens:[SOS], # a list of possible start words
                                                  self.end_tokens:EOS})
                rst[(_z_0[i], _epsilon[i])] = wm.getWordFromIdx(_ids)
                print _loss, self.Z
            return rst
                

if __name__ == "__main__":
    totaldata = "./data/reviews.txt"
    traindata = "./data/review_rate_[1, 3]_[50, 150].txt"
    traindata2 = "./data/review_rate_[8, 10]_[50, 150].txt"
    DM = DH.DataManager()
    DM.buildModel(totaldata).buildLookupTabel()
    mp = DM.wordMap
    wm = DM.data4NN(traindata2, traindata, 1)
    epoch = 50
    data = wm.getBatch()
    model = VAEmo(SHAPE, T)
    # training
    model_p = "./model2"
    model.train(wm, epoch, model=model_p)
    ## predicttion
    # model_p = "./model2"
    # prediction = model.predict(wm, [-1, -1, 1, 1], [0.01,0.001, 0.01, 0.001], model=model_p)
    # print prediction
