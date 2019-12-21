# -*- coding: utf-8 -*-
# file: main.py
# author: JinTian
# time: 11/03/2017 9:53 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from poems.model import rnn_model
from poems.poems import process_poems
import numpy as np
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import time

start_token = 'B'
end_token = 'E'
model_dir = './model/'
corpus_file = './data/poems.txt'

lr = 0.0002
poemList = []
mutex = threading.Lock()

class Resquest(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json;charset=utf-8')
        self.end_headers()
        if 0 < len(poemList):
            try:
                mutex.acquire()
                poem = poemList[0]
                poemList.pop(0)
                self.wfile.write(str(poem).encode(encoding='utf-8'))
            finally:
                mutex.release()
        else:
            #print("nothing")
            self.wfile.write(str("nothing to say").encode(encoding='utf-8'))

class fillPoem(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID

    def run(self):
        batch_size = 1
        begin_word = None
        print('## loading corpus from %s' % model_dir)
        poems_vector, word_int_map, vocabularies = process_poems(corpus_file)

        input_data = tf.placeholder(tf.int32, [batch_size, None])

        end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
            vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=lr)

        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)

            checkpoint = tf.train.latest_checkpoint(model_dir)
            saver.restore(sess, checkpoint)

            x = np.array([list(map(word_int_map.get, start_token))])
            while True:
                if 100 > len(poemList):
                    lock = False
                    try:
                        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                         feed_dict={input_data: x})
                        word = begin_word or to_word(predict, vocabularies)
                        poem_ = ''

                        i = 0
                        poem_ = ''
                        while word != end_token:
                            poem_ += word
                            i += 1
                            if i > 40:
                                break
                            x = np.array([[word_int_map[word]]])
                            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                             feed_dict={input_data: x,
                                                                        end_points['initial_state']: last_state})
                            word = to_word(predict, vocabularies)

                        poem = pretty_print_poem(poem_)
                        if "" != poem and "                                         。" != poem :
                            lock = True
                            mutex.acquire()
                            poemList.append(poem)
                    finally:
                        if lock:
                            mutex.release()
                else:
                    time.sleep(1)

def to_word(predict, vocabs):
    predict = predict[0]       
    predict /= np.sum(predict)
    sample = np.random.choice(np.arange(len(predict)), p=predict)
    if sample > len(vocabs):
        return vocabs[-1]
    else:
        return vocabs[sample]




def pretty_print_poem(poem_):
    poem = ""
    poem_sentences = poem_.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            poem = poem + s + '。'
    return poem

if __name__ == '__main__':
    thread = fillPoem(1)
    thread.start()

    host = ('0.0.0.0', 22222)
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()

