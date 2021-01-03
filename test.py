import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('deep-learning-from-scratch-2-master')
from common import config
from common.trainer import Trainer
from common.optimizer import Adam
from ch04.cbow import CBOW
from common.util import create_contexts_target, to_cpu
from dataset import ptb

window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 1　　　

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)

model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs

print(word_vecs)