import numpy as np
from common.layers import Embedding,SigmoidWithLoss
import collections


class EmbeddingDot:
    def __init__(self,W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self,h,idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W*h , axis=1 )
        self.cache = (h,target_W)

    def backwar(self,dout):
        h,target_W = self.cache
        dout = dout.reshape(dout.shape[0],1)
        dtraget_W = dout*h
        self.embed.backward(dtraget_W)
        dh = dout*target_W
        return dh


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1
        vocab_size = len(counts)
        self.vocab_size = vocab_size
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        # positive
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        # negetive
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh

    class CBOW:
        def __init__(self, vocab_size, hidden_size, window_size, corpus):
            V, H = vocab_size, hidden_size
            W_in = 0.01 * np.random.randn(V, H).astype('f')
            # 因为negetiveloss中使用了Embedding层
            W_out = 0.01 * np.random.randn(V, H).astype('f')

            self.in_layers = []
            for i in range(2 * window_size):
                layer = Embedding(W_in)
                self.in_layers.append(layer)
            self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

            layers = self.in_layers + [self.ns_loss]
            self.params, self.grads = [], []
            for layer in layers:
                self.params += layer.params
                self.grads += layer.grads

            self.word_vecs = W_in

        # 这里因为有Embedding层，输入的contexts 和 target 是单词ID 而不是one-hot vector
        def forward(self, contexts, target):
            h = 0
            for i, layer in enumerate(self.in_layers):
                h += layer.forward(contexts[:, i])
            h *= 1 / len(self.in_layers)
            loss = self.ns_loss.forward(h, target)
            return loss

        def backward(self, dout=1):
            dout = self.ns_loss.backward(dout)
            dout *= 1 / len(self.in_layers)
            for layer in self.in_layers:
                layer.backward(dout)
            return None