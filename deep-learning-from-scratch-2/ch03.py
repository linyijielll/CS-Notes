"""
import numpy as np

c = np.array([[1,0,0,0,0,0,0]])
W = np.random.rand(7,3)
h = np.dot(c, W)
print(h)
# [[0.1897929  0.39096817 0.53397147]]


# 推理过程
import numpy as np
from common.layers import MatMul
c0 = np.array([[1,0,0,0,0,0,0]])
c1 = np.array([[0,0,1,0,0,0,0]])
W_in = np.random.randn(7,3)
W_out = np.random.randn(3,7)

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5*h0+0.5*h1
s = out_layer.forward(h)
print(s)
"""


import numpy as np
from common.util import preprocess,create_contexts_target,convert_one_hot
from common.layers import MatMul, SoftmaxWithLoss
from common.trainer import Trainer
from common.optimizer import Adam

text = "You say goodbye and I say hello."
corpus,word_to_id,id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus,window_size=1)
vocab_size = len(word_to_id)
target = convert_one_hot(target,vocab_size)
contexts = convert_one_hot(contexts,vocab_size)


class SimpleCBOW:
    def __init__(self,vocab_size,hidden_size):
        V, H = vocab_size,hidden_size
        W_in = 0.01*np.random.randn(V,H).astype('f')
        W_out = 0.01*np.random.randn(H,V).astype('f')
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        # 把所有权重和梯度整理到列表里
        layers = [self.in_layer0,self.in_layer1,self.out_layer]
        self.params,self.grads = [],[]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        self.word_vecs = W_in

    def forward(self,contexts,target):
        # contexts的形状是(6,2,7) target的形状是(6,7)
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:,1])
        h = 0.5*h0 + 0.5*h1
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score,target)
        return loss

    def backward(self,dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *=0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

model = SimpleCBOW(vocab_size,hidden_size)
optimizer = Adam()
trainer = Trainer(model,optimizer)
trainer.fit(contexts,target,max_epoch,batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id,word in id_to_word.items():
    print(word,word_vecs[word_id])



class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()

        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        l1 = self.loss_layer1.forward(s, contexts[:, 0])
        l2 = self.loss_layer2.forward(s, contexts[:, 1])
        loss = l1 + l2
        return loss

    def backward(self, dout=1):
        dl1 = self.loss_layer1.backward(dout)
        dl2 = self.loss_layer2.backward(dout)
        ds = dl1 + dl2
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None