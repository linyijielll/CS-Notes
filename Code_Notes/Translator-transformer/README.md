## Transformers 的 Embedding 为什么要乘![d_k](https://latex.codecogs.com/svg.image?&space;\sqrt{d_k})

基于 pytorch 的 embedding  matrix ( vocab_size*embedding_size )默认的初始化方式是**xavier init** ,  N(0, 1/![d_k](https://latex.codecogs.com/svg.image?&space;\sqrt{d_k})), 这会导致元素分布的方差跟随 ![d_model](https://latex.codecogs.com/svg.image?&space;\sqrt{d_{model}})  变化，![d_model](https://latex.codecogs.com/svg.image?\&space;d_{model}) 大输出值的波动比较小。通过乘以  ![d_model](https://latex.codecogs.com/svg.image?&space;\sqrt{d_{model}}) ，可以使embedding matrix的分布回调到 N(0,1) ，需要主要到position_embedding因为使用sin cos 值在(-1,1)之前， 把embedding拉回N(0,1)有利于训练。

## Transformer 中 Attention 计算为什么要除以![d_k](https://latex.codecogs.com/svg.image?&space;\sqrt{d_k})
反向传播时梯度更加稳定，除![d_k](https://latex.codecogs.com/svg.image?&space;\sqrt{d_k}) 放置梯度在经过softmax之后变的更小。

## 多头注意力机制的作用？
1. 扩展了模型关注不同位置的能力
2. 多头注意力机器赋予attention层多个“子空间表示”。

