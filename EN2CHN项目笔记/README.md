## transformers 的 Embedding 为什么要乘 $\sqrt{d_k}$

基于 pytorch 的 embedding  matrix ( vocab_size*embedding_size )默认的初始化方式是**xavier init** ,  N(0, 1/$\sqrt{d_{model}}$), 这会导致元素分布的方差跟随 $\sqrt{d_{model}}$  变化，$\ d_{model}$ 大输出值的波动比较小。通过乘以  $\ d_{model}$ ，可以使embedding matrix的分布回调到 ![[公式]](https://www.zhihu.com/equation?tex=N%280%2C1%29) ，有利于训练。


