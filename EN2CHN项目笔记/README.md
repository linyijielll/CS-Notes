## transformers 的 Embedding 为什么要乘![d_k](https://latex.codecogs.com/svg.image?&space;\sqrt{d_k})

基于 pytorch 的 embedding  matrix ( vocab_size*embedding_size )默认的初始化方式是**xavier init** ,  N(0, 1/![d_k](https://latex.codecogs.com/svg.image?&space;\sqrt{d_k})), 这会导致元素分布的方差跟随 ![d_model](https://latex.codecogs.com/svg.image?&space;\sqrt{d_{model}})  变化，![d_model](https://latex.codecogs.com/svg.image?\&space;d_{model}) 大输出值的波动比较小。通过乘以  ![d_model](https://latex.codecogs.com/svg.image?&space;\sqrt{d_{model}}) ，可以使embedding matrix的分布回调到 N(0,1) ，有利于训练。


