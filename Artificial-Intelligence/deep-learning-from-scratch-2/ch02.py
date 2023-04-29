from common.util import *
import numpy as np
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus,vocab_size,window_size=1)
W = ppmi(C)

# svd
U, S, V = np.linalg.svd(W)

# 经过奇异值分解
print(C[0])
print(W[0])
print(U[0])
# 对U这个密集向量降维,例如降到二维，只需要取出前两个元素
print(U[0,:2])

#画图展示
for word,word_id in word_to_id.items():
    plt.annotate(word,(U[word_id,0],U[word_id,1]))
plt.scatter(U[:,0],U[:,1],alpha=0.5)
plt.show()
