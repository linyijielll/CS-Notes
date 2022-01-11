# import numpy as np
#
# # batch_size = 3
# # 2 W_in
# # word = 4-d vector
# arr = [[[1,1,1,1],[2,2,2,2]],
#        [[3,3,3,3],[4,4,4,4]],
#        [[5,5,5,5,],[6,6,6,6]]]
# arr = np.array(arr)
# print(arr.shape)
# print(arr[:,0])

import numpy as np
x = 107
y = np.array(x)
print(y)
z = np.array(x).reshape(1,1)
print(z)
