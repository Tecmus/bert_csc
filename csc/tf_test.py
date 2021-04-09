# import tensorflow as tf
# sess=tf.Session()
# c = tf.constant([[3.0,1.4,1], [5.0,3.0, 4.0]])
# top_2=tf.nn.top_k(c,2)

# with sess.as_default():
#     print(c.eval())
#     # print(top_2.eval())
#     sess.run(top_2)
#     print(top_2.values.eval())
    

import tensorflow as tf
import numpy as np
 
input = tf.constant(np.random.rand(3,4))
k = 2
topk,topk_indices = tf.nn.top_k(input, k)
with tf.Session() as sess:
    print(sess.run(input))
    print(sess.run(topk))
    print(sess.run(topk_indices))
    predict=list(topk_indices.eval())
    for i in predict:
        for j in i:
            print(j)
        # for j in i:
        #     print(j)
        
    
    