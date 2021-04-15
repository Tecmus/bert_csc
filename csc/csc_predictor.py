#coding:utf-8
from tensorflow.contrib import predictor


# def create_input():
#     def create_int_feature(values):
#         feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
#         return feature
#     features = collections.OrderedDict()
#     features["input_ids"] = create_int_feature(input_ids)
#     features["input_mask"] = create_int_feature(input_mask)
#     features["segment_ids"] = create_int_feature(segment_ids)
#     tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#     return tf_example

# def _serving_input_receiver_fn():
 
#     receiver_tensors = {'examples': serialized_tf_example}
#     features=create_input()
#     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

# def _serving_input_receiver_fn():
#     """Serving input_fn that builds features from placeholders
#     Returns
#     -------
#     tf.estimator.export.ServingInputReceiver
#     """
#     top_k_res = tf.placeholder(dtype=tf.string, shape=[None, None], name='top_k_res')
#     input_ids = tf.placeholder(dtype=tf.int32, shape=[None,None], name='input_ids')

#     input_ids = tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_ids')
#     input_mask = tf.placeholder(dtype=tf.int64, shape=[None,None], name='input_mask')
#     segment_ids = tf.placeholder(dtype=tf.int64, shape=[None,None], name='segment_ids')

#     receiver_tensors = {'top_k_res': top_k_res, 'input': input_ids}
#     features = {'input_ids': input_ids, 'input_mask': input_mask,'segment_ids':segment_ids}

#     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

# estimator.export_savedmodel(export_dir, _serving_input_receiver_fn)

model_dir='./save_model/1618512160'
predict_fn = predictor.from_saved_model(model_dir)

input_ids=[1]*128
feed_dict={'input_ids': input_ids, 'input_mask': input_ids,'segment_ids':input_ids}
pred = predict_fn(feed_dict)['output']
print(pred)