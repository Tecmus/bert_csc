#coding:utf-8
from tensorflow.contrib import predictor
import tensorflow as tf
import collections
import tokenization
flags = tf.app.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string(
    "model_dir", None,
    "model path")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")
    
def create_input(input_ids,input_mask,segment_ids):
    features = collections.OrderedDict()
    features["input_ids"] = input_ids
    features["input_mask"] = input_mask
    features["segment_ids"] = segment_ids
    
    return features

def encode_to_feed_dict(query,tokenizer):
    query = tokenization.convert_to_unicode(query)
    input_tokens = tokenizer.tokenize(query)
    input_tokens=['[CLS]']+input_tokens+['[SEP]']+['[SEP]']
    assert len(input_tokens) <= FLAGS.max_seq_length
    input_ids=tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    segment_ids[-1]=1

    while len(input_ids) < FLAGS.max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
            
    feed_dict=create_input([input_ids],[input_mask],[segment_ids])
    
    return feed_dict

def decode_to_query(ids,tokenizer):
    tokens=tokenizer.convert_ids_to_tokens(ids)
    
    pass
def process_output(item,tokenizer):
    seq_topk = item['top_k_res']
    input_ids = item['input']
    
    input_ids=input_ids.tolist()[0]
    seq_topk=seq_topk.tolist()[0]

    input_tokens= [token 
            for token in  tokenizer.convert_ids_to_tokens(input_ids)
            if token != '[PAD]']
    len_input= len(input_tokens)
    input_seq=input_tokens[1:len_input-2]
    output_seq=[]
    for input_token,seq in zip(input_seq, seq_topk[1:len_input-1]):
        out_candi_seq=tokenizer.convert_ids_to_tokens(seq)
        candi_set = set(out_candi_seq)
        if input_token in candi_set:
            output_seq.append(input_token)
        else:  
            output_seq.append(out_candi_seq[0])
      
    return ''.join(input_seq),''.join(output_seq)

def main():

    predict_fn = predictor.from_saved_model(FLAGS.model_dir)
    
    query='女通非常使利'
    
    tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
    feed_dict=encode_to_feed_dict(query,tokenizer)
    pred = predict_fn(feed_dict)
    res=process_output(pred,tokenizer)
    print(res)
if __name__ == "__main__":

    main()