
export BERT_BASE_DIR=/home/azureuser/projects/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12
python create_csc_data.py \
  --input_file=./sighan15/sighan8csc_release1.0/sighan8csc_release1.0/Test/test_input_tmp.txt \
  --output_file=./record_data/sighan15_test.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=1

