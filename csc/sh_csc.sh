
export BERT_BASE_DIR=/root/model/chinese_L-12_H-768_A-12
PYTHONIOENCODING=utf-8 python create_csc_data.py \
  --input_file=./ocr/data/ocr_test_1000.txt \
  --output_file=./record_data/ocr_test_no_shuf.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=1

