# bert-based 中文纠错


## 简介：
利用bert-base模型尝试进行中文纠错。
   
## 方案1:

loss: bert+ mlm

举例：
>input: 今天天晴不错。
>truth:  今天天气不错。		

训练阶段:	

* loss包括两部分，一部分是正例loss(loss(晴，气))，另外一部分随机负例的loss</br>

预测阶段:</br>

* 检错：判断当前位置输入是否在概率top k的里面。 (实际实验Top 1最理想)
* 纠错：如果错误则选top 1进行输出。 

在[FASPell](https://github.com/iqiyi/FASPell) 提供的ocr数据表现。

     Param(top k)|Type |Precision |Recall
----------------------- | :------: | :----------: | :------:
top_1 |detection | **0.793** | **0.766**
top_1|correction| **0.605**|**0.714**
top_2|detection | 0.787 | 0.637
top_2|correction| 0.586|0.567
top_3 |detection | 0.774| 0.584
top_3|correction| 0.572|0.509



       Goodcase          |Truth |Predict
----------------------- | :------: | :----------: | :------:
就是《兆园春，长沙》|就是《沁园春，长沙》 | 就是《沁园春，长沙》| 
这属于话相理不粗    | 这属于话粗理不粗         | 这属于话粗理不粗
裔葩说的高跷松    | 奇葩说的高晓松         | 奇葩说的高晓松 


      Badcase          |Truth |Predict
----------------------- | :------: | :----------: | :------:
让风继续收，不忍远离|让风继续吹，不敢远离 | 让风继续收，不敢远离| 
啊琼她究竟去哪了    | 阿琼她究竟去哪了         | 啊琼她究竟去哪了
很快就都决了    | 很快就解决了         | 很快就解决了



## 使用：
环境：
>python3 </br>
>tensorflow>=1.11

训练：

数据格式：
>3 cols: label\tinput_query\ttruth_query</br>
1\t今天天晴不错。\t今天天气不错。
			
生成数据：
			create_csc_train_data.sh
			
```shell
export BERT_BASE_DIR=./model/chinese_L-12_H-768_A-12

PYTHONIOENCODING=utf-8 python create_csc_data.py \
  --input_file=./ocr/data/ocr_test_1000.txt \
  --output_file=./record_data/ocr_test.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=1
  
```

训练:
train.sh

```shell
export BERT_BASE_DIR=./model/chinese_L-12_H-768_A-12

PYTHONIOENCODING=utf-8 python run_pretraining.py \
  --input_file=./record_data/ocr_tes.tfrecord \
  --output_ckpt=./output_model \
  --do_train=True \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=./output/model.ckpt-10000 \
  --train_batch_size=20 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=100 \
  --num_warmup_steps=100 \
  --learning_rate=2e-5 \
  --top_k=1
		
```
预测：predict.sh

```shell
export BERT_BASE_DIR=/root/model/chinese_L-12_H-768_A-12

PYTHONIOENCODING=utf-8 python csc_predictor.py \
  --model_dir=./save_model \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128
		 
```