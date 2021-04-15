#coding=utf-8
#correction position & value -> T
#detection  position -> T

import sys 
import pdb
test_file=sys.argv[1]
# pdb.set_trace()
# test_file='./ocr/data/ocr_test_1000_origin.txt'
test_lines=[line.strip() for line in open(test_file)]

pred_file=sys.argv[2]
# pred_file='./ocr_pred_2k.txt'
pred_lines=[line.strip() for line in open(pred_file)]
assert len(test_lines) == len(pred_lines)
detection_tp=0
detection_fp=0
detection_tn=0
detection_fn=0

correction_tp=0
correction_fp=0
correction_tn=0
correction_fn=0
for test_line,pred_line in zip(test_lines, pred_lines):
    arr=test_line.split('\t')
    assert len(arr) == 3
    label,raw_line,truth_line = arr
    _,pred_line = pred_line.split('\t')
    err_pos=[]
    pred_pos=[]
    if raw_line==truth_line:
        print(raw_line)
    # pdb.set_trace()
    for pos,(r_char,t_char,p_char) in enumerate(zip(raw_line,truth_line,pred_line)):
        if r_char!=t_char:
            err_pos.append((pos,t_char))
        if r_char!=p_char:
            pred_pos.append((pos,p_char))
    if not err_pos:
        if not pred_pos:
            detection_tn+=1
            correction_tn+=1
        else:
            detection_fp+=1
            correction_fp+=1
    else:
            if not pred_pos:
                detection_fn+=1
                correction_fn+=1
            else:
                if [item[0]  for item in err_pos] ==  [item[0] for item in pred_pos]:
                    detection_tp+=1
                    if [item[1]  for item  in err_pos] ==  [item[1] for item in pred_pos]:
                        correction_tp+=1
                    else:
                        correction_fp+=1
                else:
                    detection_fp+=1
                    correction_fp+=1

detection_precision=detection_tp/(detection_tp+detection_fp)
detection_recall=detection_tp/(detection_tp+detection_fn)

print(detection_tp)
print(detection_fp)
print(detection_tn)
print(detection_fn)

correction_precision=correction_tp/(correction_tp+correction_fp)
correction_recall=correction_tp/(correction_tp+correction_fn)
print(detection_precision,detection_recall)

print(correction_tp)
print(correction_fp)
print(correction_tn)
print(correction_fn)
print(correction_precision,correction_recall)

    
    
    
    
        
            
 
    
    
      
    
    
