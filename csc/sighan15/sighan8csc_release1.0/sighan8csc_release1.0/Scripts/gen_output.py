#coding:utf-8

def load_file(sighan_test, pred_file):
    raw_lines = [for x in open(sighan)]
    pred_lines = [for x in open(pred_file))]

    assert len(raw_lines) == len(pred_lines)
    return raw_lines, pred_lines

def gen_test_format(raw_lines, pred_lines):
    for origin, pred in zip(raw, pred_lines):
    pid, origin_input=origin.split('\t')
    pid=pid[1:-1]
    format_out=build_format(pid,mpred)
    print(format_out)
    
def build_format(pid,pred):
    pred_input, pred_out=pred.split('\t')
    assert pred_input == origin_input
    out_res=[pid]
    for idx,(in_char ,pred_char) in enumerate(zip(pred_input,pred_out)):
        if in_char!=pred_char:
            out_res.append(str(idx))
            out_res.append(pred_char)
    return ','.join(out_res)
                
def main():
    sighan_test=''
    pred_file=''
    raw_lines, pred_lines=load_file(sighan_test, pred_file)
    gen_test_format(raw_lines, pred_lines)
    
if __name__ == '__main__':
    main()