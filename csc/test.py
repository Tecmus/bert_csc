#coding:utf-8

line='1我很开心\t我很开行'

def create_csc_masked_lm_predictions(tokens_input,tokens_target, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    masked_lm_positions=[]
    masked_lm_labels=[]

    for idx,(input_token,target_token) in enumerate(zip(tokens_input,tokens_target)):
        if input_token!=target_token:
            masked_lm_positions.append(idx)
            masked_lm_labels.append(target_token)
        else:
            if rng.random() < 0.5:
                masked_lm_positions.append(idx)
                masked_lm_labels.append(target_token)
                                    
    return (tokens_input, masked_lm_positions, masked_lm_labels)

t=['我','开','森','t','m']
i_t=['我','开','心','t','d']
import random
rng=random.Random()

tokens_input, masked_lm_positions, masked_lm_labels=create_csc_masked_lm_predictions(t,i_t,0.8,20,'',rng)
print(tokens_input,masked_lm_labels,masked_lm_positions)


