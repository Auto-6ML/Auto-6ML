import pickle
import random
import sys
from keytotext import pipeline

nlp = pipeline.pipeline("mrm8488/t5-base-finetuned-common_gen", use_cuda=True)


def word2sentence(classnames, num=200, save_path=''):
    sentence_dict = {}
    for n in classnames:
        sentence_dict[n] = []
    for n in classnames:
        print(n)
        for i in range(num + 100):
            sentence = nlp([n], num_return_sequences=1, do_sample=True)
            sentence_dict[n].append(sentence)
            # print(sentence)

    # remove duplicate
    sampled_dict = {}
    for k, v in sentence_dict.items():
        v_unique = list(set(v))
        sampled_v = random.sample(v_unique, num)
        sampled_dict[k] = sampled_v

    r = open(save_path, "wb")
    pickle.dump(sampled_dict, r)
    r.close()


if __name__ == "__main__":
    num = sys.argv[1]
    dataset = sys.argv[2]
    save_path = "./generated_prompt/{}_data_text.pkl".format(dataset)
    label_path = './dataset/{}/cls_classes.txt'.format(dataset)
    with open(label_path, 'r') as file:
        labels = [line.strip().replace("_", " ") for line in file]
    word2sentence(labels, int(num), save_path)

'''
python LE.py 200 dataset
'''
