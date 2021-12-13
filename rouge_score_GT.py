import argparse
from collections import defaultdict
from tqdm.auto import tqdm
import sys
import json

sys.path.insert(0, "./gar")
sys.path.insert(0, "./dpr")

from gar.train_generator import calculate_rouge

def open_file(fname, mode='txt'):
    if fname is None:
        return None
    with open(fname) as f:
        if mode == 'txt':
            return [line.strip() for line in f.readlines()]
        else:
            return [line.strip().split('\t')[1] for line in f.readlines()]
    
def preprocess(gt):
    gt_new = {}
    for elem in gt:
        gt_new[elem['question_tokens']] = elem['context']
    return gt_new

def get_gt_entries(gt, keys):
    return [gt[key] for key in keys]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--queries_file", type=str, help="queries file")
    parser.add_argument("--generations_file", type=str, help="generation file")
    parser.add_argument("--gt_file", type=str, help="ground truth file")
    
    args = parser.parse_args()
    
    gar_files = []
    
    rets_title = open_file(args.queries_file, 'txt')
    rets_answer = open_file(args.generations_file, 'tsv')
    target = preprocess(json.load(open(args.gt_file))['data'])
    
    print(calculate_rouge(rets_answer, get_gt_entries(target, rets_title)))