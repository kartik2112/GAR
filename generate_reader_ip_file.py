from collections import defaultdict
from pyserini.search import SimpleSearcher
from tqdm.auto import tqdm
import argparse
import json
import sys

sys.path.insert(0, "./dpr")
from data.qa_validation import exact_match_score, has_answer
from utils.tokenizers import SimpleTokenizer

top_K = 20
def fetch_all_retrieval_doc_ids(pred_file):
    predictions = defaultdict(list)
    with open(pred_file) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            line = line.split()
            predictions[line[0]].append((line[2], line[4]))
    return predictions

def get_top_k_accuracy(top_k, predictions, searcher, target):
    hits = 0
    results = []
    for docid in tqdm(predictions.keys()):
        found = False
        answers = [ans for ans in target[int(docid)]['short_answers']]
        res = {'question': target[int(docid)]['question'], 'answers': answers, 'ctxs': []}
        for pred in predictions[docid][:top_k]:
            doc = searcher.doc(int(pred[0]))
            content = json.loads(doc.raw())['contents']
            if has_answer(answers=answers, text=content, tokenizer=SimpleTokenizer(), match_type='string'):
                found = True
            else:
                found = False
            res_obj = {'id': pred[0], 'title': json.loads(doc.raw())['title'], 'text': content, 'score': pred[1], 'has_answer': found}
            res['ctxs'].append(res_obj)
        results.append(res)
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pred_file", type=str, help="predictions file")
    parser.add_argument("--index_file", type=str, help="index file")
    parser.add_argument("--target_file", type=str, help="target file")
    parser.add_argument("--output_file", type=str, help="output file")
    
#     parser.add_argument("--top_k", type=int, default=20, help="top k retrieved results to be considered")
    
    args = parser.parse_args()
    
    PRED_FILE = args.pred_file # "/scratch/kshenoy/output/retrieval_results/gar_sample/run.sample.txt"
    INDEX_FILE = args.index_file # '/scratch/kshenoy/data/indexes/psgs_w100'
    TARGET_FILE = args.target_file # "/scratch/kshenoy/data/data/gold_passages_info/nq_test.json"

    target = json.load(open(TARGET_FILE))['data']
    searcher = SimpleSearcher(INDEX_FILE)
    predictions = fetch_all_retrieval_doc_ids(PRED_FILE)
    print(len(target), len(predictions))
    
#     print(get_top_k_accuracy(args.top_k, predictions, searcher, target))
    with open(args.output_file, 'w') as fW:
        for top_k in [100]:
#         for top_k in [5, 20, 100, 500, 1000]:
            res = get_top_k_accuracy(top_k, predictions, searcher, target)
            fW.write(json.dumps(res, indent=4))
#             print(top_k, res)
#             fW.write(str(top_k) + ': ' + str(res) + '\n')
        
        