import argparse
from collections import defaultdict
from tqdm.auto import tqdm

def fetch_all_retrieval_doc_ids(pred_file):
    predictions = defaultdict(list)
    with open(pred_file) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            line1 = line.strip().split()
            predictions[int(line1[0])].append((line1[2], float(line1[4]), line.strip()))
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--title_file", type=str, help="title retrieval file")
    parser.add_argument("--answer_file", type=str, help="answer retrieval file")
    parser.add_argument("--sentence_file", type=str, help="sentence retrieval file")
    parser.add_argument("--op_file", type=str, help="output file")
    
    args = parser.parse_args()
    
    gar_files = []
    
    rets_title = fetch_all_retrieval_doc_ids(args.title_file)
    rets_answer = fetch_all_retrieval_doc_ids(args.answer_file)
    rets_sentence = fetch_all_retrieval_doc_ids(args.sentence_file)
    
    with open(args.op_file, 'w') as fW:
        for key in tqdm(range(len(rets_title.keys()))):
            new_preds = rets_title[key][:333] + rets_answer[key][:333] + rets_sentence[key][:334]
            new_preds = sorted(new_preds, key=lambda v: -v[1])
            fW.writelines("\n".join([line[2] for line in new_preds]) + '\n')
        