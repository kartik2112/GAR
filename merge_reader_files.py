import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ip1_file", type=str, help="IP1 file")
    parser.add_argument("--ip2_file", type=str, help="IP2 file")
    parser.add_argument("--output_file", type=str, help="output file")
    
#     parser.add_argument("--top_k", type=int, default=20, help="top k retrieved results to be considered")
    
    args = parser.parse_args()
    
    IP1_FILE = args.ip1_file
    IP2_FILE = args.ip2_file
    
    ip1 = json.load(open(IP1_FILE))
    ip2 = json.load(open(IP2_FILE))
    
#     print(get_top_k_accuracy(args.top_k, predictions, searcher, target))
    with open(args.output_file, 'w') as fW:
        results = []
        for elem1, elem2 in zip(ip1, ip2):
            combined_ctxs = sorted(elem1['ctxs'] + elem2['ctxs'], key=lambda p: -float(p['score']))[:100]
            elem1['ctxs'] = combined_ctxs
            results.append(elem1)
        fW.write(json.dumps(results, indent=4))
        
        