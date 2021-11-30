import argparse

def open_file(fname):
    if fname is None:
        return None
    with open(fname) as f:
        return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--og_queries_file", type=str, help="OG queries file")
    parser.add_argument("--queries_file", type=str, help="queries file")
    parser.add_argument("--title_file", type=str, help="title predictions file")
    parser.add_argument("--answer_file", type=str, help="answer predictions file")
    parser.add_argument("--sentence_file", type=str, help="sentence predictions file")
    parser.add_argument("--op_file", type=str, help="output file")
    
    args = parser.parse_args()
    
    gar_files = []
    
    lines = open_file(args.queries_file)
    assert lines is not None
    gar_files.append(lines)
    
    lines = open_file(args.title_file)
    if lines is not None:
        gar_files.append(lines)
       
    lines = open_file(args.answer_file)
    if lines is not None:
        gar_files.append(lines)
       
    lines = open_file(args.sentence_file)
    if lines is not None:
        gar_files.append(lines)
    
    with open(args.op_file, 'w') as fW:
        lines = [(str(i) + '\t' + ' '.join(line)) for i, line in enumerate(zip(*gar_files))]
        fW.writelines("\n".join(lines))
        