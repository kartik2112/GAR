salloc --time=9:00:00 --cpus-per-task=4 --mem=16GB --partition=gpu --gres=gpu:v100:1 --priority=TOP

# Python BART Caching since discovery gpu mode cuts off internet:
# from transformers import BartTokenizer, BartModel
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Task 1
GEN_TARGET='answer' python train_generator.py --remark generator_train_nq_A  --train_batch_size 128 --eval_batch_size 256 --ckpt_metric val-ROUGE-1 --data_dir '/scratch/kshenoy/data/nq-answer/' --output_dir '/scratch/kshenoy/output/GAR_models/nq-answer/'

GEN_TARGET='title' python train_generator.py --remark generator_train_nq_A  --train_batch_size 128 --eval_batch_size 256 --ckpt_metric val-ROUGE-1 --data_dir '/scratch/kshenoy/data/nq-title/' --output_dir '/scratch/kshenoy/output/GAR_models/nq-title/'

GEN_TARGET='sentence' python train_generator.py --remark generator_train_nq_A  --train_batch_size 128 --eval_batch_size 256 --ckpt_metric val-ROUGE-1 --data_dir '/scratch/kshenoy/data/nq-sentence/' --output_dir '/scratch/kshenoy/output/GAR_models/nq-sentence/'

GEN_TARGET='answer' python train_generator.py --remark generator_train_nq_A  --train_batch_size 128 --eval_batch_size 256 --ckpt_metric val-ROUGE-1 --data_dir '/scratch/kshenoy/data/nq-answer/' --output_dir '/scratch/kshenoy/output/GAR_models_v2/nq-answer/' --learning_rate=1e-6 --num_train_epochs 150

GEN_TARGET='title' python train_generator.py --remark generator_train_nq_A  --train_batch_size 128 --eval_batch_size 256 --ckpt_metric val-ROUGE-1 --data_dir '/scratch/kshenoy/data/nq-title/' --output_dir '/scratch/kshenoy/output/GAR_models_v2/nq-title/' --learning_rate=1e-6 --num_train_epochs 150

GEN_TARGET='sentence' python train_generator.py --remark generator_train_nq_A  --train_batch_size 128 --eval_batch_size 256 --ckpt_metric val-ROUGE-1 --data_dir '/scratch/kshenoy/data/nq-sentence/' --output_dir '/scratch/kshenoy/output/GAR_models_v2/nq-sentence/' --learning_rate=1e-6 --num_train_epochs 150

GEN_TARGET='answer' python train_generator.py --remark generator_train_nq_A  --train_batch_size 128 --eval_batch_size 256 --ckpt_metric val-ROUGE-1 --data_dir '/scratch/kshenoy/data/nq-answer/' --output_dir '/scratch/kshenoy/output/GAR_models_v3/nq-answer/' --learning_rate=1e-5 --num_train_epochs 150

GEN_TARGET='title' python train_generator.py --remark generator_train_nq_A  --train_batch_size 128 --eval_batch_size 256 --ckpt_metric val-ROUGE-1 --data_dir '/scratch/kshenoy/data/nq-title/' --output_dir '/scratch/kshenoy/output/GAR_models_v3/nq-title/' --learning_rate=1e-5 --num_train_epochs 150

GEN_TARGET='sentence' python train_generator.py --remark generator_train_nq_A  --train_batch_size 128 --eval_batch_size 256 --ckpt_metric val-ROUGE-1 --data_dir '/scratch/kshenoy/data/nq-sentence/' --output_dir '/scratch/kshenoy/output/GAR_models_v3/nq-sentence/' --learning_rate=1e-5 --num_train_epochs 150

# Executed trainers on 3 parallel V100s.
# Answer: 21 hours
# Title: 30 hours
# Sentence: 35 hours

sinfo -o "%20N  %10c  %10m  %25f  %30G "


# Pyserini related:
# https://github.com/castorini/pyserini/blob/master/docs/installation.md

# Task 3A: 49 minutes

python -m pyserini.index -collection JsonCollection \
                         -generator DefaultLuceneDocumentGenerator \
                         -threads 4 \
                         -input /scratch/kshenoy/data/wikipedia_splits \
                         -index /scratch/kshenoy/data/indexes/psgs_w100 \
                         -storePositions -storeDocvectors -storeRaw


# python -m pyserini.encode input   --corpus /scratch/kshenoy/data/dense_wikipedia_splits \
# --fields text \
# output  --embeddings /scratch/kshenoy/data/indexes/dindex-sample-dpr-multi \
# --to-faiss \
# encoder --encoder facebook/dpr-ctx_encoder-multiset-base \
# --fields text \
# --batch 64


# ISI Execution:
# https://github.com/castorini/pyserini/blob/master/docs/usage-dense-indexes.md
# python -m pyserini.encode input   --corpus . --fields text --shard-id 0 --shard-num 4 output  --embeddings ../indexes --to-faiss encoder --encoder facebook/dpr-ctx_encoder-multiset-base --fields text --batch 128 --device cuda:1

# python -m pyserini.dindex --corpus . \
#                           --encoder facebook/dpr-ctx_encoder-multiset-base \
#                           --index ./indexes \
#                           --batch 64 \
#                           --device cpu \
#                           --title-delimiter '\n' 


# Task 3: NQ GAR evaluations ~1min per command

DATA_DIR='/scratch/kshenoy/data'
RET_DIR='/scratch/kshenoy/output/GAR_results'
CPT_DIR='/scratch/kshenoy/output/GAR_models'
python test_generator.py --input_path $DATA_DIR/nq-answer/test.source --reference_path $DATA_DIR/nq-answer/test.target --output_path $RET_DIR/nq-answer/gar_nq_test_answer.txt --score_path $RET_DIR/nq-answer/ROUGE-test_nq_answer.txt --bs 256 --model_ckpt $CPT_DIR/nq-answer/checkpointlast.ckpt --max_source_length 20 --max_target_length 40 --remark test_nq_answer

DATA_DIR='/scratch/kshenoy/data'
RET_DIR='/scratch/kshenoy/output/GAR_results'
CPT_DIR='/scratch/kshenoy/output/GAR_models'
python test_generator.py --input_path $DATA_DIR/nq-title/test.source.full --reference_path $DATA_DIR/nq-title/test.target.full --output_path $RET_DIR/nq-title/gar_nq_test_title.txt --score_path $RET_DIR/nq-title/ROUGE-test_nq_title.txt --bs 256 --model_ckpt $CPT_DIR/nq-title/checkpointlast.ckpt --max_source_length 20 --max_target_length 40 --remark test_nq_title

DATA_DIR='/scratch/kshenoy/data'
RET_DIR='/scratch/kshenoy/output/GAR_results'
CPT_DIR='/scratch/kshenoy/output/GAR_models'
python test_generator.py --input_path $DATA_DIR/nq-sentence/test.source --reference_path $DATA_DIR/nq-sentence/test.target --output_path $RET_DIR/nq-sentence/gar_nq_test_sentence.txt --score_path $RET_DIR/nq-sentence/ROUGE-test_nq_sentence.txt --bs 256 --model_ckpt $CPT_DIR/nq-sentence/checkpointlast.ckpt --max_source_length 20 --max_target_length 40 --remark test_nq_sentence

# Task 3: NQ GAR Predictions for sentence since its target doesn't cover all questions

DATA_DIR='/scratch/kshenoy/data'
RET_DIR='/scratch/kshenoy/output/GAR_results'
CPT_DIR='/scratch/kshenoy/output/GAR_models'
python test_generator.py --input_path $DATA_DIR/nq-sentence/test.source.full --output_path $RET_DIR/nq-sentence/gar_nq_test_sentence.full.txt --bs 256 --model_ckpt $CPT_DIR/nq-sentence/checkpointlast.ckpt --max_source_length 20 --max_target_length 40 --remark test_nq_sentence_full

# Task 3: VAL SET NQ GAR evaluations ~1min per command

DATA_DIR='/scratch/kshenoy/data'
RET_DIR='/scratch/kshenoy/output/GAR_results'
CPT_DIR='/scratch/kshenoy/output/GAR_models'
python test_generator.py --input_path $DATA_DIR/nq-answer/val.source --reference_path $DATA_DIR/nq-answer/val.target --output_path $RET_DIR/nq-answer/gar_nq_val_answer.txt --score_path $RET_DIR/nq-answer/ROUGE-val_nq_answer.txt --bs 256 --model_ckpt $CPT_DIR/nq-answer/checkpointlast.ckpt --max_source_length 20 --max_target_length 40 --remark val_nq_answer

DATA_DIR='/scratch/kshenoy/data'
RET_DIR='/scratch/kshenoy/output/GAR_results'
CPT_DIR='/scratch/kshenoy/output/GAR_models'
python test_generator.py --input_path $DATA_DIR/nq-title/val.source --reference_path $DATA_DIR/nq-title/val.target --output_path $RET_DIR/nq-title/gar_nq_val_title.txt --score_path $RET_DIR/nq-title/ROUGE-val_nq_title.txt --bs 256 --model_ckpt $CPT_DIR/nq-title/checkpointlast.ckpt --max_source_length 20 --max_target_length 40 --remark val_nq_title

DATA_DIR='/scratch/kshenoy/data'
RET_DIR='/scratch/kshenoy/output/GAR_results'
CPT_DIR='/scratch/kshenoy/output/GAR_models'
python test_generator.py --input_path $DATA_DIR/nq-sentence/val.source --reference_path $DATA_DIR/nq-sentence/val.target --output_path $RET_DIR/nq-sentence/gar_nq_val_sentence.txt --score_path $RET_DIR/nq-sentence/ROUGE-val_nq_sentence.txt --bs 256 --model_ckpt $CPT_DIR/nq-sentence/checkpointlast.ckpt --max_source_length 20 --max_target_length 40 --remark val_nq_sentence

# Task 3: NQ GAR Predictions for sentence since its target doesn't cover all questions

DATA_DIR='/scratch/kshenoy/data'
RET_DIR='/scratch/kshenoy/output/GAR_results'
CPT_DIR='/scratch/kshenoy/output/GAR_models'
python test_generator.py --input_path $DATA_DIR/nq-sentence/test.source.full --output_path $RET_DIR/nq-sentence/gar_nq_test_sentence.full.txt --bs 256 --model_ckpt $CPT_DIR/nq-sentence/checkpointlast.ckpt --max_source_length 20 --max_target_length 40 --remark test_nq_sentence_full

# Task 5: Generation of augmented queries: Instantaneous

python merge_queries.py --queries_file /scratch/kshenoy/data/nq-title/test.source.full \
--answer_file /scratch/kshenoy/output/GAR_results/nq-answer/gar_nq_test_answer.txt \
--op_file /scratch/kshenoy/output/queries/gar_trained/nq_test_queries.answer.tsv

python merge_queries.py --queries_file /scratch/kshenoy/data/nq-title/test.source.full \
--sentence_file /scratch/kshenoy/output/GAR_results/nq-sentence/gar_nq_test_sentence.full.txt \
--op_file /scratch/kshenoy/output/queries/gar_trained/nq_test_queries.sentence.tsv

python merge_queries.py --queries_file /scratch/kshenoy/data/nq-title/test.source.full \
--title_file /scratch/kshenoy/output/GAR_results/nq-title/gar_nq_test_title.txt \
--op_file /scratch/kshenoy/output/queries/gar_trained/nq_test_queries.title.tsv

python merge_queries.py --queries_file /scratch/kshenoy/data/nq-title/test.source.full \
--answer_file /scratch/kshenoy/output/GAR_results/nq-answer/gar_nq_test_answer.txt \
--sentence_file /scratch/kshenoy/output/GAR_results/nq-sentence/gar_nq_test_sentence.full.txt \
--title_file /scratch/kshenoy/output/GAR_results/nq-title/gar_nq_test_title.txt \
--op_file /scratch/kshenoy/output/queries/gar_trained/nq_test_queries.multi_input.tsv

python merge_queries.py --queries_file /scratch/kshenoy/data/nq-title/test.source.full \
--op_file /scratch/kshenoy/output/queries/gar_trained/nq_test_queries.original.tsv

# Task 6:
# Retrieve documents
# python -m pyserini.search --topics /scratch/kshenoy/output/queries/gar_sample/queries.answer_NEW.QG.test.tsv \
# --index /scratch/kshenoy/data/indexes/psgs_w100 \
# --output /scratch/kshenoy/output/retrieval_results/gar_sample/run.sample.txt \
# --bm25 \
# --threads 16

# Took 20 mins
python -m pyserini.search --topics /scratch/kshenoy/output/queries/gar_trained/nq_test_queries.answer.tsv \
--index /scratch/kshenoy/data/indexes/psgs_w100 \
--output /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.answer.txt \
--bm25 \
--threads 16

# Took 39 mins
python -m pyserini.search --topics /scratch/kshenoy/output/queries/gar_trained/nq_test_queries.sentence.tsv \
--index /scratch/kshenoy/data/indexes/psgs_w100 \
--output /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.sentence.txt \
--bm25 \
--threads 16

# Took 22 mins
python -m pyserini.search --topics /scratch/kshenoy/output/queries/gar_trained/nq_test_queries.title.tsv \
--index /scratch/kshenoy/data/indexes/psgs_w100 \
--output /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.title.txt \
--bm25 \
--threads 16

# Took 30 mins
python -m pyserini.search --topics /scratch/kshenoy/output/queries/gar_trained/nq_test_queries.multi_input.tsv \
--index /scratch/kshenoy/data/indexes/psgs_w100 \
--output /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.multi_input.txt \
--bm25 \
--threads 16

# Took 13.5 mins
python -m pyserini.search --topics /scratch/kshenoy/output/queries/gar_trained/nq_test_queries.original.tsv \
--index /scratch/kshenoy/data/indexes/psgs_w100 \
--output /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.original.txt \
--bm25 \
--threads 16

python merge_predictions.py --answer_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.answer.txt \
--sentence_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.sentence.txt \
--title_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.title.txt \
--op_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.fused.txt

# Task 7:
python retr_acc.py --pred_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.original.txt --index_file /scratch/kshenoy/data/indexes/psgs_w100 --target_file /scratch/kshenoy/data/data/gold_passages_info/nq_test.json --results_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.original.results.txt

python retr_acc.py --pred_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.answer.txt --index_file /scratch/kshenoy/data/indexes/psgs_w100 --target_file /scratch/kshenoy/data/data/gold_passages_info/nq_test.json --results_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.answer.results.txt

python retr_acc.py --pred_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.multi_input.txt --index_file /scratch/kshenoy/data/indexes/psgs_w100 --target_file /scratch/kshenoy/data/data/gold_passages_info/nq_test.json --results_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.multi_input.results.txt

python retr_acc.py --pred_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.fused.txt --index_file /scratch/kshenoy/data/indexes/psgs_w100 --target_file /scratch/kshenoy/data/data/gold_passages_info/nq_test.json --results_file /scratch/kshenoy/output/retrieval_results/gar_trained/nq_test.fused.results.txt