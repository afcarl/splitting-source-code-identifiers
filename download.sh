python download.py 1wZR5zF1GL1fVcA1gZuAN_9rSLd5ssqKV repos2ids_v3.4_stats.csv.gz
gunzip repos2ids_v3.4_stats.csv.gz
mkdir data
mv repos2ids_v3.4_stats.csv data
python preprocess.py
cat data/all_examples{0..48}.json >> data/train_examples.json
cat data/all_examples49.json >> data/test_examples.json

