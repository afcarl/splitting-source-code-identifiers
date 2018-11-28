python download.py 1wZR5zF1GL1fVcA1gZuAN_9rSLd5ssqKV repos2ids_v3.4_stats.csv.gz
gunzip repos2ids_v3.4_stats.csv.gz
mkdir data
mv repos2ids_v3.4_stats.csv data
python preprocess.py
cat all_examples{0..48}.json >> train_examples.json
cat all_examples49.json >> test_examples.json
