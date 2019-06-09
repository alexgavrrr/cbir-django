# Django image storage app powered by CBIR library

## Setup instructions
```
pipenv shell
pipenv install
python manage.py makemigrations
python manage.py migrate
```

## To run public server
 
git stash apply 0
python manage.py runserver 0.0.0.0:80


## Development

Reload from scratch command
```
yes | rm db.sqlite3;
    rm -rf photologue/migrations/*; touch photologue/migrations/__init__.py && \
    python manage.py makemigrations && cp backup/0002_photosize_data.py photologue/migrations && \
    python manage.py migrate && \
    ./createsuper.sh && \
    # ./clean_content.sh && \
    # ./clean_cbir_state.sh
```

```
yes | rm db.sqlite3;
    touch photologue/migrations/__init__.py && \
    python manage.py makemigrations && \
    python manage.py migrate && \
    ./createsuper.sh && \
    # ./clean_content.sh && \
    # ./clean_cbir_state.sh
```

## Profiling commands
`mkdir -p profs/now`

`python -m cProfile -o profs/now/create_database.prof ./manage.py create_database oxford data/Buildings/Original/Oxford/jpg`

`python -m cProfile -o profs/now/create_index_for_database.prof ./manage.py create_index_for_database oxford index_oxford_1 data/Buildings/Original/Paris/jpg`

`python -m cProfile -o profs/now/create_event.prof ./manage.py create_event oxford index_oxford_1 data/Buildings/Original/Oxford/jpg/oriel_000062.jpg --sv --qe`

`python -m cProfile -o profs/now/evaluate.prof cbir_main.py evaluate_only oxford index_oxford_1 public/media/content/oxford/database_all data/Buildings/Original/Oxford/gt --sv --qe`

profile evaluating commands?

## Start snakeviz for visualizing profs
`snakeviz [--server]`

## logs/ column names
**computing_descriptors**
elapsed

**training_clusterer**
time_copying_descriptors_to_memmap, time_fitting_vocabulary_tree, time_saving_vocabulary_tree

**add_images_to_index.txt**
time_writing_bows, time_creating_index_by_word, time_building_inv

**retrieving_candidates**
n_candidatesl, elapsed, whether_qe_stage

**preliminary_sorting.txt**
elapsed, whether_qe_stage

**answers**
query_image, result_image, rank, similarity, sv, qe

## commands
sample 10 4
`python cbir_main.py --log_prefix logs/sample_10_4 create_index --K 10 --L 4 d-sample i-1 data/Buildings/Original/Oxford_sample/jpg 14`

`python cbir_main.py --log_prefix logs/sample_10_4 search d-sample i-1 data/Buildings/Original/Oxford_sample/jpg/all_souls_000013.jpg --topk 10` 

`...`

sample 10 5
`python cbir_main.py --log_prefix logs/sample_10_5 change_params --K 10 --L 5 d-sample i-1`

`python cbir_main.py --log_prefix logs/sample_10_5 search d-sample i-1 data/Buildings/Original/Oxford_sample/jpg/all_souls_000013.jpg --topk 10` 

`...`


10k 10 4
`python cbir_main.py --log_prefix logs/10k_10_4 create_index --K 10 --L 4 d-10k i-1 data/Buildings/Original/Oxford/jpg 10000`

`python cbir_main.py --log_prefix logs/10k_10_4 search d-10k i-1 data/Buildings/Original/Oxford/jpg/all_souls_000013.jpg --topk 10` 

`...`

10k 10 5
`python cbir_main.py --log_prefix logs/10k_10_5 change_params --K 10 --L 5 d-10k i-1`

`python cbir_main.py --log_prefix logs/10k_10_5 search d-10k i-1 data/Buildings/Original/Oxford/jpg/all_souls_000013.jpg --topk 10` 

`...`



100k 10 4
`python cbir_main.py --log_prefix logs/100k_10_4 create_index --K 10 --L 4 d-100k i-1 data/Buildings/Original/Oxford/jpg 100000`

`python cbir_main.py --log_prefix logs/100k_10_4 search d-100k i-1 data/Buildings/Original/Oxford/jpg/all_souls_000013.jpg --topk 10` 

`...`

100k 10 5
`python cbir_main.py --log_prefix logs/100k_10_5 change_params --K 10 --L 5 d-100k i-1`

`python cbir_main.py --log_prefix logs/100k_10_5 search d-100k i-1 data/Buildings/Original/Oxford/jpg/all_souls_000013.jpg --topk 10` 

`...`

scp -r gavr@40.114.26.173:~/main/cbir-django/logs/100k_10_4 saved_logs

scp -r gavr@40.121.12.226:~/main/cbir-py-prototype/logs/hardnet_10_4 saved_logs

scp -r gavr@40.121.12.226:~/main/cbir-py-prototype/logs/result* saved_logs/original

scp -r gavr@13.82.228.62:~/main/cbir-django/logs/default saved_logs/my_old_10k_10_4

python cbir_main.py --log_prefix logs/100k_10_5 change_params --K 10 --L 5 d-100k i-1

--inspect index

python cbir_main.py --log_prefix logs/100k_10_5 search d-100k i-1 data/Buildings/Original/Oxford/jpg/all_souls_000013.jpg --topk 10 --p_fine_max 0.5
 or --p_fine_max 0.1
 
--inspect answers

python cbir_main.py --log_prefix logs/100k_10_5 evaluate_only d-100k i-1 data/Buildings/Original/Oxford/jpg data/Buildings/Original/Oxford/gt  --p_fine_max TODO --sv --qe

scp -r gavr@40.76.75.111:~/main/cbir-django/logs/vectorized_sparse_dists_with_idf_refactored saved_logs/vectorized_sparse_dists_with_idf_refactored

./download_flickr_100k.sh 1 40 && unrar x -y  oxc1_100k.part01.rar \
 && 
 
 scp -r gavr@137.117.96.159:~/main/cbir-django/logs/d-oxford-eval saved_logs/d-oxford-eval
 
 python cbir_main.py --log_prefix logs/d-oxford-eval-2 evaluate_only d-paris i-paris-1 data/Buildings/Original/Paris/jpg  data/Buildings/Original/Paris/gt \
 && python cbir_main.py --log_prefix logs/d-oxford-eval-sv evaluate_only d-paris i-paris-1 data/Buildings/Original/Paris/jpg  data/Buildings/Original/Paris/gt --sv \
 && python cbir_main.py --log_prefix logs/d-oxford-eval-qe evaluate_only d-paris i-paris-1 data/Buildings/Original/Paris/jpg  data/Buildings/Original/Paris/gt --sv --qe
 
scp -r gavr@137.117.96.159:~/main/cbir-django/logs/d-oxford-eval-2 saved_logs/d-oxford-eval-2 \
&&  scp -r gavr@137.117.96.159:~/main/cbir-django/logs/d-oxford-eval-sv saved_logs/d-oxford-eval-sv \
&&  scp -r gavr@137.117.96.159:~/main/cbir-django/logs/d-oxford-eval-qe saved_logs/d-oxford-eval-qe
logs/d-110k-pqkmeans

 scp -r gavr@137.117.96.159:~/main/cbir-django/logs/d-110k-pqkmeans saved_logs/d-110k-pqkmeans
  scp -r gavr@137.117.96.159:~/main/cbir-django/logs/d-110k-pqkmeans-eval-1 saved_logs/d-110k-pqkmeans-eval-1


 python cbir_main.py --log_prefix logs/d-oxford-eval-topk-11 evaluate_only d-paris i-paris-1 data/Buildings/Original/Paris/jpg  data/Buildings/Original/Paris/gt --topk 11
 
 scp -r gavr@137.117.96.159:~/main/cbir-django/answers/False_False_d-paris.txt saved_logs/answers/False_False_d-paris_eval_recent.txt
 
 scp -r gavr@137.117.96.159:~/main/cbir-django/answers/True_False_d-paris.txt saved_logs/answers/True_False_d-paris_eval_recent_sv.txt
 
 ___________
 python cbir_main.py evaluate_only d-sample i-1 data/Buildings/Original/Oxford_sample/jpg data/Buildings/Original/Oxford/gt --sv

python cbir_main.py --log_prefix logs/10k_10_5 evaluate_only d-10k i-1 data/Buildings/Original/Oxford/jpg data/Buildings/Original/Oxford/gt

python cbir_main.py --log_prefix logs/100k_10_4 evaluate_only d-100k i-1 data/Buildings/Original/Oxford/jpg data/Buildings/Original/Oxford/gt

python cbir_main.py --log_prefix logs/logs_now evaluate_only d-now i-1 data/Buildings/Original/Oxford_sample/jpg data/Buildings/Original/Oxford_sample/gt


python cbir_main.py --log_prefix logs/d-10k-now create_index d-10k i-1 data/Buildings/Original/Paris/jpg --K 10 --L 4

<---python cbir_main.py --log_prefix logs/d-10k-now create_index d-10k-distractor i-1 data/Buildings/Original/Paris/jpg --path_to_distractor_images data/Buildings/Original/distractor  --K 10 --L 4

python cbir_main.py --log_prefix logs/d-10k-now evaluate_only d-10k-now i-1 data/Buildings/Original/Paris/jpg data/Buildings/Original/Paris/gt

python cbir_main.py --log_prefix logs/10k_10_4_sparse evaluate --is_sample --des_type sift
python cbir_main.py --log_prefix logs/10k_10_4_sparse evaluate --des_type HardNetAll

python cbir_main.py --log_prefix logs/10k_10_4_sparse_cycle evaluate --des_type HardNetAll 


python cbir_main.py d-1 i-1-1 data/Buildings/Original/Paris_sample/jpg data/Buildings/Original/Paris_sample/gt