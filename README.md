# Django image storage app powered by CBIR library

## Setup instructions
```
pipenv shell
pipenv install
python manage.py makemigrations
python manage.py migrate
```

`scp -r data gavr@104.45.144.192:~/main/cbir-django`


## Development

Reload from scratch command
```
yes | rm db.sqlite3;
    rm -rf photologue/migrations/*; touch photologue/migrations/__init__.py && \
    python manage.py makemigrations && cp backup/0002_photosize_data.py photologue/migrations && \
    python manage.py migrate && \
    ./createsuper.sh && \
    ./clean_content.sh && \
    ./clean_cbir_state.sh
```

```
yes | rm db.sqlite3;
    touch photologue/migrations/__init__.py && \
    python manage.py makemigrations && \
    python manage.py migrate && \
    ./createsuper.sh && \
    ./clean_content.sh && \
    ./clean_cbir_state.sh
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

## commands
sample 10 4
`python cbir_main.py --log_prefix logs/sample_10_4 create_index --K 10 --L 4 d-sample i-1 data/Buildings/Original/Oxford_sample/jpg 14`

`python cbir_main.py --log_prefix logs/sample_10_4 search d-sample i-1 data/Buildings/Original/Oxford_sample/jpg/all_souls_000013.jpg --topk 10` 

`...`

sample 10 5
`python cbir_main.py --log_prefix logs/sample_10_5 change_params --K 10 --L 5 d-sample i-1`

`python cbir_main.py --log_prefix logs/sample_10_5 search d-sample i-1 data/Buildings/Original/Oxford_sample/jpg/all_souls_000013.jpg --topk 10` 

`...`


100k 10 4
`python cbir_main.py --log_prefix logs/100k_10_4 create_index --K 10 --L 4 d-100k i-1 data/Buildings/Original/Oxford/jpg 8`

`python cbir_main.py --log_prefix logs/100k_10_4 search d-100k i-1 data/Buildings/Original/Oxford/jpg/all_souls_000013.jpg --topk 10` 

`...`

100k 10 5
`python cbir_main.py --log_prefix logs/100k_10_5 change_params --K 10 --L 5 d-100k i-1`

`python cbir_main.py --log_prefix logs/100k_10_5 search d-100k i-1 data/Buildings/Original/Oxford/jpg/all_souls_000013.jpg --topk 10` 

`...`

scp -r data/Buildings/Original/Oxford_sample/jpg/distractor gavr@40.114.26.173:~/main/cbir-django/data/Buildings/Original/Oxford_sample/jpg/distractor
