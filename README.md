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
