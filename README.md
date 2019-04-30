# Django image storage app powered by CBIR library

## Setup instructions
```
pipenv shell
pipenv install
python manage.py makemigrations
python manage.py migrate
```


## Development

Reload from scratch command
```
yes | rm db.sqlite3;
    rm -rf photologue/migrations/*; touch photologue/migrations/__init__.py && \
    python manage.py makemigrations && \
    python manage.py migrate && \
    ./createsuper.sh && \
    ./clean_content.sh && \
    ./clean_cbir_state.sh
```

```
yes | rm db.sqlite3;
    python manage.py makemigrations && \
    python manage.py migrate && \
    ./createsuper.sh && \
    ./clean_content.sh && \
    ./clean_cbir_state.sh
```

scp -r data/ gavr@104.45.144.192:~/main/cbir-django
