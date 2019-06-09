# Photologue app powered by CBIR library

<img src="https://raw.githubusercontent.com/gavr97/cbir-django/master/screens/demo.gif" alt="Demonstration" width="700">


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


## Start snakeviz for visualizing profs
`snakeviz [--server]`
