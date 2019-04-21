# Django image storage app powered by CBIRCore library

## Setup instructions
pipenv install
python manage.py makemigrations
python manage.py migrate

python cbir_main.py prepare_cbir_directory_structure

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
