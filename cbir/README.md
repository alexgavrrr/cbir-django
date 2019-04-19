# CBIR on Python

## CLI

### Register
`python main.py build_cbir_index <name> <path>`
`python main.py build_cbir_index buildings /Users/alexgavr/main/Developer/Data/Buildings/Original/Oxford_sample/jpg`

### Search
`python main.py search buildings /Users/alexgavr/main/Developer/Data/Buildings/Original/Paris_sample/jpg/paris_defense_000004.jpg`
`python main.py search buildings /Users/alexgavr/main/Developer/Data/Buildings/Original/Paris_sample/jpg/paris_defense_000004.jpg --save --tag first`

### Testing
#### Evaluating CBIR pipeline
in root directory perform:
`env PYTHONPATH="." python tests/evaluation.py`
`env PYTHONPATH="." python tests/test.py`
`.` must be root directory

## Contributing

### Computer environment setup
Refer to `remote_setup_gpu.txt`

### Project environment setup
Use pipenv:
`pipenv shell`
`pipenv install`

### Getting data
To get oxford paris datasets:
`./scripts/get_oxford_paris.sh`
It will create `Paris` and `Oxford` dirs in current working dir.
Based on https://github.com/figitaki/deep-retrieval
