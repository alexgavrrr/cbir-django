# CBIRCore on Python

## CLI

### Testing
#### Evaluating CBIRCore pipeline
`python cbir_main.py evaluate_with_all_descriptors [--train_dir --test_dir --gt_dir] [--is_sample]`

`python cbir_main.py evaluate [--train_dir <train_dir> --test_dir <test_dir> --gt_dir <gt_dir>] [--is_sample] --des_type <des_type> [--sv] [--qe]`

#### examples

TODO: think to rewrite them so that building index is not performed if it is not needed
`python cbir_main.py evaluate_with_all_descriptors --is_sample`
`python cbir_main.py evaluate --is_sample --des_type l2net --sv --qe`

## Contributing

### Computer environment setup
Refer to `remote_setup_gpu.txt`

### Project environment setup
Use pipenv:
`pipenv shell`
`pipenv install`

### Getting data
To get oxford paris datasets execute script
`data/Buildings/Original/get_oxford_paris.sh`
It will create `Paris` and `Oxford` dirs in current working dir.
Based on https://github.com/figitaki/deep-retrieval
