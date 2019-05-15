# CBIRCore on Python

## CLI

### Testing
#### Evaluating CBIRCore pipeline
`python cbir_main.py evaluate_with_all_descriptors [--train_dir --test_dir --gt_dir] [--is_sample]`

`python cbir_main.py evaluate [--train_dir <train_dir> --test_dir <test_dir> --gt_dir <gt_dir>] [--is_sample] --des_type <des_type> [--sv] [--qe]`

`python cbir_main.py evaluate_only <database_name> <index_name> <database_photos_dir> <gt_dir> [--sv] [--qe]`


#### examples

TODO: think to rewrite them so that building index is not performed if it is not needed
`python cbir_main.py evaluate_with_all_descriptors --is_sample`
`python cbir_main.py evaluate --is_sample --des_type l2net --sv --qe`
`python cbir_main.py evaluate_only oxford index_oxford_1 public/media/content/oxford/database_all data/Buildings/Original/Oxford/gt --sv --qe`

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


python cbir_main.py evaluate_only d-sample i-1 data/Buildings/Original/Oxford_sample/jpg data/Buildings/Original/Oxford/gt --sv

python cbir_main.py evaluate_only d-10k i-1 data/Buildings/Original/Oxford/jpg data/Buildings/Original/Oxford/gt
