# CBIRCore on Python

## CLI

### Testing
#### Evaluating CBIRCore pipeline
`python cbir_main.py evaluate_with_all_descriptors [--train_dir --test_dir --gt_dir] [--is_sample]`

`python cbir_main.py evaluate [--train_dir <train_dir> --test_dir <test_dir> --gt_dir <gt_dir>] [--is_sample] --des_type <des_type> [--sv] [--qe]`

`python cbir_main.py evaluate_only <database_name> <index_name> <database_photos_dir> <gt_dir> [--sv] [--qe]`


## Contributing

### Environment setup
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

To get oxford flickr distractor:
```
cd data/Buildings/Original/Oxford/jpg/distractor
./download_flickr_100k.sh
unrar x -y oxc1_100k.part01.rar
```
