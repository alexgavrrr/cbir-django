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

To get oxford flickr distractor:
```
cd data/Buildings/Original/Oxford/jpg/distractor
./download_flickr_100k.sh
unrar x -y oxc1_100k.part01.rar  # not sure
```

## More commands
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
