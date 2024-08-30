## Environment
Recommend python==3.6.13 and torch==1.10.1

## Data
Please unzip the files in the data directory to get the data.

## Train
'''shell
cd pas_code/{dataset}/{pre-trained model}/{circumstance}/{paradigm}
'''

Run the corresponding python file.

Take IMDB dataset, BERT-base, DA0, PFSF for example.
'''shell
cd pas_code/imdb/base/DA0/PFSF

python pre+fine+self+fine.py --data_dir data/imdb/DA0.txt --save_dir save_path
'''

## Eval
'''shell
cd pas_code/{dataset}

python eval.py --data_dir your_data --checkpoint_dir checkpoint_path
'''
