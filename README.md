## Environment
Recommend python==3.6.13 and torch==1.10.1

## Data
Please unzip the files in the data directory to get the data.

## Train
cd pas_code/{dataset}/{pre-trained model}/{circumstance}/{paradigm}
Run the corresponding python file

Take IMDB dataset, BERT-base, DA0, PFSF for example.
cd pas_code/imdb/base/DA0/PFSF
python pre+fine+self+fine.py --data_dir data/imdb/DA0.txt --save_dir {SAVE_DIR}
