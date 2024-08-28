#!/bin/bash

fairseq-preprocess --source-lang de --target-lang en --trainpref /mnt/lustrefs/home/wangyh/home1/imdb/imdbback --testpref /mnt/lustrefs/home/wangyh/home1/imdb/imdbback --srcdict /mnt/lustrefs/home/wangyh/models/wmt19/wmt19.de-en/dict.de.txt --tgtdict /mnt/lustrefs/home/wangyh/models/wmt19/wmt19.de-en/dict.en.txt