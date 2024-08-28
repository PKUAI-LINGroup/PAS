#!/bin/bash

fairseq-preprocess --source-lang en --target-lang de --trainpref /mnt/lustrefs/home/wangyh/home1/sst/sst1000 --testpref /mnt/lustrefs/home/wangyh/home1/sst/sst1000 --srcdict /mnt/lustrefs/home/wangyh/models/wmt19/wmt19.en-de/dict.en.txt --tgtdict /mnt/lustrefs/home/wangyh/models/wmt19/wmt19.en-de/dict.de.txt