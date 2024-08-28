#!/bin/bash

fairseq-generate data-bin --path /mnt/lustrefs/home/wangyh/models/wmt19/wmt19.en-de/model1.pt --remove-bpe --skip-invalid-size-inputs-valid-test > result.txt