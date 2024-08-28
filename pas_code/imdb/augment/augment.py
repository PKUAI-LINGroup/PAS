import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import os
import nlpaug.flow as nafc
from nlpaug.util import Action

os.environ["MODEL_DIR"] = '/home1/wangyh/ppdb/'
aug = naw.BackTranslationAug(from_model_name='/mnt/lustrefs/home/wangyh/models/wmt19/wmt19.en-de', from_model_checkpt='model1.pt', to_model_name='/mnt/lustrefs/home/wangyh/models/wmt19/wmt19.de-en', to_model_checkpt='model1.pt', is_load_from_github=False)
with open('/mnt/lustrefs/home/wangyh/home1/imdb/imdb1000.en', 'r', encoding='utf-8') as f1:
    L = f1.readlines()
    L = [line.split('\t') for line in L[:2]]
    L1 = [line[1] for line in L]
    print(L1)
    L2 = aug.augment(L1)
    L3 = ['\t'.join([L[i][0], L2[i]])+'\n' for i in range(len(L))]
with open('/mnt/lustrefs/home/wangyh/home1/imdb/imdb_back.en', 'w', encoding='utf-8') as f2:
    f2.writelines(L3)