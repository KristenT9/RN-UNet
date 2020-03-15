from glob import glob
import pandas as pd
from tensorflow.python.keras import optimizers,initializers
import random
from tensorflow import set_random_seed
set_random_seed(10)
import numpy as np
np.random.seed(10)
h=256
w=256
c=32
patch_h=128
patch_w=128
patch_c=32
modality=7
patches3d_per_pat=200
patches2d_per_pat=16
num_pat=43
test_num=32
n_splits=7
batch_size = 4
epochs =100
lr=5e-5
lr_decay=1e-6
initializer="he_normal"
adam=optimizers.Adam(lr=lr,decay=lr_decay)
sgd=optimizers.SGD(lr=lr, decay=lr_decay, momentum=0.9)
name_list=["ADC","MTT","rCBF","rCBV","Tmax","TTP","OT"]
model_name="Res Nonlocal dsv Unet"
model_name_list=["Unet","Nonlocal dsv Unet","Res Nonlocal dsv Unet"]
mode_list=["Training","Evaluating","Testing","Predicting"]
TICI_scores=["0","1","2","2a","2b","3"]
augs = ["flip", "rotate", "scale", "translate"]
metrics=["Loss","Dice","Precision","Recall","Hausdoff Distance","Average Symmetric Surface Distance"]
sample_all=True
sample_prop=False
over_sample=False
two_phase_sample=False
half_lesion=False
set_prob=0.5#less than the prob will be augmented
hdf5_path="dataio/saved_data/"
model_save_path="models/saved_models"
filepath=glob("/home/huangli/ISLES2017/"+"*training*")[:num_pat]
testpath=glob("/home/huangli/ISLES2017_Testing/"+"*test_*")[0:test_num]
csv_path="/home/huangli/ISLES2017/ISLES2017_Training.xlsx"
test_img_path=hdf5_path+"test_img/"
pred_saved_path='/home/huangli/ISLES2017_'+model_name+'_my_result/'
clinical_info = pd.read_excel(csv_path)

