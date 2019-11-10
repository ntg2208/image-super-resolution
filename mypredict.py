import numpy as np
from PIL import Image

img = Image.open('data/DIV2K_train_LR_x8/0009x8.png')
lr_img = np.array(img)

from ISR.models import RRDN

rrdn = RRDN(arch_params={'C':4, 'D':3, 'G':64, 'G0':64, 'T':10, 'x':2})
rrdn.model.load_weights('rrdn_90_500_16_valPSNR_Y.hdf5')

sr_img = rrdn.predict(lr_img)
tmp = Image.fromarray(sr_img)

tmp.save("test2.jpg")
