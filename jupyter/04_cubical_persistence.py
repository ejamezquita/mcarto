import matplotlib.pyplot as plt
import numpy as np

import gudhi as gd
import json

from glob import glob
import os
import tifffile as tf

from scipy import ndimage

maximgsize = 1200

src = '../proc/'
dst = '../gudhi/'

sample = 'B1'

tsrc = src + sample + '/'
sdst = dst + sample + '/'
if not os.path.isdir(sdst):
    os.mkdir(sdst)

transcriptomes = sorted([foo.split('/')[-2] for foo in glob(tsrc + '*/')])
print(len(transcriptomes), 'transcriptomes')

tidx = 15
for tidx in range(len(transcriptomes)):
    tiffs = sorted(glob(tsrc + transcriptomes[tidx] + '/*.tif'))
    tdst = sdst + transcriptomes[tidx] + '/'
    if not os.path.isdir(tdst):
        os.mkdir(tdst)
        
    for cidx in range(len(tiffs)):
        
        filename = tdst + 'sublevel_' + os.path.splitext(os.path.split(tiffs[cidx])[1])[0] + '.json'
        
        if not os.path.isfile(filename):        
            img = tf.imread(tiffs[cidx])
            if max(img.shape) > maximgsize:
                zoom = maximgsize/max(img.shape)
                print(tiffs[cidx], '\nResized', img.shape, 'by a factor of ', zoom)
                img = ndimage.zoom(img, zoom = zoom, order=1, mode='reflect')
                print('Now', img.shape, '\n----')

            cc = gd.CubicalComplex(top_dimensional_cells = img)
            pers = cc.persistence(homology_coeff_field=2, min_persistence=1)
            filename = tdst + 'sublevel_' + os.path.splitext(os.path.split(tiffs[cidx])[1])[0] + '.json'
            with open(filename, 'w') as f:
                json.dump(pers,f)

            cc = gd.CubicalComplex(top_dimensional_cells = 255 - img)
            pers = cc.persistence(homology_coeff_field=2, min_persistence=1)
            filename = tdst + 'superlevel_' + os.path.splitext(os.path.split(tiffs[cidx])[1])[0] + '.json'
            with open(filename, 'w') as f:
                json.dump(pers,f)
