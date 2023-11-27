import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tifffile as tf
from glob import glob
import os
import tifffile as tf

from scipy import ndimage
from KDEpy import FFTKDE
from KDEpy.bw_selection import improved_sheather_jones as ISJ


struc1 = ndimage.generate_binary_structure(2,1)
struc2 = ndimage.generate_binary_structure(2,2)
fs = 15

wsrc = '../cell_dams/'
nsrc = '../nuclear_mask/'
csrc = '../data/'
dst = '../kde/'
sample = 'B1'

kdst = dst + sample + '/'
if not os.path.isdir(kdst):
    os.mkdir(kdst)

pows2 = 2**np.arange(20) + 1
wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)


#### prepare the KDE grid #####
pad = 5
cy,cx = wall.shape
cx += int(np.where( cx % pad, cx%pad, pad))
cy += int(np.where( cy % pad, cy%pad, pad))

xaxis = np.arange(0, cx, pad)
yaxis = np.arange(0, cy, pad)
mx = np.nonzero(pows2 > 1.5*len(xaxis))[0][0]
my = np.nonzero(pows2 > 1.5*len(yaxis))[0][0]

foo = pows2[mx] - len(xaxis)
xneg = foo//2
if foo % 2 == 0:
    xpos = foo//2
else:
    xpos = foo//2 + 1

foo = pows2[my] - len(yaxis)
yneg = foo//2
if foo % 2 == 0:
    ypos = foo//2
else:
    ypos = foo//2 + 1

xaxes = np.hstack((np.arange(-xneg, 0, 1)*pad, xaxis, np.arange(len(xaxis), len(xaxis)+xpos, 1)*pad))
yaxes = np.hstack((np.arange(-yneg, 0, 1)*pad, yaxis, np.arange(len(yaxis), len(yaxis)+ypos, 1)*pad))
X,Y = np.meshgrid(xaxis, yaxis)
XX, YY = np.meshgrid(xaxes, yaxes)

grid = np.column_stack((np.ravel(XX, 'F'), np.ravel(YY, 'F')))
mask = (grid[:,0] > -1) & (grid[:,0] < cx) & (grid[:,1] > -1) & (grid[:,1] < cy)

##### load transcriptomic data #######

edt = tf.imread(nsrc + sample + '_EDT.tif')
nuclei = edt < 2
wcoords = np.array(np.nonzero(~wall))


transcriptomes = sorted([foo.split('/')[-2] for foo in glob(csrc + sample + '_data/*/')])
print(len(transcriptomes), 'transcriptomes')

filename = csrc + sample + '_data/32771-slide1_' + sample + '_results.txt'
data = pd.read_csv(filename, header=None, sep='\t').drop(columns=[4])
data.columns = ['X', 'Y', 'Z', 'T']

tsize = np.arange(len(transcriptomes))
for tidx in range(len(tsize)):
    tcoords = data.loc[ data['T'] == transcriptomes[tidx] , ['X', 'Y', 'Z'] ].values.T
    tsize[tidx] = tcoords.shape[1]

targsort = np.argsort(tsize)[::-1]
for tidx in range(10):
    print(tidx+1, targsort[tidx], transcriptomes[targsort[tidx]], tsize[targsort[tidx]] , sep='\t')

# # Do KDE

tidx = 84
for tidx in range(len(transcriptomes)):

    tcoords = data.loc[ data['T'] == transcriptomes[tidx] , ['X', 'Y', 'Z'] ].values.T
    nmask = ~nuclei[ tcoords[1], tcoords[0] ]
    coords = tcoords[:2, nmask]
    
    if len(coords[0]) > 250:
        
        filename = kdst + 'kde_' + sample + '_' + transcriptomes[tidx] + '.npy'
        print(filename)
        
        if not os.path.isfile(filename):
            
            bw = np.zeros(len(coords))
            for i in range(len(bw)):
                bw[i] = ISJ(coords[ i ].reshape(-1,1))
            
            coords_scaled = coords/bw.reshape(-1,1)
            kde_scaled = FFTKDE(kernel='gaussian', bw=1, norm=2).fit(coords_scaled.T).evaluate(grid/bw)
            isj0 = kde_scaled / np.prod(bw)
            isj = isj0[mask]/(np.sum(isj0[mask])*pad**2)
            isj = isj.reshape((len(yaxis), len(xaxis)), order='F')
            
            ### Save NPY with KDE matrix
            
            filename = kdst + 'meta_' + sample + '_' + transcriptomes[tidx] + '.csv'
            meta = [np.sum(nmask), *bw, isj.min(), isj.max()]
            pd.DataFrame(meta, columns=[transcriptomes[tidx]]).T.to_csv(filename, index=True, header=False)
            filename = kdst + 'kde_' + sample + '_' + transcriptomes[tidx] + '.npy'
            np.save(filename, isj)
        
#        else:
#            isj = np.load(filename, allow_pickle=True)
#            filename = kdst + 'meta_' + sample + '_' + transcriptomes[tidx] + '.csv'
#            meta = pd.read_csv(filename, header=None, index_col=0)
#            bw = meta.iloc[0, 1:3].values
            
            ### Save diagnostic figure

            fig, ax = plt.subplots(1,2,figsize=(10,8), sharex=True, sharey=True)
            ax = np.atleast_1d(ax).ravel()

            ax[0].set_facecolor('gainsboro')
            ax[0].scatter(wcoords[1], wcoords[0], c='lime', s=.01, marker='.', alpha=0.1, zorder=1);
            ax[0].scatter(*coords, c='magenta', marker='o', s=1, zorder=2, alpha=0.25)

            ax[1].contourf(xaxis, yaxis, isj, 16, cmap='plasma', vmin=0, zorder=1)

            for i in range(len(ax)):
                ax[i].set_aspect('equal')
                ax[i].margins(0)
                ax[i].tick_params(labelsize=fs-5)

            ax[0].set_title('Grid size: {}'.format(pad), fontsize=fs)
            ax[1].set_title('ISJ: {}'.format(np.round(bw,1)), fontsize=fs)

            fig.suptitle('{}: {} (N={})'.format(sample, transcriptomes[tidx], np.sum(nmask)), fontsize=fs);
            fig.tight_layout();

            filename = kdst + '{}_-_{}_kde'.format(sample, transcriptomes[tidx])
            print(filename)
            plt.savefig(filename + '.jpg', format='jpg', dpi=150, bbox_inches='tight', pil_kwargs={'optimize':True})
            plt.close()
