import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tifffile as tf
from glob import glob
import os
import tifffile as tf

from scipy import ndimage, interpolate, spatial

struc1 = ndimage.generate_binary_structure(2,1)
struc2 = ndimage.generate_binary_structure(2,2)

wsrc = '../cell_dams/'
nsrc = '../nuclear_mask/'
ksrc = '../kde/'
csrc = '../data/'
dst = '../proc/'

sample = 'B1'
sdst = dst + sample + '/'
if not os.path.isdir(sdst):
    os.mkdir(sdst)

wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
edt = tf.imread(nsrc + sample + '_EDT.tif')
nuclei = edt < 2
wcoords = np.array(np.nonzero(~wall))
label, cellnum = ndimage.label(wall, struc1)
objss = ndimage.find_objects(label)
cnuclei = np.asarray(ndimage.center_of_mass(wall, label, range(1,cellnum+1)))

pad = 5
cy,cx = wall.shape
cx += int(np.where( cx % pad, cx%pad, pad))
cy += int(np.where( cy % pad, cy%pad, pad))
xaxis = np.arange(0, cx, pad)
yaxis = np.arange(0, cy, pad)


transcriptomes = sorted([os.path.splitext(foo)[0].split('_' + sample + '_')[-1] for foo in glob(ksrc + sample + '/meta*.csv')])
filename = csrc + sample + '_data/32771-slide1_' + sample + '_results.txt'
data = pd.read_csv(filename, header=None, sep='\t').drop(columns=[4])
data.columns = ['X', 'Y', 'Z', 'T']

filename = csrc + sample + '_data/' + transcriptomes[0] + '/' + transcriptomes[0] + ' - localization results by cell.csv'
df = pd.read_csv(filename)
df = df.iloc[~np.any(df.iloc[:, :5].isnull().values, axis=1)]
df = df[df['Cell.Area..px.'] > 9]
sel = [0,3,4,5,6,7,8,9]
df = df.astype(dict(zip(df.columns[np.array(sel)], [int for i in range(len(sel))])))
dcoords = np.round(df.iloc[:, 1:3].values).astype(int)


cdist = spatial.distance.cdist(np.flip(cnuclei, axis=1), dcoords, metric='euclidean')
cmatches = np.argmin(cdist, axis=1)

tidx = 0
for tidx in range(len(transcriptomes)):

    tcoords = data.loc[ data['T'] == transcriptomes[tidx] , ['X', 'Y', 'Z'] ].values.T
    nmask = ~nuclei[ tcoords[1], tcoords[0] ]

    coords = tcoords[:2, nmask]
    tdst = sdst + transcriptomes[tidx] + '/'
    if not os.path.isdir(tdst):
        os.mkdir(tdst)
    print(tdst)

    # # Load KDE

    filename = ksrc + sample + '/kde_' + sample + '_' + transcriptomes[tidx] + '.npy'
    isj = np.load(filename, allow_pickle=True)
    filename = ksrc + sample + '/meta_' + sample + '_' + transcriptomes[tidx] + '.csv'
    meta = pd.read_csv(filename, header=None, index_col=0)

    # # Individual cells

    interp = interpolate.RegularGridInterpolator((yaxis, xaxis), isj, method='linear', bounds_error=True, fill_value=None)
    bins = np.linspace(0, isj.max(), 256)

    cellcount = np.zeros(cellnum, dtype=int)
    cellid = cellcount.copy()
    
    for cidx in range(cellnum):
        cdata = df.iloc[cmatches[cidx]]
        ccoords = coords[:2, label[ coords[1], coords[0] ] == cidx + 1 ].copy()
        cellcount[cidx] = len(ccoords[0])
        cellid[cidx] = int(cdata['Cell.ID..'])
        filename = tdst + transcriptomes[tidx] + '_-_{:05d}_-_{:05d}.tif'.format(cidx, int(cdata['Cell.ID..']))
        
        if (len(ccoords[0]) > 2) and not os.path.isfile(filename):
            
            print('Processing', cidx)
            ss = objss[cidx]
            css = (np.s_[ss[0].start // pad : ss[0].stop // pad + 2], np.s_[ss[1].start // pad : ss[1].stop // pad + 2 ])
            cxs = xaxis[css[1]]
            cys = yaxis[css[0]]
            cisj = isj[css]

            ss = (np.s_[ cys[0]:cys[-1] ], np.s_[ cxs[0]:cxs[-1] ])
            extent = (ss[1].start, ss[1].stop, ss[0].start, ss[0].stop)

            cell = wall[ss].copy().astype(np.uint8)
            cell[ label[ss] == cidx+1 ] += 1

            wcellcoords = np.asarray(np.nonzero(~wall[ss]))
            wcellcoords[0] += ss[0].start
            wcellcoords[1] += ss[1].start
            
            cellcoords = np.asarray(np.nonzero(label == cidx + 1))
            cvals = interp(cellcoords.T)

            cpdf = np.zeros(cell.shape)
            cpdf[cellcoords[0]-ss[0].start, cellcoords[1] - ss[1].start] = cvals
            dig = np.digitize(cpdf, bins, right=True).astype(np.uint8)

            
            # Diagnostic image
            
            fig, ax = plt.subplots(1,3,figsize=(10,5), sharex=True, sharey=True)
            ax = np.atleast_1d(ax).ravel()

            ax[0].imshow(cell, cmap='plasma', origin='lower', extent=extent)
            ax[0].scatter(*cdata[1:3].values, c='blue', marker='v', zorder=2, s=50)
            ax[0].scatter(cnuclei[cidx,1], cnuclei[cidx,0], c='black', marker='^', zorder=2, s=50)
            ax[0].scatter(ccoords[0], ccoords[1], c='red', marker='o', zorder=3, alpha=0.5);
            ax[0].scatter(ccoords[0], ccoords[1], c='red', marker='o', zorder=3, alpha=0.5);

            ax[1].contourf(cxs, cys, cisj, 16, cmap='plasma', vmin=0, zorder=1)
            ax[1].scatter(wcellcoords[1], wcellcoords[0], c='lime', marker='.', s=0.1)
            ax[1].scatter(ccoords[0], ccoords[1], c='cyan', marker='*', zorder=3, alpha=0.5);

            ax[2].imshow(dig, origin='lower', cmap='plasma', extent=extent, vmax=255)

            step = int(np.ediff1d(ax[0].get_xticks())[0])
            for i in range(len(ax)):
                ax[i].set_aspect('equal')
                ax[i].margins(0)

            title = '{}: {}: Cell ID {} [{}]'.format(sample, transcriptomes[tidx], cellid[cidx], cidx)
            fig.suptitle(title, fontsize=18)
            fig.tight_layout();
            filename = tdst + 'diagnostic_' + transcriptomes[tidx] + '_-_{:05d}_-_{:05d}'.format(cidx, cellid[cidx])
            plt.savefig(filename + '.jpg', format='jpg', dpi=150, bbox_inches='tight', pil_kwargs={'optimize':True})
            plt.close()

            filename = tdst + transcriptomes[tidx] + '_-_{:05d}_-_{:05d}.tif'.format(cidx, cellid[cidx])
            print(filename)
            tf.imwrite(filename, dig, photometric='minisblack')
            
    filename = tdst + 'metacount_' + transcriptomes[tidx] + '.csv'
    foo = pd.DataFrame()
    foo['Label.ID.'] = np.arange(cellnum) + 1
    foo['Cell.ID.'] = cellid
    foo['Transcript.Count'] = cellcount
    foo.to_csv(filename, index=False)
    print(filename)
             
