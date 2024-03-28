import numpy as np
import pandas as pd
import tifffile as tf
from glob import glob
import os
from scipy import ndimage, stats

from KDEpy import FFTKDE
#from KDEpy.bw_selection import improved_sheather_jones as ISJ
import argparse

pows2 = 2**np.arange(20) + 1
bw = 10
PP = 6
stepsize = 5
pp = 0

wsrc = '../cell_dams/'
nsrc = '../nuclear_mask/'
tsrc = '../translocs/'
psrc = '../proc/'
osrc = '../data/'

def generate_transcript_metadata(translocs, transcriptomes, invidx, orig_size, tsize):
    isj = np.zeros((len(transcriptomes), 2))
    for tidx in range(len(isj)):
        coords = translocs.loc[invidx == tidx , ['X', 'Y'] ].values.T
        for i in range(len(coords)):
            isj[tidx, i] = ISJ(coords[ i ].reshape(-1,1))
    meta = pd.DataFrame(isj, columns=['ISJ1', 'ISJ2'])
    meta['total_number'] = orig_size
    meta['cyto_number'] = tsize
    meta['nuclei_number'] = orig_size - tsize
    meta['ratio'] = tsize/orig_size
    meta['gene'] = transcriptomes
    
    return meta

def kde_grid_generator(stepsize, maxdims, pows2 = 2**np.arange(20) + 1, pad=1.5):
    axes = [ np.arange(0, maxdims[i], stepsize) for i in range(len(maxdims)) ]
    AXES = [ None for i in range(len(axes)) ]
    
    for i in range(len(axes)):
        m = np.nonzero(pows2 > pad*len(axes[i]))[0][0]
        foo = pows2[m] - len(axes[i])
        neg = foo//2
        pos = np.where(foo%2==0, foo//2, foo//2 + 1) + 0
        AXES[i] = np.hstack((np.arange(-neg, 0, 1)*stepsize, axes[i], np.arange(len(axes[i]), len(axes[i])+pos, 1)*stepsize))
    
    AXES = np.meshgrid(*AXES, indexing='ij')
    grid = np.column_stack([ np.ravel(AXES[i]) for i in range(len(AXES)) ])
    
    mask = np.ones(len(grid), dtype=bool)
    for i in range(len(axes)):
        mask = mask & (grid[:,i] >= 0) & (grid[:,i] < maxdims[i])

    return axes, grid, mask

def cardinal_distance_transform(img):
    PAD = 1
    pss = np.s_[PAD:-PAD,PAD:-PAD]
    pad = np.pad(img, PAD, constant_values=0)
    initd = np.full(pad.shape, max(pad.shape)+1, dtype=int)
    initd[~pad] = 0
    left = np.copy(initd)
    for j in range(1,pad.shape[1]):
        left[:, j] = np.minimum(left[:, j], left[:, j-1] + 1)
    right = np.copy(initd)
    for j in range(pad.shape[1]-2, -1, -1):
        right[:, j] = np.minimum(right[:, j], right[:, j+1] + 1)
    bottom = np.copy(initd)
    for j in range(1,pad.shape[0]):
        bottom[j] = np.minimum(bottom[j], bottom[j-1] + 1)
    top = np.copy(initd)
    for j in range(pad.shape[0]-2, -1, -1):
        top[j] = np.minimum(top[j], top[j+1] + 1)

    return top[pss], right[pss], bottom[pss], left[pss]
    
def cell_img_preparation(cidx, wall, label, metacell, PP = 6):
    ss = (np.s_[max([0, metacell.loc[cidx, 'y0'] - PP]) : min([wall.shape[0], metacell.loc[cidx, 'y1'] + PP])],
          np.s_[max([1, metacell.loc[cidx, 'x0'] - PP]) : min([wall.shape[1], metacell.loc[cidx, 'x1'] + PP])])
    cell = wall[ss].copy().astype(np.uint8)
    cell[ label[ss] == cidx+1 ] = 2
    cell[~wall[ss]] = 0
    
    return cell, ss
    
def cell_grid_preparation(cell, ss, zmax, stepsize, pows2):
    
    maxdims = ( cell.shape[1], cell.shape[0], zmax )
    axes, grid, gmask = kde_grid_generator(stepsize=stepsize, maxdims=maxdims, pows2 = pows2, pad=1.5)
    grid[:, :2] = grid[:, :2] + np.array([ss[1].start, ss[0].start])
    
    cgrid = grid[gmask].copy()
    cgrid[:,:2] = grid[gmask][:,:2] - np.array([ss[1].start, ss[0].start])
    cgridmask = cell[cgrid[:,1],cgrid[:,0]] != 2
    
    return axes, grid, gmask, cgrid, cgridmask
        
    
def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('sample', metavar='raw_dir', type=str, help='directory where raw images are located')
    parser.add_argument('cstart', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('cfinish', metavar='raw_dir', type=int, help='directory where raw images are located')
    args = parser.parse_args()

    sample = args.sample
    cstart = args.cstart
    cfinish = args.cfinish
    
    dst = '../kde/'
    dst = dst + sample + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)

    # # Load all general data

    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    label, cellnum = ndimage.label(wall, ndimage.generate_binary_structure(2,1))

    wall[tf.imread(nsrc + sample + '_EDT.tif') < 2] = False
    print('Detected',cellnum,'cells')

    metacell = pd.read_csv(dst + sample + '_cells_metadata.csv')
    metatrans = pd.read_csv(dst + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(dst + sample + '_transcells_metadata.csv')
    tcumsum = np.hstack(([0], np.cumsum(metatrans['cyto_number'].values)))

    transcriptomes = list(metatrans['gene'])
    translocs = [None for i in range(len(transcriptomes))]
    for i in range(len(transcriptomes)):
        filename = tsrc + sample + os.sep + 'location_corrected_D2_-_' + transcriptomes[i] + '.csv'
        translocs[i] = pd.read_csv(filename, header=None, names=['X', 'Y', 'Z'])
    tlocs = pd.concat(translocs)
    zmax = np.max(tlocs['Z']+stepsize)

    # # Compute weights
    
    filename = dst + sample + '_border_weights.npy'
    if not os.path.isfile(filename):
        top, right, bottom, left = cardinal_distance_transform(wall)

        wv = stats.norm.cdf(top[tlocs['Y'].values, tlocs['X'].values]+pp, loc=0, scale=bw)
        wv-= stats.norm.cdf(-bottom[tlocs['Y'].values, tlocs['X'].values]-pp, loc=0, scale=bw)
        wh = stats.norm.cdf(right[tlocs['Y'].values, tlocs['X'].values]+pp, loc=0, scale=bw) 
        wh-= stats.norm.cdf(-left[tlocs['Y'].values, tlocs['X'].values]-pp, loc=0, scale=bw)
        
        weight = 2-(wv*wh)
        np.save(dst + sample + '_border_weights.npy', weight)

    weight = np.load(filename, allow_pickle=True)

    # # Select a transcript and a cell

    for cidx in range(cstart,cfinish):
        
        cell, ss = cell_img_preparation(cidx, wall, label, metacell, PP = 6)
        axes, grid, gmask, cgrid, cgridmask = cell_grid_preparation(cell, ss, zmax, stepsize, pows2)
        
        # # Prepare the KDE grid

        for tidx in np.nonzero(transcell.iloc[:, cidx].values > 5)[0]:
            
            kdst = dst + transcriptomes[tidx] + os.sep
            if not os.path.isdir(kdst):
                os.mkdir(kdst)
            if not os.path.isfile(filename):
                tmask = invidx == tidx
            
                coords = translocs.loc[ tmask , ['X', 'Y'] ].values.T
                cmask = label[ coords[1], coords[0] ] == cidx + 1
                ccoords = coords[:, cmask ].copy()
                
                w = weight[tmask][cmask]
                kde = FFTKDE(kernel='gaussian', bw=bw, norm=2).fit(ccoords.T, w).evaluate(grid)
                kde = kde[gridmask]/(np.sum(kde[gridmask])*stepsize*stepsize)
                kde[ cgridmask ] = 0
                kde = kde/(np.sum(kde)*stepsize*stepsize)
                kde = kde.reshape( ( len(yaxis), len(xaxis) ), order='F')
                maxkde = np.max(kde)
                
                #ratio = ccoords.shape[1]/coords.shape[1]
                
                # # Save results
                
                
                meta = [cidx, PP, stepsize, bw, maxkde]
                filename = kdst + 'c{}_p{}_s{}_b{}_m{:.25E}.npy'.format(*meta)
                np.save(filename, kde)
                print('Generated', filename)
                    
if __name__ == '__main__':
    main()
