import numpy as np
import pandas as pd
import tifffile as tf
from glob import glob
import os
from scipy import ndimage, stats
import gudhi as gd
import json

from KDEpy import FFTKDE
#from KDEpy.bw_selection import improved_sheather_jones as ISJ
import argparse

pows2 = 2**np.arange(20) + 1
bw = 10
PP = 6
stepsize = 3
pp = 0
ndims = 3

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
        
def cell_weighted_kde(coords, grid, weights, bw, gmask, stepsize, cgridmask, axes):

    kde = FFTKDE(kernel='gaussian', bw=bw, norm=2).fit(coords, weights).evaluate(grid)
    kde = kde[gmask]/( np.sum(kde[gmask]) * (stepsize**len(coords)) )
    kde[ cgridmask ] = 0
    kde = kde/( np.sum(kde) * (stepsize**len(coords)) )
    kde = kde.reshape( list(map(len, axes))[::-1], order='F')
    
    return kde
    
def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('sample', metavar='raw_dir', type=str, help='directory where raw images are located')
    parser.add_argument('cstart', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('cfinish', metavar='raw_dir', type=int, help='directory where raw images are located')
    args = parser.parse_args()

    sample = args.sample
    cstart = args.cstart
    cfinish = args.cfinish
    
    ksrc = '../kde/'
    dst = '../suplevel/'
    
    ksrc += sample + os.sep
    dst += sample + os.sep
    
    if not os.path.isdir(dst):
        os.mkdir(dst)
    
    # # Load all general data
    
    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv')
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv')
    tcumsum = np.hstack(([0], np.cumsum(metatrans['cyto_number'].values)))
    
    cfinish = np.min([cfinish, len(metacell)])

    transcriptomes = list(metatrans['gene'])
    translocs = [None for i in range(len(transcriptomes))]
    for i in range(len(transcriptomes)):
        filename = tsrc + sample + os.sep + 'location_corrected_D2_-_' + transcriptomes[i] + '.csv'
        translocs[i] = pd.read_csv(filename, header=None, names=['X', 'Y', 'Z'])
    tlocs = pd.concat(translocs)
    zmax = np.max(tlocs['Z']+stepsize)
    
    for i in range(len(transcriptomes)):
        tdst = dst + transcriptomes[i]
        if not os.path.isdir(tdst):
            os.mkdir(tdst)

    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    label, cellnum = ndimage.label(wall, ndimage.generate_binary_structure(2,1))

    wall[tf.imread(nsrc + sample + '_EDT.tif') < 2] = False
    print('Detected',cellnum,'cells')
    
    # # Compute weights
    
    filename = ksrc + sample + '_border_weights.npy'
    if not os.path.isfile(filename):
        top, right, bottom, left = cardinal_distance_transform(wall)

        wv = stats.norm.cdf(top[tlocs['Y'].values, tlocs['X'].values]+pp, loc=0, scale=bw)
        wv-= stats.norm.cdf(-bottom[tlocs['Y'].values, tlocs['X'].values]-pp, loc=0, scale=bw)
        wh = stats.norm.cdf(right[tlocs['Y'].values, tlocs['X'].values]+pp, loc=0, scale=bw) 
        wh-= stats.norm.cdf(-left[tlocs['Y'].values, tlocs['X'].values]-pp, loc=0, scale=bw)
        
        weight = 2-(wv*wh)
        np.save(filename, weight, allow_pickle=True)

    weight = np.load(filename, allow_pickle=True)

    # # Select a transcript and a cell
    
    tidxs = np.arange(10)

    for cidx in range(cstart,cfinish):
        #filename = '{}maxkde_p{}_s{}_bw{}_c{:06d}.csv'.format(ksrc,PP,stepsize,bw,cidx)
        
        #if not os.path.isfile(filename):
        maxkde = np.zeros(len(transcriptomes)) - 1
        cell, ss = cell_img_preparation(cidx, wall, label, metacell, PP = 6)
        axes, grid, gmask, cgrid, cgridmask = cell_grid_preparation(cell, ss, zmax, stepsize, pows2)
        print('Cell {}: ( {} , {} , {} )'.format(cidx, *list(map(len, axes))[::-1]))
        
        # # Prepare the KDE grid
        tvals = np.nonzero(transcell.iloc[tidxs, cidx].values > 0)[0]
        tcounter = 1
        for tidx in tvals:
            
            tidx = tidxs[tidx]
            tdst = dst + transcriptomes[tidx] + os.sep
            filename = '{}{}_-_{}_p{}_s{}_bw{}_c{:06d}.json'.format(tdst, transcriptomes[tidx], dst.split('/')[1], PP,stepsize,bw,cidx)
            
            if not os.path.isfile(filename):
            
                print('Transcript {:05d} --\t-- {}/{}'.format(tidx, tcounter, len(tvals)))
                
                coords = translocs[tidx].values.T
                cmask = label[ coords[1], coords[0] ] == cidx + 1
                ccoords = coords[:, cmask ].copy()
                
                w = weight[tcumsum[tidx]:tcumsum[tidx+1]][cmask]
                kde = cell_weighted_kde(ccoords.T, grid, w, bw, gmask, stepsize, cgridmask, axes)
                maxkde[tidx] = np.max(kde)
                
                #filename = '{}{}_-_{}_p{}_s{}_bw{}_c{:06d}.json'.format(tdst, transcriptomes[tidx], dst.split('/')[1], PP,stepsize,bw,cidx)
                #if not os.path.isfile(filename):                    
                                
                cc = gd.CubicalComplex(top_dimensional_cells = np.max(kde) - kde)
                pers = cc.persistence(homology_coeff_field=2, min_persistence=1e-15)
                print(filename)
                with open(filename, 'w') as f:
                    json.dump(pers,f)
                
                tcounter += 1
        
            #filename = '{}maxkde_p{}_s{}_bw{}_c{:06d}.csv'.format(ksrc,PP,stepsize,bw,cidx)
            #print(filename)
            #pd.DataFrame(maxkde.reshape(1,-1)).to_csv(filename, index=False, header=False)
                
if __name__ == '__main__':
    main()
