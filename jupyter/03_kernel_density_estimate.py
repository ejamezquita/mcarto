import numpy as np
import pandas as pd
import tifffile as tf
from glob import glob
import os
from scipy import ndimage, stats
import gudhi as gd
import json

from KDEpy import FFTKDE
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
                
                cc = gd.CubicalComplex(top_dimensional_cells = np.max(kde) - kde)
                pers = cc.persistence(homology_coeff_field=2, min_persistence=1e-15)
                print(filename)
                with open(filename, 'w') as f:
                    json.dump(pers,f)
                
                tcounter += 1
        
                 
if __name__ == '__main__':
    main()
