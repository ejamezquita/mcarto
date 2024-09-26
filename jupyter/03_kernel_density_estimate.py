import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6


import numpy as np
import pandas as pd
import tifffile as tf
from glob import glob
from scipy import ndimage, stats

import gudhi as gd
import json

from KDEpy import FFTKDE
import argparse
import utils

pows2 = 2**np.arange(20) + 1
PP = 6
pp = 0
BW = [10,15,20,25,30]

def main():
    
    parser = argparse.ArgumentParser(description="Produce cell and gene metadata that will be useful later.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='Last major update: May 2024. © Erik Amezquita')
    parser.add_argument("sample", type=str,
                        help="Cross-section sample name")
    parser.add_argument("-b", "--kde_bandwidth", type=int, default=10,
                        help="bandwidth to compute KDE")
    parser.add_argument("-s", "--grid_stepsize", type=int, default=3,
                        help="grid size to evaluate the KDE")                        
    parser.add_argument("-c", "--cell_focus", type=str,
                        help="file or single ID with cell to evaluate")
    parser.add_argument("-g", "--gene_focus", type=str,
                        help="file or single ID with gene to evaluate")
    parser.add_argument("-l", "--level_filtration", type=str, choices=['sub','sup'],
                        help="level filtration to use to compute persistent homology")
    parser.add_argument("-n", "--nuclei_mask_cutoff", type=int, default=1,
                        help="Consider a transcript as part of the nucleus if it is within this distance from one")
    parser.add_argument("--cell_wall_directory", type=str, default="cell_dams",
                        help="directory containing cell wall TIFs")
    parser.add_argument("--nuclear_directory", type=str, default="nuclear_mask",
                        help="directory containing nuclei TIFs")
    parser.add_argument("--location_directory", type=str, default="translocs",
                        help="directory to contain corrected spatial location data")
    parser.add_argument("--kde_directory", type=str, default="kde",
                        help="directory to contain data related to KDE computations")
    parser.add_argument("-w", "--rewrite_results", action="store_true",
                        help="prior results will be rewritten")
    args = parser.parse_args()
    
    rewrite = args.rewrite_results
    sample = args.sample

    if args.level_filtration is None:
        Levels = ['sub','sup']
    else:
        Levels = [args.level_filtration]

    wsrc = '..' + os.sep + args.cell_wall_directory + os.sep
    nsrc = '..' + os.sep + args.nuclear_directory + os.sep
    tsrc = '..' + os.sep + args.location_directory + os.sep
    ksrc = '..' + os.sep + args.kde_directory + os.sep + sample + os.sep

    bw = args.kde_bandwidth
    stepsize = args.grid_stepsize

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv', index_col = 0)
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv')
    transcriptomes = np.asarray(metatrans['gene'])

    Cells = utils.get_range_cell_values(args.cell_focus, metacell, startval=1)
    Genes = utils.get_range_gene_values(args.gene_focus, transcriptomes, startval=0)
    
    if (Cells is None) or (Genes is None):
        print('Make sure that the ID value is an integer')
        print('Or make sure that the specified file exists and is formatted correctly')
        return 0
    
    tcumsum = np.hstack(([0], np.cumsum(metatrans['cyto_number'].values)))
    translocs = [None for i in range(len(transcriptomes))]
    for i in range(len(transcriptomes)):
        filename = tsrc + sample + os.sep + 'location_corrected_D2_-_' + transcriptomes[i] + '.csv'
        translocs[i] = pd.read_csv(filename, header=None, names=['X', 'Y', 'Z'])
    tlocs = pd.concat(translocs)
    zmax = np.max(tlocs['Z']+stepsize)
    
    for level in Levels:
        dst = '..' + os.sep + level + 'level' + os.sep + sample + os.sep
        if not os.path.isdir(dst):
            os.mkdir(dst)
        for i in range(len(transcriptomes)):
            tdst = dst + transcriptomes[i] + os.sep
            if not os.path.isdir(tdst):
                os.mkdir(tdst)
                
    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    label, cellnum = ndimage.label(wall, ndimage.generate_binary_structure(2,1))

    wall[tf.imread(nsrc + sample + '_EDT.tif') < args.nuclei_mask_cutoff] = False
    print('Detected',cellnum,'cells')

    # # Compute transcript weights for KDE

    filename = ksrc + sample + '_border_weights.npy'

    if not os.path.isfile(filename):
        top, right, bottom, left = utils.cardinal_distance_transform(wall)
        wv = stats.norm.cdf(top[tlocs['Y'].values, tlocs['X'].values]+pp, loc=0, scale=bw)
        wv-= stats.norm.cdf(-bottom[tlocs['Y'].values, tlocs['X'].values]-pp, loc=0, scale=bw)
        
        wh = stats.norm.cdf(right[tlocs['Y'].values, tlocs['X'].values]+pp, loc=0, scale=bw) 
        wh-= stats.norm.cdf(-left[tlocs['Y'].values, tlocs['X'].values]-pp, loc=0, scale=bw)
        
        weight = 2-(wv*wh)
        np.save(filename, weight)
        print('Saved',filename)

    weight = np.load(filename, allow_pickle=True)

    # # Select a cell and then a gene

    for cidx in Cells:
        
        s_ = (np.s_[max([0, metacell.loc[cidx, 'y0'] - PP]) : min([wall.shape[0], metacell.loc[cidx, 'y1'] + PP])],
              np.s_[max([1, metacell.loc[cidx, 'x0'] - PP]) : min([wall.shape[1], metacell.loc[cidx, 'x1'] + PP])])
        extent = (s_[1].start, s_[1].stop, s_[0].start, s_[0].stop)
        cell = wall[s_].copy().astype(np.uint8)
        cell[ label[s_] == cidx ] = 2
        cell[~wall[s_]] = 0
        maxdims = ( cell.shape[1], cell.shape[0], zmax )
        axes, grid, gmask = utils.kde_grid_generator(stepsize=stepsize, maxdims=maxdims, pows2 = pows2, pad=1.5)
        grid[:, :2] = grid[:, :2] + np.array([ extent[0], extent[2] ])
        
        cgrid = grid[gmask].copy()
        cgrid[:,:2] = grid[gmask][:,:2] - np.array([ extent[0], extent[2] ])
        cgridmask = cell[cgrid[:,1],cgrid[:,0]] != 2
        
        for tidx in Genes:

            coords = translocs[tidx].values.T
            cmask = label[ coords[1], coords[0] ] == cidx
            
            if np.sum(cmask) > 1:
                w = weight[tcumsum[tidx]:tcumsum[tidx+1]][cmask]
                foo = glob('..' + os.sep + '*level' + os.sep + sample + os.sep + transcriptomes[tidx] + os.sep + '*c{:06d}.json'.format(cidx))
                
                if rewrite or ( len(foo) != len(Levels)*len(BW) ):
            
                    ccoords = coords[:, cmask ].copy()
                    # # Compute, crop, and correct the KDE
                    
                    for bw in BW:

                        kde = FFTKDE(kernel='gaussian', bw=bw, norm=2).fit(ccoords.T, w).evaluate(grid)
                        kde = kde[gmask]/( np.sum(kde[gmask]) * (stepsize**len(coords)) )
                        kde[ cgridmask ] = 0
                        kde = kde/( np.sum(kde) * (stepsize**len(coords)) )
                        kde = kde.reshape( list(map(len, axes))[::-1], order='F')
                        
                        # # Cubical persistence

                        for level in Levels:
                            tdst = '..' + os.sep + level + 'level' + os.sep + sample + os.sep + transcriptomes[tidx] + os.sep
                            filename = tdst + transcriptomes[tidx] + '_-_{}_p{}_s{}_bw{}_c{:06d}.json'.format(level,PP,stepsize,bw,cidx)
                            if not os.path.isfile(filename):
                                cc = gd.CubicalComplex(top_dimensional_cells = utils.get_level_filtration(kde, level) )
                                pers = cc.persistence(homology_coeff_field=2, min_persistence=1e-15)
                                print(filename)
                                with open(filename, 'w') as f:
                                    json.dump(pers,f)

    return 0

if __name__ == '__main__':
    main()
