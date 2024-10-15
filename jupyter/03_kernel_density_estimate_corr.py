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
        Levels = ['sub']
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
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv', index_col=-1)
    cell_nuc = pd.read_csv(ksrc + sample + '_nuclei_limits.csv')
    transcriptomes = np.asarray(metatrans['gene'])

    Cells = utils.get_range_cell_values(args.cell_focus, metacell, startval=1)
    Genes = utils.get_range_gene_values(args.gene_focus, transcriptomes, startval=0)
    
    if (Cells is None) or (Genes is None):
        print('Make sure that the ID value is an integer')
        print('Or make sure that the specified file exists and is formatted correctly')
        return 0
    
    tcumsum = np.hstack(([0], np.cumsum(metatrans['cyto_number'].values)))
    translocs = [None for _ in range(len(transcriptomes))]
    for i in range(len(transcriptomes)):
        filename = tsrc + sample + os.sep + 'location_corrected_D2_-_' + transcriptomes[i] + '.csv'
        translocs[i] = pd.read_csv(filename, header=None, names=['X', 'Y', 'Z'])
    tlocs = pd.concat(translocs)
    zmax = np.max(tlocs['Z']+stepsize)
    
    for level in Levels:
        dst = '..' + os.sep + level + 'level' + os.sep + sample + os.sep
        if not os.path.isdir(dst):
            os.mkdir(dst)
        for i in range(len(Genes)):
            tdst = dst + transcriptomes[Genes[i]] + os.sep
            if not os.path.isdir(tdst):
                os.mkdir(tdst)
                
    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    label, cellnum = ndimage.label(wall, ndimage.generate_binary_structure(2,1))
    print('Detected',cellnum,'cells')
    
    lnuc, nnuc = ndimage.label(tf.imread(nsrc + sample + '_EDT.tif') < args.nuclei_mask_cutoff, ndimage.generate_binary_structure(2,1))
    print('Detected',nnuc,'nuclei')

    # # Select a cell and then a gene

    for cidx in Cells:
    
        s_ = (np.s_[max([0, metacell.loc[cidx, 'y0'] - PP]) : min([wall.shape[0], metacell.loc[cidx, 'y1'] + PP])],
              np.s_[max([1, metacell.loc[cidx, 'x0'] - PP]) : min([wall.shape[1], metacell.loc[cidx, 'x1'] + PP])])
        extent = (s_[1].start, s_[1].stop, s_[0].start, s_[0].stop)


        cell = label[s_].copy()
        cell[ label[s_] > 0 ] = 0
        cell[ label[s_] == cidx ] = nnuc + 1
        cell[ lnuc[s_] > 0 ] = lnuc[s_][lnuc[s_] > 0]
        cell[ label[s_] == 0 ] = -1
        
        nuc_lims = cell_nuc.loc[ (cell_nuc['ndimage_ID'] == cidx), ['ndimage_ID','nuc_ID','N_inside','n_bot','n_top']]
        foo = np.setdiff1d( np.unique(cell), nuc_lims['nuc_ID'].values)[:-1]
        
        if len(foo) <= 2:
            print('NO:\t{}'.format(cidx), foo)
        
        else:
            print('Yes:\t{}'.format(cidx))
            maxdims = ( cell.shape[1], cell.shape[0], zmax) 
            axes, grid, gmask = utils.kde_grid_generator(stepsize=stepsize, maxdims=maxdims, pows2 = pows2, pad=1.5)
            grid[:, :2] = grid[:, :2] + np.array([s_[1].start, s_[0].start])
            
            cgrid = grid[gmask].copy()
            cgrid[:,:2] = grid[gmask][:,:2] - np.array([s_[1].start, s_[0].start])

            
            outside_walls = cell[cgrid[:,1],cgrid[:,0]] < 1
        
            for v in foo:
                outside_walls |= cell[cgrid[:,1], cgrid[:,0]] == v
            
            for j in range(len(nuc_lims)):
                _, nidx, N_inside, n_bot, n_top = nuc_lims.iloc[j]
                if n_bot < n_top:
                    thr_mask = (cgrid[:,2] >= n_bot) & (cgrid[:,2] <= n_top)
                else:
                    thr_mask = (cgrid[:,2] <= n_top) | (cgrid[:,2] >= n_bot)

                outside_walls |= ((cell[cgrid[:,1],cgrid[:,0]] == nidx) & thr_mask)
            
            for tidx in Genes:

                coords = translocs[tidx].values.T
                cmask = label[ coords[1], coords[0] ] == cidx
                
                if np.sum(cmask) > 5:
                    
                    ccoords = coords[:, cmask ].copy()
                    
                    # # Compute, crop, and correct the KDE
                    
                    for bw in BW:

                        kde = FFTKDE(kernel='gaussian', bw=bw, norm=2).fit(ccoords.T).evaluate(grid)
                        kde = kde[gmask]/(np.sum(kde[gmask])*(stepsize**len(coords)))
                        kde[outside_walls] = 0

                        kde = kde/(np.sum(kde)*(stepsize**len(coords)))
                        kde = kde.reshape( list(map(len, axes))[::-1], order='F')
                        
                        # # Cubical persistence

                        for level in Levels:
                            tdst = '..' + os.sep + level + 'level' + os.sep + sample + os.sep + transcriptomes[tidx] + os.sep
                            filename = tdst + transcriptomes[tidx] + '_-_{}_p{}_s{}_bw{}_c{:06d}.json'.format(level,PP,stepsize,bw,cidx)
                            cc = gd.CubicalComplex(top_dimensional_cells = utils.get_level_filtration(kde, level) )
                            pers = cc.persistence(homology_coeff_field=2, min_persistence=1e-15)
                            print(filename)
                            with open(filename, 'w') as f:
                                json.dump(pers,f)

    return 0

if __name__ == '__main__':
    main()
