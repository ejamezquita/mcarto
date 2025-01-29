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
KBINS_NO = 27

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
    parser.add_argument("--geometry_directory", type=str, default="geometry",
                        help="directory to contain data related to geometry computations")
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
    gsrc = '..' + os.sep + args.geometry_directory + os.sep + sample + os.sep

    bw = args.kde_bandwidth
    stepsize = args.grid_stepsize

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv', index_col = 0)
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv', index_col=-1)
    cell_nuc = pd.read_csv(ksrc + sample + '_nuclei_limits.csv')
    transcriptomes = np.asarray(metatrans['gene'])
    
    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    label, cellnum = ndimage.label(wall, ndimage.generate_binary_structure(2,1))
    print('Detected',cellnum,'cells')
    
    lnuc, nnuc = ndimage.label(tf.imread(nsrc + sample + '_EDT.tif') < args.nuclei_mask_cutoff, ndimage.generate_binary_structure(2,1))
    print('Detected',nnuc,'nuclei')


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
        translocs[i]['cidx'] = label[ translocs[i]['Y'], translocs[i]['X'] ]
        translocs[i]['nidx'] =  lnuc[ translocs[i]['Y'], translocs[i]['X'] ]

    tlocs = pd.concat(translocs)
    
    zmax = np.max(tlocs['Z']+stepsize)

    for level in Levels:
        dst = '..' + os.sep + level + 'level' + os.sep + sample + os.sep
        if not os.path.isdir(dst):
            os.mkdir(dst)
        for gidx in Genes:
            tdst = dst + transcriptomes[gidx] + os.sep
            if not os.path.isdir(tdst):
                os.mkdir(tdst)
    
    if not os.path.isdir(gsrc):
        os.mkdir(gsrc)
    for gidx in Genes:
        tdst = gsrc + transcriptomes[gidx] + os.sep
        if not os.path.isdir(tdst):
            os.mkdir(tdst)
            
    
    # # Select a cell and then a gene
    
    level = 'sub'
    for cidx in Cells:
        
        # all_files = True
        # for tidx in Genes:
            # for level in Levels:
                # for bw in BW:
                    # filename = '..' + os.sep + '{}level'.format(level) + os.sep + sample + os.sep + transcriptomes[tidx] + os.sep 
                    # filename +='{}_-_{}_p{}_s{}_bw{}_c{:06d}.json'.format(transcriptomes[tidx], level, PP, stepsize, bw, cidx)
                    # #print(filename, os.path.isfile(filename))
                    # if not os.path.isfile(filename):
                        # all_files = False
                        # break
        
        if rewrite or not all_files:
            
            cell, cextent = utils.get_cell_img(cidx, metacell, label, lnuc, nnuc, PP=PP, pxbar=True)
            s_ = np.s_[ cextent[2]:cextent[3] , cextent[0]:cextent[1] ]
            edt = ndimage.distance_transform_edt(label[s_] == cidx)

            axes, grid, kdegmask, cgrid, outside_walls = utils.cell_grid_preparation(cidx, cell, label, cextent, zmax, stepsize, cell_nuc)
            outw = outside_walls.copy().reshape( list(map(len, axes))[::-1], order='F')
            zfactor = np.divide(  list(map(len, axes))[::-1][1:] , np.asarray(cell.shape) )
            kbins = np.linspace(0, np.max(edt), KBINS_NO)

            cellhist = np.digitize(edt, kbins, right=True)
            zoom = ndimage.zoom(cellhist, zfactor, mode='grid-constant', grid_mode=True)
            zoom = (~outw)*np.tile(zoom, reps=(len(axes[-1]), 1,1))

            peripherality = pd.DataFrame(index=range(1,len(kbins)))
            peripherality['count'] =  ndimage.histogram(zoom, 1, len(kbins)-1, len(kbins)-1)
            
            for tidx in Genes:

                coords = translocs[tidx].loc[ translocs[tidx]['cidx'] == cidx , ['X','Y', 'Z'] ].values.T
                tdst = '..' + os.sep + level + 'level' + os.sep + sample + os.sep + transcriptomes[tidx] + os.sep
                        
                # # Compute, crop, and correct the KDE
                
                for bw in BW:

                    kde = FFTKDE(kernel='gaussian', bw=bw, norm=2).fit(coords.T).evaluate(grid)
                    kde = kde[kdegmask]/(np.sum(kde[kdegmask])*(stepsize**len(coords)))
                    kde[outside_walls] = 0

                    kde = kde/(np.sum(kde)*(stepsize**len(coords)))
                    kde = kde.reshape( list(map(len, axes))[::-1], order='F')
                    
                    foo = '_{}_-_{}'.format(bw, transcriptomes[tidx])
                    peripherality['sum' + foo] = ndimage.sum_labels(kde, zoom, range(1, len(kbins)))
                    peripherality['mean' + foo] = peripherality['sum'+foo] / peripherality['count']
                    
                    # # Cubical persistence

                    #for level in Levels:
                    if metacell.loc[cidx, 'nuclei_area'] > 0:
                        filename = tdst + transcriptomes[tidx] + '_-_{}_p{}_s{}_bw{}_c{:06d}.json'.format(level,PP,stepsize,bw,cidx)                        
                        cc = gd.CubicalComplex(top_dimensional_cells = utils.get_level_filtration(kde, level) )
                        pers = cc.persistence(homology_coeff_field=2, min_persistence=1e-15)
                        print(filename)
                        with open(filename, 'w') as f:
                            json.dump(pers,f)
                                
            filename = gsrc + transcriptomes[tidx] + os.sep + transcriptomes[tidx] + '_-_peripherality_c{:06d}.csv'.format(cidx)
            peripherality.to_csv(filename, index=False)
            

    return 0

if __name__ == '__main__':
    main()
