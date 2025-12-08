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

min_mrna_num = 0
pows2 = 2**np.arange(20) + 1
PP = 6
pp = 0
BW = [10,15,20,25,30]
KBINS_NO = 27
pericols = []
for bw in BW:
    pericols += ['{}_{}'.format(c,bw) for c in ['mean','sum']]
level = 'sub'

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
    parser.add_argument("-N", "--include_nuclei_mrna", action="store_true",
                        help="include mRNA located in nuclear regions")
    args = parser.parse_args()
    
    rewrite = args.rewrite_results
    sample = args.sample
    stepsize = args.grid_stepsize
    exclude_nuclei = not args.include_nuclei_mrna
    print('Exclude nuclear mRNA:', exclude_nuclei)

    if args.level_filtration is None:
        Levels = ['sub','sup']
    else:
        Levels = [args.level_filtration]

    wsrc = os.pardir + os.sep + args.cell_wall_directory + os.sep
    nsrc = os.pardir + os.sep + args.nuclear_directory + os.sep
    tsrc = os.pardir + os.sep + args.location_directory + os.sep + sample + os.sep
    ksrc = os.pardir + os.sep + args.kde_directory + os.sep + sample + os.sep
    gsrc = os.pardir + os.sep + args.geometry_directory + os.sep + sample + os.sep

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv', index_col = 'ndimage_cellID')
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv', index_col='gene')
    cell_nuc = pd.read_csv(ksrc + sample + '_nuclei_limits.csv')
    transcriptomes = np.asarray(metatrans.index, dtype=str)

    
    label, cellnum = ndimage.label(tf.imread(wsrc + sample + '_dams.tif').astype(bool), ndimage.generate_binary_structure(2,1))
    print('Detected',cellnum,'cells')

    lnuc, nnuc = ndimage.label(tf.imread(nsrc + sample + '_EDT.tif') < args.nuclei_mask_cutoff, ndimage.generate_binary_structure(2,1))
    print('Detected',nnuc,'nuclei')

    Cells = utils.get_range_cell_values(args.cell_focus, metacell, startval=1)
    Genes = utils.get_range_gene_values(args.gene_focus, transcriptomes, startval=0)
    
    if (Cells is None) or (Genes is None):
        print('Make sure that the ID value is an integer')
        print('Or make sure that the specified file exists and is formatted correctly')
        return 0
    
    translocs = [None for i in range(len(transcriptomes))]
    
    if exclude_nuclei:
        transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv', index_col='gene').rename(columns=int)
        for i in range(len(transcriptomes)):
            filename = tsrc + 'location_corrected_D2_-_' + transcriptomes[i] + '.csv'
            translocs[i] = pd.read_csv(filename, header=None, names=['X', 'Y', 'Z'])
    else:
        transcell = pd.read_csv(ksrc + sample + '_transcells_metadata_w_nucleus.csv', index_col='gene').rename(columns=int)
        tsrc = os.pardir + os.sep + 'Bacteria Info for Erik' + os.sep
        for i in range(len(transcriptomes)):
            filename = tsrc + transcriptomes[i] + '_v2.txt'
            translocs[i] = pd.read_csv(filename, sep='\t')
    
    for i in range(len(transcriptomes)):
        translocs[i]['cidx'] = label[ translocs[i]['Y'], translocs[i]['X'] ]
        translocs[i]['nidx'] =  lnuc[ translocs[i]['Y'], translocs[i]['X'] ]

    tlocs = pd.concat(translocs)
    
    zmax = np.max(tlocs['Z']+stepsize)

    for level in Levels:
        dst = os.pardir + os.sep + level + 'level' + os.sep + sample + os.sep
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
    
    for cidx in Cells:
        
        # Check if geometry and persistence files have already been computed for the cell
                
        isgfile = dict()
        ispfile = dict()

        for t in range(len(Genes)):
            tidx = Genes[t]
            foo = gsrc + '{}/{}_bins_peripherality_c{:06d}.csv'.format(transcriptomes[tidx], KBINS_NO-1, cidx)
            isgfile[foo] = os.path.isfile(foo) | (transcell.loc[transcriptomes[tidx], cidx] <= min_mrna_num)
            
            tdst = os.pardir + os.sep + level + 'level' + os.sep + sample + os.sep + transcriptomes[tidx] + os.sep
            for b in range(len(BW)):
                foo = tdst + transcriptomes[tidx] + '_-_{}_p{}_s{}_bw{}_c{:06d}.json'.format(level,PP,stepsize,BW[b],cidx)
                ispfile[foo] = os.path.isfile(foo) | (transcell.loc[transcriptomes[tidx], cidx] <= min_mrna_num)
        
        # If geometry OR persistence files are missing, focus on the cell
        
        #print(cidx, isgfile, ispfile, all(isgfile.values()), all(ispfile.values()), not (all(isgfile.values()) & all(ispfile.values())), sep='\n')
        
        if not (all(isgfile.values()) & all(ispfile.values())):
            
            cell, cextent = utils.get_cell_img(cidx, metacell, label, lnuc, nnuc, PP=PP)
            s_ = np.s_[ cextent[2]:cextent[3] , cextent[0]:cextent[1] ]
            axes, grid, kdegmask, cgrid, outside_walls = utils.cell_grid_preparation(cidx, cell, label[s_], cextent, zmax, stepsize, cell_nuc, exclude_nuclei=exclude_nuclei)
            
            # If geometry is missing, do the EDT
            
            if not all(isgfile.values()):
            
                outw = outside_walls.copy().reshape( list(map(len, axes))[::-1], order='F')
                zfactor = np.divide(  list(map(len, axes))[::-1][1:] , np.asarray(cell.shape) )
                edt = ndimage.distance_transform_edt(label[s_] == cidx)
                kbins = np.linspace(0, np.max(edt), KBINS_NO)

                cellhist = np.digitize(edt, kbins, right=True)
                zoom = ndimage.zoom(cellhist, zfactor, mode='grid-constant', grid_mode=True)
                zoom = (~outw)*np.tile(zoom, reps=(len(axes[-1]), 1,1))
                pcount =  ndimage.histogram(zoom, 1, len(kbins)-1, len(kbins)-1)
            
            # Loop through all the genes
            
            for tidx in Genes:
                
                if transcell.loc[transcriptomes[tidx], cidx] > min_mrna_num:
                    
                    coords = translocs[tidx].loc[ translocs[tidx]['cidx'] == cidx , ['X','Y', 'Z'] ].values.T
                    tdst = os.pardir + os.sep + level + 'level' + os.sep + sample + os.sep + transcriptomes[tidx] + os.sep
                    perifile = gsrc + '{}/{}_bins_peripherality_c{:06d}.csv'.format(transcriptomes[tidx], KBINS_NO-1, cidx)
                    peripherality = pd.DataFrame(index=range(1,KBINS_NO), columns=pericols)
                            
                    # # Compute, crop, and correct the KDE
                    
                    for bw in BW:
                        
                        # If either the persistence file OR the geometry files are missing
                        
                        persfile = tdst + transcriptomes[tidx] + '_-_{}_p{}_s{}_bw{}_c{:06d}.json'.format(level,PP,stepsize,bw,cidx)
                        if (not ispfile[persfile]) | (not isgfile[perifile]) :

                            kde = FFTKDE(kernel='gaussian', bw=bw, norm=2).fit(coords.T).evaluate(grid)
                            kde = kde[kdegmask]/(np.sum(kde[kdegmask])*(stepsize**len(coords)))
                            kde[outside_walls] = 0

                            kde = kde/(np.sum(kde)*(stepsize**len(coords)))
                            kde = kde.reshape( list(map(len, axes))[::-1], order='F')
                            
                        if not isgfile[perifile]:
                        
                            peripherality['sum_{}'.format(bw)] = ndimage.sum_labels(kde, zoom, range(1, len(kbins)))
                            peripherality['mean_{}'.format(bw)] = peripherality['sum_{}'.format(bw)] / pcount
                            
                        
                        if not ispfile[persfile]:
                            cc = gd.CubicalComplex(top_dimensional_cells = utils.get_level_filtration(kde, level) )
                            pers = cc.persistence(homology_coeff_field=2, min_persistence=1e-15)
                            print(persfile)
                            with open(persfile, 'w') as f:
                                json.dump(pers,f)
                                
                    if not isgfile[perifile]:
                        peripherality.to_csv(perifile, index=False)
                        print(perifile)
                
                else:
                    print('Found',transcell.loc[transcriptomes[tidx], cidx],'transcripts for', transcriptomes[tidx],'in cell', cidx)
                

    return 0

if __name__ == '__main__':
    main()
