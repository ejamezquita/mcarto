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
from scipy import ndimage
import argparse
import utils

PP = 0
min_n_ratio = 0.05
max_N_ratio = 0.55

# The no. of nuclei in a cell is determined as the number of separate nuclei
# connected components that lie inside the cell such each of these components
# represent at least `min_n_ratio` of the total nuclear pixels inside the cell
# AND `max_N_ratio` of these nuclei is inside the cell

def main():
    
    parser = argparse.ArgumentParser(description="Produce cell and gene metadata that will be useful later.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='Last major update: May 2024. © Erik Amezquita')
    parser.add_argument("sample", type=str,
                        help="Cross-section sample name")
    parser.add_argument("-n", "--nuclei_mask_cutoff", type=int, default=1,
                        help="Consider a transcript as part of the nucleus if it is within this distance from one")
    parser.add_argument("--cell_wall_directory", type=str, default="cell_dams",
                        help="directory containing cell wall TIFs")
    parser.add_argument("--nuclear_directory", type=str, default="nuclear_mask",
                        help="directory containing nuclei TIFs")
    parser.add_argument("--initial_data_directory", type=str, default="data",
                        help="directory containing spatial location data")
    parser.add_argument("--location_directory", type=str, default="translocs",
                        help="directory to contain corrected spatial location data")
    parser.add_argument("--kde_directory", type=str, default="kde",
                        help="directory to contain data related to KDE computations")
    parser.add_argument("-w", "--rewrite_results", action="store_true",
                        help="prior results will be rewritten")
    args = parser.parse_args()

    wsrc = os.pardir + os.sep + args.cell_wall_directory + os.sep
    nsrc = os.pardir + os.sep + args.nuclear_directory + os.sep
    csrc = os.pardir + os.sep + args.initial_data_directory + os.sep
    tsrc = os.pardir + os.sep + args.location_directory + os.sep
    dst = os.pardir + os.sep + args.kde_directory + os.sep
    
    rewrite = args.rewrite_results
    sample = args.sample
        
    dst = dst + sample + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)

    # # Load all general data

    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    nuclei = tf.imread(nsrc + sample + '_EDT.tif') < args.nuclei_mask_cutoff

    filenames = sorted(glob(tsrc + sample + os.sep + '*.csv'))
    tsize = np.zeros(len(filenames), dtype=int)
    transcriptomes = [os.path.splitext(filenames[i])[0].split('_-_')[-1] for i in range(len(filenames)) ]
    translocs = [None for i in range(len(filenames))]
    for i in range(len(filenames)):
        translocs[i] = pd.read_csv(filenames[i], header=None, names=['X', 'Y', 'Z'])
        tsize[i] = len(translocs[i])

    filename = dst + sample + '_cells_metadata.csv'
    if rewrite or (not os.path.isfile(filename)):
        label, cellnum = ndimage.label(wall, ndimage.generate_binary_structure(2,1))
        objss = ndimage.find_objects(label)
        print('Detected',cellnum,'cells')
        
        celllocs = utils.celllocs_read(csrc + sample + '_data' + os.sep + transcriptomes[1] + os.sep + transcriptomes[1] + ' - localization results by cell.csv')
        dcoords, cnuclei, argmatches, orig_cellID = utils.match_original_ndimage(celllocs, wall, label, cellnum)
        
        meta = utils.generate_cell_metadata(label, objss, nuclei)
        meta['ndimage_cellID'] = np.arange(1,cellnum+1)
        meta = meta.set_index(keys=['ndimage_cellID'])
        
        lnuc, nnuc = ndimage.label(nuclei, ndimage.generate_binary_structure(2,1))
        nuc_area, _ = np.histogram(lnuc, bins=np.arange(nnuc + 2))
        nuc_area[0] = 0
        print('Detected',nnuc,'nuclei')
        
        number_nuclei = np.zeros(len(meta), dtype=int)
        for i in np.nonzero(meta['nuclei_area'] > 100)[0]:
            cidx = meta.iloc[i].name
            ss = (np.s_[max([0, meta.loc[cidx, 'y0'] - PP]) : min([wall.shape[0], meta.loc[cidx, 'y1'] + PP])],
                  np.s_[max([1, meta.loc[cidx, 'x0'] - PP]) : min([wall.shape[1], meta.loc[cidx, 'x1'] + PP])])
            
            uq, ct = np.unique(lnuc[ss][(lnuc[ss] > 0) & (label[ss] == cidx)], return_counts=True)
            number_nuclei[i] = np.sum( ( (ct/np.sum(ct)) > min_n_ratio ) & (ct/nuc_area[uq] > max_N_ratio) )
        
        meta['number_nuclei'] = number_nuclei
        orig_com = np.round(dcoords[argmatches], 2)
        orig_com[orig_cellID == 0] = 0
        meta = meta.join(pd.DataFrame( orig_com, columns=['orig_comX', 'orig_comY'], index=meta.index))
        meta = meta.join(pd.DataFrame(np.round(np.flip(cnuclei, axis=1),2), columns=['ndimage_comX', 'ndimage_comY'], index=meta.index))
        meta['orig_cellID'] = orig_cellID
        meta.to_csv(filename, index_label='ndimage_cellID')
        print('Created:\t', filename)
    
    metacell = pd.read_csv(filename, index_col=0)
    print('Number of unmatched ndimage labels:\t', len(metacell.loc[metacell['orig_cellID'] == 0]) )
    uq, ct = np.unique(metacell.loc[metacell['orig_cellID'] > 0, 'orig_cellID'].values, return_counts=True)
    print('IDs with more than one ndimage cell:', np.sum(ct > 1))
    d = metacell[metacell['orig_cellID'] != 0].loc[:, ['orig_comX','orig_comY','ndimage_comX','ndimage_comY']].values
    print( pd.Series(np.sqrt((d[:,0] - d[:,2])**2 + (d[:,1] - d[:,3])**2)).describe() )
    
    filename = dst + sample + '_transcripts_metadata.csv'
    if rewrite or (not os.path.isfile(filename)):
        data = pd.read_csv(csrc + sample + '_data'+os.sep+'32771-slide1_' + sample + '_results.txt', header=None, sep='\t').drop(columns=[4])
        _, orig_size = np.unique(data.iloc[:,-1], return_index = False, return_inverse=False, return_counts=True) 
        meta = pd.DataFrame()
        meta['total_number'] = orig_size
        meta['cyto_number'] = tsize
        meta['nuclei_number'] = orig_size - tsize
        meta['ratio'] = tsize/orig_size
        meta['gene'] = transcriptomes
        meta.to_csv(filename, index=False)
        print('Created:\t', filename)

    filename = dst + sample + '_transcells_metadata.csv'
    if rewrite or (not os.path.isfile(filename)):
        meta = utils.generate_transcell_metadata(translocs, transcriptomes, cellnum, label)
        meta.to_csv(filename, index=False)
        print('Created:\t', filename)
        
    return 0
    
if __name__ == '__main__':
    main()
