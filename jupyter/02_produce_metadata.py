import numpy as np
import pandas as pd
import tifffile as tf
from glob import glob
import os
from scipy import ndimage, spatial
import argparse
import utils

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

    wsrc = '..' + os.sep + args.cell_wall_directory + os.sep
    nsrc = '..' + os.sep + args.nuclear_directory + os.sep
    csrc = '..' + os.sep + args.initial_data_directory + os.sep
    tsrc = '..' + os.sep + args.location_directory + os.sep
    dst = '..' + os.sep + args.kde_directory + os.sep
    
    rewrite = args.rewrite_results
    sample = args.sample
        
    dst = dst + sample + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)

    # # Load all general data

    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    nuclei = tf.imread(nsrc + sample + '_EDT.tif') < args.nuclei_mask_cutoff
    
    label, cellnum = ndimage.label(wall, ndimage.generate_binary_structure(2,1))

    print('Detected',cellnum,'cells')

    filenames = sorted(glob(tsrc + sample + os.sep + '*.csv'))
    tsize = np.zeros(len(filenames), dtype=int)
    transcriptomes = [os.path.splitext(filenames[i])[0].split('_-_')[-1] for i in range(len(filenames)) ]
    translocs = [None for i in range(len(filenames))]
    for i in range(len(filenames)):
        translocs[i] = pd.read_csv(filenames[i], header=None, names=['X', 'Y', 'Z'])
        tsize[i] = len(translocs[i])

    filename = dst + sample + '_cells_metadata.csv'
    if rewrite or (not os.path.isfile(filename)):
        celllocs = utils.celllocs_read(csrc + sample + '_data/' + transcriptomes[1] + '/' + transcriptomes[1] + ' - localization results by cell.csv')
        dcoords, cnuclei, cmatches = utils.match_original_ndimage(celllocs, wall, label, cellnum)
        objss = ndimage.find_objects(label)

        meta = utils.generate_cell_metadata(label, objss, nuclei)
        meta = meta.join(pd.DataFrame(np.round(dcoords[cmatches], 2), columns=['orig_comX', 'orig_comY']))
        meta = meta.join(pd.DataFrame(np.round(np.flip(cnuclei, axis=1),2), columns=['ndimage_comX', 'ndimage_comY']))
        meta['orig_cellID'] = celllocs['Cell.ID..'].values[cmatches]
        meta['ndimage_cellID'] = np.arange(1,cellnum+1)
        meta.set_index(keys=['ndimage_cellID']).to_csv(filename, index_label='ndimage_cellID')
        print('Created:\t', filename)

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
