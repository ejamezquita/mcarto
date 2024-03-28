import numpy as np
import pandas as pd
import tifffile as tf
from glob import glob
import os
from scipy import ndimage, spatial

import argparse

wsrc = '../cell_dams/'
nsrc = '../nuclear_mask/'
psrc = '../proc/'
osrc = '../data/'
tsrc = '../translocs/'

verbose = False

def celllocs_read(filename):
    celllocs = pd.read_csv(filename)
    sel = [0,3,4,5,6,7,8,9]
    celllocs = celllocs.iloc[~np.any(celllocs.iloc[:, :5].isnull().values, axis=1)]
    celllocs = celllocs[celllocs['Cell.Area..px.'] > 9]
    celllocs = celllocs.astype(dict(zip(celllocs.columns[np.array(sel)], [int for i in range(len(sel))])))
    return celllocs

def match_original_ndimage(celllocs, wall, label, cellnum):
    cnuclei = np.asarray(ndimage.center_of_mass(wall, label, range(1,cellnum+1)))
    dcoords = celllocs.iloc[:, 1:3].values
    cdist = spatial.distance.cdist(np.flip(cnuclei, axis=1), dcoords, metric='euclidean')
    cmatches = np.argmin(cdist, axis=1)
    foo = len(np.unique(cmatches))
    print("Matched {} ndimage.cells to {} unique cells in the metadata".format(cellnum,foo))
    print("Out of {} cells in the metadata\n{}".format(len(celllocs),foo>=cellnum) )

    return dcoords, cnuclei, cmatches

def generate_cell_metadata(label, objss, nuclei):
    meta = np.zeros((len(objss), 8), dtype=int)
    for i in range(len(meta)):
        meta[i, :4] = objss[i][1].start, objss[i][1].stop, objss[i][0].start, objss[i][0].stop
        meta[i, 4] = meta[i,1] - meta[i,0]
        meta[i, 5] = meta[i,3] - meta[i,2]
    meta[:, 6], _ = np.histogram(label, bins=np.arange(1, len(objss) + 2))
    meta[:, 7], _ = np.histogram(label[nuclei], bins=np.arange(1, len(objss) + 2))
    meta = pd.DataFrame(meta, columns=['x0', 'x1', 'y0', 'y1', 'length', 'height', 'total_area', 'nuclei_area'])
    meta['cyto_area'] = meta['total_area'] - meta['nuclei_area']
    meta['c2t_area_ratio'] = meta['cyto_area']/meta['total_area']
    return meta

def generate_transcell_metadata(translocs, transcriptomes, cellnum, label):
    meta = np.zeros((len(transcriptomes), cellnum), dtype=int)
    bins = np.arange(1, cellnum + 2)
    for tidx in range(len(meta)):
        coords = translocs[tidx].loc[:, ['X', 'Y']].values.T
        meta[tidx], _ = np.histogram(label[coords[1], coords[0]], bins=bins)
        
    meta = pd.DataFrame(meta, columns=bins[:-1])
    meta['gene'] = transcriptomes

    return meta
    
def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('sample', metavar='raw_dir', type=str, help='directory where raw images are located')
    args = parser.parse_args()

    sample = args.sample
    
    dst = '../kde/'
    dst = dst + sample + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)

    # # Load all general data

    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    nuclei = tf.imread(nsrc + sample + '_EDT.tif') < 2
    
    label, cellnum = ndimage.label(wall, ndimage.generate_binary_structure(2,1))

    print('Detected',cellnum,'cells')

    filenames = sorted(glob(tsrc + sample + os.sep + '*.csv'))
    tsize = np.zeros(len(filenames), dtype=int)
    transcriptomes = [os.path.splitext(filenames[i])[0].split('_-_')[-1] for i in range(len(filenames)) ]
    translocs = [None for i in range(len(filenames))]
    for i in range(len(filenames)):
        translocs[i] = pd.read_csv(filenames[i], header=None, names=['X', 'Y', 'Z'])
        tsize[i] = len(translocs[i])

    # # Compute bandwidth via ISJ

    filename = dst + sample + '_cells_metadata.csv'
    if not os.path.isfile(filename):
        celllocs = celllocs_read(osrc + sample + '_data/' + transcriptomes[1] + '/' + transcriptomes[1] + ' - localization results by cell.csv')
        dcoords, cnuclei, cmatches = match_original_ndimage(celllocs, wall, label, cellnum)
        objss = ndimage.find_objects(label)

        meta = generate_cell_metadata(label, objss, nuclei)
        meta = meta.join(pd.DataFrame(np.round(dcoords[cmatches], 2), columns=['orig_comX', 'orig_comY']))
        meta = meta.join(pd.DataFrame(np.round(np.flip(cnuclei, axis=1),2), columns=['ndimage_comX', 'ndimage_comY']))
        meta['orig_cellID'] = celllocs['Cell.ID..'].values[cmatches]
        meta['ndimage_cellID'] = np.arange(1,cellnum+1)
        meta.to_csv(filename, index=False)
        print('Created:\t', filename)

    metacell = pd.read_csv(filename)

    filename = dst + sample + '_transcripts_metadata.csv'
    if not os.path.isfile(filename):
        data = pd.read_csv(osrc + sample + '_data/32771-slide1_' + sample + '_results.txt', header=None, sep='\t').drop(columns=[4])
        _, orig_size = np.unique(data.iloc[:,-1], return_index = False, return_inverse=False, return_counts=True) 
        meta = pd.DataFrame()
        meta['total_number'] = orig_size
        meta['cyto_number'] = tsize
        meta['nuclei_number'] = orig_size - tsize
        meta['ratio'] = tsize/orig_size
        meta['gene'] = transcriptomes
        meta.to_csv(filename, index=False)
        print('Created:\t', filename)

    metatrans = pd.read_csv(filename)

    filename = dst + sample + '_transcells_metadata.csv'
    if not os.path.isfile(filename):
        meta = generate_transcell_metadata(translocs, transcriptomes, cellnum, label)
        meta.to_csv(filename, index=False)
        print('Created:\t', filename)
        
    transcell = pd.read_csv(filename)
    
    
if __name__ == '__main__':
    main()
