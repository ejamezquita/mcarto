import numpy as np
import pandas as pd
import tifffile as tf
from glob import glob
import os

from scipy import ndimage
from sklearn import neighbors
import argparse

radius = 30
maxdwall = 6
minneighs = 5
minprob = 74

def correct_boundary_transcripts(tlabs, coords, label, tpercell, R = 25):
    for i in np.nonzero(tlabs == 0)[0]:
        x,y = coords[:2,i]
        ss = np.s_[y - R : y + R, x - R : x + R]
        cells = np.unique(label[ss])[1:]
        newlab = cells[np.argmax(tpercell[cells])]
        com = np.flip(np.mean(np.asarray(np.nonzero(label[ss] == newlab)), axis=1))
        com[0] += x - R
        com[1] += y - R
        dv = com - coords[:2,i]
        dv = dv/np.linalg.norm(dv)
        delta = 1
        x,y = (coords[:2,i] + delta*dv).astype(int)
        
        while(label[y,x] != newlab) and (delta < 50):
            delta += 1
            x,y = (coords[:2,i] + delta*dv).astype(int)
        if delta < 50:
            coords[:2,i] = [x,y]
            tlabs[i] = newlab
        else:
            print('Review index', i)

    return 0

def transcript_shift(i, ndist, nidxs, cat, cdtlabs, cdtmask, cdtcoords, edtmask, edtvals, label):
    mask = cdtlabs[nidxs[i]] == cat[i,1]
    nearest = np.average(cdtcoords[:2,nidxs[i][mask]], axis=1, weights = radius - ndist[i][mask])
    dv = nearest - cdtcoords[:2,edtvals[i]]
    dv = dv/np.linalg.norm(dv)
    x,y = cdtcoords[:2,edtvals[i]]
    
    delta = 1
    x,y = (cdtcoords[:2,edtvals[i]] + delta*dv).astype(int)
    while(label[y,x] != cat[i,1]) and (delta < radius):
        delta += 1
        x,y = (cdtcoords[:2,edtvals[i]] + delta*dv).astype(int)
    if delta < radius:
        return [ [x,y], cat[i,1], True ]
    else:
        nearest = cdtcoords[:2,nidxs[i][mask][0]]
        dv = nearest - cdtcoords[:2,edtvals[i]]
        dv = dv/np.linalg.norm(dv)
        x,y = cdtcoords[:2,edtvals[i]]
        
        delta = 1
        x,y = (cdtcoords[:2,edtvals[i]] + delta*dv).astype(int)
        while(label[y,x] != cat[i,1]) and (delta < radius):
            delta += 1
            x,y = (cdtcoords[:2,edtvals[i]] + delta*dv).astype(int)
        if delta < radius:
            return [ [x,y], cat[i,1], True ]
        else:
            return [ [x,y], cat[i,1], False ]

# Produce a N x 4 matrix with metadata.
# For each transcript, consider its nearest neighbors
# Count how many neighbors belong to what cells
# IF neighbors include transcripts belonging to more than 1 cell,
# THEN 1st Col: Number of cells to which nearest neighbors belong
#      2nd Col: Index of most popular cell (Cell with the most neighbors)
#      3rd Col: BOOL: Current transcript cell location different from majority of neighbors
#      4th Col: Percentage of neighbors belonging to most popular cell

def get_neighbor_data(nidxs, indexing, minneighs, cdtlabs, cdtmask, cdtcoords, edtmask, edtvals):
    
    cat = np.zeros((len(nidxs),4), dtype=int)

    for i in indexing:
        foo, bar = np.unique(cdtlabs[nidxs[i][1:]], return_counts=True)
        if len(foo) > 1:
            cts = bar/np.sum(bar)
            
            cat[i,0] = len(foo)
            cat[i,1] = foo[np.argmax(cts)]
            cat[i,2] = cat[i,1] != cdtlabs[edtvals[i]]
            cat[i,3] = 100*np.max(cts)
    
    return cat
            
def correct_shifted_transcripts(cdtlabs, cdtmask, cdtcoords, edtmask, edtvals, label, maxdwall=maxdwall, minneighs=minneighs, minprob = minprob):
    neigh = neighbors.NearestNeighbors(radius=radius)
    neigh.fit(cdtcoords.T)
    
    ndist, nidxs = neigh.radius_neighbors(cdtcoords[:, edtmask].T, sort_results=True)
    nneighs = np.array(list(map(len,nidxs))) - 1
    indexing = np.nonzero(nneighs > minneighs)[0]

    cat = get_neighbor_data(nidxs, indexing, minneighs, cdtlabs, cdtmask, cdtcoords, edtmask, edtvals)
    
    indexing = np.nonzero((cat[:,2] == 1) & (cat[:,3] > 70))[0]
    for i in indexing:
        shift = transcript_shift(i, ndist, nidxs, cat, cdtlabs, cdtmask, cdtcoords, edtmask, edtvals, label)
        if shift[2]:
            cdtcoords[:2,edtvals[i]], cdtlabs[edtvals[i]] = shift[0], shift[1]
        else:
            print('Pay attention to index\t',i)

    return len(indexing)

def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('sample', metavar='raw_dir', type=str, help='directory where raw images are located')
    args = parser.parse_args()

    struc1 = ndimage.generate_binary_structure(2,1)
    struc2 = ndimage.generate_binary_structure(2,2)
    theta = np.linspace(-np.pi, np.pi, 50)

    wsrc = '../cell_dams/'
    nsrc = '../nuclear_mask/'
    csrc = '../data/'
    dst = '../translocs/'
    sample = args.sample

    dst += sample + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)
        
    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    nuclei = tf.imread(nsrc + sample + '_EDT.tif') < 2

    edt = ndimage.distance_transform_cdt(wall, 'chessboard')
    label, cellnum = ndimage.label(wall, struc1)
    print('Detected',cellnum,'cells')

    filename = csrc + sample + '_data/32771-slide1_' + sample + '_results.txt'
    data = pd.read_csv(filename, header=None, sep='\t').drop(columns=[4])
    data.columns = ['X', 'Y', 'Z', 'T']

    transcriptomes, invidx, tsize = np.unique(data.iloc[:,-1], return_index = False, return_inverse=True, return_counts=True) 
    print(len(transcriptomes), 'transcriptomes')

    for tidx in range(len(transcriptomes)):
        
        filename =  dst + 'location_corrected_' + sample +'_-_' + transcriptomes[tidx] + '.csv'
        if not os.path.isfile(filename):
            print('----', tidx, ' : \t', transcriptomes[tidx], ':\n')
            tcoords = data.loc[invidx == tidx , ['X', 'Y', 'Z'] ].values.T
            coords = tcoords[:, ~nuclei[ tcoords[1], tcoords[0] ]]
            tlabs = label[coords[1], coords[0] ].astype(int)
            tpercell, _ = np.histogram(tlabs, np.arange(cellnum+2))
            
            # # Deal with transcripts on the edge

            foo = np.sum(tlabs == 0)
            print('Initially, there are\t',foo,'\ttranscripts on the walls',sep='')
            if foo > 0:
                correct_boundary_transcripts(tlabs, coords, label, tpercell, R=5)
                foo = np.sum(tlabs == 0)
                print('Now there are\t',foo,'\ttranscripts on the walls',sep='')

            # # Deal with misplaced transcripts

            foo = 100
            iters = 0

            cdtmask = np.nonzero(edt[coords[1], coords[0]] < radius)[0]
            if np.sum(cdtmask) > 0:
                cdtlabs = tlabs[cdtmask].copy()
                cdtcoords = coords[ :,  cdtmask].copy()
                edtmask = edt[cdtcoords[1], cdtcoords[0]] < maxdwall
                if np.sum(edtmask) > 0:
                    edtvals = np.nonzero(edtmask)[0]
                
                    while (foo  > 0) and (iters < 25):
                        iters += 1    
                        foo = correct_shifted_transcripts(cdtlabs, cdtmask, cdtcoords, edtmask, edtvals, label, maxdwall, minneighs, minprob)
                        print('Iteration: ', iters, '\tShifted\t',foo,' transcripts', sep='')
                    if iters == 25:
                        print('Max number of iterations reached')

                    shiftmask = np.any(cdtcoords != coords[ :,  cdtmask], axis=0)
                    print('Shifted\t',np.sum(shiftmask),'\ttranscripts in total', sep='')

                    # # Save Results

                    coords[:, cdtmask] = cdtcoords
                    coords = coords[:, ~nuclei[coords[1], coords[0]]]
                    
            print('Saved file', filename,'\n')
            df = pd.DataFrame(coords.T)
            #df['foo'] = transcriptomes[tidx]
            df.to_csv(filename, header=False, index=False)

if __name__ == '__main__':
    main()
