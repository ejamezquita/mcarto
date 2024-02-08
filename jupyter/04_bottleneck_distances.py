import numpy as np
import pandas as pd

import gudhi as gd
import json

from glob import glob
import os
import wasserstein as ws

import argparse

def get_diagrams(jsonfiles):
    diagh0 = [ np.empty((0,2)) for i in range(len(jsonfiles)) ]
    diagh1 = [ np.empty((0,2)) for i in range(len(jsonfiles)) ]

    for i in range(len(jsonfiles)):
        with open(jsonfiles[i]) as f:
            diag = [tuple(x) for x in json.load(f)]
        h1mask = np.sum([ diag[j][0] == 1 for j in range(len(diag)) ])
        diagh0[i] = np.asarray( [ x[1] for x in diag[h1mask:] ] )
        if h1mask > 0:
            diagh1[i] = np.asarray( [ x[1] for x in diag[:h1mask] ] )

    return diagh0, diagh1

def bottleneck_matrix(diagh0, diagh1, dfunction=gd.bottleneck_distance, **kwargs):
    bottleneck_h0 = np.zeros( (len(diagh0), len(diagh0)) )
    bottleneck_h1 = np.zeros( (len(diagh0), len(diagh0)) )
    
    for i in range(len(bottleneck_h0) - 1):
        for j in range(i+1, len(bottleneck_h0)):
    
            ## H0 ##
            d = dfunction(diagh0[i], diagh0[j], **kwargs)
            bottleneck_h0[i,j] = d
            bottleneck_h0[j,i] = d
    
            ## H1 ##
            d = dfunction(diagh1[i], diagh1[j], **kwargs)
            bottleneck_h1[i,j] = d
            bottleneck_h1[j,i] = d

    return bottleneck_h0, bottleneck_h1

def save_dmatrix(mtrx, filename):
    N = len(mtrx)
    dflat = mtrx[np.triu_indices(N, k=1)]
    pd.Series(dflat).to_csv(filename, index=False, header=None)
    return dflat

def main():
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('sample', metavar='raw_dir', type=str, help='directory where raw images are located')
    parser.add_argument('level', metavar='name_dir', type=str, help='name of plate origin')
    parser.add_argument('tstart', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('tend', metavar='name_dir', type=int, help='name of plate origin')
    args = parser.parse_args()

    gsrc = '../gd_trans/'
    sample = args.sample
    tsrc = gsrc + sample + '/'
    dst = tsrc
    ws_order = 1
    tstart = args.tstart
    tend = args.tend

    transcriptomes = sorted([foo.split('/')[-2] for foo in glob(tsrc + '*/')])
    print(len(transcriptomes), 'transcriptomes')

    level = args.level

    for tidx in range(tstart, tend):
        print('Checking', transcriptomes[tidx], '\t[{}]'.format(tidx))
        
        jsonfiles = sorted(glob(tsrc + transcriptomes[tidx] + '/*' + level + 'level.json'))
        diag0, diag1 = get_diagrams(jsonfiles)
        
        if len(jsonfiles) == 0:
            print('\n****\nNo JSONs detected for',transcriptomes[tidx], '\t[{}]\n****\n'.format(tidx))
        
        else:
            # 1-Wasserstein
            filename = tsrc + transcriptomes[tidx] + '_-_' + level + 'level_wasserstein{}.csv'.format(ws_order)
            if not os.path.isfile(filename):
                h0, h1 = bottleneck_matrix(diag0, diag1, ws.wasserstein_distance, order=ws_order, keep_essential_parts=True)
                dmatrix = h0 + h1
                _ = save_dmatrix(dmatrix, filename)
                print('Generated',filename)

            # Bottleneck
            filename = tsrc + transcriptomes[tidx] + '_-_' + level + 'level_bottleneck.csv'
            if not os.path.isfile(filename):
                h0, h1 = bottleneck_matrix(diag0, diag1, gd.bottleneck_distance)
                dmatrix = np.maximum(h0, h1)
                _ = save_dmatrix(dmatrix, filename)
                print('Generated',filename)

if __name__ == '__main__':
    main()
