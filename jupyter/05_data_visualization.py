import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gudhi as gd
import tifffile as tf
import json

from glob import glob
import os

from scipy import ndimage, interpolate
from sklearn import manifold, cluster, decomposition, preprocessing

import argparse

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def get_MDS(filebase, bottle, mds_params):
    mds = dict()
    for key in bottle.keys():
        filename = filebase + '_' + key + '_MDS.csv' 
        if not os.path.isfile(filename):
            MDS = manifold.MDS(**mds_params).fit_transform(bottle[key])
            np.savetxt(filename, MDS, delimiter=',', fmt='%.6f')
        else:
            MDS = np.loadtxt(filename, delimiter=',')
        mds[key] = MDS
    return mds

def main():
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('sample', metavar='raw_dir', type=str, help='directory where raw images are located')
    args = parser.parse_args()
    
    alpha = 1
    fs = 16
    N = 10
    marker = ['D', 'o', 's', '^', 'v', 'P']
    marker = dict(zip(range(N*len(marker)), np.tile(marker,N)))
    marker[-1] = '*'

    levels = ['sublevel', 'superlevel']
    ksrc = '../kde/'
    gsrc = '../gd_trans/'
    sample = args.sample
    selection = 'all'
    distance = 'bottleneck'
    
    wall = tf.imread('../cell_dams/' + sample + '_dams.tif').astype(bool)
    wall[tf.imread('../nuclear_mask/' + sample + '_EDT.tif') < 2] = False
    wcoords = np.asarray(np.nonzero(~wall))
    wallshape = wall.shape
    wc = wcoords[:, ~np.all(wcoords%50, axis=0)]

    tsrc = gsrc + sample + '/'

    ksrc += sample + os.sep
    dst = '../distances/'
    dst += sample + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv')
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv')
    transcriptomes = sorted([foo.split('/')[-2] for foo in glob(ksrc + '*/')])
    print(len(transcriptomes), 'transcriptomes')
    dbscan_params = {'eps':10, 'min_samples':3}


    for tidx in np.sum(transcell.iloc[:,1:-1] > 5, axis=1).sort_values(ascending=False).index[:25]:

        ddst = dst + transcriptomes[tidx] + os.sep
        if not os.path.isdir(ddst):
            os.mkdir(ddst)

        kdefiles = glob(ksrc + transcriptomes[tidx] + os.sep + '*.npy')
        ratios = transcell.iloc[tidx, :-1].values.astype(float) / metatrans.loc[tidx, 'cyto_number']
        metakde = np.zeros((len(kdefiles), 4), dtype=int)
        rawkdemax = np.zeros(len(metakde))
        for i in range(len(kdefiles)):
            foo = (os.path.splitext(os.path.split(kdefiles[i])[1])[0]).split('_')
            for j in range(metakde.shape[1]):
                metakde[i,j] = int(foo[j][1:])
            rawkdemax[i] = float(foo[-1][1:])
        metakde = pd.DataFrame(metakde, columns=['ndimage_cellID', 'pad', 'stepsize', 'bandwidth'])
        metakde['rawkdemax'] = rawkdemax
        metakde['transnum'] = transcell.iloc[tidx, metakde['ndimage_cellID'].values].values.astype(int)
        metakde['ratio_all'] = ratios[metakde['ndimage_cellID'].values]
        metakde['ratio_select'] = metakde['transnum']/np.sum(metakde['transnum'])
        metakde['kdemax_all'] = metakde['rawkdemax']*metakde['ratio_all']
        metakde['kdemax_select'] = metakde['rawkdemax']*metakde['ratio_select']
        metakde = metakde[metakde['ndimage_cellID'] != 0]

        print('Working with',transcriptomes[tidx])

        bottle = dict()
        for key in levels:
            filename = tsrc + transcriptomes[tidx] + '_-_' + key + '_' + distance + '.csv'
            if not os.path.isfile(filename):
                N = 0
            else:        
                dflat = np.squeeze(pd.read_csv(filename, header=None).values)
                N = int(1 + np.sqrt(1 + 8*len(dflat)))//2
                A = np.zeros((N,N))
                A[np.triu_indices(N,k=1)] = dflat
                A += A.T
                bottle[key] = A
        print('Working with',N,'cells')

        jsonfiles = sorted(glob(tsrc + transcriptomes[tidx] + '/*' + key + '.json'))
        cidxs = np.array([int(jsonfiles[i].split('_')[-4][1:]) for i in range(len(jsonfiles))])

        mds_params = {'n_components':2, 'metric':True, 'random_state':42, 'dissimilarity':'precomputed', 'normalized_stress':False}
        filebase = dst + transcriptomes[tidx] + '_' + distance 
        mds = get_MDS(filebase, bottle, mds_params)

        filename = dst + sample + '_-_' + transcriptomes[tidx] + '_-_' + distance + '_distance.jpg'
        if not os.path.isfile(filename):
            
            clust = dict()
            for key in mds.keys():
                clust[key] = cluster.DBSCAN(**dbscan_params).fit(mds[key][:,:2])

            meta = dict()
            for key in levels:
                labs = clust[key].labels_
                ulabs, cts = np.unique(labs, return_counts=True)
                if -1 in ulabs:
                    ulabs = np.hstack(([-1], ulabs[1:][np.argsort(cts[1:])[::-1]]))
                else:
                    ulabs = ulabs[np.argsort(cts)[::-1]]
                meta[key] = {'labels': ulabs}
            data = dict()
            for key in mds.keys():
                data[key] = pd.DataFrame(mds[key], columns=['MDS1', 'MDS2'])
                labs = np.full_like(clust[key].labels_, -2)
                for i,lab in enumerate(range(min(meta[key]['labels']), max(meta[key]['labels'])+1)):
                    labs[clust[key].labels_ == lab] = meta[key]['labels'][i]
                
                data[key]['DBSCAN'] = labs
                
                scaler = preprocessing.StandardScaler().fit(mds[key][data[key]['DBSCAN'] > -1])
                train = scaler.transform(mds[key])
                #PCA = decomposition.KernelPCA(n_components=1, random_state=42, kernel='rbf', gamma=0.1).fit(train[data[key]['DBSCAN'] > -1])
                #PCA = decomposition.PCA(n_components=1, random_state=42, whiten=False).fit(train[data[key]['DBSCAN'] > -1])
                #colors = PCA.transform(train).squeeze()
                data[key]['PCA1'] = metakde.set_index('ndimage_cellID').loc[cidxs, 'kdemax_all'].values

                data[key]['ndimage_ID'] = cidxs
                data[key] = data[key].join(metacell.loc[cidxs, ['orig_cellID', 'orig_comX', 'orig_comY']].reset_index(drop=True))
                data[key] = data[key].iloc[:, [4,5,6,7,0,1,3,2]]

            for key in meta:
                meta[key]['vmin'] = np.min(data[key][data[key]['DBSCAN'] > -1 ]['PCA1'])
                meta[key]['vmax'] = np.max(data[key][data[key]['DBSCAN'] > -1 ]['PCA1'])


            fig, ax = plt.subplots(2,2, figsize=(10,12))
            ax = np.atleast_1d(ax).ravel()

            for i,key in enumerate(levels):
                ax[2*i].scatter(wc[1], wc[0], c='gray', marker='.', s=0.1, zorder=1, alpha=1)
                
                for j in range(len(meta[key]['labels'])):
                    label = meta[key]['labels'][j]
                    foo = data[key][data[key]['DBSCAN'] == label]
                    ax[2*i].scatter(foo['orig_comX'], foo['orig_comY'], s=50, edgecolor='k',
                                    vmax=meta[key]['vmax'], vmin=meta[key]['vmin'],
                                  c=foo['PCA1'], marker=marker[j-1], cmap='plasma', alpha=alpha, zorder = label+3)
                    ax[2*i+1].scatter(foo['MDS1'], foo['MDS2'], s=50, alpha=alpha, vmax=meta[key]['vmax'], vmin=meta[key]['vmin'],
                                  c=foo['PCA1'], marker=marker[j-1], edgecolor='k', cmap='plasma')
                    
                
                ax[2*i].set_title(key.title() + ' set persistence', fontsize=fs)
            
            for i in range(len(ax)):
                ax[i].set_aspect('equal', 'datalim');
                ax[i].margins(0)
                ax[i].set_facecolor('snow')
                ax[i].tick_params(labelbottom=False, labelleft=False)
                
            fig.supxlabel('MDS 1', fontsize=fs)
            fig.supylabel('MDS 2', fontsize=fs)

            title = sample + ' - ' + transcriptomes[tidx]
            title += ': {} cells into {} clusters (DBSCAN [$\\varepsilon = $ {}])'.format(len(kdefiles), len(ulabs)-1, dbscan_params['eps'])
            fig.suptitle(title, fontsize=fs)

            fig.tight_layout();

            filename = dst + sample + '_-_' + transcriptomes[tidx] + '_-_' + distance + '_distance.jpg'
            plt.savefig(filename, format='jpg', dpi=96, bbox_inches='tight', pil_kwargs={'optimize':True})
            plt.close()
            
if __name__ == '__main__':
    main()
