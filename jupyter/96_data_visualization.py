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
        arr[os.pardir.,i] = a
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
    ksrc = os.pardir + os.sep + 'kde' + os.sep
    gsrc = os.pardir + os.sep + 'gd_trans' + os.sep
    sample = args.sample
    selection = 'all'
    distance = 'bottleneck'
    
    wall = tf.imread(os.pardir + os.sep + 'cell_dams' + os.sep + sample + '_dams.tif').astype(bool)
    wall[tf.imread(os.pardir + os.sep + 'nuclear_mask' + os.sep + sample + '_EDT.tif') < 2] = False
    wcoords = np.asarray(np.nonzero(~wall))
    wallshape = wall.shape
    wc = wcoords[:, ~np.all(wcoords%50, axis=0)]

    tsrc = gsrc + sample + '' + os.sep

    ksrc += sample + os.sep
    dst = os.pardir + os.sep + 'distances' + os.sep
    dst += sample + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv')
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv')
    transcriptomes = sorted([foo.split('' + os.sep)[-2] for foo in glob(ksrc + '*' + os.sep)])
    print(len(transcriptomes), 'transcriptomes')
    dbscan_params = {'eps':10, 'min_samples':3}


    for tidx in np.sum(transcell.iloc[:,1:-1] > 5, axis=1).sort_values(ascending=False).index[4:15]:
    #for tidx in []:
        
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
            data[key]['PCA1'] = metakde.set_index('ndimage_cellID').loc[cidxs, 'kdemax_all'].values

            data[key]['ndimage_ID'] = cidxs
            data[key] = data[key].join(metacell.loc[cidxs, ['orig_cellID', 'orig_comX', 'orig_comY']].reset_index(drop=True))
            data[key] = data[key].iloc[:, [4,5,6,7,0,1,3,2]]

        for key in meta:
            meta[key]['vmin'] = np.min(data[key][data[key]['DBSCAN'] > -1 ]['PCA1'])
            meta[key]['vmax'] = np.max(data[key][data[key]['DBSCAN'] > -1 ]['PCA1'])


        filename = dst + sample + '_-_' + transcriptomes[tidx] + '_-_' + distance + '_distance.jpg'
        if not os.path.isfile(filename):
            fig, ax = plt.subplots(2,2, figsize=(10,12))
            ax = np.atleast_1d(ax).ravel()

            for i,key in enumerate(levels):
                ax[2*i].scatter(wc[1], wc[0], c='gray', marker='.', s=0.1, zorder=1, alpha=1)
                
                for j in range(len(meta[key]['labels'])):
                    label = meta[key]['labels'][j]
                    foo = data[key][data[key]['DBSCAN'] == label]
                    ax[2*i].scatter(foo['orig_comX'], foo['orig_comY'], s=50, edgecolor='k',
                                    vmax=meta[key]['vmax'], vmin=meta[key]['vmin'],
                                  c=foo['PCA1'], marker=marker[label], cmap='plasma', alpha=alpha, zorder = label+3)
                    ax[2*i+1].scatter(foo['MDS1'], foo['MDS2'], s=50, alpha=alpha, vmax=meta[key]['vmax'], vmin=meta[key]['vmin'],
                                  c=foo['PCA1'], marker=marker[label], edgecolor='k', cmap='plasma')
                    
                
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
            
        vmax = np.max(metakde['kdemax_' + selection])
        bins = np.linspace(0, vmax, 256)

        key = 'sublevel'
        psrt = data[key]['PCA1'].sort_values().index
        
        def get_plot_info(idx):
            scatter = data[key].iloc[psrt[:idx+1], :].values
            cidx = int(scatter[-1,0])
            
            non = data[key].iloc[psrt[idx+1:], :].values
            kdemeta = metakde[metakde['ndimage_cellID'] == cidx]
            PP = kdemeta['pad'].iloc[0]
            stepsize = kdemeta['stepsize'].iloc[0]
            ss = (np.s_[max([0, metacell.loc[cidx, 'y0'] - PP]) : min([wallshape[0], metacell.loc[cidx, 'y1'] + PP])],
                  np.s_[max([1, metacell.loc[cidx, 'x0'] - PP]) : min([wallshape[1], metacell.loc[cidx, 'x1'] + PP])])
            extent = (ss[1].start, ss[1].stop, ss[0].start, ss[0].stop)
            cy = ss[0].stop - ss[0].start
            cx = ss[1].stop - ss[1].start
            xaxis = np.arange(0, cx, stepsize); yaxis = np.arange(0, cy, stepsize)
            xaxis += ss[1].start; yaxis += ss[0].start
            kde = np.load(kdefiles[kdemeta.index[0]], allow_pickle=True)*kdemeta['ratio_' + selection].iloc[0]
            cp = cartesian_product(np.arange(yaxis[0],yaxis[-1]), np.arange(xaxis[0],xaxis[-1]))
            interp = interpolate.RegularGridInterpolator((yaxis, xaxis), kde, method='linear', bounds_error=True, fill_value=None)
            img = interp(cp)
            img = np.digitize(img, bins, right=True).astype(np.uint8).reshape(yaxis[-1]-yaxis[0],xaxis[-1]-xaxis[0])

            return scatter, non, img, extent
                    
        
        for idx in range(len(kdefiles)):
                fig, ax = plt.subplots(1,3, figsize=(10,4))
                ax = np.atleast_1d(ax).ravel()
                scatter, non, img, extent = get_plot_info(idx)
                cidx = int(scatter[-1,0])
                
                wmask = (wcoords[1] > extent[0]) & (wcoords[1] < extent[1]) & (wcoords[0] > extent[2]) & (wcoords[0] < extent[3])
                
                ax[0].scatter(wc[1], wc[0], c='darkgray', marker='.', s=0.1, zorder=1, alpha=1)
                ax[0].scatter(scatter[:-1,2], scatter[:-1,3], c = scatter[:-1,6], marker = 'D', cmap='plasma',
                              edgecolor='k', vmax=meta[key]['vmax'], vmin=meta[key]['vmin']);
                ax[0].scatter(scatter[-1,2], scatter[-1,3], c = 'lime', marker = 's', s=50, edgecolor='k');
                ax[1].scatter(scatter[:-1,4], scatter[:-1,5], c = scatter[:-1,6], marker = 'D', cmap='plasma',
                              edgecolor='k', vmax=meta[key]['vmax'], vmin=meta[key]['vmin']);
                ax[1].scatter(non[:,4], non[:,5], alpha=0, marker='D')
                ax[1].scatter(scatter[-1,4], scatter[-1,5], c = 'lime', marker = 's', s=50, edgecolor='k');
                
                ax[2].imshow(img, cmap='plasma', origin='lower', vmin=0, vmax=255, extent=extent)
                ax[2].set_xlim((extent[0], extent[1])); ax[2].set_ylim((extent[2], extent[3]))
                ax[2].scatter(wcoords[1, wmask], wcoords[0, wmask], c='darkgray', marker='.', s=1, zorder=2, alpha=1)
                
                for i in range(len(ax)):
                    ax[i].set_aspect('equal', 'datalim');
                    ax[i].set_facecolor('snow')
                    ax[i].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

                ax[0].set_title(sample + ' - ' + transcriptomes[tidx], fontsize=fs)
                ax[1].set_title('{} cells in total'.format(len(kdefiles)), fontsize=fs)
                ax[2].set_title('Cell ID {} [{}]'.format(metacell.loc[cidx, 'orig_cellID'], cidx), loc='left', fontsize=fs)

                ax[0].set_xlabel('Nodal cross section', fontsize=fs)
                ax[1].set_xlabel('MDS of {} distance'.format(distance), fontsize=fs)
                ax[2].set_xlabel('KDE and {} pers.'.format(key), fontsize=fs)
                
                fig.tight_layout()
                
                filename = ddst + transcriptomes[tidx] + '_-_{}_{}_{:05d}_c{}.png'.format(key, distance, idx, cidx)
                plt.savefig(filename, format='png', dpi=96, bbox_inches='tight')
                plt.close()
            
if __name__ == '__main__':
    main()
