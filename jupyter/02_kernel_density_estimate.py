import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tifffile as tf
from glob import glob
import os
from scipy import ndimage, spatial, stats

from KDEpy import FFTKDE
from KDEpy.bw_selection import improved_sheather_jones as ISJ
import argparse

struc1 = ndimage.generate_binary_structure(2,1)
struc2 = ndimage.generate_binary_structure(2,2)
pows2 = 2**np.arange(20) + 1
bdmatrix = ndimage.generate_binary_structure(2,1).astype(int)
bdmatrix[1,1] = 1 - np.sum(bdmatrix)
fs = 15
bw = 15
PP = 6
stepsize = 5
pp = 0

wsrc = '../cell_dams/'
nsrc = '../nuclear_mask/'
psrc = '../proc/'
osrc = '../data/'
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

def generate_transcript_metadata(translocs, transcriptomes, invidx, orig_size, tsize):
    isj = np.zeros((len(transcriptomes), 2))
    for tidx in range(len(isj)):
        coords = translocs.loc[invidx == tidx , ['X', 'Y'] ].values.T
        for i in range(len(coords)):
            isj[tidx, i] = ISJ(coords[ i ].reshape(-1,1))
    meta = pd.DataFrame(isj, columns=['ISJ1', 'ISJ2'])
    meta['total_number'] = orig_size
    meta['cyto_number'] = tsize
    meta['nuclei_number'] = orig_size - tsize
    meta['ratio'] = tsize/orig_size
    meta['gene'] = transcriptomes
    
    return meta

def generate_transcell_metadata(translocs, transcriptomes, cellnum, invidx, label):
    meta = np.zeros((len(transcriptomes), cellnum), dtype=int)
    bins = np.arange(1, cellnum + 2)
    for tidx in range(len(meta)):
        coords = translocs.loc[invidx == tidx , ['X', 'Y'] ].values.T
        meta[tidx], _ = np.histogram(label[coords[1], coords[0]], bins=bins)
        
    meta = pd.DataFrame(meta, columns=bins[:-1])
    meta['gene'] = transcriptomes

    return meta

def kde_grid_generator(stepsize, cellshape, pows2, pad=1.5):
    cy,cx = cellshape
    #cx += int(np.where( cx % stepsize, cx%stepsize, stepsize))
    #cy += int(np.where( cy % stepsize, cy%stepsize, stepsize))
    
    xaxis = np.arange(0, cx, stepsize); yaxis = np.arange(0, cy, stepsize)
    mx = np.nonzero(pows2 > pad*len(xaxis))[0][0]; my = np.nonzero(pows2 > pad*len(yaxis))[0][0]
    
    foo = pows2[mx] - len(xaxis)
    xneg = foo//2
    xpos = np.where(foo%2==0, foo//2, foo//2 + 1) + 0
    
    foo = pows2[my] - len(yaxis)
    yneg = foo//2
    ypos = np.where(foo%2==0, foo//2, foo//2 + 1) + 0
    
    xaxes = np.hstack((np.arange(-xneg, 0, 1)*stepsize, xaxis, np.arange(len(xaxis), len(xaxis)+xpos, 1)*stepsize))
    yaxes = np.hstack((np.arange(-yneg, 0, 1)*stepsize, yaxis, np.arange(len(yaxis), len(yaxis)+ypos, 1)*stepsize))
    
    XX, YY = np.meshgrid(xaxes, yaxes)
    
    grid = np.column_stack((np.ravel(XX, 'F'), np.ravel(YY, 'F')))
    mask = (grid[:,0] >= 0) & (grid[:,0] < cx) & (grid[:,1] >= 0) & (grid[:,1] < cy)
    return xaxis, yaxis, grid, mask

def cardinal_distance_transform(img):
    PAD = 1
    pss = np.s_[PAD:-PAD,PAD:-PAD]
    pad = np.pad(img, PAD, constant_values=0)
    initd = np.full(pad.shape, max(pad.shape)+1, dtype=int)
    initd[~pad] = 0
    left = np.copy(initd)
    for j in range(1,pad.shape[1]):
        left[:, j] = np.minimum(left[:, j], left[:, j-1] + 1)
    right = np.copy(initd)
    for j in range(pad.shape[1]-2, -1, -1):
        right[:, j] = np.minimum(right[:, j], right[:, j+1] + 1)
    bottom = np.copy(initd)
    for j in range(1,pad.shape[0]):
        bottom[j] = np.minimum(bottom[j], bottom[j-1] + 1)
    top = np.copy(initd)
    for j in range(pad.shape[0]-2, -1, -1):
        top[j] = np.minimum(top[j], top[j+1] + 1)

    return top[pss], right[pss], bottom[pss], left[pss]
    
def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('sample', metavar='raw_dir', type=str, help='directory where raw images are located')
    parser.add_argument('cstart', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('cfinish', metavar='raw_dir', type=int, help='directory where raw images are located')
    args = parser.parse_args()

    sample = args.sample
    cstart = args.cstart
    cfinish = args.cfinish
    
    dst = '../kde/'
    dst = dst + sample + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)

    # # Load all general data

    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    edt = tf.imread(nsrc + sample + '_EDT.tif')
    nuclei = edt < 2
    label, cellnum = ndimage.label(wall, struc1)
    wall[nuclei] = False
    print('Detected',cellnum,'cells')

    filename = dst + 'location_corrected_' + sample + '.csv'
    translocs = pd.read_csv(filename, header=None)
    translocs.columns = ['X', 'Y', 'T']

    transcriptomes, invidx, tsize = np.unique(translocs.iloc[:,-1], return_index = False, return_inverse=True, return_counts=True) 
    print(len(transcriptomes), 'transcriptomes')

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

    metacell = pd.read_csv(filename)

    filename = dst + sample + '_transcripts_metadata.csv'
    if not os.path.isfile(filename):
        data = pd.read_csv(osrc + sample + '_data/32771-slide1_' + sample + '_results.txt', header=None, sep='\t').drop(columns=[4])
        _, orig_size = np.unique(data.iloc[:,-1], return_index = False, return_inverse=False, return_counts=True) 
        meta = generate_transcript_metadata(translocs, transcriptomes, invidx, orig_size, tsize)
        meta.to_csv(filename, index=False)

    metatrans = pd.read_csv(filename)

    filename = dst + sample + '_transcells_metadata.csv'
    if not os.path.isfile(filename):
        meta = generate_transcell_metadata(translocs, transcriptomes, cellnum, invidx, label)
        meta.to_csv(filename, index=False)
    transcell = pd.read_csv(filename)
    
    # # Compute weights
    
    top, right, bottom, left = cardinal_distance_transform(wall)
    
    wv = stats.norm.cdf(top[translocs['Y'].values, translocs['X'].values]+pp, loc=0, scale=bw)
    wv-= stats.norm.cdf(-bottom[translocs['Y'].values, translocs['X'].values]-pp, loc=0, scale=bw)
    wh = stats.norm.cdf(right[translocs['Y'].values, translocs['X'].values]+pp, loc=0, scale=bw) 
    wh-= stats.norm.cdf(-left[translocs['Y'].values, translocs['X'].values]-pp, loc=0, scale=bw)

    weight = 2-(wv*wh)

    # # Select a transcript and a cell

    for cidx in range(cstart,cfinish):
        #ratios = transcell.iloc[:, cidx]/metatrans['cyto_number']
        ss = (np.s_[max([0, metacell.loc[cidx, 'y0'] - PP]) : min([wall.shape[0], metacell.loc[cidx, 'y1'] + PP])],
              np.s_[max([1, metacell.loc[cidx, 'x0'] - PP]) : min([wall.shape[1], metacell.loc[cidx, 'x1'] + PP])])
        extent = (ss[1].start, ss[1].stop, ss[0].start, ss[0].stop)
        cell = wall[ss].copy().astype(np.uint8)
        cell[ label[ss] == cidx+1 ] = 2
        cell[~wall[ss]] = 0
        
        # # Prepare the KDE grid

        xaxis, yaxis, grid, gridmask = kde_grid_generator(stepsize, cell.shape, pows2)

        grid = grid + np.array([ss[1].start, ss[0].start])
        cgrid = grid[gridmask] - np.array([ss[1].start, ss[0].start])
        cgridmask = cell[cgrid[:,1],cgrid[:,0]] != 2
                    
        X,Y = np.meshgrid(xaxis, yaxis)
        xaxis += ss[1].start; yaxis += ss[0].start    

        for tidx in np.nonzero(transcell.iloc[:, cidx].values > 5)[0]:
            kdst = dst + transcriptomes[tidx] + os.sep
            if not os.path.isdir(kdst):
                os.mkdir(kdst)
            #filename = kdst + 'c{}_p{}_s{}_b{}.jpg'.format(cidx, PP, stepsize, bw)
            filename = 'bar.txt'
            if not os.path.isfile(filename):
                tmask = invidx == tidx
            
                coords = translocs.loc[ tmask , ['X', 'Y'] ].values.T
                cmask = label[ coords[1], coords[0] ] == cidx + 1
                ccoords = coords[:, cmask ].copy()
                
                w = weight[tmask][cmask]
                kde = FFTKDE(kernel='gaussian', bw=bw, norm=2).fit(ccoords.T, w).evaluate(grid)
                kde = kde[gridmask]/(np.sum(kde[gridmask])*stepsize*stepsize)
                kde[ cgridmask ] = 0
                kde = kde/(np.sum(kde)*stepsize*stepsize)
                kde = kde.reshape( ( len(yaxis), len(xaxis) ), order='F')
                maxkde = np.max(kde)
                
                #ratio = ccoords.shape[1]/coords.shape[1]
                
                # # Save results
                
                if verbose:
                
                    uwkde = FFTKDE(kernel='gaussian', bw=bw, norm=2).fit(ccoords.T).evaluate(grid)
                    uwkde = uwkde[gridmask]/(np.sum(uwkde[gridmask])*stepsize*stepsize)
                    uwkde = uwkde.reshape( ( len(yaxis), len(xaxis) ), order='F')
                    maxuwkde = np.max(uwkde)
                    
                    diff = kde - uwkde
                    maxd = np.max(np.abs(diff))/2
                    vmax = max([maxkde, maxuwkde])

                    fig, ax = plt.subplots(1,4,figsize=(13,7), sharex=True, sharey=True)
                    
                    ax[0].imshow(cell, cmap='plasma', origin='lower', extent=extent)
                    ax[0].scatter(ccoords[0], ccoords[1], c=w, cmap='copper', marker='o', zorder=3, s=20, alpha=1); i = 1
                    for density in [uwkde,kde]:
                        ax[i].contourf(xaxis,yaxis, density, 16, cmap='plasma', vmin=0, vmax=vmax, zorder=1)
                        i += 1
                    ax[3].contourf(xaxis, yaxis, diff, 32, cmap='coolwarm', vmin=-maxd, vmax=maxd)
                    for i in [1,2,3]:
                        ax[i].scatter(ccoords[0], ccoords[1], c=w, cmap='copper', marker='*', s=1, alpha=1, zorder=3)
                    for i in range(len(ax)):
                        ax[i].set_aspect('equal')
                        ax[i].margins(0)
                        
                    ax[0].set_xlabel('Original', fontsize=fs)
                    ax[1].set_xlabel('KDE', fontsize=fs)
                    ax[2].set_xlabel('Weighted & Cropped', fontsize=fs)
                    ax[3].set_xlabel('Difference', fontsize=fs)

                    ax[0].set_title('{} ({})'.format(sample, len(ccoords[0])), fontsize=fs)
                    ax[1].set_title(transcriptomes[tidx], fontsize=fs)
                    ax[2].set_title('Cell ID: {} [{}]'.format(metacell.loc[cidx, 'orig_cellID'], cidx), fontsize=fs)
                    ax[3].set_title('gs = {}; bw = {}'.format(stepsize, bw), fontsize=fs)

                    fig.tight_layout();
                    filename = kdst + 'c{}_p{}_s{}_b{}.jpg'.format(cidx, PP, stepsize, bw)
                    plt.savefig(filename, format='jpg', dpi=96, bbox_inches='tight', pil_kwargs={'optimize':True})
                    plt.close()

                meta = [cidx, PP, stepsize, bw, maxkde]
                filename = kdst + 'c{}_p{}_s{}_b{}_m{:.25E}.npy'.format(*meta)
                np.save(filename, kde)
                print('Generated', filename)
                    
if __name__ == '__main__':
    main()
