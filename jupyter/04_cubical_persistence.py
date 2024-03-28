import numpy as np
import pandas as pd
from glob import glob
import os
import gudhi as gd
import json
import argparse
from scipy import ndimage, interpolate

maximgsize = 1255

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('sample', metavar='raw_dir', type=str, help='directory where raw images are located')
    parser.add_argument('selection', metavar='raw_dir', type=str, help='directory where raw images are located')
    parser.add_argument('initt', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('endt', metavar='raw_dir', type=int, help='directory where raw images are located')
    
    args = parser.parse_args()

    sample = args.sample
    selection = args.selection
    
    ksrc = '../kde/'
    dst = '../gd_trans/'
    initt = args.initt
    endt = args.endt

    ksrc += sample + os.sep
    dst = dst + sample + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv')
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv')
    transcriptomes = sorted([foo.split('/')[-2] for foo in glob(ksrc + '*/')])
    wallshape = (np.max(metacell['x1']), np.max(metacell['x1']))
    print(len(transcriptomes), 'transcriptomes')

    for tidx in range(initt, endt):
        
        sdst = dst + transcriptomes[tidx] + os.sep
        if not os.path.isdir(sdst):
            os.mkdir(sdst)

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
        
        vmax = np.max(metakde['kdemax_' + selection])
        bins = np.linspace(0, vmax, 256)
        counter = 0
        tcounter = len(metakde.index)

        for idx in metakde.index:
            counter += 1
            jsname = '_'.join(np.asarray(os.path.splitext(os.path.split(kdefiles[idx])[1])[0].split('_'))[[0,3]])
            filename = sdst + transcriptomes[tidx] + '_' + jsname + '_' + selection + '_sublevel.json'
            
            if not os.path.isfile(filename):
                cidx = metakde.loc[idx,'ndimage_cellID']
                kde = np.load(kdefiles[idx], allow_pickle=True)*metakde.loc[idx, 'ratio_' + selection]

                PP = metakde.loc[idx, 'pad']
                stepsize = metakde.loc[idx, 'stepsize']
                ss = (np.s_[max([0, metacell.loc[cidx, 'y0'] - PP]) : min([wallshape[0], metacell.loc[cidx, 'y1'] + PP])],
                      np.s_[max([1, metacell.loc[cidx, 'x0'] - PP]) : min([wallshape[1], metacell.loc[cidx, 'x1'] + PP])])
                cy = ss[0].stop - ss[0].start
                cx = ss[1].stop - ss[1].start
                xaxis = np.arange(0, cx, stepsize); yaxis = np.arange(0, cy, stepsize)

                cp = cartesian_product(np.arange(yaxis[0],yaxis[-1]), np.arange(xaxis[0],xaxis[-1]))
                interp = interpolate.RegularGridInterpolator((yaxis, xaxis), kde, method='linear', bounds_error=True, fill_value=None)
                img = interp(cp)
                img = np.digitize(img, bins, right=True).astype(np.uint8).reshape(yaxis[-1]-yaxis[0],xaxis[-1]-xaxis[0])
                img = np.pad(img, 5)
                
                if max(img.shape) > maximgsize:
                    zoom = maximgsize/max(img.shape)
                    print(kdefiles[idx], '\nResized', img.shape, 'by a factor of ', zoom)
                    img = ndimage.zoom(img, zoom = zoom, order=1, mode='reflect')
                    print('Now', img.shape, '\n----')

                cc = gd.CubicalComplex(top_dimensional_cells = img)
                pers = cc.persistence(homology_coeff_field=2, min_persistence=1)
                filename = sdst + transcriptomes[tidx] + '_' + jsname + '_' + selection + '_sublevel.json'
                with open(filename, 'w') as f:
                    json.dump(pers,f)

                maxv = img.max()
                cc = gd.CubicalComplex(top_dimensional_cells = maxv-img)
                pers = cc.persistence(homology_coeff_field=2, min_persistence=1)
                filename = sdst + transcriptomes[tidx] + '_' + jsname + '_' + selection + '_superlevel.json'
                with open(filename, 'w') as f:
                    json.dump(pers,f)
                    
                print('Processed',transcriptomes[tidx], jsname, '\t{}/{}'.format(counter,tcounter))

if __name__ == '__main__':
    main()
