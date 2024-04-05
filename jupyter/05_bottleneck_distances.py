import numpy as np
import pandas as pd
import json
import os
import persim
import argparse

def pers2numpy(pers):
    bd = np.zeros((len(pers), 3), dtype=float)
    for i in range(len(bd)):
        bd[i, 0] = pers[i][0]
        bd[i, 1:] = pers[i][1]
    return bd

def get_diagrams(jsonfiles, ndims, remove_inf = False):
    # diag[j-th cell][k-th dimension]
    diags = [ [np.empty((0,2)) for k in range(ndims)] for j in range(len(jsonfiles))]

    for j in range(len(jsonfiles)):
        
        if jsonfiles[j] is not None:
            with open(jsonfiles[j]) as f:
                diag = [tuple(x) for x in json.load(f)]
            diag = pers2numpy(diag)
        
            for k in range(ndims):
                diags[j][k] = diag[diag[:,0] == k, 1:]
    
    if remove_inf:
        for j in range(len(diags)):
            for k in range(ndims):
                diags[j][k]  = np.atleast_2d(diags[j][k][np.all(diags[j][k] < np.inf, axis=1), :].squeeze())

    return diags

stepsize, PP, bw = 3,6,10
ndims = 3
minlife = 8
wsrc = '../cell_dams/'
nsrc = '../nuclear_mask/'
tsrc = '../translocs/'


def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('sample', metavar='raw_dir', type=str, help='directory where raw images are located')
    parser.add_argument('level', metavar='raw_dir', type=str, help='directory where raw images are located')
    parser.add_argument('normtype', metavar='raw_dir', type=str, help='directory where raw images are located')
    parser.add_argument('cstart', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('cfinish', metavar='raw_dir', type=int, help='directory where raw images are located')
    args = parser.parse_args()

    sample = args.sample
    level = args.level
    initrow = args.cstart
    endrow = args.cfinish
    normtype = args.normtype

    gsrc = '../{}level/'.format(level)
    ksrc = '../kde/'
    
    ksrc += sample + os.sep
    gsrc += sample + os.sep

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv')
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv')
    transcriptomes = np.asarray(list(metatrans['gene']))

    TT = ['GLYMA_05G092200','GLYMA_17G195900']
    tidxs = np.array([np.argwhere(transcriptomes == TT[i])[0][0] for i in range(len(TT))])

    dst = '../distance/{}/{}_vs_{}_{}/'.format(sample, *transcriptomes[tidxs], normtype)
    if not os.path.isdir(dst):
        os.mkdir(dst)

    focus = pd.read_csv('../data/D2_data/scattersutton.csv')
    focus = focus[ focus['Bact'] == 'Infected' ]
    print(focus.shape)

    metafocus = np.zeros( (len(focus),1+metacell.shape[1]) )
    for i in range(len(metafocus)):
        foo = metacell[metacell['orig_cellID'] == focus.iloc[i,0]].iloc[0,:]
        metafocus[i,0] = foo.name
        metafocus[i,1:] = foo.values

    transfocus = transcell.loc[tidxs, metacell.loc[metafocus[:,0].astype(int), 'ndimage_cellID'].values.astype(str)]
    if   normtype == 'both':
        ratios = transfocus.values/np.sum(transfocus.values, axis=None)
    elif normtype == 'cell':
        ratios = transfocus.values/np.sum(transfocus.values, axis=0)
    elif normtype == 'gene':
        ratios = transfocus.values/np.sum(transfocus.values, axis=1).reshape(-1,1)
    jsonfiles = [ [ None for j in range(ratios.shape[1]) ] for i in range(ratios.shape[0]) ]

    for i in range(len(jsonfiles)):
        foo = '{}{}/{}_-_{}level_p{}_s{}_bw{}_c{:06d}.json'
        for j in range(len(metafocus)):
            filename = foo.format(gsrc, transcriptomes[tidxs[i]],transcriptomes[tidxs[i]],level,PP,stepsize,bw,int(metafocus[j,0]))
            if os.path.isfile(filename):
                jsonfiles[i][j] = filename

    orig_diags = [None for i in range(len(jsonfiles))]

    for i in range(len(orig_diags)):
        orig_diags[i] = get_diagrams(jsonfiles[i], ndims, remove_inf=True)
        print(i, len(orig_diags), len(orig_diags[i]), len(orig_diags[i][0]), sep='\t')

    maxk = np.zeros(ndims)
    maxlife = np.zeros((len(orig_diags), len(orig_diags[0]), len(orig_diags[0][0])))

    for i in range(len(orig_diags)):
        for j in range(len(orig_diags[i])):
            for k in range(len(orig_diags[i][j])):
                orig_diags[i][j][k] *= ratios[i][j]
                if len(orig_diags[i][j][k]) > 0:
                    maxlife[i,j,k] = orig_diags[i][j][k][0,1] - orig_diags[i][j][k][0,0]
                    if maxk[k] < np.max(orig_diags[i][j][k]):
                        maxk[k] = np.max(orig_diags[i][j][k])
    
    maxx = np.max(maxk)
    rescale = 256/maxx
    maxlife *= rescale
    argmaxlife = np.argmax(maxlife, axis=-1)
    
    numpairs,reduced = 0,0
    diags = [ [ [ rescale*orig_diags[i][j][k].copy()  for k in range(ndims) ] for j in range(len(orig_diags[i])) ] for i in range(len(orig_diags)) ]

    for i in range(len(diags)):
        for j in range(len(diags[i])):
            for k in range(len(diags[i][j])):
                diags[i][j][k] = np.atleast_2d(diags[i][j][k][np.diff(diags[i][j][k]).squeeze() > minlife, :].squeeze())
                numpairs += len(diags[i][j][k])
             
            k = argmaxlife[i,j]
            if (len(diags[i][j][k]) == 0) & (len(orig_diags[i][j][k]) > 0):
                diags[i][j][k] = rescale*np.atleast_2d(orig_diags[i][j][k][0])
                numpairs += 1
                reduced +=1
                
    print('Post number of life-birth pairs\t:', numpairs)
    print('Reduced to null diagrams:\t', reduced)

    diagh = [ [ None for k in range(ndims) ] for i in range(np.sum(ratios > 0)) ]

    counter = 0
    for i in range(len(diags)):
        for j in np.nonzero(ratios[i] > 0)[0]:
            for k in range(len(diags[i][j])):
                diagh[counter][k] = diags[i][j][k]
            counter += 1
    
    endrow = np.min([endrow, len(diagh)])
    
    bottleneck = np.zeros((endrow - initrow, len(diagh)))
    
    ix = 0
    for i in range(initrow, endrow, 1):
        for j in range(i+1, bottleneck.shape[1]):
            dk = np.zeros(ndims)
            for k in range(len(dk)):
                if (len(diagh[i][k]) > 0) | (len(diagh[j][k]) > 0):
                    dk[k] = persim.bottleneck(diagh[i][k], diagh[j][k], matching=False)
                    
            bottleneck[ix,j] = np.max(dk)
        ix += 1
        print('Row {} [{}]'.format(i, ix))

    filename = dst + dst.split(os.sep)[-2] + '_{}level_bottleneck_{:05d}_{:05d}.csv'.format(level, initrow, endrow)
    pd.DataFrame(bottleneck).to_csv(filename, index=False, header=None)

if __name__ == '__main__':
    main()
