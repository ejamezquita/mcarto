import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import argparse

import json

from glob import glob
import os

import utils
import persim
import itertools

import tifffile as tf
from KDEpy import FFTKDE
from scipy import ndimage,stats,spatial

seed = 42
ndims = 3
fs = 12
PP = 6
N = 6
dpi = 96
minlife = 0.0
rng = np.random.default_rng(seed)
pxs = 75
marker = ['D', '8', 's', '^', 'v', 'P', 'X', '*']
color = ['#56b4e9', '#f0e442', '#009e73', '#0072b2', '#d55e00', '#cc79a7', '#e69f00', '#e0e0e0', '#000000']
cmap = ['Blues_r', 'Wistia', 'Greens_r', 'BuPu_r', 'Oranges_r', 'RdPu_r', 'YlOrBr_r', 'gray', 'gist_gray']
method = 'PCA'
order = 1
hdims = np.array([1,2])
steps = 1
qq = 0.1
Cmap = 'inferno'
GIDX = [0,5]

def main():
    
    parser = argparse.ArgumentParser(description="Produce diagnostic images.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='Last major update: May 2024. © Erik Amezquita')
    parser.add_argument("sample", type=str,
                        help="Cross-section sample name")
    parser.add_argument("level", type=str, choices=['sup','sub'],
                        help="filtration to use")
    parser.add_argument("norm_type", type=str, choices=['both','gene'],
                        help="how to normalize all the KDEs")
    parser.add_argument("-c", "--scale", type=int, default=16,
                        help="scaling factor for easier treatment")
    parser.add_argument("-b", "--bandwidth", type=int, default=10,
                        help="KDE bandwidth")
    parser.add_argument("-z", "--stepsize", type=int, default=3,
                        help="KDE stepsize")
    parser.add_argument("-p", "--pers_w", type=int, default=1,
                        help="power for weight function")                        
    parser.add_argument("-s", "--sigma", type=float, default=1,
                        help="sigma for persistent images")  
    parser.add_argument("-x", "--pixel_size", type=int, default=1,
                        help="pixel size for persistent images")
    parser.add_argument("-n", "--nuclei_mask_cutoff", type=int, default=1,
                        help="Consider a transcript as part of the nucleus if it is within this distance from one")
    parser.add_argument("--cell_wall_directory", type=str, default="cell_dams",
                        help="directory containing cell wall TIFs")
    parser.add_argument("--nuclear_directory", type=str, default="nuclear_mask",
                        help="directory containing nuclei TIFs")
    parser.add_argument("--location_directory", type=str, default="translocs",
                        help="directory to contain corrected spatial location data")
    parser.add_argument("--kde_directory", type=str, default="kde",
                        help="directory to contain data related to KDE computations")
    parser.add_argument("--pca_directory", type=str, default="infected_focus_summer24",
                        help="directory to contain data related to KDE computations")
    parser.add_argument("-w", "--rewrite_results", action="store_true",
                        help="prior results will be rewritten")
    args = parser.parse_args()
    
    rewrite = args.rewrite_results
    sample = args.sample
    normtype = args.norm_type
    level = args.level
    sigma = args.sigma
    pers_w = args.pers_w
    pixel_size = args.pixel_size
    bw = args.bandwidth
    stepsize = args.stepsize
    SCALE = args.scale
    
    wsrc = '..' + os.sep + args.cell_wall_directory + os.sep
    nsrc = '..' + os.sep + args.nuclear_directory + os.sep
    tsrc = '..' + os.sep + args.location_directory + os.sep
    ksrc = '..' + os.sep + args.kde_directory + os.sep
    isrc = '..' + os.sep + args.pca_directory + os.sep 

    isrc += sample + os.sep
    ksrc += sample + os.sep

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv', index_col='ndimage_cellID')
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv').set_index('gene')
    weight = np.load(ksrc + sample + '_border_weights.npy', allow_pickle=True)
    transcriptomes = np.asarray(list(metatrans['gene']))
    metatrans = metatrans.set_index('gene')

    Cells = utils.get_range_cell_values(isrc + 'infected_cells_ids.csv', metacell, startval=1)
    Cells = np.setdiff1d( Cells, metacell[metacell['number_nuclei'] > 1].index)

    Genes = utils.get_range_gene_values(isrc + 'genes_to_focus_infection.csv', transcriptomes, startval=0)
    invGenes = dict(zip(Genes, range(len(Genes))))
    invCells = dict(zip(Cells, range(len(Cells))))
    transfocus = transcell.loc[transcriptomes[Genes], Cells.astype(str)]

    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    label, cellnum = ndimage.label(wall, ndimage.generate_binary_structure(2,1))
    css = ndimage.find_objects(label)
    wall[tf.imread(nsrc + sample + '_EDT.tif') < args.nuclei_mask_cutoff] = False
    wcoords = np.asarray(np.nonzero(~wall))
    wallshape = wall.shape
    wc = wcoords[:, ~np.all(wcoords%100, axis=0)]

    translocs = [None for i in range(len(transcriptomes))]
    for i in range(len(transcriptomes)):
        filename = tsrc + sample + os.sep + 'location_corrected_D2_-_' + transcriptomes[i] + '.csv'
        translocs[i] = pd.read_csv(filename, header=None, names=['X', 'Y', 'Z'])
    tlocs = pd.concat(translocs)
    tcumsum = np.hstack(([0], np.cumsum(metatrans['cyto_number'].values)))

    zmax = np.max(tlocs['Z']+stepsize)
    zbins = np.arange(0, zmax+stepsize, stepsize)

    dsrc = isrc + 'G{}_{}level_{}_step{}_bw{}'.format(len(Genes), level, normtype, stepsize, bw) + os.sep
    ratios = utils.normalize_counts(transfocus, normtype)
    print(ratios)

    jsonfiles = [ [ None for j in range(ratios.shape[1]) ] for i in range(ratios.shape[0]) ]
    gsrc = '../{}level/'.format(level) + sample + os.sep
    for i in range(len(jsonfiles)):
        foo = '{}{}/{}_-_{}_p{}_s{}_bw{}_c{:06d}.json'
        for j in range(len(jsonfiles[0])):
            filename = foo.format(gsrc, transcriptomes[Genes[i]],transcriptomes[Genes[i]],level,PP,stepsize,bw,Cells[j])
            if os.path.isfile(filename):
                jsonfiles[i][j] = filename

    orig_diags = [ utils.get_diagrams(jsonfiles[i], ndims, remove_inf=True) for i in range(len(jsonfiles))]
    orig_diags, rescale, maxlife, focus_dim = utils.normalize_persistence_diagrams(orig_diags, ratios, normtype, SCALE)

    kde_max = np.zeros(ratios.size)
    for i in range(len(orig_diags)):
        for j in range(len(orig_diags[i])):
            if len(orig_diags[i][j][focus_dim]) > 0:
                kde_max[i*len(orig_diags[i]) + j] = np.max(orig_diags[i][j][focus_dim])

    kmax = np.sort(kde_max)[-20]

    lt_mask = np.any(maxlife > minlife, axis=2)
    gmask, cmask = np.nonzero(lt_mask)

    bsummary = pd.DataFrame()
    bsummary['gene_ID'] = Genes[gmask]
    bsummary['ndimage_ID'] = Cells[cmask]
    uq , cts = np.unique(gmask, return_counts=True)
    nzcumsum = np.hstack(([0], np.cumsum(cts) ))

    if normtype == 'gene':
        diags = [ [ [ rescale[i][0][0]*orig_diags[i][j][k].copy() for k in range(len(orig_diags[i][j])) ] for j in range(len(orig_diags[i])) ] for i in range(len(orig_diags)) ]
    elif normtype == 'both':
        diags = [ [ [ rescale*orig_diags[i][j][k].copy() for k in range(len(orig_diags[i][j])) ] for j in range(len(orig_diags[i])) ] for i in range(len(orig_diags)) ]
    for i in range(len(diags)):
        for j in range(len(diags[i])):
            for k in range(len(diags[i][j])):
                diags[i][j][k] = np.atleast_2d(diags[i][j][k][ diags[i][j][k][:,1] - diags[i][j][k][:,0] > minlife, : ])

    full_lt_coll = [ [None for _ in range(np.sum(lt_mask)) ] for _ in range(ndims) ]
    for i in range(len(gmask)):
        for k in range(len(full_lt_coll)):
            d = diags[gmask[i]][cmask[i]][k]
            full_lt_coll[k][i] = np.column_stack( (d[:, 0], d[:, 1] - d[:, 0])  )

    maxbirth = 0
    for k in range(len(full_lt_coll)):
        for i in range(len(full_lt_coll[k])):
            if len(full_lt_coll[k][i]) > 0:
                b = np.max(full_lt_coll[k][i][:,0])
                if b > maxbirth:
                    maxbirth = b

    print(maxlife)
    return 0

if __name__ == '__main__':
    main()

