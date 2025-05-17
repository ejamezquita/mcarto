import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import numpy as np
import pandas as pd
import argparse

import json
import os

import utils
import persim

import tifffile as tf
from KDEpy import FFTKDE
from scipy import ndimage,stats

seed = 42
ndims = 3
fs = 15
PP = 6
dpi = 96

Cmap = 'plasma'
pxs = 75
plot_pxbar = np.s_[-25:-5, 5:5 + pxs]
hdims = np.array([1,2])
steps = 1
qq = 0.125

s = 50
alphaNmax = 10
alphaNmin = 0.1
wong = ['#d81b60', '#b5b5b5', '#6b6b6b', '#000000']
normtype = 'both'

BW = [10,15,20,25,30]
SCALES = [8,16,24,32,40,48]
smcolumns=['PC 1', 'PC 2', 'N', 'Density', 'N(05G203100)', 'X', 'Y', 'Cell Size', 'Eccentricity']
crcolumns=['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'intercept_stderr', 'spearson', 'ppearson', 'sspearman', 'pspearman']
cnames = ['N', 'Density', 'PC 1', 'PC 2']

stepsize = 3
sbkw = dict(label='', size=pxs, loc='upper left', pad=0.5, color='k', frameon=False, size_vertical=7.5)
sbkw1 = dict(label='', size=pxs/stepsize, loc='upper left', pad=0.5, color='lime', frameon=False, size_vertical=7.5/stepsize)
rtkw = dict(ha='right', va='bottom', c='navy', bbox=dict(facecolor='cornsilk', alpha=0.75, boxstyle=mpl.patches.BoxStyle("Square", pad=0.05)))

k = 0
vs = ['' for _ in range( len(smcolumns)*( len(smcolumns) - 1)//2) ]
for i in range(len(smcolumns)-1):
    for j in range(i+1, len(smcolumns)):
        vs[k] = '{}-vs-{}'.format(smcolumns[i] , smcolumns[j] )
        k += 1

hdims = np.array([1,2])
Pname = ' [$' + ' \\oplus '.join(['H_{}'.format(k) for k in hdims]) + '$]'
pname = 'H' + '+'.join(hdims.astype(str))

def main():
    
    parser = argparse.ArgumentParser(description="Produce diagnostic images.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='Last major update: May 2024. © Erik Amezquita')
    parser.add_argument("sample", type=str,
                        help="Cross-section sample name")
    parser.add_argument("level", type=str, choices=['sup','sub'],
                        help="filtration to use")
    parser.add_argument("-c", "--scale", type=int,
                        help="scaling factor for easier treatment")
    parser.add_argument("-b", "--bandwidth", type=int,
                        help="KDE bandwidth")
    parser.add_argument("-g", "--gene_focus", type=int,
                        help="file or single ID with gene to evaluate")
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
    parser.add_argument("-N", "--include_nuclei_mrna", action="store_true",
                        help="include mRNA located in nuclear regions")
    args = parser.parse_args()
    
    rewrite = args.rewrite_results
    sample = args.sample
    level = args.level
    sigma = args.sigma
    pers_w = args.pers_w
    pixel_size = args.pixel_size
    stepsize = args.stepsize
    SCALE = args.scale
    bw = args.bandwidth
    exclude_nuclei = not args.include_nuclei_mrna
    print('Exclude nuclear mRNA:', exclude_nuclei)
        
    wsrc = '..' + os.sep + args.cell_wall_directory + os.sep
    nsrc = '..' + os.sep + args.nuclear_directory + os.sep
    ksrc = '..' + os.sep + args.kde_directory + os.sep + sample + os.sep
    isrc = '..' + os.sep + args.pca_directory + os.sep + sample + os.sep
    gsrc = '..' + os.sep + level + 'level' + os.sep + sample + os.sep

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv', index_col='ndimage_cellID')
    metaecc = pd.read_csv(ksrc + sample + '_nodule_root_eccentricity.csv', index_col='ndimage_cellID')
    cell_nuc = pd.read_csv(ksrc + sample + '_nuclei_limits.csv')

    Cells = utils.get_range_cell_values(isrc + 'infected_cells_ids.csv', metacell, startval=1)
    Cells = np.setdiff1d( Cells, metacell[metacell['number_nuclei'] > 1].index)

    label, cellnum = ndimage.label(tf.imread(wsrc + sample + '_dams.tif').astype(bool), ndimage.generate_binary_structure(2,1))
    print('Detected',cellnum,'cells')
    wcoords = np.loadtxt(ksrc + sample + '_icoords.csv', delimiter=',', dtype=int)
    wc = wcoords[:, ~np.all(wcoords%100, axis=0)]

    lnuc, nnuc = ndimage.label(tf.imread(nsrc + sample + '_EDT.tif') < args.nuclei_mask_cutoff, ndimage.generate_binary_structure(2,1))
    print('Detected',nnuc,'nuclei')
    
    foo = [ wong[-1], wong[-2] ] + np.repeat(wong[1], nnuc).tolist() + ['#f0f0f0']
    cellular_cmap = mpl.colors.ListedColormap(foo)

    if exclude_nuclei:
        filenameb = '..' + os.sep + 'translocs' + os.sep + sample + os.sep + 'location_corrected_D2_-_{}.csv'
        ex_nuclei = ''
        pdkw = {'header':None, 'names':['X', 'Y', 'Z']}
    else:
        filenameb = '..' + os.sep + 'Bacteria Info for Erik' + os.sep + '{}_v2.txt'
        ex_nuclei = '_w_nucleus'
        pdkw = {'sep':'\t'}

    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata' + ex_nuclei + '.csv', index_col='gene').rename(columns=int)
    transcriptomes = np.asarray(transcell.index, dtype=str)
    Genes = utils.get_range_gene_values(isrc + 'genes_to_focus_infection' + ex_nuclei + '.csv', transcriptomes, startval=0)
    if args.gene_focus is None:
        gene_focus = range(len(Genes))
    else:
        gene_focus = [args.gene_focus]

    translocs = dict()
    for key in transcriptomes:
        filename = filenameb.format(key)
        translocs[key] = pd.read_csv(filename, **pdkw)
        translocs[key]['cidx'] = label[ translocs[key]['Y'], translocs[key]['X'] ]
        translocs[key]['nidx'] =  lnuc[ translocs[key]['Y'], translocs[key]['X'] ]
    tlocs = pd.concat(translocs)
    zmax = np.max(tlocs['Z']+stepsize)
    zbins = np.arange(0, zmax+stepsize, stepsize)

    for tidx in gene_focus:
        
        transfocus = transcell.loc[ np.atleast_1d( transcriptomes[Genes[ tidx ]]), Cells]
        ratios = utils.normalize_counts(transfocus, normtype)
        genes = '_-_'.join(sorted([ g.replace('GLYMA_', 'Glyma.') for g in ratios.index ]))
        print(genes, 'Max ratio by {}:\t{:.2f}%'.format(normtype, 100*np.max(ratios) ) )
        
        for bw in BW:
            
            foo = '{}{}/{}_-_{}_p{}_s{}_bw{}_c{:06d}.json'
            jsonfiles = dict()
            for t in transfocus.index:
                jsonfiles[t] = [ foo.format(gsrc, t, t, level, PP, stepsize, bw, Cells[i]) for i in range(ratios.shape[1]) ]

            orig_diags = utils.get_diagrams(jsonfiles, ndims, remove_inf=True)
            
            for SCALE in SCALES:
                
                diags, rescale, maxlife, focus_dim = utils.normalize_persistence_diagrams(orig_diags, ratios, normtype, SCALE)
                maxxlife = max( list(map(np.max, iter(maxlife.values()))) )

                kde_max = np.zeros(ratios.size); i = 0
                for gene in diags:
                    for cidx in diags[gene]:
                        if len(diags[gene][cidx][2]) > 0:
                            kde_max[i] = np.max(diags[gene][cidx][2])
                            i += 1
                kmax = np.sort(kde_max)[-10]

                maxbirth = 0
                lt_diags = dict()
                for gene in diags:
                    lt_diags[gene] = dict()
                    for cidx in diags[gene]:
                        lt_diags[gene][cidx] = [ None for _ in range(len(diags[gene][cidx])) ]
                        for k in range(len(diags[gene][cidx])): 
                            foo = rescale*diags[gene][cidx][k].copy()
                            lt_diags[gene][cidx][k] = np.column_stack( (foo[:,0], foo[:,1] - foo[:,0]) )
                            if (len(lt_diags[gene][cidx][k]) > 0) and (lt_diags[gene][cidx][k][:,0].max() > maxbirth):
                                maxbirth = lt_diags[gene][cidx][k][:,0].max()
                print(list(map(len, iter(lt_diags.values()))), maxbirth)
                
              
                # Bring in persistence images to the party

                pi_params = {'birth_range':(0,min([SCALE, int(np.ceil(maxbirth + sigma))] )),
                             'pers_range':(0,min([SCALE, int(np.ceil(maxxlife + sigma))])),
                             'pixel_size': pixel_size,
                             'weight': 'persistence',
                             'weight_params': {'n': pers_w},
                             'kernel':'gaussian',
                             'kernel_params':{'sigma': [[sigma, 0.0], [0.0, sigma]]} }
                           
                pimgr = persim.PersistenceImager(**pi_params)
                extent = np.array([ pimgr.birth_range[0], pimgr.birth_range[1], pimgr.pers_range[0], pimgr.pers_range[1] ]).astype(int)
                
                full_img = dict()
                for gene in lt_diags:
                    full_img[gene] = dict()
                    for cidx in lt_diags[gene]:
                        full_img[gene][cidx] = np.asarray( [ pimgr.transform( lt_diags[gene][cidx][k] , skew=False) for k in range(len(lt_diags[gene][cidx])) ])
                        full_img[gene][cidx][ full_img[gene][cidx] < 0 ] = 0

                bname = isrc + '{}_bw{}_{}level'.format(genes.replace('Glyma.',''), bw, level) + os.sep + 'PI_scale{}_'.format(SCALE)
                Bname = genes + ' PIs: KDE bandwidth {}. {}level persistence. Scale {}'.format(bw, level.title(), SCALE)
                print(bname, Bname, sep='\n')
                
                # PCA sampling
                
                embedding = pd.read_csv(bname + 'pca.csv')
                zero_val = embedding.iloc[-1, 2:].values
                zs = [0, 0, zero_val[0], zero_val[1]]
                embedding = embedding.iloc[:-1]
                pca = embedding.iloc[:,2:4].values
                
                steps = 2
                reps = np.arange(100)
                while len(reps) > 20:
                    _, _, grid0, minmask0, _ = utils.grid_representatives(pca, pca, steps)
                    nrow, ncol, grid, minmask, reps = utils.grid_representatives(grid0[minmask0], pca, steps)
                    steps -= 0.05
                    
                Ns = transcell.loc[gene, embedding.loc[reps, 'ndimage_ID'].values].values
                rhos = Ns/metacell.loc[embedding.loc[reps, 'ndimage_ID'], 'cyto_area'].values
                expo = (np.floor(np.log10(rhos))).astype(int)
                base = np.round(rhos*np.power(10., -expo),1)                    
                print(steps, nrow, ncol, len(reps), sep='\t')
    
                # Gridded PCA
                filename = bname + 'pca_gridded.png'
                if rewrite or (not os.path.isfile(filename)):
                    fig, ax = plt.subplots( 1,1, figsize=(10, 4), sharex=True, sharey=True)
                    ax = np.atleast_1d(ax).ravel(); i = 0

                    ax[i].set_title(Bname, fontsize=fs)
                    ax[i].scatter(pca[:,0], pca[:,1], c='lightgray', marker='o', s=10, alpha=1, zorder=1)
                    ax[i].scatter(*zero_val[:2], c='r', marker='x', s=25, alpha=1, zorder=3)
                    ax[i].scatter(pca[ reps,0], pca[ reps,1], c='k', marker='*', s=100, alpha=1, zorder=4, edgecolor='yellow', linewidth=0.5)
                    ax[i].set_xlabel(embedding.columns[2]+Pname, fontsize=fs)
                    ax[i].set_ylabel(embedding.columns[3], fontsize=fs)
                    ax[i].set_facecolor('snow')
                    print(filename)
                    plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                    plt.close()

                # Gridded PCA: Persistence Images
                filename = bname + 'PI_sample.png'
                if rewrite or (not os.path.isfile(filename)):
                    
                    lt_coll = [None for _ in range(len(reps))]
                    img = np.zeros((len(reps), len(hdims)*extent[1], extent[3]))

                    for i in range(len(lt_coll)):
                        gene, cidx = embedding.loc[reps[i], ['gene', 'ndimage_ID']]
                        lt_coll[i] = np.vstack([ np.array([(k-1)*extent[1], 0]) + lt_diags[gene][cidx][k] for k in hdims ])    
                        img[i] = np.vstack(full_img[gene][cidx][hdims])
                    vmax = np.quantile(img[img > 0], 0.99)
                    
                    fig, ax = plt.subplots( nrow, ncol, figsize=(13, (0+0.5*len(hdims))*nrow), sharex=True, sharey=True)
                    ax = np.atleast_1d(ax).ravel();

                    for i, j in enumerate(np.nonzero(minmask)[0]):
                        ax[j].imshow(img[i].T, cmap=Cmap, vmin=0, vmax=vmax, origin='lower')
                        for k in range(1, len(hdims)):
                            ax[j].axvline(k*extent[1] - .5, c='white', lw=0.5)
                        ax[j].scatter(lt_coll[i][:,0], lt_coll[i][:,1], c='w', marker='o', s=10, edgecolor='k', linewidth=0.5)

                    for j in np.nonzero(~minmask)[0]:
                        fig.delaxes(ax[j])

                    fig.suptitle(Bname, fontsize=1.2*fs)
                    fig.tight_layout();
                    print(filename)
                    plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                    plt.close()
        
                # Pre-computing KDEs
                filename = bname + 'cell_sample.png'
                if rewrite or (not os.path.isfile(filename)):
                    
                    hkdes, hcells, hcoords, hextent = [ [None for _ in range(len(reps))] for _ in range(4) ]
                    for i in range(len(hkdes)):
                        gene, cidx = embedding.loc[reps[i], ['gene', 'ndimage_ID']]
                        hcoords[i] = translocs[gene].loc[ translocs[gene]['cidx'] == cidx , ['X', 'Y', 'Z'] ].values.T
                        
                        hcells[i], hextent[i] = utils.get_cell_img(cidx, metacell, label, lnuc, nnuc, PP=6)
                        s_ = (np.s_[hextent[i][2]:hextent[i][3]], np.s_[hextent[i][0]:hextent[i][1]])
                        
                        axes, kgrid, kdegmask, cgrid, outside_walls = utils.cell_grid_preparation(cidx, hcells[i], label[s_], hextent[i], zmax, stepsize, cell_nuc, exclude_nuclei=exclude_nuclei)
                        
                        kde = FFTKDE(kernel='gaussian', bw=bw, norm=2).fit(hcoords[i].T).evaluate(kgrid)
                        kde = kde[kdegmask]/(np.sum(kde[kdegmask])*(stepsize**len(hcoords[i])))
                        kde[outside_walls] = 0
                        kde = kde/(np.sum(kde)*(stepsize**len(hcoords[i])))
                        kde = kde.reshape( list(map(len, axes))[::-1], order='F')
                        hkdes[i] = np.max(kde * ratios.loc[gene,cidx], axis=0)

                    # Gridded PCA: Cells
                
                    fig, ax = plt.subplots( nrow, ncol, figsize=(10, 1.55*nrow), sharex=False, sharey=False)
                    ax = np.atleast_1d(ax).ravel();

                    for i, j in enumerate(np.nonzero(minmask)[0]):
                        ax[j].imshow(hcells[i]+1, cmap=cellular_cmap, origin='lower', extent=hextent[i], vmin=0, vmax=nnuc+2);
                        ax[j].scatter(*hcoords[i][:2], color='r', marker='o', alpha=min([0.5,75/Ns[i]]), s=int(4e6/hcells[i].size))
                        ax[j].set_facecolor(wong[2])
                        ax[j].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                        ax[j].set_aspect('equal','datalim')
                        ax[j].text(0.99,0, Ns[i], transform=ax[j].transAxes, **rtkw)
                        ax[j].add_artist(AnchoredSizeBar(ax[j].transData, **sbkw))

                    for j in np.nonzero(~minmask)[0]:
                        fig.delaxes(ax[j])

                    fig.suptitle(Bname, fontsize=1.2*fs)
                    fig.tight_layout();
                    print(filename)
                    plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                    plt.close()
                    
                    # Gridded PCA: KDEs
                    
                    filename = bname + 'kde_sample.png'
                    
                    fig, ax = plt.subplots( nrow, ncol, figsize=(10, 1.55*nrow), sharex=False, sharey=False)
                    ax = np.atleast_1d(ax).ravel();

                    for i, j in enumerate(np.nonzero(minmask)[0]):
                        ax[j].imshow(hkdes[i], origin='lower', cmap=Cmap, vmin=0, vmax=kmax, zorder=1)
                        ax[j].set_facecolor( mpl.colormaps[Cmap](0) )
                        ax[j].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                        ax[j].set_aspect('equal','datalim')
                        ax[j].text(0.99,0, '{:.1f}$\\times$10$^{{{}}}$'.format(base[i], expo[i]), transform=ax[j].transAxes, **rtkw)
                        ax[j].add_artist(AnchoredSizeBar(ax[j].transData, **sbkw1))

                    for j in np.nonzero(~minmask)[0]:
                        fig.delaxes(ax[j])

                    fig.suptitle(Bname, fontsize=1.2*fs)
                    fig.tight_layout();
                    print(filename)
                    plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                    plt.close()

                # Summarize correlations
                
                cmask = embedding.loc[ embedding['gene'] == ratios.index[0] , 'ndimage_ID' ]
                summary0 = embedding.iloc[:, 1:4].copy().set_index('ndimage_ID')
                density = transcell.loc[ratios.index[0], Cells] / metacell.loc[Cells, 'cyto_area']
                density.name = 'density'
                NN = transcell.loc[ratios.index[0], Cells].copy()
                NN.name = 'N'
                summary0 = summary0.join(pd.concat([NN, density], axis=1), how='outer').join(transcell.loc['GLYMA_05G203100', Cells] + 1)
                summary0 = summary0.join(metacell.loc[Cells, ['ndimage_comX','ndimage_comY','cyto_area']], how='outer')
                summary0.iloc[ pd.isna(summary0.iloc[:,0]).values , :2] = zero_val[:2]
                summary0 = summary0.join(metaecc['eccentricity'].max() - metaecc.loc[Cells, 'eccentricity'])
                summary0.columns = smcolumns
                
                corr = pd.DataFrame(index=vs, columns=crcolumns)
                corr0 = pd.DataFrame(index=vs, columns=crcolumns)
                for idx in corr0.index:
                    x,y = summary0[ idx.split('-vs-') ].T.values
                    corr0.loc[idx, ['sspearman', 'pspearman'] ] = stats.spearmanr(x,y)
                    corr0.loc[idx, ['spearson', 'ppearson'] ] = stats.pearsonr(x,y)
                    c = stats.linregress(x,y)
                    corr0.loc[idx, ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']] = c
                    corr0.loc[idx, 'intercept_stderr'] = c.intercept_stderr
                    
                    x,y = summary0.loc[ cmask, idx.split('-vs-') ].T.values
                    corr.loc[idx, ['sspearman', 'pspearman'] ] = stats.spearmanr(x,y)
                    corr.loc[idx, ['spearson', 'ppearson'] ] = stats.pearsonr(x,y)
                    c = stats.linregress(x,y)
                    corr.loc[idx, ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']] = c
                    corr.loc[idx, 'intercept_stderr'] = c.intercept_stderr
                    
                filename = bname + 'corr_summary.csv'
                if rewrite or (not os.path.isfile(filename)):
                    corr.to_csv(filename, index=True, index_label='Comparison')
                
                filename = bname + 'corr0_summary.csv'
                if rewrite or (not os.path.isfile(filename)):
                    corr0.to_csv(filename, index=True, index_label='Comparison')
                    
                filename = bname + 'nodule_locations.png'
                if rewrite or (not os.path.isfile(filename)):
                    s = 30
                    zs = [0, 0, zero_val[0], zero_val[1]]
                    fig, ax = plt.subplots(1, 4, figsize=(12,4), sharex=True, sharey=True)
                    ax = np.atleast_1d(ax).ravel(); i = 0

                    for i,cname in enumerate(cnames):
                        c = summary0.loc[:, cname]
                        zmask = c != zs[i] 
                        vmax = utils.maximum_qq_size(c[zmask], alpha=0.25, iqr_factor=1.5)
                        vmin = utils.minimum_qq_size(c[zmask], alpha=0.25, iqr_factor=1.5)
                        delta = 0.25*(vmax-vmin)
                        vmax += delta

                        ax[i].scatter(wc[1], wc[0], c='#808080', marker='.', s=0.5, zorder=1)
                        ax[i].scatter(*summary0.loc[~zmask,['X','Y']].T.values, c='k', marker='D', s=0.75*s, edgecolor='#808080', zorder=2)
                        ax[i].scatter(*summary0.loc[zmask,['X','Y']].T.values, c=c[zmask]+delta, marker='o', cmap=Cmap, s=s,
                                      edgecolor='k', linewidth=0.5, zorder=3, vmax=vmax, vmin=vmin)
                        ax[i].set_aspect('equal')
                        ax[i].set_facecolor('snow')
                        ax[i].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                        ax[i].set_title(c.name, fontsize=fs)
                    fig.suptitle(Bname, fontsize=1.2*fs)
                    fig.tight_layout()
                    print(filename)
                    plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                    plt.close()
                    
                filename = bname + 'eccentricity_correlations.png'
                if rewrite or (not os.path.isfile(filename)):
                
                    for yaxis,yscale in zip(['Eccentricity', 'N(05G203100)'],['linear','log']):
                        
                        filename = bname + yaxis.replace('(','').replace(')','').lower() + '_correlations.png'
                        ecc = summary0[yaxis]
                        fig, ax = plt.subplots(1, 4, figsize=(12,3.75), sharex=False, sharey=True)
                        ax = np.atleast_1d(ax).ravel(); i = 0
                        
                        for i,cname in enumerate(cnames):
                            c = summary0.loc[:, cname]
                            zmask = c != zs[i] 
                            vmax = utils.maximum_qq_size(c[zmask], alpha=0.25, iqr_factor=1.5)
                            vmin = utils.minimum_qq_size(c[zmask], alpha=0.25, iqr_factor=1.5)
                            delta = 0.25*(vmax-vmin)
                            vmax += delta
                        
                            ax[i].scatter(c[zmask], ecc[zmask], c=c[zmask]+delta, marker='o', cmap=Cmap, s=s,
                                          edgecolor='k', linewidth=0.5, zorder=3, vmax=vmax, vmin=vmin)
                            ax[i].scatter(c[~zmask], ecc[~zmask], c='k', marker='D', s=s, edgecolor='w', zorder=2)
                            ax[i].set_facecolor('snow')
                            ax[i].tick_params(labelsize=0.9*fs)
                            ax[i].set_xlabel(c.name, fontsize=fs)
                            ax[i].set_xlim(c.min(), 1.25*vmax)
                            ax[i].set_yscale(yscale)
                            r,p = corr.loc[cname + '-vs-' + ecc.name, ['sspearman', 'pspearman'] ]
                            ll = 'r$_s = ${:.2f} [{}]'.format(r, utils.star_signif(p, mx=3))
                            ax[i].set_title(ll, fontsize=fs)
                        
                        ax[0].set_ylabel(ecc.name, fontsize=fs)
                        fig.suptitle(Bname, fontsize=1.2*fs)
                        fig.tight_layout()
                        print(filename)
                        plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                        plt.close()
                        
                    filename = bname + 'fulleccentricity_correlations.png'
                    if rewrite or (not os.path.isfile(filename)):
                        ecc = summary0['Eccentricity']
                        fig, ax = plt.subplots(2, 4, figsize=(13.5,7), sharex=False, sharey=False)

                        for i,cname in enumerate(cnames):
                            j = 0
                            c = summary0.loc[:, cname]
                            zmask = c != zs[i] 
                            vmax = utils.maximum_qq_size(c[zmask], alpha=0.25, iqr_factor=1.5)
                            vmin = utils.minimum_qq_size(c[zmask], alpha=0.25, iqr_factor=1.5)
                            delta = 0.25*(vmax-vmin)
                            vmax += delta

                            ax[j,i].scatter(wc[1], wc[0], c='#808080', marker='.', s=0.5, zorder=1)
                            ax[j,i].scatter(*summary0.loc[~zmask,['X','Y']].T.values, c='k', marker='D', s=0.75*s, edgecolor='#808080', zorder=2)
                            ax[j,i].scatter(*summary0.loc[zmask,['X','Y']].T.values, c=c[zmask]+delta, marker='o', cmap=Cmap, s=s,
                                          edgecolor='k', linewidth=0.5, zorder=3, vmax=vmax, vmin=vmin)
                            ax[j,i].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                            r,p = corr.loc[cname + '-vs-' + ecc.name, ['sspearman', 'pspearman'] ]
                            ll = 'r$_s = ${:.2f} [{}]'.format(r, utils.star_signif(p, mx=3))
                            ax[j,i].text(0.99,0, ll, transform=ax[j,i].transAxes, fontsize=fs, **rtkw)
                            
                            j = 1
                            ax[j,i].scatter(c[zmask], ecc[zmask], c=c[zmask]+delta, marker='o', cmap=Cmap, s=s, edgecolor='k', linewidth=0.5, vmax=vmax, vmin=vmin)
                            ax[j,i].scatter(c[~zmask], ecc[~zmask], c='k', marker='D', s=s, edgecolor='w', zorder=2)
                            ax[j,i].set_xlim(c.min(), 1.25*vmax)
                            ax[j,i].set_ylim(0, 1.05*ecc.max())
                            ax[j,i].tick_params(bottom=True, labelbottom=True, left=True, labelleft=False)
                            ax[j,i].set_xlabel(c.name, fontsize=fs)

                        ax[0,0].set_ylabel('Nodule ' + sample, fontsize=fs)
                        ax[1,0].tick_params(labelleft=True)
                        ax[1,0].set_ylabel('Eccentricity [px]', fontsize=fs)

                        for a in ax.ravel():
                            a.set_facecolor('snow')
                            a.tick_params(labelsize=0.9*fs)

                        fig.align_ylabels()
                        fig.suptitle(Bname, fontsize=1.2*fs)
                        fig.tight_layout()
                        print(filename)
                        plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                        plt.close()
                        
                        
    
    return 0

if __name__ == '__main__':
    main()

