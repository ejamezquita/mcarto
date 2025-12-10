import argparse
import json
import functools
from glob import glob
import itertools

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import persim
import utils
from sklearn import decomposition, preprocessing

jsonfile = '{}{}/{}_-_{}_p{}_s{}_bw{}_c{:06d}.json'
marker = ['D', '8', 's', '^', 'v', 'P', 'X', '*']
color = ['#56b4e9', '#f0e442', '#009e73', '#0072b2', '#d55e00', '#cc79a7', '#e69f00', '#e0e0e0', '#000000']
cmap = ['Blues_r', 'Wistia', 'Greens_r', 'BuPu_r', 'Oranges_r', 'RdPu_r', 'YlOrBr_r', 'gray', 'gist_gray']
dest_directory = 'infected_focus_summer24'
seed = 42
ndims = 3
dpi = 96
PP = 6
alpha = 0.25
iqr_factor = 1.5
pcacol = 3
fs = 12

#perms = [np.nonzero(p)[0] for p in itertools.product(range(2), repeat=ndims)][1:]
perms = [ np.array([1,2]) ]
nrows = 1

def main():
    
    parser = argparse.ArgumentParser(description="Produce Persistence Image summaries.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='Last major update: May 2024. © Erik Amezquita')
    parser.add_argument("sample", type=str,
                        help="Cross-section sample name")
    parser.add_argument("level", type=str, choices=['sup','sub'],
                        help="filtration to use")
    parser.add_argument("norm_type", type=str, choices=['both','gene'],
                        help="how to normalize all the KDEs")
    parser.add_argument("-c", "--scale", type=int,
                        help="scaling factor for easier treatment")
    parser.add_argument("-b", "--bandwidth", type=int,
                        help="KDE bandwidth")
    parser.add_argument("-z", "--stepsize", type=int, default=3,
                        help="KDE stepsize")
    parser.add_argument("-p", "--pers_w", type=int, default=1,
                        help="power for weight function")                        
    parser.add_argument("-s", "--sigma", type=float, default=1,
                        help="sigma for persistent images")  
    parser.add_argument("-x", "--pixel_size", type=int, default=1,
                        help="pixel size for persistent images")  
    parser.add_argument("--cell_wall_directory", type=str, default="cell_dams",
                        help="directory containing cell wall TIFs")
    parser.add_argument("--nuclear_directory", type=str, default="nuclear_mask",
                        help="directory containing nuclei TIFs")
    parser.add_argument("--location_directory", type=str, default="translocs",
                        help="directory to contain corrected spatial location data")
    parser.add_argument("--kde_directory", type=str, default="kde",
                        help="directory to contain data related to KDE computations")
    parser.add_argument("-w", "--rewrite_results", action="store_true",
                        help="prior results will be rewritten")
    parser.add_argument("-N", "--include_nuclei_mrna", action="store_true",
                        help="include mRNA located in nuclear regions")
    args = parser.parse_args()
    
    rewrite = args.rewrite_results
    sample = args.sample
    normtype = args.norm_type
    level = args.level
    sigma = args.sigma
    pers_w = args.pers_w
    pixel_size = args.pixel_size
    stepsize = args.stepsize
    exclude_nuclei = not args.include_nuclei_mrna
    print('Exclude nuclear mRNA:', exclude_nuclei)
    
    if args.scale is None:
        SCALES = [8,16,24,32,40,48]
    else:
        SCALES = [args.scale]
        
    if args.bandwidth is None:
        BW = [10,15,20,25,30]
    else:
        BW = [args.bandwidth]

    wsrc = 'os.pardir' + os.sep + args.cell_wall_directory + os.sep
    nsrc = 'os.pardir' + os.sep + args.nuclear_directory + os.sep
    tsrc = 'os.pardir' + os.sep + args.location_directory + os.sep + sample + os.sep
    ksrc = 'os.pardir' + os.sep + args.kde_directory + os.sep + sample + os.sep
    gsrc = 'os.pardir' + os.sep + level + 'level' + os.sep + sample + os.sep
    dst = 'os.pardir' + os.sep + dest_directory + os.sep + sample + os.sep

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv', index_col='ndimage_cellID')
    
    Cells = utils.get_range_cell_values(dst + 'infected_cells_ids.csv', metacell, startval=1)
    Cells = np.setdiff1d( Cells, metacell[metacell['number_nuclei'] > 1].index)
    
    if exclude_nuclei:
        ex_nuclei = ''
    else:
        ex_nuclei = '_w_nucleus'
    
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata' + ex_nuclei + '.csv', index_col='gene').rename(columns=int)
    transcriptomes = np.asarray(transcell.index, dtype=str)
    Genes = utils.get_range_gene_values(dst + 'genes_to_focus_infection' + ex_nuclei + '.csv', transcriptomes, startval=0)
    
    for gidx in range(len(Genes)):
        #transfocus = transcell.loc[ np.atleast_1d( transcriptomes[Genes]), Cells]
        transfocus = transcell.loc[ np.atleast_1d( transcriptomes[Genes[ gidx ]]), Cells]
        ratios = utils.normalize_counts(transfocus, normtype)
        if ratios is None:
            print('ERROR: ratios is None')
            return 0
        print('Max ratio by {}:\t{:.2f}%'.format(normtype, 100*np.max(ratios) ), np.sum(ratios > 0, axis=1) )
        genes = '_-_'.join(sorted([ g.split('_')[-1] for g in ratios.index ]))
        
        for bw in BW:
            
            jsonfiles = dict()

            for t in transfocus.index:
                jsonfiles[t] = [ jsonfile.format(gsrc, t, t, level, PP, stepsize, bw, Cells[i]) for i in range(ratios.shape[1]) ]

            orig_diags = utils.get_diagrams(jsonfiles, ndims, remove_inf=True)
            nzcumsum = np.hstack(([0], np.cumsum(list(map(len, iter(orig_diags.values()))))))
            bsummary = pd.DataFrame(index=range(nzcumsum[-1]), columns=['gene','ndimage_ID'] )

            for SCALE in SCALES:
                
                diags, rescale, maxlife, _ = utils.normalize_persistence_diagrams(orig_diags, ratios, normtype, SCALE)
                maxxlife = max( list(map(np.max, iter(maxlife.values()))) )
                
                i = 0
                maxbirth = 0
                lt_diags = dict()
                for gene in diags:
                    lt_diags[gene] = dict()
                    for cidx in diags[gene]:
                        bsummary.iloc[i] = [gene, cidx]
                        i += 1
                        lt_diags[gene][cidx] = [ None for _ in range(len(diags[gene][cidx])) ]
                        for k in range(len(diags[gene][cidx])): 
                            foo = rescale*diags[gene][cidx][k].copy()
                            lt_diags[gene][cidx][k] = np.column_stack( (foo[:,0], foo[:,1] - foo[:,0]) )
                            if (len(lt_diags[gene][cidx][k]) > 0) and (lt_diags[gene][cidx][k][:,0].max() > maxbirth):
                                maxbirth = lt_diags[gene][cidx][k][:,0].max()

                bsummary = bsummary.astype({'ndimage_ID':int})
                
                # # Persistence Images

                pi_params = {'birth_range':(0,min([SCALE, int(np.ceil(maxbirth + sigma))] )),
                             'pers_range':(0,min([SCALE, int(np.ceil(maxxlife + sigma))])),
                             'pixel_size': pixel_size,
                             'weight': 'persistence',
                             'weight_params': {'n': pers_w},
                             'kernel':'gaussian',
                             'kernel_params':{'sigma': [[sigma, 0.0], [0.0, sigma]]} }
                                           
                pimgr = persim.PersistenceImager(**pi_params)
                extent = np.array([ pimgr.birth_range[0], pimgr.birth_range[1], pimgr.pers_range[0], pimgr.pers_range[1] ]).astype(int)
                
                tdst = dst + '{}_bw{}_{}level'.format(genes, bw, level) + os.sep
                bname = tdst + 'PI_scale{}_'.format(SCALE)
                Bname = 'PIs: KDE bandwidth {}. {}level persistence. Scale {}'.format(bw, level.title(), SCALE)
                print('------\n',Bname,'\n-------')
                if not os.path.isdir(tdst):
                    os.mkdir(tdst)
                
                img = dict()
                for gene in lt_diags:
                    img[gene] = dict()
                    for cidx in lt_diags[gene]:
                        img[gene][cidx] = np.asarray( [ pimgr.transform( lt_diags[gene][cidx][k] , skew=False) for k in range(len(lt_diags[gene][cidx])) ])
                        img[gene][cidx][ img[gene][cidx] < 0 ] = 0

                pi = np.zeros((ndims, sum(list(map(len, iter(lt_diags.values())))), extent[1]*extent[3]))
                for k in range(len(pi)):
                    i = 0
                    for gene in img:
                        for cidx in img[gene]:
                            pi[k,i] = img[gene][cidx][k].ravel()
                            i += 1
                            
                maxpis = np.max(pi, axis=2)
                maxxpis = maxpis.max()
                boxes = [ [ maxpis[k, nzcumsum[i]:nzcumsum[i+1]] for i in range(len(nzcumsum)-1) ] for k in range(len(maxpis)) ]
                qq = np.asarray([ [ np.quantile(boxes[k][i], [alpha, 1-alpha]) for i in range(len(boxes[k])) ] for k in range(len(boxes)) ])
                thr = np.max(qq[:,:,1] + iqr_factor*(qq[:,:,1] - qq[:,:,0]), axis=1)

                avg = np.zeros( (ndims, len(nzcumsum) - 1, pimgr.resolution[1], pimgr.resolution[0]))
                for k in range(len(avg)):
                    for i in range(avg.shape[1]):
                        avg[k,i] = np.mean(pi[ k, nzcumsum[i]:nzcumsum[i+1] ], axis=0).reshape(avg.shape[3], avg.shape[2]).T
                avgmax = avg.max()
                
                filename = bname + 'average.png'
                if rewrite or (not os.path.isfile(filename)):
                    fig, ax = plt.subplots(len(ratios), len(avg), figsize=(7, 2*len(ratios)), sharex=True, sharey=True)
                    ax = np.atleast_2d(ax)

                    for i in range(len(ratios)):
                        for k in range(len(avg)):
                            ax[i,k].text((extent[1] - extent[0])*.975, 0, '$H_{}$'.format(k), fontsize=fs, color='w', ha='right', va='bottom')
                            ax[i,k].imshow(avg[k,i], cmap='inferno', origin='lower', vmin=0, vmax=avgmax, extent=extent)
                            ax[i,k].text((extent[1] - extent[0])*.975, (extent[3] - extent[2])*.95, 
                                         'Max val:\n{:.2f}'.format(np.max(avg[k,i])), color='w', ha='right', va='top')
                        ax[i,1].set_xlabel(ratios.index[i], fontsize=fs)

                    ax[0,1].set_title('Avg. ' + Bname, fontsize=fs)
                    fig.tight_layout()
                    print(filename)
                    plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                    plt.close()
                
                filename = bname + 'max_vals.png'
                if rewrite or (not os.path.isfile(filename)):
                    fig, ax = plt.subplots(1, len(thr), figsize=(12, max([1.25*len(ratios), 2])), sharex=True, sharey=True)
                    ax = np.atleast_1d(ax).ravel(); 
                    for k in range(len(ax)):
                        ax[k].axvline(thr[k], c='r', ls='--', zorder=1)
                        ax[k].boxplot(boxes[k], vert=False, zorder=2, widths=0.75)
                        ax[k].text(maxxpis, 0.8, '$H_{}$'.format(k), fontsize=1.15*fs, ha='right', va='top')

                    ax[0].set_yticks(range(1, len(lt_diags)+1), iter(lt_diags.keys()), fontsize=fs)

                    ax[1].set_title(Bname, fontsize=1.15*fs)
                    ax[1].set_xlabel('Max PI value', fontsize=fs)
                    fig.tight_layout()
                    print(filename)
                    plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                    plt.close()
                
                # # Reduce dimension
                
                for perm in perms:
                    
                    pname = 'H' + '+'.join(perm.astype(str))
                    Pname = ' [$' + ' \\oplus '.join(['H_{}'.format(k) for k in perm]) + '$]'
                    full_pi = np.hstack(pi[perm])
                    maxmask = np.ones(len(full_pi), dtype=bool)
                    maxmask[functools.reduce(np.union1d, [np.nonzero(maxpis[k] > thr[k])[0] for k in perm])] = False
                    scaler = preprocessing.StandardScaler(copy=True, with_std=False, with_mean=True)
                    data = scaler.fit_transform(full_pi[maxmask].copy())
                    fulldata = scaler.transform(full_pi)

                    ## PCA
                
                    PCA = decomposition.PCA(n_components=min([6, data.shape[1]//20]), random_state=seed)
                    print('Considering the first', PCA.n_components,'PCs')
                    PCA.fit(data)
                    pca = PCA.transform(fulldata).astype('float32')
                    zero_val = PCA.transform(scaler.transform(np.zeros((1,fulldata.shape[1]))))
                    loadings = PCA.components_.T * np.sqrt(PCA.explained_variance_)
                    explained_ratio = 100*PCA.explained_variance_ratio_
                    print(explained_ratio)
                    print('Total explained var:\t', np.sum(explained_ratio), np.sum(explained_ratio[:2]))
                    pcacols = ['PC {:02d} ({:.2f})'.format(i+1,explained_ratio[i]) for i in range(pca.shape[1])]
                        
                    ### Loadings
                    
                    filename = bname + 'loadings.png'
                    if rewrite or (not os.path.isfile(filename)):
                    
                        pcarow = np.where(pca.shape[1] % pcacol == 0, pca.shape[1]//pcacol, pca.shape[1]//pcacol + 1) + 0
                        xlabs = np.tile(np.arange(0, extent[1]-2, extent[1]//3), len(perm))
                        xticks = np.hstack([ np.arange(0, extent[1]-2, xlabs[1]-xlabs[0]) + i*extent[1] for i in range(len(perm)) ]) - 0.5

                        fig, ax = plt.subplots(pcarow, pcacol, figsize=(10, 2.0*pcarow), sharex=True, sharey=True)
                        ax = np.atleast_1d(ax).ravel(); i = 0
                        for i in range(loadings.shape[1]):
                            ll = loadings[:,i].reshape( len(perm)*extent[1], extent[3], order='C').T
                            vmax = np.max(np.abs(ll))
                            ax[i].imshow(ll, cmap='coolwarm', vmax=vmax, vmin=-vmax, origin='lower')
                            ax[i].set_xlabel('PC {:02d} ({:.1f}%)'.format(i+1, explained_ratio[i]), fontsize=fs)
                            for j in range(1, len(perm)):
                                ax[i].axvline(j*extent[1] - 0.5, c='k', lw=0.5)
                            ax[i].set_xticks(xticks, xlabs, fontsize=0.85*fs)
                            for k in perm:
                                ax[i].text(.975*(k*extent[1]), 0, '$H_{}$'.format(k), c='k', va='bottom', ha='right', fontsize=fs)

                        for i in range(loadings.shape[1], len(ax)):
                            fig.delaxes(ax[i])
                        ax[pcacol//2].set_title('PCA Loadings ' + Bname, fontsize=1.15*fs)
                        fig.supylabel(ratios.index[0], fontsize=fs)
                        fig.tight_layout();
                        plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                        print(filename)
                        plt.close()
                        
                    ### CSVs
                    
                    filename = bname + 'pca.csv'
                    if rewrite or (not os.path.isfile(filename)):
                                                
                        summary = bsummary.join(pd.DataFrame(pca, columns=pcacols))
                        summary.loc[len(summary)] = ['ZERO', 0] + zero_val[0].tolist()
                        summary.to_csv(filename, index=False)
                        print(filename)
                        minspos = 1.2*summary[pcacols[:2]].min().values
                        
                        ncols = len(ratios)//nrows
                        fig, ax = plt.subplots(nrows, ncols, figsize=(7*ncols,2*nrows+1), sharex=True, sharey=True)
                        ax = np.atleast_1d(ax).ravel()

                        for i in range(len(ax)):
                            gene = ratios.index[i]
                            ax[i].set_facecolor('snow')
                            ax[i].scatter(*summary.loc[ summary['gene'] != gene, pcacols[:2] ].T.values, c='lightgray', marker='.', s=1, alpha=1, zorder=1)
                            ax[i].scatter(*summary.loc[ summary['gene'] == gene, pcacols[:2] ].T.values, c=color[i], marker=marker[i], alpha=0.75, zorder=2,
                                          edgecolor='k', linewidth=0.5)
                            ax[i].set_title(gene)
                            ax[i].text(*minspos, Pname, ha='left', va='bottom', fontsize=0.85*fs)
                            ax[i].margins(0.1)
                            ax[i].scatter(zero_val[0,0], zero_val[0,1], c='r', marker='x', zorder=3)

                        for i in range( len(ax) - len(nzcumsum)+1 , 0, -1):
                            fig.delaxes(ax[-i])

                        fig.suptitle(Bname, fontsize=fs)
                        fig.supxlabel('PC 01 [{:.1f}%]'.format(explained_ratio[0]), fontsize=fs)
                        fig.supylabel('PC 02 [{:.1f}%]'.format(explained_ratio[1]), fontsize=fs)

                        fig.tight_layout();
                        filename = bname + 'pca_' + pname + '.png'
                        plt.savefig(filename, dpi=dpi, bbox_inches='tight', format='png')
                        print(filename)
                        plt.close()
            
    return 0

if __name__ == '__main__':
    main()

