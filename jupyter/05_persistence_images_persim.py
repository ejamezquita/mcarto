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

marker = ['D', '8', 's', '^', 'v', 'P', 'X', '*']
color = ['#56b4e9', '#f0e442', '#009e73', '#0072b2', '#d55e00', '#cc79a7', '#e69f00', '#e0e0e0', '#000000']
cmap = ['Blues_r', 'Wistia', 'Greens_r', 'BuPu_r', 'Oranges_r', 'RdPu_r', 'YlOrBr_r', 'gray', 'gist_gray']
dest_directory = 'infected_focus_summer24'
seed = 42
ndims = 3
minlife = 0.0
dpi = 96
PP = 6
alpha = 0.25
iqr_factor = 1.5
pcacol = 3
fs = 12

perms = [np.nonzero(p)[0] for p in itertools.product(range(2), repeat=ndims)][1:]

perms = [ np.array([1,2]) ]

nrows, ncols = 1,2
def plot_embedding(nzcumsum, titles, embedding, label=None, alpha=0.0, nrows=nrows, ncols=ncols, ticks=True):
    
    q1, q3 = np.quantile(embedding[:,:2], [alpha, 1-alpha], axis=0)
    iqr = q3 - q1
    mn = np.maximum( q1 - 1.5*iqr, np.min(embedding[:,:2], axis=0) )
    mx = np.minimum( q3 + 1.5*iqr, np.max(embedding[:,:2], axis=0) )
        
    fig, ax = plt.subplots(nrows, ncols, figsize=(3*ncols,2*nrows+1), sharex=True, sharey=True)
    ax = np.atleast_1d(ax).ravel()
    
    for i in range(len(nzcumsum) - 1):
        ax[i].scatter(embedding[:,0], embedding[:,1], c='lightgray', marker='.', alpha=1, zorder=1, s=3)
        ax[i].set_facecolor('snow')
        s_ = np.s_[nzcumsum[i]:nzcumsum[i+1]]
        ax[i].scatter(embedding[s_,0], embedding[s_,1], c=color[i], marker=marker[i], alpha=0.5, zorder=2,
                      linewidth=0.5, edgecolor='k', s=50)
        ax[i].set_title(titles[i])
        ax[i].set_xlim(mn[0],mx[0])
        ax[i].set_ylim(mn[1],mx[1])
        ax[i].tick_params(left=ticks, labelleft=ticks, labelbottom=ticks, bottom=ticks);
    
    for i in range( len(ax) - len(nzcumsum)+1 , 0, -1):
        fig.delaxes(ax[-i])

    if label is not None:
        fig.supxlabel(label + ' 01')
        fig.supylabel(label + ' 02')

    return fig, ax

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
    args = parser.parse_args()
    
    rewrite = args.rewrite_results
    sample = args.sample
    normtype = args.norm_type
    level = args.level
    sigma = args.sigma
    pers_w = args.pers_w
    pixel_size = args.pixel_size
    stepsize = args.stepsize
    
    if args.scale is None:
        SCALES = [16,24,32,40,48]
    else:
        SCALES = [args.scale]
        
    if args.bandwidth is None:
        BW = [10,15,20,25,30]
    else:
        BW = [args.bandwidth]

    wsrc = '..' + os.sep + args.cell_wall_directory + os.sep
    nsrc = '..' + os.sep + args.nuclear_directory + os.sep
    tsrc = '..' + os.sep + args.location_directory + os.sep
    ksrc = '..' + os.sep + args.kde_directory + os.sep + sample + os.sep
    dst = '..' + os.sep + dest_directory + os.sep 

    dst += sample + os.sep

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv', index_col=0)
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv')
    transcriptomes = np.asarray(list(metatrans['gene']))

    gsrc = '../{}level/'.format(level)
    gsrc += sample + os.sep

    Cells = utils.get_range_cell_values(dst + 'infected_cells_ids.csv', metacell, startval=1)
    Cells = np.setdiff1d( Cells, metacell[metacell['number_nuclei'] > 1].index)
    Genes = utils.get_range_gene_values(dst + 'genes_to_focus_infection.csv', transcriptomes, startval=0)
    titles = transcriptomes[Genes]

    transfocus = transcell.loc[Genes, Cells.astype(str)]
    ratios = utils.normalize_counts(transfocus, normtype)
    if ratios is None:
        print('ERROR: ratios is None')
        return 0
    print('Max ratio by {}:\t{:.2f}%'.format(normtype, 100*np.max(ratios) ), np.sum(ratios > 0, axis=1) )
    
    for bw in BW:
        
        jsonfiles = [ [ None for j in range(ratios.shape[1]) ] for i in range(ratios.shape[0]) ]
        for i in range(len(jsonfiles)):
            foo = '{}{}/{}_-_{}_p{}_s{}_bw{}_c{:06d}.json'
            for j in range(len(jsonfiles[0])):
                filename = foo.format(gsrc, transcriptomes[Genes[i]],transcriptomes[Genes[i]],level,PP,stepsize,bw,Cells[j])
                if os.path.isfile(filename):
                    jsonfiles[i][j] = filename

        for SCALE in SCALES:
            orig_diags = [ utils.get_diagrams(jsonfiles[i], ndims, remove_inf=True) for i in range(len(jsonfiles))]
            orig_diags, rescale, maxlife, focus_dim = utils.normalize_persistence_diagrams(orig_diags, ratios, normtype, SCALE)
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

            lt_coll = [ [None for _ in range(np.sum(lt_mask)) ] for _ in range(ndims) ]
            for i in range(len(gmask)):
                for k in range(len(lt_coll)):
                    d = diags[gmask[i]][cmask[i]][k]
                    lt_coll[k][i] = np.column_stack( (d[:, 0], d[:, 1] - d[:, 0])  )

            maxbirth = 0
            for k in range(len(lt_coll)):
                for i in range(len(lt_coll[k])):
                    if len(lt_coll[k][i]) > 0:
                        b = np.max(lt_coll[k][i][:,0])
                        if b > maxbirth:
                            maxbirth = b
            
            
            # # Persistence Images

            pi_params = {'birth_range':(0,min([SCALE, int(np.ceil(maxbirth + sigma))] )),
                         'pers_range':(0,min([SCALE, int(np.ceil(maxlife[:,:,focus_dim].max()+sigma))])),
                         'pixel_size': pixel_size,
                         'weight': 'persistence',
                         'weight_params': {'n': pers_w},
                         'kernel':'gaussian',
                         'kernel_params':{'sigma': [[sigma, 0.0], [0.0, sigma]]} }
                                       
            pimgr = persim.PersistenceImager(**pi_params)
            extent = np.array([ pimgr.birth_range[0], pimgr.birth_range[1], pimgr.pers_range[0], pimgr.pers_range[1] ]).astype(int)
            bname = 'scale{}_-_PI_{}_{}_{}_'.format(SCALE, sigma, pers_w, pixel_size)
            foo = [bw, stepsize, level.title(), normtype.title(), sigma, pers_w]
            Bname = 'KDE bandwidth {}, stepsize {}. {}level persistence. {} normalized. PIs $\sigma = {}$. Weighted by $n^{{{}}}$.'.format(*foo)
            tdst = dst + 'G{}_{}level_{}_step{}_bw{}'.format(len(Genes), level, normtype, stepsize, bw) + os.sep
            if not os.path.isdir(tdst):
                os.mkdir(tdst)
                print(tdst)
            
            img = np.zeros((len(lt_coll), len(lt_coll[0]), extent[1], extent[3]))
            for k in range(len(img)):
                img[k] = np.asarray(pimgr.transform(lt_coll[k], skew=False))
            img[img < 0] = 0
            pi = np.zeros((img.shape[0], img.shape[1], img.shape[2]*img.shape[3]))
            for k in range(len(pi)):
                pi[k] = img[k].reshape(pi.shape[1], pi.shape[2])
            maxpis = np.max(pi, axis=2)
            boxes = [ [ maxpis[k, nzcumsum[i]:nzcumsum[i+1]] for i in range(len(nzcumsum)-1) ] for k in range(len(maxpis)) ]
            qq = np.asarray([ [ np.quantile(boxes[k][i], [alpha, 1-alpha]) for i in range(len(boxes[k])) ] for k in range(len(boxes)) ])
            thr = np.max(qq[:,:,1] + iqr_factor*(qq[:,:,1] - qq[:,:,0]), axis=1)
            
            filename = tdst + bname + 'average_PI'
            if rewrite or (not os.path.isfile(filename+'.png')):
                avg = np.zeros( (len(img), len(nzcumsum) - 1, pimgr.resolution[1], pimgr.resolution[0]))
                for k in range(len(avg)):
                    for i in range(avg.shape[1]):
                        s_ = np.s_[nzcumsum[i]:nzcumsum[i+1]]
                        avg[k,i] = np.mean(img[k,s_], axis=0).T
                        
                vmax = avg.max()
                fig, ax = plt.subplots(len(avg), avg.shape[1], figsize=(2*avg.shape[1], 6), sharex=True, sharey=True)

                for k in range(avg.shape[0]):
                    for i in range(len(nzcumsum)-1):
                        ax[k,i].imshow(avg[k,i], cmap='inferno', origin='lower', vmin=0, vmax=vmax, extent=extent)
                        ax[k,i].text((extent[1] - extent[0])*.95, (extent[3] - extent[2])*.95, 
                                     'Max val:\n{:.2f}'.format(np.max(avg[k,i])), color='w', ha='right', va='top')
                    ax[k,0].set_ylabel('$H_{}$'.format(k), fontsize=fs, rotation=0)
                for i in range(avg.shape[1]):
                    ax[0,i].set_title(transcriptomes[Genes[i]], fontsize=fs)

                fig.supxlabel('Birth', y=.04, fontsize=fs); 
                fig.supylabel('Lifetime', fontsize=fs)
                fig.suptitle(Bname, fontsize=fs)

                fig.tight_layout()
                plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
                plt.close()
            
            filename = tdst + bname + 'max_PI_val_boxplot'
            if rewrite or (not os.path.isfile(filename+'.png')):
                fig, ax = plt.subplots(1, len(thr), figsize=(12, 2*len(Genes)/3), sharex=True, sharey=True)
                ax = np.atleast_1d(ax).ravel(); 
                for k in range(len(ax)):
                    ax[k].axvline(thr[k], c='r', ls='--', zorder=1)
                    ax[k].boxplot(boxes[k], vert=False, zorder=2, widths=0.75)
                    ax[k].set_title('$H_{}$'.format(k), fontsize=fs)

                ax[0].set_yticks(range(1, len(Genes)+1), transcriptomes[Genes], fontsize=fs)

                fig.suptitle(Bname, fontsize=fs)
                fig.supxlabel('Max PI value', fontsize=fs)
                plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
                plt.close()
            
            # # Reduce dimension
            
            for perm in perms:
                
                xlabs = np.tile(np.arange(0, img.shape[2], img.shape[2]//3), len(perm))
                xticks = np.hstack([ np.arange(0, img.shape[2], xlabs[1]-xlabs[0]) + i*img.shape[2] for i in range(len(perm)) ])
                xlabs = xlabs.astype(str)
                xlabs[xlabs == '0'] = [ '$H_{}$'.format(perm[i]) for i in range(len(perm)) ]

                pname = 'H' + '+'.join(perm.astype(str))
                Pname = ' [$' + ' \\oplus '.join(['H_{}'.format(k) for k in perm]) + '$]'
                full_pi = np.hstack(pi[perm])
                maxmask = np.ones(len(full_pi), dtype=bool)
                maxmask[functools.reduce(np.union1d, [np.nonzero(maxpis[k] > thr[k])[0] for k in perm])] = False

                scaler = preprocessing.StandardScaler(copy=True, with_std=False, with_mean=True)
                data = scaler.fit_transform(full_pi[maxmask].copy())
                fulldata = scaler.transform(full_pi)
                print(perm, data.shape, fulldata.shape)
                
                ## PCA
            
                method = 'PCA'
                filename = tdst + bname + method.lower() + '_' + pname + '.csv'
                
                if rewrite or (not os.path.isfile(filename+'.png')):
                    
                    PCA = decomposition.PCA(n_components=min([6, data.shape[1]//20]), random_state=seed)
                    print('Considering the first', PCA.n_components,'PCs')
                    PCA.fit(data)
                    pca = PCA.transform(fulldata).astype('float32')
                    loadings = PCA.components_.T * np.sqrt(PCA.explained_variance_)
                    explained_ratio = 100*PCA.explained_variance_ratio_
                    print('{}\tTotal explained var:\t{:.2f} [{:.2f}]'.format(perm, np.sum(explained_ratio), np.sum(explained_ratio[:2])) )
                    
                    summary = bsummary.join(pd.DataFrame(pca, columns=['{}{:02d} ({:.2f})'.format(method,i+1,explained_ratio[i]) for i in range(pca.shape[1])]))
                    summary.to_csv(filename, index=False)
                    
                    ###
                    
                    pcarow = np.where(pca.shape[1] % pcacol == 0, pca.shape[1]//pcacol, pca.shape[1]//pcacol + 1) + 0
                    fig, ax = plt.subplots(pcarow, pcacol, figsize=(10, 2.5*pcarow), sharex=True, sharey=True)
                    ax = np.atleast_1d(ax).ravel(); i = 0
                    for i in range(loadings.shape[1]):
                        ll = loadings[:,i].reshape( len(perm)*img.shape[2], img.shape[3], order='C').T
                        vmax = np.max(np.abs(ll))
                        ax[i].imshow(ll, cmap='coolwarm', vmax=vmax, vmin=-vmax, origin='lower')
                        ax[i].set_title('{} {:02d} ({:.2f}%)'.format(method, i+1, explained_ratio[i]), fontsize=fs)
                        for j in range(1, len(perm)):
                            ax[i].axvline(j*img.shape[2] - .5, c='k', lw=0.5)
                        ax[i].set_xticks(xticks, xlabs, fontsize=fs)

                    for i in range(loadings.shape[1], len(ax)):
                        fig.delaxes(ax[i])
                    fig.supxlabel('Birth', y=.04, fontsize=fs); 
                    fig.supylabel('Lifetime', fontsize=fs)
                    fig.suptitle(Bname, fontsize=fs)
                    fig.tight_layout();
                    filename = tdst + bname + method.lower() + '_' + pname +'_loadings'
                    plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
                    plt.close()
                    
                    ###
                    
                    fig, ax = plot_embedding(nzcumsum, titles, pca, alpha=0.00, label=None, nrows=nrows, ncols=ncols)
                    fig.suptitle(Bname+Pname, fontsize=fs)
                    fig.supxlabel('PC 01 [{:.1f}%]'.format(explained_ratio[0]), fontsize=fs)
                    fig.supylabel('PC 02 [{:.1f}%]'.format(explained_ratio[1]), fontsize=fs)

                    fig.tight_layout();
                    filename = tdst + bname + method.lower() + '_' + pname
                    plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
                    plt.close()
            
    return 0

if __name__ == '__main__':
    main()

