import argparse
import json
from glob import glob
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
import umap

from sklearn import decomposition, preprocessing, manifold

marker = ['D', '8', 's', '^', 'v', 'P', 'X', '*']
color = ['#56b4e9', '#f0e442', '#009e73', '#0072b2', '#d55e00', '#cc79a7', '#e69f00', '#f0e442', '#e0e0e0', '#000000']
cmap = ['Blues_r', 'Wistia', 'Greens_r', 'BuPu_r', 'Oranges_r', 'RdPu_r']
dest_directory = 'infected_focus_summer24'
seed = 42
ndims = 3
minlife = 0.05
dpi = 96
PP = 6
alpha = 0.25
iqr_factor = 1.5
pcacol = 6
fs = 12

nrows, ncols = 2,3

def plot_embedding(nzcumsum, titles, embedding, label=None, alpha=0.0, nrows=2, ncols=4, ticks=True):
    
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
    
    parser = argparse.ArgumentParser(description="Produce cell and gene metadata that will be useful later.",
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
    bw = args.bandwidth
    stepsize = args.stepsize
    SCALE = args.scale

    wsrc = '..' + os.sep + args.cell_wall_directory + os.sep
    nsrc = '..' + os.sep + args.nuclear_directory + os.sep
    tsrc = '..' + os.sep + args.location_directory + os.sep
    ksrc = '..' + os.sep + args.kde_directory + os.sep + sample + os.sep
    dst = '..' + os.sep + dest_directory + os.sep 

    dst += sample + os.sep

    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv')
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv')
    transcriptomes = np.asarray(list(metatrans['gene']))

    gsrc = '../{}level/'.format(level)
    gsrc += sample + os.sep

    Cells = utils.get_range_cell_values(dst + 'infected_cells_ids.csv', metacell, startval=1)
    Genes = utils.get_range_gene_values(dst + 'genes_to_focus_infection.csv', transcriptomes, startval=0)
    titles = transcriptomes[Genes]

    transfocus = transcell.loc[Genes, Cells.astype(str)]
    ratios = utils.normalize_counts(transfocus, normtype)
    if ratios is None:
        print('ERROR: ratios is None')
        return 0

    print('Max ratio by {}:\t{:.2f}%'.format(normtype, 100*np.max(ratios) ) )
    jsonfiles = [ [ None for j in range(ratios.shape[1]) ] for i in range(ratios.shape[0]) ]

    for i in range(len(jsonfiles)):
        foo = '{}{}/{}_-_{}_p{}_s{}_bw{}_c{:06d}.json'
        for j in range(len(jsonfiles[0])):
            filename = foo.format(gsrc, transcriptomes[Genes[i]],transcriptomes[Genes[i]],level,PP,stepsize,bw,Cells[j])
            if os.path.isfile(filename):
                jsonfiles[i][j] = filename

    orig_diags = [None for i in range(len(jsonfiles))]
    for i in range(len(orig_diags)):
        orig_diags[i] = utils.get_diagrams(jsonfiles[i], ndims, remove_inf=True)

    numpairs = 0
    genemaxk = np.zeros((len(orig_diags), ndims))
    maxlife = np.zeros((len(orig_diags), len(orig_diags[0]), len(orig_diags[0][0])))

    # [gene][cell][dimension]

    for i in range(len(orig_diags)):
        for j in range(len(orig_diags[i])):
            for k in range(len(orig_diags[i][j])):
                orig_diags[i][j][k] *= ratios[i][j]
                numpairs += len(orig_diags[i][j][k])
                if len(orig_diags[i][j][k]) > 0:
                    maxlife[i,j,k] = orig_diags[i][j][k][0,1] - orig_diags[i][j][k][0,0]
                    if genemaxk[i,k] < np.max(orig_diags[i][j][k]):
                        genemaxk[i,k] = np.max(orig_diags[i][j][k])

    print('Initial number of life-birth pairs\t:', numpairs)

    if normtype == 'gene':
        maxx = np.max(genemaxk,axis=1).reshape(len(maxlife),1,1)
    elif normtype == 'both':
        maxx = np.max(genemaxk) 

    rescale = SCALE/maxx
    maxlife *= rescale
    argmaxlife = np.argmax(maxlife, axis=-1)
    print(np.histogram(argmaxlife.ravel(), bins=range(ndims+1)))

    for i in range(len(orig_diags)):
        foo = np.sum(transfocus.loc[Genes[i]].values)
        print(i, transcriptomes[Genes[i]], foo, np.max(maxlife[i]), sep='\t')

    mhist, _ = np.histogram(argmaxlife.ravel(), bins=range(ndims+1))
    focus_dim = np.argmax(mhist)

    if normtype == 'gene':
        diags = [ [ rescale[i][0][0]*orig_diags[i][j][focus_dim].copy() for j in range(len(orig_diags[i])) ] for i in range(len(orig_diags)) ]
    elif normtype == 'both':
        diags = [ [ rescale*orig_diags[i][j][focus_dim].copy() for j in range(len(orig_diags[i])) ] for i in range(len(orig_diags)) ]

    for i in range(len(diags)):
        for j in range(len(diags[i])):
            diags[i][j] = np.atleast_2d(diags[i][j][ diags[i][j][:,1] - diags[i][j][:,0] > minlife, : ])

    nonzerodiags = np.zeros(1+len(diags), dtype=int)
    nzmask = [None for i in range(len(diags)) ]

    for i in range(len(diags)):
        nzmask[i] = np.nonzero( np.array(list(map(len, diags[i]))) > 0 )[0]
        nonzerodiags[i+1] += len(nzmask[i])
        diags[i] = [ diags[i][j] for j in nzmask[i] ]

    nzcumsum = np.cumsum(nonzerodiags)
    lt_coll = [ None for _ in range(nzcumsum[-1]) ]

    k = 0
    maxbirth = 0
    for i in range(len(diags)):
        for j in range(len(diags[i])):
            lt_coll[k] = np.column_stack( (diags[i][j][:, 0], diags[i][j][:, 1] - diags[i][j][:, 0]) )
            foo = np.max(diags[i][j][:, 0])
            if foo > maxbirth:
                maxbirth = foo
            k += 1

    foo = nzcumsum[-1]/(len(diags)*len(diags[0]))*100
    print('Non-zero diagrams:\t', nzcumsum[-1],'\nCompared to all diagrams:\t',len(diags)*len(diags[0]),'\t[{:.2f}%]'.format(foo), sep='')

    # # Persistence Images

    pi_params = {'birth_range':(0,min([SCALE, maxbirth + 10])),
                 'pers_range':(0,min([SCALE,maxlife[:,:,focus_dim].max()+10])),
                 'pixel_size': pixel_size,
                 'weight': 'persistence',
                 'weight_params': {'n': pers_w},
                 'kernel':'gaussian',
                 'kernel_params':{'sigma': [[sigma, 0.0], [0.0, sigma]]} }
                               
    pimgr = persim.PersistenceImager(**pi_params)
    extent = np.array([ pimgr.birth_range[0], pimgr.birth_range[1], pimgr.pers_range[0], pimgr.pers_range[1] ]).astype(int)
    img = np.asarray(pimgr.transform(lt_coll, skew=False))
    img[img < 0] = 0
    pi = img.reshape(img.shape[0], img.shape[1]*img.shape[2])
    maxpis = np.max(pi, axis=1)

    avg = np.zeros( (len(nzcumsum) - 1, pimgr.resolution[1], pimgr.resolution[0]))
    for i in range(len(avg)):
        s_ = np.s_[nzcumsum[i]:nzcumsum[i+1]]
        avg[i] = np.mean(img[s_], axis=0).T

    boxes = [ maxpis[nzcumsum[i]:nzcumsum[i+1]] for i in range(len(Genes)) ]
    qq = np.asarray([ np.quantile(boxes[i], [alpha, 1-alpha]) for i in range(len(boxes)) ])
    thr = np.max(qq[:,1] + iqr_factor*(qq[:,1] - qq[:,0]))
    maxmask = maxpis < thr
    
    foo = [bw, stepsize, level.title(), normtype.title(), sigma, pers_w]
    bname = 'scale{}_-_PI_{}_{}_{}_'.format(SCALE, sigma, pers_w, pixel_size)
    Bname = 'KDE bandwidth {}, stepsize {}. {}level persistence. {} normalized. PIs $\sigma = {}$. Weighted by $n^{{{}}}$.'.format(*foo)
    tdst = dst + 'G{}_{}level_{}_step{}_bw{}'.format(len(Genes), level, normtype, stepsize, bw) + os.sep
    if not os.path.isdir(tdst):
        os.mkdir(tdst)
        print(tdst)

    filename = tdst + bname + 'average_PI'
    if rewrite or (not os.path.isfile(filename)):
        vmax = np.max(np.max(avg, axis=(1,2)))
        fig, ax = plt.subplots(nrows, ncols, figsize=(2.75*ncols, 2.15*nrows+1), sharex=True, sharey=True)
        ax = np.atleast_1d(ax).ravel()

        for i in range(len(nzcumsum)-1):
            ax[i].imshow(avg[i], cmap='inferno', origin='lower', vmin=0, vmax=vmax, extent=extent)
            ax[i].text((extent[1] - extent[0])*.95, (extent[3] - extent[2])*.95, 
                       'Max val:\n{:.2f}'.format(np.max(avg[i])), color='w', ha='right', va='top')
            ax[i].set_title(transcriptomes[Genes[i]], fontsize=fs)

        for i in range( len(ax) - len(nzcumsum)+1 , 0, -1):
            fig.delaxes(ax[-i])

        fig.supxlabel('Birth', y=.04, fontsize=fs); 
        fig.supylabel('Lifetime', fontsize=fs)
        fig.suptitle(Bname, fontsize=fs)

        fig.tight_layout()
        plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
        plt.close()

    # # Reduce dimension
    filename = tdst + bname + 'max_PI_val_boxplot'
    if rewrite or (not os.path.isfile(filename)):
    
        fig, ax = plt.subplots(1, 1, figsize=(10, 2*len(Genes)/3), sharex=True, sharey=True)
        ax = np.atleast_1d(ax).ravel(); i = 0
        ax[i].axvline(thr, c='r', ls='--', zorder=1)
        ax[i].boxplot(boxes, vert=False, zorder=2, widths=0.75)
        ax[i].set_xlabel('Max PI value', fontsize=fs)
        ax[i].set_title(Bname, fontsize=fs)
        ax[i].set_yticks(range(1,len(boxes)+1), transcriptomes[Genes], fontsize=fs)

        filename = tdst + bname + 'max_PI_val_boxplot'
        plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
        
    scaler = preprocessing.StandardScaler(copy=True, with_std=False, with_mean=True)
    data = scaler.fit_transform(pi[maxmask].copy())
    nz = np.hstack( ( [0], np.cumsum([ np.sum(maxmask[nzcumsum[i]:nzcumsum[i+1]]) for i in range(len(Genes)) ])))
    fulldata = scaler.transform(pi)
    
    ## PCA
    
    method = 'PCA'
    filename = tdst + bname + method.lower() + '.csv'
    
    if rewrite or (not os.path.isfile(filename)):
        
        PCA = decomposition.PCA(n_components=data.shape[1]//20, random_state=seed)
        print('Considering the first',data.shape[1]//20,'PCs')
        PCA.fit(data)
        pca = PCA.transform(fulldata).astype('float32')
        loadings = PCA.components_.T * np.sqrt(PCA.explained_variance_)
        explained_ratio = 100*PCA.explained_variance_ratio_
        
        summary = pd.DataFrame(pca, columns=['{}{:02d} ({:.2f})'.format(method, i+1, explained_ratio[i]) for i in range(pca.shape[1])])
        summary['gene_ID'] = np.repeat(Genes, list(map(len, nzmask)) )
        summary['ndimage_ID'] = np.hstack([ Cells[ nzmask[i] ] for i in range(len(nzmask)) ])
        summary.iloc[:, [-1,-2] + list(range(pca.shape[1]) )]
        summary.to_csv(tdst + bname + method.lower() + '.csv')
        
        ###
        
        pcarow = np.where(pca.shape[1] % pcacol == 0, pca.shape[1]//pcacol, pca.shape[1]//pcacol + 1) + 0
        fig, ax = plt.subplots(pcarow, pcacol, figsize=(12, 2.25*pcarow), sharex=True, sharey=True)
        ax = np.atleast_1d(ax).ravel(); i = 0
        for i in range(loadings.shape[1]):
            ll = loadings[:,i].reshape(img.shape[1],img.shape[2]).T
            vmax = np.max(np.abs(ll))
            ax[i].imshow(ll, cmap='coolwarm', vmax=vmax, vmin=-vmax, origin='lower', extent=extent)
            ax[i].set_title('{} {:02d} ({:.2f}%)'.format(method, i+1, explained_ratio[i]), fontsize=fs)

        for i in range(loadings.shape[1], len(ax)):
            fig.delaxes(ax[i])
        fig.supxlabel('Birth', y=.04, fontsize=fs); 
        fig.supylabel('Lifetime', fontsize=fs)
        fig.suptitle(Bname, fontsize=fs)
        fig.tight_layout();
        filename = tdst + bname + method.lower() + '_loadings'
        plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
        
        ###
        
        fig, ax = plot_embedding(nzcumsum, titles, pca, method, nrows=nrows, ncols=ncols)
        fig.suptitle(Bname)
        fig.tight_layout();
        filename = tdst + bname + method.lower()
        plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
    
    ## t-SNE
    
    method = 'tSNE'
    t_sne = manifold.TSNE(
        n_components=2,
        perplexity=25,
        init="random",
        n_iter=250,
        random_state=seed,
    )
    params = t_sne.get_params()
    filename = tdst + bname + '{}_{}.csv'.format(method.lower(), params['perplexity'])
    
    if rewrite or (not os.path.isfile(filename)):
        tsne = t_sne.fit_transform(fulldata).astype('float32')
        
        summary = pd.DataFrame(tsne, columns=['{}{:02d}'.format(method, i+1) for i in range(tsne.shape[1])])
        summary['gene_ID'] = np.repeat(Genes, list(map(len, nzmask)) )
        summary['ndimage_ID'] = np.hstack([ Cells[ nzmask[i] ] for i in range(len(nzmask)) ])
        summary.iloc[:, [-1,-2] + list(range(tsne.shape[1]) )]
        summary.to_csv(tdst + bname + '{}_{}.csv'.format(method.lower(), params['perplexity']))

        fig, ax = plot_embedding(nzcumsum, titles, tsne, method, nrows=nrows, ncols=ncols)
        fig.suptitle(Bname)
        fig.tight_layout();
        filename = tdst + bname + '{}_{}.csv'.format(method.lower(), params['perplexity'])
        plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
    
    for n_neighbors in [8,16,32]:
    
        ## Locally Linear Embeddings
        
        method = 'LLE'
        params = {"n_neighbors": n_neighbors,"n_components": 2, "eigen_solver": "auto","random_state": seed}
        LLE = manifold.LocallyLinearEmbedding(method="standard", **params)
        params = LLE.get_params()
        filename = tdst + bname + '{}_{}_{}.csv'.format(method.lower(), LLE.method, params['n_neighbors'])
        
        if rewrite or (not os.path.isfile(filename)):
        
            LLE.fit(data)
            lle = LLE.transform(fulldata).astype('float32')
            
            summary = pd.DataFrame(lle, columns=['{}{:02d}'.format(method, i+1) for i in range(lle.shape[1])])
            summary['gene_ID'] = np.repeat(Genes, list(map(len, nzmask)) )
            summary['ndimage_ID'] = np.hstack([ Cells[ nzmask[i] ] for i in range(len(nzmask)) ])
            summary.iloc[:, [-1,-2] + list(range(lle.shape[1]) )]
            summary.to_csv(tdst + bname + '{}_{}_{}.csv'.format(method.lower(), LLE.method, params['n_neighbors']))

            fig, ax = plot_embedding(nzcumsum, titles, lle, method, nrows=nrows, ncols=ncols)
            fig.suptitle(Bname)
            fig.tight_layout();
            filename = tdst + bname + '{}_{}_{}'.format(method.lower(), LLE.method, params['n_neighbors'])
            plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
        
        ## Isomap
        
        method = 'ISO'
        params = {"n_neighbors": n_neighbors,"n_components": 2,"eigen_solver": "auto"}
        ISO = manifold.Isomap(**params)
        params = ISO.get_params()
        filename = tdst + bname + '{}_{}.csv'.format(method.lower(), params['n_neighbors'])
        
        if rewrite or (not os.path.isfile(filename)):
            ISO.fit(data)
            iso = ISO.transform(fulldata).astype('float32')
            
            summary = pd.DataFrame(iso, columns=['{}{:02d}'.format(method, i+1) for i in range(iso.shape[1])])
            summary['gene_ID'] = np.repeat(Genes, list(map(len, nzmask)) )
            summary['ndimage_ID'] = np.hstack([ Cells[ nzmask[i] ] for i in range(len(nzmask)) ])
            summary.iloc[:, [-1,-2] + list(range(iso.shape[1]) )]
            summary.to_csv(tdst + bname + '{}_{}.csv'.format(method.lower(), params['n_neighbors']))

            fig, ax = plot_embedding(nzcumsum, titles, iso, method, nrows=nrows, ncols=ncols)
            fig.suptitle(Bname)
            fig.tight_layout();
            filename = tdst + bname + '{}_{}.csv'.format(method.lower(), params['n_neighbors'])
            plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
            plt.close()
        
        ## UMAP 
        
        method = 'UMAP'
        ufit = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.2, n_components=2, metric='euclidean', random_state=seed, n_jobs=1)
        params = ufit.get_params()
        filename = tdst + bname + '{}_{}_{}_{}_{}.csv'.format(method.lower(), params['n_neighbors'], params['min_dist'],params['metric'],params['n_components'])
            
        if rewrite or (not os.path.isfile(filename)):
            u_umap = ufit.fit_transform(fulldata);
            summary = pd.DataFrame(u_umap, columns=['{}{:02d}'.format(method, i+1) for i in range(u_umap.shape[1])])
            summary['gene_ID'] = np.repeat(Genes, list(map(len, nzmask)) )
            summary['ndimage_ID'] = np.hstack([ Cells[ nzmask[i] ] for i in range(len(nzmask)) ])
            summary.iloc[:, [-1,-2] + list(range(u_umap.shape[1]) )]
            summary.to_csv(tdst + bname + '{}_{}_{}_{}_{}.csv'.format(method.lower(), params['n_neighbors'], params['min_dist'],params['metric'],params['n_components']))

            fig, ax = plot_embedding(nzcumsum, titles, u_umap, method, nrows=nrows, ncols=ncols)
            fig.suptitle(Bname)
            fig.tight_layout();
            filename = tdst + bname + '{}_{}_{}_{}_{}'.format(method.lower(), params['n_neighbors'], params['min_dist'],params['metric'],params['n_components'])
            plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
            plt.close()
    
    return 0

if __name__ == '__main__':
    main()

