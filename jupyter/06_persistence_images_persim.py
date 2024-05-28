import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json

from glob import glob
import os
import persim

import utils
import umap

from sklearn import decomposition, preprocessing, manifold
import argparse

dest_directory = 'infected_focus_summer24'
n_components = 12
nrows, ncols = 2,3
seed = 42
SCALE = 256
ndims = 3
minlife = 0.5
dpi = 96

colors = ['r','b','forestgreen','magenta','gold','cyan','brown','orange']
markers = ['o','d', '^','v','s','D','p','<','>']

def plot_embedding(nzcumsum, titles, embedding, alpha=0.05, label=None, nrows=2, ncols=4, ticks=False):
    
    q1, q3 = np.quantile(embedding[:,:2], [alpha, 1-alpha], axis=0)
    iqr = q3 - q1
    mn = np.maximum( q1 - 1.5*iqr, np.min(embedding[:,:2], axis=0) )
    mx = np.minimum( q3 + 1.5*iqr, np.max(embedding[:,:2], axis=0) )
        
    fig, ax = plt.subplots(nrows, ncols, figsize=(3*ncols,2*nrows+1), sharex=True, sharey=True)
    ax = np.atleast_1d(ax).ravel()
    
    for i in range(len(nzcumsum) - 1):
        ax[i].scatter(embedding[:,0], embedding[:,1], c='gray', marker='.', alpha=0.15, zorder=1)
        ax[i].set_facecolor('snow')
        s_ = np.s_[nzcumsum[i]:nzcumsum[i+1]]
        ax[i].scatter(embedding[s_,0], embedding[s_,1], c=colors[i], marker=markers[i], alpha=0.5, zorder=2)
        ax[i].set_title(titles[i])
        ax[i].set_xlim(mn[0],mx[0])
        ax[0].set_ylim(mn[1],mx[1])
        ax[i].tick_params(left=ticks, labelleft=ticks, labelbottom=ticks, bottom=ticks);
    
    for i in range( len(ax) - len(nzcumsum)+1 , 0, -1):
        fig.delaxes(ax[-i])

    if label is not None:
        fig.supxlabel(label + ' 01')
        fig.supylabel(label + ' 02')
    
    fig.tight_layout();

    return fig, ax

def main():
    
    parser = argparse.ArgumentParser(description="Produce cell and gene metadata that will be useful later.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='Last major update: May 2024. © Erik Amezquita')
    parser.add_argument("sample", type=str,
                        help="Cross-section sample name")
    parser.add_argument("level", type=str, choices=['sup','sub'],
                        help="filtration to use")
    parser.add_argument("-n", "--norm_type", type=str, choices=['both','gene'],
                        help="how to normalize all the KDEs")
    parser.add_argument("-p", "--pers_w", type=int, default=1,
                        help="power for weight function")                        
    parser.add_argument("-s", "--sigma", type=float, default=2,
                        help="sigma for persistent images")  
    parser.add_argument("-x", "--pixel_size", type=int, default=4,
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

    foo = glob(gsrc + transcriptomes[Genes[0]] + os.sep + '*')[0]
    bar = os.path.split(foo)[1].split('_')
    PP = int(bar[4][1:])
    stepsize = int(bar[5][1:])
    bw = int(bar[6][2:])

    transfocus = transcell.loc[Genes, Cells.astype(str)]
    ratios = utils.normalize_counts(transfocus, normtype)
    if ratios is None:
        print('ERROR')
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
        nzmask[i] = np.nonzero( np.array(list(map(len, diags[i]))) > 1  )[0]
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
    extent = [ pimgr.birth_range[0], pimgr.birth_range[1], pimgr.pers_range[0], pimgr.pers_range[1] ]
    img = np.asarray(pimgr.transform(lt_coll, skew=False, n_jobs=1))

    # # Explore

    bname = '{}level_-_by_{}_-_sigma_{}_-_pers+n_{}_-_pixel+size_{}'.format(level, normtype, int(sigma), pers_w, pixel_size)
    Bname = '{}level persistence. {} normalized. $\sigma = {}$. Weighted by $n^{}$. Pixel size {}.'.format(level.title(), normtype.title(), int(sigma), pers_w, pixel_size)
    tdst = dst + bname + os.sep
    if not os.path.isdir(tdst):
        os.mkdir(tdst)

    avg = np.zeros( (len(nzcumsum) - 1, pimgr.resolution[1], pimgr.resolution[0]))
    for i in range(len(avg)):
        s_ = np.s_[nzcumsum[i]:nzcumsum[i+1]]
        avg[i] = np.mean(img[s_], axis=0).T


    fig, ax = plt.subplots(nrows, ncols, figsize=(2.75*ncols, 2.15*nrows+1), sharex=True, sharey=True)
    ax = np.atleast_1d(ax).ravel()

    for i in range(len(nzcumsum)-1):
        ax[i].imshow(avg[i], cmap='inferno', origin='lower', vmin=0,  extent=extent)
        ax[i].text((extent[1] - extent[0])*.95, (extent[3] - extent[2])*.95, 
                   'Max val:\n{:.1f}'.format(np.max(avg[i])), color='w', ha='right', va='top')
        ax[i].set_title(transcriptomes[Genes[i]])

    for i in range( len(ax) - len(nzcumsum)+1 , 0, -1):
        fig.delaxes(ax[-i])

    fig.supxlabel('Birth', y=.04); 
    fig.supylabel('Persistence')
    fig.suptitle(Bname)

    fig.tight_layout()
    filename = tdst + 'average_persistence_images'
    plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
    plt.close()

    # # Reduce dimension

    pi = img.reshape(img.shape[0], img.shape[1]*img.shape[2])
    
    for with_std in [True, False]:
    
        scaler = preprocessing.StandardScaler(with_mean=True, with_std=with_std, copy=True)
        data = scaler.fit_transform(pi)

        # # PCA

        PCA = decomposition.PCA(n_components=n_components, random_state=seed)
        method = 'PCA'
        bname = '{}_-_with+std_{}'.format(method.lower(), str(with_std) )
        filename = tdst+bname+'.npy'
        
        if rewrite or (not os.path.isfile(filename)):

            pca = PCA.fit_transform(data).astype('float32')

            np.save(tdst+bname+'.npy', pca, allow_pickle=True)

            fig, ax = plot_embedding(nzcumsum, titles, pca, 0.01, method, nrows=nrows, ncols=ncols)
            fig.suptitle( Bname + ' [ {:.1f}% , {:.1f}%, {:.1f}% ]'.format(*PCA.explained_variance_ratio_[:3]*100) )
            fig.tight_layout();
            filename = tdst + bname
            plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
            plt.close()

        if False:
            
            # # Locally Linear Embeddings
        
            n_neighbors = 12  # neighborhood which is used to recover the locally linear structure
            params = {"n_neighbors": n_neighbors,"n_components": n_components,"eigen_solver": "auto","random_state": seed, "n_jobs":1}
            LLE = manifold.LocallyLinearEmbedding(method="standard", **params)
            params = LLE.get_params()
            method = 'LLE'
            bname = '{}_-_method_{}_-_n+neighbors_{}_-_with+std_{}'.format(method.lower(), LLE.method, params['n_neighbors'], str(with_std))
            filename = tdst+bname+'.npy'
            
            if rewrite or (not os.path.isfile(filename)):
                lle = LLE.fit_transform(data).astype('float32')
            
                np.save(tdst+bname+'.npy', lle, allow_pickle=True)

                fig, ax = plot_embedding(nzcumsum, titles, lle, 0.01, method, nrows=nrows, ncols=ncols)
                fig.suptitle(Bname)
                fig.tight_layout();
                filename = tdst + bname
                plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
                plt.close()

            # # tSNE
            
            for perplexity in [30]:
                
                t_sne = manifold.TSNE(n_components=2, perplexity=perplexity, init="pca", n_iter=250, random_state=seed, metric='euclidean', n_jobs=1)
                params = t_sne.get_params()
                method = 'tSNE'
                bname = '{}_-_perplexity_{}_-_init_{}_-_n+iter_{}_-_with+std_{}'.format(method.lower(), params['perplexity'], params['init'],params['n_iter'],str(with_std))
                filename = tdst+bname+'.npy'

                if rewrite or (not os.path.isfile(filename)):
                    
                    tsne = t_sne.fit_transform(data).astype('float32')
                    print(bname)
                    np.save(tdst+bname+'.npy', tsne, allow_pickle=True)

                    fig, ax = plot_embedding(nzcumsum, titles, tsne, 0.01, method, nrows=nrows, ncols=ncols)
                    fig.suptitle(Bname)
                    fig.tight_layout();
                    filename = tdst + bname
                    plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
                    plt.close()

        # # UMAP
        
        for n_neighbors in [12,24]:
            for min_dist in [0.0, 0.1]:
                for metric in ['euclidean', 'manhattan', 'chebyshev']:
                    
                    ufit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state=seed, n_jobs=1)
                    params = ufit.get_params();
                    method = 'UMAP'
                    bname = '{}_-_n+neighbors_{}_-_min+dist_{}_-_metric_{}_-_with+std_{}'.format(method.lower(), params['n_neighbors'], params['min_dist'],params['metric'], str(with_std) )                
                    filename = tdst+bname+'.npy'

                    if rewrite or (not os.path.isfile(filename)):
                        
                        u_umap = ufit.fit_transform(data);

                        print(bname)
                        np.save(tdst+bname+'.npy', u_umap, allow_pickle=True)

                        fig, ax = plot_embedding(nzcumsum, titles, u_umap, 0.01, method, nrows=nrows, ncols=ncols)
                        fig.suptitle(Bname)
                        fig.tight_layout();
                        filename = tdst + bname
                        plt.savefig(filename + '.png', dpi=dpi, bbox_inches='tight', format='png')
                        plt.close()

    return 0

if __name__ == '__main__':
    main()

