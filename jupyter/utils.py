import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tifffile as tf
from glob import glob
import os

from scipy import ndimage, spatial, stats
from sklearn import neighbors

from KDEpy import FFTKDE
import json

# ======================================================================
# PART 0
# ======================================================================

def is_type_tryexcept(s, testtype=int):
    """ Returns True if string is a number. """
    try:
        testtype(s)
        return True
    except ValueError:
        return False

def get_range_cell_values(arginput=None, meta=None, startval=0):
    
    if arginput is None:
        Vals = meta.index.values[startval:]
    elif is_type_tryexcept(arginput, int):
        Vals = [int(arginput)]
    elif os.path.isfile(arginput):
        focus = pd.read_csv(arginput)
        if 'ndimage_cellID' in focus.columns:
            Vals = focus['ndimage_cellID'].values
        elif 'orig_cellID' in focus.columns:
            Vals = np.zeros(len(focus), dtype=int)
            for i in range(len(Vals)):
                Vals[i] = meta[meta['orig_cellID'] == focus.iloc[i]['orig_cellID']].index[0]
        else:
            Vals = None
            print('ERROR: Unable to choose cell ID values from input')
    else:
        Vals = None
        print('ERROR: Unable to choose cell ID values from input')
        
    return Vals
    
def get_range_gene_values(arginput=None, meta=None, startval=0):
    
    if arginput is None:
        Vals = range(startval, len(meta), 1)
    elif is_type_tryexcept(arginput, int):
        Vals = [int(arginput)]
    elif os.path.isfile(arginput):
        focus = pd.read_csv(arginput)
        if 'gene_ID' in focus.columns:
            Vals = focus['gene_ID'].values
        else:
            Vals = np.zeros(len(focus), dtype=int)
            for i in range(len(Vals)):
                Vals[i] = np.nonzero(meta == focus.iloc[i,0])[0][0]
    else:
        Vals = None
        print('ERROR: Unable to choose gene ID values from input')
        
    return Vals

pxs = 75
plot_pxbar = np.s_[-25:-5, 5 : (5 + pxs)]
def get_cell_img(cidx, metacell, label, lnuc, nnuc, PP=10, pxbar=False):
    s_ = (np.s_[max([0, metacell.loc[cidx, 'y0'] - PP]) : min([label.shape[0], metacell.loc[cidx, 'y1'] + PP])],
          np.s_[max([1, metacell.loc[cidx, 'x0'] - PP]) : min([label.shape[1], metacell.loc[cidx, 'x1'] + PP])])
    extent = (s_[1].start, s_[1].stop, s_[0].start, s_[0].stop)
    
    cell = label[s_].copy()
    cell[ label[s_] > 0 ] = 0
    cell[ label[s_] == cidx ] = nnuc + 1
    cell[ lnuc[s_] > 0 ] = lnuc[s_][lnuc[s_] > 0]
    cell[ label[s_] == 0 ] = -1
    
    if pxbar:
        cell[plot_pxbar] = -1

    return cell, extent
    
# ======================================================================
# PART I
# ======================================================================

def correct_boundary_transcripts(tlabs, coords, label, tpercell, R = 25, deltamax=50):
    for i in np.nonzero(tlabs == 0)[0]:
        x,y = coords[:2,i]
        ss = np.s_[max([0, y - R]) : min([label.shape[0], y + R]), 
                   max([0, x - R]) : min([label.shape[1], x + R])]
        cells = np.unique(label[ss])[1:]
        newlab = cells[ np.argmax(tpercell[cells]) ]
        com = np.flip(np.mean(np.asarray(np.nonzero(label[ss] == newlab)), axis=1))
        com[0] += x - R
        com[1] += y - R
        dv = com - coords[:2,i]
        dv = dv/np.linalg.norm(dv)
        delta = 1
        x,y = (coords[:2,i] + delta*dv).astype(int)
        
        while(label[y,x] != newlab) and (delta < deltamax):
            delta += 1
            x,y = (coords[:2,i] + delta*dv).astype(int)
        if delta < deltamax:
            coords[:2,i] = [x,y]
            tlabs[i] = newlab
        else:
            print('Review index', i)

    return tlabs, coords

def transcript_shift(i, ndist, nidxs, cat, cdtlabs, cdtmask, cdtcoords, edtmask, edtvals, label, radius):
    mask = cdtlabs[nidxs[i]] == cat[i,1]
    nearest = np.average(cdtcoords[:2,nidxs[i][mask]], axis=1, weights = radius - ndist[i][mask])
    dv = nearest - cdtcoords[:2,edtvals[i]]
    dv = dv/np.linalg.norm(dv)
    x,y = cdtcoords[:2,edtvals[i]]
    
    delta = 1
    x,y = (cdtcoords[:2,edtvals[i]] + delta*dv).astype(int)
    while(label[y,x] != cat[i,1]) and (delta < radius):
        delta += 1
        x,y = (cdtcoords[:2,edtvals[i]] + delta*dv).astype(int)
    if delta < radius:
        return [ [x,y], cat[i,1], True ]
    else:
        nearest = cdtcoords[:2,nidxs[i][mask][0]]
        dv = nearest - cdtcoords[:2,edtvals[i]]
        dv = dv/np.linalg.norm(dv)
        x,y = cdtcoords[:2,edtvals[i]]
        
        delta = 1
        x,y = (cdtcoords[:2,edtvals[i]] + delta*dv).astype(int)
        while(label[y,x] != cat[i,1]) and (delta < radius):
            delta += 1
            x,y = (cdtcoords[:2,edtvals[i]] + delta*dv).astype(int)
        if delta < radius:
            return [ [x,y], cat[i,1], True ]
        else:
            return [ [x,y], cat[i,1], False ]

# Produce a N x 4 matrix with metadata.
# For each transcript, consider its nearest neighbors
# Count how many neighbors belong to what cells
# IF neighbors include transcripts belonging to more than 1 cell,
# THEN 1st Col: Number of cells to which nearest neighbors belong
#      2nd Col: Index of most popular cell (Cell with the most neighbors)
#      3rd Col: BOOL: Current transcript cell location different from majority of neighbors
#      4th Col: Percentage of neighbors belonging to most popular cell

def get_neighbor_data(nidxs, indexing, minneighs, cdtlabs, cdtmask, cdtcoords, edtmask, edtvals):
    
    cat = np.zeros((len(nidxs),4), dtype=int)

    for i in indexing:
        foo, bar = np.unique(cdtlabs[nidxs[i][1:]], return_counts=True)
        if (len(foo) > 1) or ( (len(foo) == 1) & (cat[i,1] != cdtlabs[edtvals[i]]) ):
            cts = bar/np.sum(bar)
            
            cat[i,0] = len(foo)
            cat[i,1] = foo[np.argmax(cts)]
            cat[i,2] = cat[i,1] != cdtlabs[edtvals[i]]
            cat[i,3] = 100*np.max(cts)
    
    return cat
            
def correct_shifted_transcripts(cdtlabs, cdtmask, cdtcoords, edtmask, edtvals, label, maxdwall=6, minneighs=5, minprob=74, radius=30):
    neigh = neighbors.NearestNeighbors(radius=radius)
    neigh.fit(cdtcoords.T)
    
    ndist, nidxs = neigh.radius_neighbors(cdtcoords[:, edtmask].T, sort_results=True)
    nneighs = np.array(list(map(len,nidxs))) - 1
    indexing = np.nonzero(nneighs > minneighs)[0]

    cat = get_neighbor_data(nidxs, indexing, minneighs, cdtlabs, cdtmask, cdtcoords, edtmask, edtvals)
    
    indexing = np.nonzero((cat[:,2] == 1) & (cat[:,3] > minprob))[0]
    for i in indexing:
        shift = transcript_shift(i, ndist, nidxs, cat, cdtlabs, cdtmask, cdtcoords, edtmask, edtvals, label, radius)
        if shift[2]:
            cdtcoords[:2, edtvals[i]] = shift[0] 
            cdtlabs[edtvals[i]] = shift[1]
        else:
            print('Pay attention to index\t',i)

    return len(indexing), cdtlabs, cdtcoords
    
# ======================================================================
# PART II
# ======================================================================

    
def celllocs_read(filename):
    celllocs = pd.read_csv(filename)
    sel = [0,3,4,5,6,7,8,9]
    celllocs = celllocs.iloc[~np.any(celllocs.iloc[:, :5].isnull().values, axis=1)]
    celllocs = celllocs[celllocs['Cell.Area..px.'] > 9]
    celllocs = celllocs.astype(dict(zip(celllocs.columns[np.array(sel)], [int for i in range(len(sel))])))
    return celllocs

def match_original_ndimage(celllocs, wall, label, cellnum):
    cnuclei = np.asarray(ndimage.center_of_mass(wall, label, range(1,cellnum+1)))
    dcoords = celllocs.loc[:, ['X.location', 'Y.location']].values
    cdist = spatial.distance.cdist(np.flip(cnuclei, axis=1), dcoords, metric='euclidean')
    argmatches = np.argmin(cdist, axis=1)
    matches = np.min(cdist, axis=1)
    orig_cellID = celllocs['Cell.ID..'].values[argmatches]
    orig_cellID[ matches > 5+0.1*(np.sqrt(celllocs.iloc[argmatches].loc[:, 'Cell.Area..px.'])) ] = 0

    return dcoords, cnuclei, argmatches, orig_cellID

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

def generate_transcell_metadata(translocs, transcriptomes, cellnum, label):
    meta = np.zeros((len(transcriptomes), cellnum), dtype=int)
    bins = np.arange(1, cellnum + 2)
    for tidx in range(len(meta)):
        coords = translocs[tidx].loc[:, ['X', 'Y']].values.T
        if coords.shape[1] > 0:
            meta[tidx], _ = np.histogram(label[coords[1], coords[0]], bins=bins)
        
    meta = pd.DataFrame(meta, columns=bins[:-1])
    meta['gene'] = transcriptomes

    return meta

# ======================================================================
# PART III
# ======================================================================

pows2 = 2**np.arange(20) + 1

def kde_grid_generator(stepsize, maxdims, pows2 = pows2, pad=1.5):
    axes = [ np.arange(0, maxdims[i], stepsize) for i in range(len(maxdims)) ]
    AXES = [ None for i in range(len(axes)) ]
    
    for i in range(len(axes)):
        m = np.nonzero(pows2 > pad*len(axes[i]))[0][0]
        foo = pows2[m] - len(axes[i])
        neg = foo//2
        pos = np.where(foo%2==0, foo//2, foo//2 + 1) + 0
        AXES[i] = np.hstack((np.arange(-neg, 0, 1)*stepsize, axes[i], np.arange(len(axes[i]), len(axes[i])+pos, 1)*stepsize))
    
    AXES = np.meshgrid(*AXES, indexing='ij')
    grid = np.column_stack([ np.ravel(AXES[i]) for i in range(len(AXES)) ])
    
    mask = np.ones(len(grid), dtype=bool)
    for i in range(len(axes)):
        mask = mask & (grid[:,i] >= 0) & (grid[:,i] < maxdims[i])

    return axes, grid, mask

# labels_ = label[s_]
def cell_grid_preparation(cidx, cell, labels_, extent, zmax, stepsize, cell_nuc, pows2=pows2, maxdims=None, exclude_nuclei=True):
    
    if maxdims is None:
        maxdims = ( cell.shape[1], cell.shape[0], zmax )
    
    axes, grid, gmask = kde_grid_generator(stepsize=stepsize, maxdims=maxdims, pows2 = pows2, pad=1.5)
    grid[:, :2] = grid[:, :2] + np.array([ extent[0], extent[2] ])
    cgrid = grid[gmask].copy()
    
    cgrid[:,:2] = grid[gmask][:,:2] - np.array([extent[0], extent[2]])
    
    if exclude_nuclei:
        outside_walls = cell[cgrid[:,1],cgrid[:,0]] < 1
        outside_walls |= labels_[cgrid[:,1] , cgrid[:,0]] != cidx
        
        nuc_lims = cell_nuc.loc[ (cell_nuc['ndimage_ID'] == cidx), ['ndimage_ID','nuc_ID','N_inside','n_bot','n_top']]
        foo = np.setdiff1d( np.unique(cell), nuc_lims['nuc_ID'].values)[:-1]
        for v in foo:
            outside_walls |= cell[cgrid[:,1], cgrid[:,0]] == v

        for j in range(len(nuc_lims)):
            _, nidx, N_inside, n_bot, n_top = nuc_lims.iloc[j]
            if n_bot < n_top:
                thr_mask = (cgrid[:,2] >= n_bot) & (cgrid[:,2] <= n_top)
            else:
                thr_mask = (cgrid[:,2] <= n_top) | (cgrid[:,2] >= n_bot)

            outside_walls |= ((cell[cgrid[:,1],cgrid[:,0]] == nidx) & thr_mask)
    else:
        outside_walls = labels_[cgrid[:,1] , cgrid[:,0]] != cidx
    
    return axes, grid, gmask, cgrid, outside_walls
        
def cell_weighted_kde(coords, grid, weights, bw, gmask, stepsize, cgridmask, axes):

    kde = FFTKDE(kernel='gaussian', bw=bw, norm=2).fit(coords, weights).evaluate(grid)
    kde = kde[gmask]/( np.sum(kde[gmask]) * (stepsize**len(coords)) )
    kde[ cgridmask ] = 0
    kde = kde/( np.sum(kde) * (stepsize**len(coords)) )
    kde = kde.reshape( list(map(len, axes))[::-1], order='F')
    
    return kde
    
def get_level_filtration(arr, level):
    if level == 'sub':
        return arr
    elif level == 'sup':
        return np.max(arr) - arr
    else:
        print('ERROR: `level` can only be `sub` or `sup` at the moment')
    return 0
        
# ======================================================================
# PART V
# ======================================================================

def pers2numpy(pers):
    bd = np.zeros((len(pers), 3), dtype=float)
    for i in range(len(bd)):
        bd[i, 0] = pers[i][0]
        bd[i, 1:] = pers[i][1]
    return bd

def get_diagrams(jsonfiles, ndims, remove_inf = False):
    # diag[j-th cell][k-th dimension]
    
    diags = dict()

    for gene in jsonfiles:
        diags[gene] = dict()
        for i in range(len(jsonfiles[gene])):
            filename = jsonfiles[gene][i]
            if os.path.isfile(filename):
                with open(filename) as f:
                    diag = [tuple(x) for x in json.load(f)]
                diag = pers2numpy(diag)
                cidx = int( os.path.splitext(os.path.split(filename)[1])[0].split('_c')[-1] )
                diags[gene][cidx] = [np.empty((0,2)) for k in range(ndims)]
                for k in range(ndims):
                    diags[gene][cidx][k] = diag[diag[:,0] == k, 1:]
    
    if remove_inf:
        for gene in diags:
            for cidx in diags[gene]:
                for k in range(ndims):
                    diags[gene][cidx][k]  = np.atleast_2d(diags[gene][cidx][k][np.all(diags[gene][cidx][k] < np.inf, axis=1), :].squeeze())
                if sum(list(map(len, diags[gene][cidx]))) == 0:
                    del diags[gene][cidx]
    return diags

def normalize_counts(transfocus, normtype):
    if   normtype == 'both':
        ratios = transfocus/np.sum(transfocus.values, axis=None)
    elif normtype == 'cell':
        ratios = transfocus/np.sum(transfocus.values, axis=0)
    elif normtype == 'gene':
        ratios = transfocus/np.sum(transfocus.values, axis=1).reshape(-1,1)
    else:
        print('Invalid normtype\nReceived', normtype,'\nExpected one of `both`, `cell`, or `gene`', sep='')
        ratios = None
    return ratios

# ======================================================================
# PART VI
# ======================================================================

def normalize_persistence_diagrams(orig_diags, ratios, norm_type, SCALE=256):
    
    numpairs = 0
    num_diags = ratios.size
    
    for g in orig_diags:
        for c in orig_diags[g]:
            ndims = len(orig_diags[g][c])
            break
    
    genemaxk = pd.DataFrame(0, index=ratios.index, columns=range(ndims), dtype=float)
    maxlife = dict() 
    
    diags = dict()
    for gene in orig_diags:
        diags[gene] = dict()
        maxlife[gene] = pd.DataFrame(0, index=iter(orig_diags[gene].keys()), columns=range(ndims), dtype=float)
        for cidx in orig_diags[gene]:
            diags[gene][cidx] = [ ratios.loc[gene,cidx]*orig_diags[gene][cidx][k] for k in range(ndims) ]
            for k in range(len(orig_diags[gene][cidx])):
                numpairs += len(diags[gene][cidx][k])
                if len(orig_diags[gene][cidx][k]) > 0:
                    maxlife[gene].loc[cidx , k] = np.max(diags[gene][cidx][k][:,1] - diags[gene][cidx][k][:,0])
                    if genemaxk.loc[gene,k] < np.max(diags[gene][cidx][k]):
                        genemaxk.loc[gene,k] = np.max(diags[gene][cidx][k])

    print('Initial number of life-birth pairs\t:', numpairs)
    
    if norm_type == 'gene':
        maxx = np.max(genemaxk,axis=1).reshape(len(maxlife),1,1)
    elif norm_type == 'both':
        maxx = genemaxk.max(axis=None)
    rescale = SCALE/maxx
    
    mhist = np.zeros(ndims, dtype=int)
    for gene in maxlife:
        maxlife[gene] *= rescale
        bar, _ = np.histogram(maxlife[gene].idxmax(axis=1), bins=range(ndims+1))
        mhist += bar
    focus_dim = np.argmax(mhist)
    print('\nNo. of diagrams s.t. H_k had the most persistent component')
    for i in range(len(mhist)):
        print('H_{}:\t{} [ {:.1f}% ]'.format(i,mhist[i], 100*mhist[i]/num_diags) )
    print('\nWill focus just on dimension k = {}\n'.format(focus_dim) )
    
    return diags, rescale, maxlife, focus_dim

def grid_representatives(array, arrayd, steps=1, eps=4):
    llim,blim = np.min(array, axis=0)
    rlim,tlim = np.max(array, axis=0)
    nrow = int((tlim - blim)*steps)+1
    ncol = int((rlim - llim)*steps)+1
    
    AXES = np.meshgrid( np.linspace(llim, rlim, ncol), np.linspace(blim, tlim, nrow)[::-1])
    grid = np.column_stack([ np.ravel(AXES[i]) for i in range(len(AXES)) ])
    
    dists = spatial.distance.cdist(grid, arrayd, metric='euclidean')
    argmindist = np.argmin(dists, axis=1)
    minmask = np.min(dists, axis=1) < 1/(eps*steps)
    return nrow, ncol, grid, minmask, argmindist[minmask]
    

def reduce_num_of_diagrams(orig_diags, rescale, focus_dim, norm_type, minlife=1, keepall=False):
    
    num_diags = len(orig_diags)*len(orig_diags[0])
    if norm_type == 'gene':
        diags = [ [ rescale[i][0][0]*orig_diags[i][j][focus_dim].copy() for j in range(len(orig_diags[i])) ] for i in range(len(orig_diags)) ]
    elif norm_type == 'both':
        diags = [ [ rescale*orig_diags[i][j][focus_dim].copy() for j in range(len(orig_diags[i])) ] for i in range(len(orig_diags)) ]

    for i in range(len(diags)):
        for j in range(len(diags[i])):
            diags[i][j] = np.atleast_2d(diags[i][j][ diags[i][j][:,1] - diags[i][j][:,0] > minlife, : ])
    
    nonzerodiags = np.zeros(1+len(diags), dtype=int)
    nzmask = [None for i in range(len(diags)) ]
    
    if not keepall:
        for i in range(len(diags)):
            nzmask[i] = np.nonzero( np.array(list(map(len, diags[i]))) > 0  )[0]
            nonzerodiags[i+1] += len(nzmask[i])
            diags[i] = [ diags[i][j] for j in nzmask[i] ]
    else:
        nonzerodiags[1:] = np.full(len(diags), len(diags[0]), dtype=int)
        for i in range(len(diags)):
            nzmask[i] = np.arange(len(diags[i]))
        
    nzcumsum = np.cumsum(nonzerodiags)
    
    foo = nzcumsum[-1]/num_diags*100
    print('Non-zero diagrams:\t', nzcumsum[-1],'\nCompared to all diagrams:\t',num_diags,'\t[{:.2f}%]'.format(foo), sep='')
    
    return diags, nzcumsum, nzmask
    
def birthdeath_to_flattened_lifetime(diags, num_diags):
    
    lt_coll = [ None for _ in range(num_diags) ]

    k = 0
    maxbirth = 0
    for i in range(len(diags)):
        for j in range(len(diags[i])):
            lt_coll[k] = np.column_stack( (diags[i][j][:, 0], diags[i][j][:, 1] - diags[i][j][:, 0]) )
            if len(diags[i][j]) > 0:
                foo = np.max(diags[i][j][:, 0])
                if foo > maxbirth:
                    maxbirth = foo
            k += 1

    return lt_coll, maxbirth
    
def get_diagram_match_coordinates(dgm1, dgm2, dm):
    xy = []
    for m in dm:
        x1,y1 = dgm1[m[0]]
        x2,y2 = dgm2[m[1]]
        xy.append([x1,y1,x2,y2])
    
    dm = np.atleast_2d(dm)
    if dm.shape[1] > 0:
        for j in np.setdiff1d(range(len(dgm1)), dm[:,0]):
            mid = np.mean(dgm1[j])
            x1,y1 = dgm1[j]
            x2,y2 = mid,mid
            xy.append([x1,y1,x2,y2])
        for j in np.setdiff1d(range(len(dgm2)), dm[:,1]):
            mid = np.mean(dgm2[j])
            x1,y1 = dgm2[j]
            x2,y2 = mid,mid
            xy.append([x1,y1,x2,y2])
    else:
        for j in range(len(dgm1)):
            mid = np.mean(dgm1[j])
            x1,y1 = dgm1[j]
            x2,y2 = mid,mid
            xy.append([x1,y1,x2,y2])
        for j in range(len(dgm2)):
            mid = np.mean(dgm2[j])
            x1,y1 = dgm2[j]
            x2,y2 = mid,mid
            xy.append([x1,y1,x2,y2])
    
    return np.array(xy)
    

def minimum_qq_size(arr, min_thr=None, alpha=0.15, iqr_factor=1.25, ignore=None):
    if ignore is not None:
        foo = arr[arr > ignore]
    else:
        foo = arr
    if min_thr is None:
        min_thr = np.min(foo)
    q1, q3 = np.quantile(foo, [alpha, 1-alpha])
    
    return max([q1-iqr_factor*(q3 - q1) , min_thr])
    
def maximum_qq_size(arr, max_thr=None, alpha=0.15, iqr_factor=1.25, ignore=None):
    
    if ignore is not None:
        foo = arr[arr < ignore]
    else:
        foo = arr
        
    if max_thr is None:
        max_thr = np.max(foo)
    q1, q3 = np.quantile(foo, [alpha, 1-alpha])
    
    return min([q3 + iqr_factor*(q3 - q1) , max_thr])

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

def signifscalar(scalar, limits=[1e-5,1e-4,1e-3,1e-2]):
    if scalar <= limits[0]:
        return '****'
    if scalar <= limits[1]:
        return '***'
    if scalar <= limits[2]:
        return '**'
    if scalar <= limits[3]:
        return '*'
    return ''

def get_largest_element(comp, thr=0.1, minsize=None, outlabels=False, verbose=False):
    tot = np.sum(comp > 0)
    labels,num = ndimage.label(comp, structure=ndimage.generate_binary_structure(comp.ndim, 1))
    hist,bins = np.histogram(labels, bins=num, range=(1,num+1))
    argsort_hist = np.argsort(hist)[::-1]

    if minsize is None:
        minsize = np.max(hist) + 1

    where = np.where((hist/tot > thr) | (hist > minsize))[0] + 1
    if verbose:
        print(num,'components\t',len(where),'preserved')
        print(np.sort(hist)[::-1][:20])

    mask = labels == where[0]
    for w in where[1:]:
        mask = mask | (labels == w)
    box0 = comp.copy()
    box0[~mask] = 0

    if outlabels:
        return box0, labels, where

    return box0

def borderize(img, neighbor_structure=None):
    
    mborder = ndimage.generate_binary_structure(img.ndim, img.ndim).astype(int)
    mborder[mborder == 1] = -1
    mborder[1,1] = -np.sum(mborder) - 1
    
    bimg = img.copy().astype(int)
    bimg[bimg > 0]  = 1
    border = ndimage.convolve(bimg, mborder, mode='constant', cval=0)
    border[border < 0] = 0
    border[border > 0] = 1
    
    return border

