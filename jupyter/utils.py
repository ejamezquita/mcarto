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

pxs = 75
pxbar = np.s_[-15:-5, 5:5 + pxs]

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
        if 'ndimage_ID' in focus.columns:
            Vals = focus['ndimage_ID'].values
        else:
            Vals = np.zeros(len(focus), dtype=int)
            for i in range(len(cid)):
                Vals[i] = meta[meta['orig_cellID'] == focus.iloc[i,0]].index[0]
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
            for i in range(len(cid)):
                Vals[i] = np.nonzero(meta == focus.iloc[i,0])[0][0]
    else:
        Vals = None
        print('ERROR: Unable to choose gene ID values from input')
        
    return Vals

def get_cell_img(cidx, metacell, wall, label, PP=10, pxbar=False):
    s_ = (np.s_[max([0, metacell.loc[cidx, 'y0'] - PP]) : min([wall.shape[0], metacell.loc[cidx, 'y1'] + PP])],
          np.s_[max([1, metacell.loc[cidx, 'x0'] - PP]) : min([wall.shape[1], metacell.loc[cidx, 'x1'] + PP])])
    extent = (s_[1].start, s_[1].stop, s_[0].start, s_[0].stop)
    cell = wall[s_].copy().astype(np.uint8)
    cell[ label[s_] == cidx ] = 2
    cell[~wall[s_]] = 0
    
    if pxbar:
        cell[pxbar] = 0

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
        if len(foo) > 1:
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
    
    indexing = np.nonzero((cat[:,2] == 1) & (cat[:,3] > 70))[0]
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

def generate_transcell_metadata(translocs, transcriptomes, cellnum, label):
    meta = np.zeros((len(transcriptomes), cellnum), dtype=int)
    bins = np.arange(1, cellnum + 2)
    for tidx in range(len(meta)):
        coords = translocs[tidx].loc[:, ['X', 'Y']].values.T
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

def cell_grid_preparation(cell, extent, zmax, stepsize, pows2=pows2):
    
    maxdims = ( cell.shape[1], cell.shape[0], zmax )
    axes, grid, gmask = kde_grid_generator(stepsize=stepsize, maxdims=maxdims, pows2 = pows2, pad=1.5)
    grid[:, :2] = grid[:, :2] + np.array([ extent[0], extent[2] ])
    
    cgrid = grid[gmask].copy()
    cgrid[:,:2] = grid[gmask][:,:2] - np.array([ extent[0], extent[2] ])
    cgridmask = cell[cgrid[:,1],cgrid[:,0]] != 2
    
    return axes, grid, gmask, cgrid, cgridmask
        
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

def normalize_counts(transfocus, normtype):
    if   normtype == 'both':
        ratios = transfocus.values/np.sum(transfocus.values, axis=None)
    elif normtype == 'cell':
        ratios = transfocus.values/np.sum(transfocus.values, axis=0)
    elif normtype == 'gene':
        ratios = transfocus.values/np.sum(transfocus.values, axis=1).reshape(-1,1)
    else:
        print('Invalid normtype\nReceived', normtype,'\nExpected one of `both`, `cell`, or `gene`', sep='')
        ratios = None
    return ratios

# ======================================================================
# PART VI
# ======================================================================

def normalize_persistence_diagrams(orig_diags, ratios, norm_type, SCALE=256):
    
    numpairs = 0
    num_diags = len(orig_diags)*len(orig_diags[0])
    ndims = len(orig_diags[0][0])
    genemaxk = np.zeros((len(orig_diags), ndims))
    maxlife = np.zeros((len(orig_diags), len(orig_diags[0]), len(orig_diags[0][0])))

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
    
    if norm_type == 'gene':
        maxx = np.max(genemaxk,axis=1).reshape(len(maxlife),1,1)
    elif norm_type == 'both':
        maxx = np.max(genemaxk) 

    rescale = SCALE/maxx
    maxlife *= rescale
    argmaxlife = np.argmax(maxlife, axis=-1)
    mhist, _ = np.histogram(argmaxlife.ravel(), bins=range(ndims+1))
    focus_dim = np.argmax(mhist)
    print('\nNo. of diagrams s.t. H_k had the most persistent component')
    for i in range(len(mhist)):
        print('H_{}:\t{} [ {:.1f}% ]'.format(i,mhist[i], 100*mhist[i]/num_diags) )
    print('\nWill focus just on dimension k = {}\n'.format(focus_dim) )
    
    return orig_diags, rescale, maxlife, focus_dim
    
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
    
