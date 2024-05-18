import os
import pandas as pd
import numpy as np
import argparse

levels = ['sub', 'sup']
PP = 6
pp = 0

def is_type_tryexcept(s, testtype=int):
    """ Returns True if string is a number. """
    try:
        testtype(s)
        return True
    except ValueError:
        return False

def get_range_cell_values(arginput, meta, startval=0):
    
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
    
def get_range_gene_values(arginput, meta, startval=0):
    
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

def main():
    
    parser = argparse.ArgumentParser(description="Produce cell and gene metadata that will be useful later.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='Last major update: May 2024. © Erik Amezquita')
    parser.add_argument("sample", type=str,
                        help="Cross-section sample name")
    parser.add_argument("-b", "--kde_bandwidth", type=int, default=10,
                        help="bandwidth to compute KDE")
    parser.add_argument("-s", "--grid_stepsize", type=int,
                        help="grid size to evaluate the KDE")                        
    parser.add_argument("-c", "--cell_focus", type=str,
                        help="file or single ID with cell to evaluate")
    parser.add_argument("-g", "--gene_focus", type=str,
                        help="file or single ID with gene to evaluate")
    parser.add_argument("-l", "--level_filtration", type=str, choices=['sub','sup'],
                        help="level filtration to use to compute persistent homology")
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
    parser.add_argument("-w", "--rewrite_results", action="store_true",
                        help="prior results will be rewritten")
    args = parser.parse_args()
    
    rewrite = args.rewrite_results
    sample = args.sample
    
    if args.level_filtration is None:
        level = ['sub','sup']
    else:
        level = [args.level_filtration]

    wsrc = '..' + os.sep + args.cell_wall_directory + os.sep
    nsrc = '..' + os.sep + args.nuclear_directory + os.sep
    tsrc = '..' + os.sep + args.location_directory + os.sep
    ksrc = '..' + os.sep + args.kde_directory + os.sep + sample + os.sep
    
    bw = args.kde_bandwidth
    stepsize = args.grid_stepsize
    
    metacell = pd.read_csv(ksrc + sample + '_cells_metadata.csv', index_col = 0)
    metatrans = pd.read_csv(ksrc + sample + '_transcripts_metadata.csv')
    transcell = pd.read_csv(ksrc + sample + '_transcells_metadata.csv')
    tcumsum = np.hstack(([0], np.cumsum(metatrans['cyto_number'].values)))

    transcriptomes = np.asarray(metatrans['gene'])
    
    Cells = get_range_cell_values(args.cell_focus, metacell, startval=1)
    Genes = get_range_gene_values(args.gene_focus, transcriptomes, startval=0)
    
    if (Cells is None) or (Genes is None):
        print('Make sure that the ID value is an integer')
        print('Or make sure that the specified file exists and is formatted correctly')
        return 0
    
    print('Cell Focus', Cells)
    print('Gene Focus', Genes)
    
    return 0
    
if __name__ == '__main__':
    main()
