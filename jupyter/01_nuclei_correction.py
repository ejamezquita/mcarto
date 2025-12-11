import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6


import numpy as np
import pandas as pd
import tifffile as tf
from glob import glob

from scipy import ndimage
from sklearn import neighbors
import argparse
import utils

struc1 = ndimage.generate_binary_structure(2,1)

def main():
    
    parser = argparse.ArgumentParser(description="Correct spatial location of transcripts where needed.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='Last major update: May 2024. © Erik Amezquita')
    parser.add_argument("sample", type=str,
                        help="Cross-section sample name")
    parser.add_argument("-r", "--radius", type=int, default=30,
                        help="Limit scope to neighbors within this radius")
    parser.add_argument("-m", "--max_distance_to_wall", type=int, default=6,
                        help="Consider for correction only those transcripts this close to a wall")
    parser.add_argument("-k", "--minimum_neighbor_number", type=int, default=5,
                        help="Consider for correction only those transcripts with at least this number of nearest neighbors")
    parser.add_argument("-p", "--minimum_ratio", type=int, default=74,
                        help="New label considered if at least p percent of neighbors belong to other cell")
    parser.add_argument("-n", "--nuclei_mask_cutoff", type=int, default=1,
                        help="Consider a transcript as part of the nucleus if it is within this distance from one")
    parser.add_argument("--cell_wall_directory", type=str, default="cell_dams",
                        help="directory containing cell wall TIFs")
    parser.add_argument("--nuclear_directory", type=str, default="nuclear_mask",
                        help="directory containing nuclei TIFs")
    parser.add_argument("--initial_data_directory", type=str, default="data",
                        help="directory containing spatial location data")
    parser.add_argument("--location_directory", type=str, default="translocs",
                        help="directory to contain corrected spatial location data")
    parser.add_argument("-w", "--rewrite_results", action="store_true",
                        help="prior results will be rewritten")
    args = parser.parse_args()

    rewrite = args.rewrite_results
    wsrc = os.pardir + os.sep + args.cell_wall_directory + os.sep
    nsrc = os.pardir + os.sep + args.nuclear_directory + os.sep
    csrc = os.pardir + os.sep + args.initial_data_directory + os.sep
    dst = os.pardir + os.sep + args.location_directory + os.sep
    
    sample = args.sample
    radius = args.radius
    maxdwall = args.max_distance_to_wall
    minneighs = args.minimum_neighbor_number
    minprob = args.minimum_ratio


    dst += sample + os.sep
    if not os.path.isdir(dst):
        os.mkdir(dst)
        
    wall = tf.imread(wsrc + sample + '_dams.tif').astype(bool)
    nuclei = tf.imread(nsrc + sample + '_EDT.tif') < args.nuclei_mask_cutoff

    edt = ndimage.distance_transform_cdt(wall, 'chessboard')
    label, cellnum = ndimage.label(wall, struc1)
    print('Detected',cellnum,'cells')

    filenames = sorted(glob(os.pardir + os.sep + 'Bacteria Info for Erik/*_v2.txt'))

    transcriptomes = [ None for _ in range(len(filenames)) ]
    translocs = [ None for _ in range(len(filenames)) ]

    for i in range(len(filenames)):
        transcriptomes[i] = os.path.splitext(os.path.split(filenames[i])[1])[0][:-3]
        translocs[i] = pd.read_csv(filenames[i], sep='\t').iloc[:,:4]

    transcriptomes = np.asarray(transcriptomes)
    print(len(transcriptomes), 'transcriptomes')
    
    data = pd.concat(translocs)
    foo = len(data)
    data = data[data['L'] == 'c'].drop(columns='L')
    print('Reduced to', len(data), 'after removing those in nuclei [kept {:.02f}% of the originals]'.format(len(data)/foo*100) )

    for tidx in range(len(transcriptomes)):
        
        filename =  dst + 'location_corrected_' + sample +'_-_' + transcriptomes[tidx] + '.csv'
        if rewrite or (not os.path.isfile(filename)):
            print('----', tidx, ' : \t', transcriptomes[tidx], ':\n')
            
            coords = translocs[tidx].loc[ translocs[tidx]['L'] == 'c' , ['X', 'Y', 'Z'] ].values.T
            tlabs = label[ coords[1], coords[0] ].astype(int)
            tpercell, _ = np.histogram(tlabs, np.arange(cellnum+2))
            foo = [ len(translocs[tidx]), coords.shape[1], 100*coords.shape[1]/len(translocs[tidx]) ]
            print('Originally {} transcripts. Reduced to {} cytosolics [{:.2f}%]'.format(*foo))
            
            # # Deal with transcripts on the edge

            foo = np.sum(tlabs == 0)
            print('Initially, there are\t',foo,'\ttranscripts on the walls',sep='')
            if foo > 0:
                tlabs, coords = utils.correct_boundary_transcripts(tlabs, coords, label, tpercell, R=5)
                foo = np.sum(tlabs == 0)
                print('Now there are\t',foo,'\ttranscripts on the walls',sep='')

            # # Deal with misplaced transcripts

            foo = 100
            iters = 0

            if np.sum(edt[coords[1], coords[0]] < maxdwall) > 0:

                cdtmask = np.nonzero(edt[coords[1], coords[0]] < radius)[0]
                cdtlabs = tlabs[cdtmask].copy()
                cdtcoords = coords[ :,  cdtmask].copy()
                edtmask = edt[cdtcoords[1], cdtcoords[0]] < maxdwall
                edtvals = np.nonzero(edtmask)[0]

                while (foo  > 0) and (iters < 20):
                    iters += 1    
                    foo, cdtlabs, cdtcoords = utils.correct_shifted_transcripts(cdtlabs, cdtmask, cdtcoords, edtmask, edtvals, label, maxdwall, minneighs, minprob)
                    print('Iteration: ', iters, '\tShifted\t',foo,' transcripts', sep='')
                
                shiftmask = np.any(cdtcoords != coords[ :,  cdtmask], axis=0)
                print('Shifted\t',np.sum(shiftmask),'\ttranscripts in total', sep='')

                # # Save Results

                coords[:, cdtmask] = cdtcoords
                    
                print('Saved file', filename,'\n')
                df = pd.DataFrame(coords.T)
                df.to_csv(filename, header=False, index=False)
            
            else:
                df = translocs[tidx].loc[ translocs[tidx]['L'] == 'c' , ['X', 'Y', 'Z'] ]
                df.to_csv(filename, header=False, index=False)
                
    return 0

if __name__ == '__main__':
    main()
