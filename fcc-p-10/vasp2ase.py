#!/usr/bin/env python

"""Naive Script to convert VASP output files to any ase supported file.
"""

import sys

import argparse
from ase.io import read
from ase.io import write


def convert(files: list, 
            outfil: str, scf: 
            bool = False, 
            pbc: bool = True):
    
    traj = []

    for filname in files:
        with open(filname, 'r', errors="replace") as file:
            traj += read(filname, index=':')
    
    for i in traj:
        i.set_pbc(pbc)
        
    """NOTE make append=True and use try-except OSError with open(file,"w").close() and pass 
    if unexpected behaviour occurs""" 
    write(outfil, traj, append=False)
    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Naive Script to convert VASP vasprun.xml file(s) to any ase supported file.' 
    )

    parser.add_argument('infiles', type=str, nargs='+')
    parser.add_argument('-o', '--outfile', type=str, default='output.db')
    parser.add_argument('-s', '--scf', action='store_true',
        help='Is the input file a restult of an scf calculation')
    parser.add_argument('-v', '--vacuum', action='store_true',
        help='Don\'t use pbc')
    parser.add_argument('-f', '--finite_set_correction', action='store_true',
        help='Use if no finite set correction was used in the calculation') 
    
    
    args = parser.parse_args()
    convert(args.infiles, args.outfile, args.scf, (not args.vacuum))
