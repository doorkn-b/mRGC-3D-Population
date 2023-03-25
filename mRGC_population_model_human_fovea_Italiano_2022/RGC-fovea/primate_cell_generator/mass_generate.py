#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 8 19:25:28 2020

Given a population of cell .txt files, produces the corresponding .swc and .hoc
files.

@author: M.L. Italiano
"""

import os
import fnmatch
import subprocess
import time
from joblib import Parallel, delayed
import multiprocessing

TRIAL = 0
PARALLEL = True  # Parallel file creation
SWC = True
HOC = True

num_cores = multiprocessing.cpu_count()


def write_hoc_file(cell_id: int, path: str, text_source: str) -> None:
    """Performs .hoc file cell/init creation."""
    if cell_id % 1000 == 0:
        print(f"INFO: Cell {cell_id} .hoc complete...")
    hoc_cmd = (
        f"python {path}/primate_cell_generator/generate_hoc.py "
        f"{text_source}/cell_{cell_id:04d}.txt cell_{cell_id:04d} {cell_id+1}"
    )
    os.system(hoc_cmd)


def convert_txt_to_swc(
    cell_id: int, fovea_path: str, txt_source: str, swc_dest: str
) -> None:
    """
    Converts .txt cell file to .swc using bash script.

        - `cell_id`: cell id/number.
        - `fovea_path`: path to RGC-fovea.
        - `txt_source`: path to .txt file.
        - `swc_dest`: path to save .swc file to.

    """
    cell_txt_file = f"{txt_source}/cell_{cell_id:04d}.txt"
    cell_swc_file = f"{swc_dest}/cell_{cell_id:04d}.swc"
    subprocess.call(
        [
            "bash",
            f"{fovea_path}/primate_cell_generator/txt_to_swc.sh",
            cell_txt_file,
            cell_swc_file,
        ]
    )


def main():

    # Grabs absolute directory path to RGC/primateRetina
    CUR_PATH = f"{os.path.abspath(os.getcwd()).split('/RGC')[0]}/RGC-fovea"
    HOC_DESTINATION = f"{CUR_PATH}/results/trial_{TRIAL:03d}/cellHoc"
    SWC_DESTINATION = f"{CUR_PATH}/results/trial_{TRIAL:03d}/cellSwc"
    CELL_TEXT_SOURCE = f"{CUR_PATH}/results/trial_{TRIAL:03d}/cellText"
    # Directory checks and determining number of cells in .txt format
    assert os.path.isdir(
        CELL_TEXT_SOURCE
    ), f"Cell text file directory not found at {CELL_TEXT_SOURCE}. Run foveal_tiles.py to generate cell text files."
    n_cells = len(fnmatch.filter(os.listdir(CELL_TEXT_SOURCE), "cell*.txt"))
    assert (
        n_cells
    ), f"No cell text files found at {CELL_TEXT_SOURCE}. Run foveal_tiles.py to generate cell text files."
    if os.path.isdir(HOC_DESTINATION) is False:
        os.makedirs(HOC_DESTINATION)
    if os.path.isdir(SWC_DESTINATION) is False:
        os.makedirs(SWC_DESTINATION)

    print(f"INFO: Generating .swc and .hoc files for {n_cells} cells...")

    # Convert text files to swc
    if SWC:
        start_swc = time.time()

        if PARALLEL:
            Parallel(n_jobs=num_cores)(
                delayed(convert_txt_to_swc)(
                    cell_id=i,
                    fovea_path=CUR_PATH,
                    txt_source=CELL_TEXT_SOURCE,
                    swc_dest=SWC_DESTINATION,
                )
                for i in range(n_cells)
            )

        else:
            for i in range(n_cells):
                convert_txt_to_swc(
                    cell_id=i,
                    fovea_path=CUR_PATH,
                    txt_source=CELL_TEXT_SOURCE,
                    swc_dest=SWC_DESTINATION,
                )

        # Check .swc generation was successful
        n_swc = len(fnmatch.filter(os.listdir(SWC_DESTINATION), "cell*.swc"))
        assert n_cells == n_swc, ".swc generation was unsuccessful!"
        print(f"INFO: .swc creation took {(time.time() - start_swc):.2f} s.")

    # Create each cell's hoc/init-hoc file
    if HOC:
        start_hoc = time.time()

        if PARALLEL:
            Parallel(n_jobs=num_cores)(
                delayed(write_hoc_file)(
                    cell_id=i, path=CUR_PATH, text_source=CELL_TEXT_SOURCE
                )
                for i in range(n_cells)
            )

        else:
            for i in range(n_cells):
                write_hoc_file(cell_id=i, path=CUR_PATH, text_source=CELL_TEXT_SOURCE)

        end_hoc = time.time()
        print(f"INFO: .hoc creation took {(end_hoc - start_hoc):.2f} s.")

        # Move all hoc files
        mv_cmd = f"mv *cell_*.hoc {HOC_DESTINATION}/"
        os.system(mv_cmd)
        print(f"INFO: .hoc move took {(time.time() - end_hoc):.2f} s.")

        # Check .hoc generation was successful
        n_hoc = len(fnmatch.filter(os.listdir(HOC_DESTINATION), "cell*.hoc"))
        assert n_cells == n_hoc, (
            ".hoc generation was unsuccessful: ",
            f"{n_cells} cells but {n_hoc} .hoc files!",
        )

    print("INFO: ...cells successfully saved as .swc and .hoc files.")


if __name__ == "__main__":
    main()
