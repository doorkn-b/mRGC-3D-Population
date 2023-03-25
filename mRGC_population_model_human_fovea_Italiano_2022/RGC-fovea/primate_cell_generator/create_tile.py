"""Analyses the cellHoc directory to create a .hoc tile of all the cells."""

import os
import fnmatch

# TRIAL NUMBER
TRIAL = 0


def main():

    # Grabs absolute directory path to RGC-fovea/
    CUR_PATH = f"{os.path.abspath(os.getcwd()).split('/RGC')[0]}/RGC-fovea/"
    HOC_SOURCE = f"{CUR_PATH}/results/trial_{TRIAL:03d}/cellHoc"

    # Directory checks and determining number of cells in .hoc format
    assert os.path.isdir(HOC_SOURCE), (
        f"Cell file directory not found at {HOC_SOURCE}.\n"
        "Run mass_generate.py to generate cell .hoc (and .swc) files."
    )
    n_cells = len(fnmatch.filter(os.listdir(HOC_SOURCE), "cell*.hoc"))
    assert n_cells, (
        f"No cell .hoc files found at {HOC_SOURCE}.\n"
        "Run mass_generate.py to generate cell .hoc (and .swc) files."
    )

    ################################ TILE #########################################

    tile_file_path = f"{CUR_PATH}tiles/foveal-tile-{TRIAL:03d}.hoc"
    if os.path.isdir(tile_file_path.split("/foveal")[0]) is False:
        os.makedirs(tile_file_path.split("/foveal")[0])
    tile_file = open(tile_file_path, "w")

    # Load cells
    for i in range(n_cells):
        cell = f"cell_{i:04d}.hoc"
        cell_path = f"../RGC-fovea/results/trial_{TRIAL:03d}/cellHoc/{cell}"
        load_line = f'{{load_file("{cell_path}")}}\n'
        tile_file.write(load_line)

    # Load global.hoc
    tile_file.write('{load_file("common/global.hoc")}\n')

    # Create cell array
    tile_file.write(f"\nobjref rgc[{n_cells}]\n")
    for i in range(n_cells):
        tile_file.write(f"rgc[{i}] = new cell_{i:04d}()\n")

    # Access soma
    tile_file.write("access rgc[0].soma\n")
    tile_file.write("\n")

    # Visualisation
    tile_file.write("// visualisation and UI\n")
    tile_file.write('{load_file("common/gui.hoc")}\n')
    tile_file.write("FIELD_LEFT = -250\n")
    tile_file.write("FIELD_BOTTOM = -100\n")
    tile_file.write("FIELD_WIDTH = 500\n")
    tile_file.write("FIELD_HEIGHT = 200\n")
    tile_file.write(
        "// showCell(FIELD_LEFT, FIELD_BOTTOM, FIELD_WIDTH, FIELD_HEIGHT)\n"
    )
    for i in range(n_cells):
        tile_file.write(f'guiGraph.addvar("rgc[{i}].soma.v(0.5)", {i+2}, 1)\n')
    tile_file.write("\n")

    # Simulation
    tile_file.write("// instrumentation\n")
    tile_file.write('{load_file("../RGC-fovea/stim/instr.hoc")}\n')
    tile_file.write("\n")

    tile_file.close()

    print(f"INFO: Created foveal-tile.hoc at {tile_file_path}")


if __name__ == "__main__":
    main()
