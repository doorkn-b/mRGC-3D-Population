echo 'Running test_installation.sh'
make clean -C ../RGC/ && make -C ../RGC/ 
poetry run python ./foveal_tiles.py
poetry run python ./stimulate_tile.py
