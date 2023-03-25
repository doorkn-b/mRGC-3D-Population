FILES = ./mod/capump.mod ./mod/spike.mod ./mod/Xtra.mod \
        ./mod/icat.mod ./mod/ih.mod ./mod/inap.mod ./mod/spikeOptm.mod \
		./mod/vchange.mod ./mod/Synapse.mod 

all:
	nrnivmodl $(FILES)

mpi:
	nrnivmodl-mpi $(FILES)

clean:
	rm -rf umac i386 x86_64

