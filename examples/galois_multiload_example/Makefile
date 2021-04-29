all:
	#cd parla_galois && make clean && make && cd -
	GALOIS_DIR=/usr/local python3 setup.py build
	cp build/lib.linux-x86_64-3.8/parla_galois/core.cpython-38-x86_64-linux-gnu.so parla_galois/
	cp build/lib.linux-x86_64-3.8/parla_galois/core.pyx parla_galois/
	cp build/lib.linux-x86_64-3.8/parla_galois/core.pxd parla_galois/