all: pgm.o	hough houghC houghS

hough: houghBase.cu common/pgm.cpp
	nvcc houghBase.cu common/pgm.cpp -o hough 

houghC: houghConstante.cu common/pgm.cpp
	nvcc houghConstante.cu common/pgm.cpp -o houghc 

houghS: houghCompartida.cu common/pgm.cpp
	nvcc houghCompartida.cu common/pgm.cpp -o houghs 	

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o


clean:
	rm -f hough