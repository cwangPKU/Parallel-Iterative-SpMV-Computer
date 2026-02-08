CC = g++ -fopenmp -O3 -std=c++11
OBJ = *.o
EXE = serial parallel_row parallel_BCSR_sparse

all:${EXE}

serial: serial.cpp
	$(CC) -o $@ $^ $(FLAGS) 

parallel_BCSR_sparse: parallel_BCSR_sparse.cpp
	$(CC) -o $@ $^ $(FLAGS)

parallel_BCSR_dense: parallel_BCSR_dense.cpp
	$(CC) -o $@ $^ $(FLAGS)

parallel_row: parallel_row.cpp
	$(CC) -o $@ $^ $(FLAGS)
 
clean:
	rm -f $(OBJ) $(EXE)