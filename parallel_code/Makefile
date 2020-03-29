CC = nvcc -std=c++11
ARCH=-arch sm_30
NVOPTS=-O3 $(ARCH) -x cu
LIBS = -lcublas

programs = main
objects = Neural_Network.o gpu_functions.o helper.o

all: $(programs)

%.o: %.cpp %.h
	$(CC) $(NVOPTS) -c $< -o $@

%.o: %.cpp
	$(CC) $(NVOPTS) -c $< -o $@

$(programs): %: %.o $(objects)
	$(CC) $(LIBS) $(objects) $< -o $@

.PHONY: clean
clean:
	rm -rf $(objects) $(programs) main.o *~
