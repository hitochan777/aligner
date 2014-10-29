CC=g++
CFLAG=-std=c++11 -Wall
OBJGROUP=src/fast_align.o src/LM.o src/LogDouble.o src/Vocab.o src/ttables.o src/FracType.o

all: fast_align

fast_align: $(OBJGROUP)
	$(CC) -o fast_align $(OBJGROUP)
clean:
	rm *.o
