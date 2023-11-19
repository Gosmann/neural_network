all: neural.o main.o 
	g++ neural.o main.o -o output

neural.o :
	g++ -c neural.cpp -o neural.o

main.o :
	g++ -c main.cpp -o main.o

clean :
	rm *.o