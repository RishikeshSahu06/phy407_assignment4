CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall

all: nvt

nvt: nvt.cpp
	$(CXX) $(CXXFLAGS) -o nvt nvt.cpp

clean:
	rm -f nvt *.o

run_anderson: nvt
	./nvt 0 10
	./nvt 0 50
	./nvt 0 100

run_nosehoover: nvt
	./nvt 1 5
	./nvt 1 20
	./nvt 1 100

run_all: nvt
	./nvt 0 10
	./nvt 0 50
	./nvt 0 100
	./nvt 1 5
	./nvt 1 20
	./nvt 1 100

.PHONY: all clean run_anderson run_nosehoover run_all