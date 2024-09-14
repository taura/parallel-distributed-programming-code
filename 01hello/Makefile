tbb_dir := /home/tau/parallel2/sys/inst/g

all : hello hello_tbb

hello : hello.c
	gcc -fopenmp -o $@ $< 

hello_tbb : hello_tbb.cc
	g++ -std=gnu++11 -I$(tbb_dir)/include -L$(tbb_dir)/lib -Wl,-R$(tbb_dir)/lib -Wall -Wextra -O3 -march=native -o $@ $< -ltbb

clean :
	rm -f hello
