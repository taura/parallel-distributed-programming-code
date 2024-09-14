cc_files := $(wildcard *.cc)
exe := $(patsubst %.cc,%,$(cc_files))
asm := $(addsuffix .s,$(exe))

opts := 
opts += -O3
opts += -Wall -Wextra
opts += -march=native
#opts += -mavx2
opts += -funroll-loops
opts += -fopenmp
opts += -I/home/tau/libpfm-4.10.1/include

ld_opts :=
ld_opts += -L/home/tau/libpfm-4.10.1/lib
ld_opts += -Wl,-R/home/tau/libpfm-4.10.1/lib

libs :=
libs += -lpfm

CXX := g++
CXXFLAGS := $(opts)
LDFLAGS := $(ld_opts)
LIBS := $(libs)

all : $(exe) $(asm)

$(exe) : % : %.cc event.h
	$(CXX) -o $@ $(CXXFLAGS) $< $(LDFLAGS) $(LIBS)

$(asm) : %.s : %.cc event.h
	$(CXX) -o $@ -S $(CXXFLAGS) $<

clean :
	rm -f mem

