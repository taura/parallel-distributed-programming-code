ms_srcs := $(wildcard ms_?.cc)
idx := $(patsubst ms_%.cc,%,$(ms_srcs))
exes := $(addprefix ms_,$(idx))
asms := $(addsuffix .S,$(exes))

all :=
all += $(exes)
all += $(asms)

opts := 
opts += -march=native 
#opts += -mavx512f
#opts += -mavx2
#opts += -funroll-loops 
#opts += --param max-unroll-times=100
#opts += -fopenmp-simd
#opts += -fopt-info-vec-optimized 
#opts += -fopt-info-vec-missed
#opts += -axMIC-AVX512
#opts += -mkl
opts += -O3
#opts += -O0 -g
opts += -Wall -Wextra
opts += -DDAG_RECORDER=2
opts += -DTO_TBB
opts += -I/home/tau/parallel2/sys/inst/g/include
opts += -L/home/tau/parallel2/sys/inst/g/lib
opts += -Wl,-R/home/tau/parallel2/sys/inst/g/lib
opts += -std=gnu++11

libs :=
libs += -ldr
libs += -ltbb
libs += -lpthread

CXX := g++
CC := gcc
CXXFLAGS := $(opts)
CFLAGS := $(opts)
LIBS := $(libs)

all : $(all)

$(exes) : ms_% : ms_%.cc
	$(CXX) -o $@ $(CXXFLAGS) $< $(LIBS)

$(asms) : ms_%.S : ms_%.cc
	$(CXX) -o $@ $(CXXFLAGS) -S $<

$(exes) $(asms) : Makefile

clean :
	rm -f $(exes) $(objs) $(asms)


