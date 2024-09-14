mm_srcs := $(wildcard mm_*.cc)
idx := $(patsubst mm_%.cc,%,$(mm_srcs))
exes := $(addprefix mm_,$(idx))
asms := $(addsuffix .S,$(exes))

all :=
all += $(exes)
all += $(asms)

opts := 
#opts += -march=native 
opts += -mavx512f
opts += -mfma
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
//opts += -faligned-new
opts += -fopenmp

CXX := clang++
CC := clang
CXXFLAGS := $(opts)
CFLAGS := $(opts)

all : $(all)

$(exes) : mm_% : mm_%.cc mm_main.h
	$(CXX) -o $@ $(CXXFLAGS) $<

$(asms) : mm_%.S : mm_%.cc mm_main.h
	$(CXX) -o $@ $(CXXFLAGS) -S $<

$(exes) $(asms) : Makefile mm_main.h

clean :
	rm -f $(exes) $(objs) $(asms)


