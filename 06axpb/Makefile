CXX := g++
#CXX := icpc
#CXX := clang
#CXX:=nvcc

cc_files := $(wildcard axpb.cc)
exes := $(patsubst %.cc,%.$(CXX),$(cc_files))
asms := $(patsubst %.cc,%.S,$(cc_files))
ptxs := $(patsubst %.cc,%.ptx,$(cc_files))
cubins := $(patsubst %.cc,%.cubin,$(cc_files))
fatbins := $(patsubst %.cc,%.fatbin,$(cc_files))
cuasms := $(patsubst %.cc,%.asm,$(cc_files))
fatasms := $(patsubst %.cc,%.fatasm,$(cc_files))

common_opts := 
#common_opts += -Wall
#common_opts += -Wextra
common_opts += -O3
#common_opts += -O0 -g

# GCC
g++_opts :=
g++_opts += -fopenmp
g++_opts += -march=native
#g++_opts += -mavx512f

clang_opts :=
clang_opts += -fopenmp
clang_opts += -march=native
#clang_opts += -mavx512f

# ICC
icc_opts :=
icc_opts += -qopenmp
icc_opts += -xMIC-AVX512

# NVCC
nvcc_opts :=
nvcc_opts += -x cu
nvcc_opts += --generate-code arch=compute_60,code=sm_60
nvcc_opts += --generate-code arch=compute_70,code=sm_70
#nvcc_opts += --maxrregcount 64
#nvcc_opts += -Xptxas -O3,-v
#nvcc_opts += -O0 -g -G -Xptxas -O0
nvcc_opts += -Xptxas -O3
nvcc_opts += --compiler-options=-mavx2

CXXFLAGS := $(common_opts) $($(CXX)_opts)

targets :=
targets += $(exes)
ifeq ($(CXX),nvcc)
#targets += $(ptxs)
#targets += $(cubins)
#targets += $(cuasms)
targets += $(fatbins)
targets += $(fatasms)
else
targets += $(asms)
endif

all : $(targets)
exe : $(exes)
asm : $(asms)
ptx : $(ptxs)
cubin : $(cubins)
cuasm : $(cuasms)
fatasm : $(fatasms)

$(exes) : %.$(CXX) : %.cc Makefile
	$(CXX) -o $@ $(CXXFLAGS) $<
$(asms) : %.S : %.cc Makefile
	$(CXX) -o $@ -S $(CXXFLAGS) $<
$(ptxs) : %.ptx : %.cc Makefile
	$(CXX) -o $@ --ptx $(CXXFLAGS) $<
$(cubins) : %.cubin : %.cc Makefile
	$(CXX) -o $@ --cubin $(CXXFLAGS) $<
$(fatbins) : %.fatbin : %.cc Makefile
	$(CXX) -o $@ --fatbin $(CXXFLAGS) $<
$(cuasms) : %.asm : %.cubin Makefile
	cuobjdump --dump-sass $< > $@ 
$(fatasms) : %.fatasm : %.fatbin Makefile
	cuobjdump --dump-sass $< > $@ 

clean :
	rm -f $(exes) $(asms) $(ptxs) $(cubins) $(cuasms)
