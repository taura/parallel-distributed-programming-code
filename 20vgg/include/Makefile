layers := 
layers += convolution
layers += batchnormalization
layers += dropout
layers += maxpooling
layers += linear
layers += relu
layers += softmaxcrossentropy
layers += block
layers += vgg

real_types := 
real_types += float
real_types += double

cxxs :=
cxxs += g++
#cxxs += nvcc

g++ := g++
nvcc := nvcc
#nvcc := /usr/local/bin/nvcc

flags :=
flags += -O3
#flags += -O0 -g
flags += -DARRAY_INDEX_CHECK=0
#flags += -DARRAY_INDEX_CHECK=1
flags += -DMAX_BATCH_SIZE=64
#flags += -DMAX_BATCH_SIZE=1

g++flags += -Wall -Wextra
g++flags += -Wno-strict-overflow

nvccflags := 
nvccflags += --gpu-code sm_60
nvccflags += --gpu-architecture compute_60
#nvccflags += --maxrregcount 64
#nvccflags += -Xptxas -O3,-v
nvccflags += -Xptxas -O0,-v -G
nvccflags += -x cu

#
# template of compilation rules
#
define compile
$(layer).$(real_type).$(cxx) : $(layer).h vgg_util.h vgg_arrays.h cuda_util.h Makefile
	$($(cxx)) $(flags) $($(cxx)flags) -o $$@ unit_check.cc -Dreal_type=$(real_type) -DINC_H=\"$(layer).h\" -D$(layer)_main=main
endef

targets:=$(foreach layer,$(layers),\
$(foreach real_type,$(real_types),\
$(foreach cxx,$(cxxs),\
$(layer).$(real_type).$(cxx))))

all : $(targets)

$(foreach layer,$(layers),\
$(foreach real_type,$(real_types),\
$(foreach cxx,$(cxxs),\
$(eval $(call compile)))))

clean :
	rm -f $(targets)

.DELETE_ON_ERROR :

