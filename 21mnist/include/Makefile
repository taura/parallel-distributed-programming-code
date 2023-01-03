####

flags :=
flags += -O3
#flags += -O0 -g
flags += -DMAX_BATCH_SIZE=64
#flags += -DMAX_BATCH_SIZE=1

g++flags += -Wall -Wextra
g++flags += -Wno-strict-overflow
g++flags += -fno-aggressive-loop-optimizations

nvccflags := 
nvccflags += --gpu-code sm_80
nvccflags += --gpu-architecture compute_80
#nvccflags += --maxrregcount 64
#nvccflags += 
#nvccflags += -Xptxas -O0,-v -G
nvccflags += -x cu

####

layers := 
layers += mnist_util
layers += tensor
layers += mnist_data
layers += ada_delta
layers += convolution
layers += linear
layers += relu
layers += dropout
layers += max_pooling
layers += nll_log_softmax
layers += mnist

real_types := 
real_types += float
#real_types += double

cxxs :=
#cxxs += g++
#cxxs += clang++
cxxs += nvcc

checks :=
checks += 0
#checks += 1

g++ := g++
clang++ := clang++
nvcc := nvcc

#
# template of compilation rules
#
define compile
exe/$(layer).$(check).$(real_type).$(cxx) : $(layer).h $(headers) exe/dir
	$($(cxx)) $(flags) $($(cxx)flags) -o $$@ -Dreal_type=$(real_type) -DARRAY_INDEX_CHECK=$(check) -DINC_H=\"$(layer).h\" -D$(layer)_main=main main.cc
exe/$(layer).$(check).$(real_type).$(cxx).s : $(layer).h $(headers) exe/dir
	$($(cxx)) $(flags) $($(cxx)flags) -S -o $$@ -Dreal_type=$(real_type) -DARRAY_INDEX_CHECK=$(check) -DINC_H=\"$(layer).h\" -D$(layer)_main=main main.cc
endef

targets:=$(foreach layer,$(layers),\
$(foreach check,$(checks),\
$(foreach real_type,$(real_types),\
$(foreach cxx,$(cxxs),\
exe/$(layer).$(check).$(real_type).$(cxx)))))
#  exe/$(layer).$(check).$(real_type).$(cxx).s

all : $(targets)

$(foreach layer,$(layers),\
$(foreach check,$(checks),\
$(foreach real_type,$(real_types),\
$(foreach cxx,$(cxxs),\
$(eval $(call compile))))))

exe/dir :
	mkdir -p $@

clean :
	rm -rf exe

.DELETE_ON_ERROR :

