####

flags :=
#flags += -O3
flags += -O0 -g
flags += -DMAX_BATCH_SIZE=64
#flags += -DMAX_BATCH_SIZE=1
#flags += -DGENERIC_GRAD_CHECK=1

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

apps :=
apps += mnist

real_types := 
#real_types += float
real_types += double

cxxs :=
#cxxs += g++
cxxs += clang++
cxxs += nvcc

checks :=
checks += 0
#checks += 1

####

headers := $(wildcard include/*.h)

g++ := g++
clang++ := clang++
nvcc := nvcc

#
# template of compilation rules
#
define compile
exe/$(app).$(check).$(real_type).$(cxx) : $(app).cc $(headers) exe/dir
	$($(cxx)) $(flags) $($(cxx)flags) -o $$@ -Dreal_type=$(real_type) -DARRAY_INDEX_CHECK=$(check) $(app).cc
exe/$(app).$(check).$(real_type).$(cxx).s : $(app).cc $(headers) exe/dir
	$($(cxx)) $(flags) $($(cxx)flags) -S -o $$@ -Dreal_type=$(real_type) -DARRAY_INDEX_CHECK=$(check) $(app).cc
endef

targets:=$(foreach app,$(apps),\
$(foreach check,$(checks),\
$(foreach real_type,$(real_types),\
$(foreach cxx,$(cxxs),\
exe/$(app).$(check).$(real_type).$(cxx)))))

all : $(targets)

$(foreach app,$(apps),\
$(foreach check,$(checks),\
$(foreach real_type,$(real_types),\
$(foreach cxx,$(cxxs),\
$(eval $(call compile))))))

exe/dir :
	mkdir -p $@

clean :
	rm -rf exe

.DELETE_ON_ERROR :

