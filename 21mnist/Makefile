#
# makefile for mnist
#

#
# versions you want to get
#
# - "base" is the baseline code I provide
# - you can add your version(s) by adding lines like
#   vers += name (e.g., vers += fast)
# - by default, the version you added will be compiled
#   by the same compiler and flags, some of which you 
#   surely want to change
# - to do this, you can set various flags.
#   let's say your version is named "simd". then,
#   simd_cc := the compiler used for "simd"
#   simd_flags := compiler flags for "simd"
#   simd_ldflags := linker flags for "simd"
# - let's say you chose nvc++ for "simd", then
#   simd_nvc++_flags := compiler flags for "simd", applied
#     only for nvc++
#   simd_nvc++_ldflags := linker flags for "simd", applied
#     only for nvc++
#   these two variables are useful when you switch between
#   different compilers and some flags are available only
#   for specific compilers
#
vers :=
vers += cpu_base
vers += cuda_base
#vers += fast

#
# C++ compiler flags
#
# flags common to all C++ compilers and versions
#
# uncomment -O0 -g for running in the debugger
# (performance difference between -O3 and -O0
# is HUGE)
#
flags :=
#flags += -O0
flags += -O3
flags += -DMAX_BATCH_SIZE=64
flags += -DARRAY_INDEX_CHECK=0
#flags += -DARRAY_INDEX_CHECK=1
flags += -Dreal_type=float
#flags += -Dreal_type=double

ldflags :=

#
# flags given to all versions compiled by g++ 
#
g++_flags :=
g++_flags += -Wall
g++_flags += -Wextra
g++_flags += -Wno-strict-overflow
g++_ldflags :=

#
# flags given to all versions compiled by clang++ 
#
clang++_flags :=
clang++_flags += -Wall
clang++_flags += -Wextra
clang++_flags += -Wno-strict-overflow
clang++_flags += -gdwarf-4
clang++_ldflags :=

#
# flags given to all versions compiled by nvc++ 
#
nvc++_flags :=
nvc++_flags += -Wall
nvc++_flags += -Wextra
nvc++_ldflags :=

#
# flags given to all versions compiled by nvcc
#
nvcc_flags := 
nvcc_flags += -x cu
nvcc_flags += --gpu-code sm_80
nvcc_flags += --gpu-architecture compute_80
#nvcc_flags += --maxrregcount 64
#nvcc_flags += 
#nvcc_flags += -Xptxas -O0,-v -G
nvcc_ldflags := 

#
# C++ compiler for versions
#

cpu_base_cxx := clang++
cuda_base_cxx := nvcc
fast_cxx := clang++

#
# toplevel C++ file of each version
# you can change it if you want to
#

cpu_base_cc := mnist.cc
cuda_base_cc := mnist.cc
fast_cc := mnist.cc

#
# version-specific flags
#

cpu_base_flags := 
cpu_base_ldflags := 

cuda_base_flags := 
cuda_base_ldflags := 

fast_flags := 
fast_ldflags := 

#
# version- and compiler-specific flags
#

cpu_base_g++_flags     :=
cpu_base_g++_ldflags     :=

cpu_base_clang++_flags :=
cpu_base_clang++_ldflags :=

cpu_base_nvc++_flags   :=
cpu_base_nvcc_ldflags    :=

cuda_base_g++_flags     :=
cuda_base_g++_ldflags     :=

cuda_base_clang++_flags :=
cuda_base_clang++_ldflags :=

cuda_base_nvc++_flags   :=
cuda_base_nvc++_ldflags   :=

cuda_base_nvcc_flags    :=
cuda_base_nvcc_ldflags    :=

fast_g++_flags     :=
fast_g++_ldflags     :=

fast_clang++_flags :=
fast_clang++_ldflags :=

fast_nvc++_flags   :=
fast_nvc++_ldflags   :=

fast_nvcc_flags    :=
fast_nvcc_ldflags    :=

#
# you are unlikely to change anything below
#

headers := $(wildcard include/*.h)

#
# template of compilation rules
#
define compile
exe/mnist_$(ver) : $($(ver)_cc) $(headers) exe/dir
	$($(ver)_cxx) $(flags) $($($(ver)_cxx)_flags) $($(ver)_flags) $($(ver)_$($(ver)_cxx)_flags) -o $$@ $($(ver)_cc) $(ldflags) $($($(ver)_cxx)_ldflags) $($(ver)_ldflags) $($(ver)_$($(ver)_cxx)_ldflags) 
endef

targets := $(foreach ver,$(vers),exe/mnist_$(ver))

all : $(targets)

$(foreach ver,$(vers),$(eval $(call compile)))

exe/dir :
	mkdir -p $@

tag_srcs := $(patsubst %,tag_dir/%,$(headers) mnist.cc)

$(tag_srcs) : tag_dir/% : %
	install -D $< $@

docs/doxy/html/index.html : $(headers) mnist.cc
	doxygen docs/Doxyfile

docs/tags/HTML/index.html : docs/tags/dir $(tag_srcs)
	htags -C tag_dir --suggest2 ../docs/tags

docs/tags/dir :
	mkdir -p $@

doc : docs/doxy/html/index.html 
doc : docs/tags/HTML/index.html

clean :
	rm -rf exe

cleandoc :
	rm -rf docs/doxy docs/tags tag_dir

.DELETE_ON_ERROR :

