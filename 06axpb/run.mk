include stuff/psweep.mk

parameters := host exe algo m n c bs threads

input = out/created
output = out/out_$(host)_$(algo)_$(m)_$(c)_$(bs)_$(threads).txt
cmd = OMP_NUM_THREADS=$(threads) ./$(exe) -a $(algo) -m $(m) -n $(n) -c $(c) --cuda-block-size $(bs) > $(output)

out/created :
	mkdir -p $@

# various algos
host:=$(shell hostname | tr -d [0-9])
n:=1000000

ifeq ($(host),big)
processor:=cpu
endif
ifeq ($(host),knm)
processor:=cpu
endif
ifeq ($(host),p)
processor:=gpu
endif
ifeq ($(host),v)
processor:=gpu
endif

# single variable

ifeq ($(processor),cpu)
exe := axpb.g++
algo := scalar simd simd_c
m := 0
c := 1
bs := 1
threads := 1
$(define_rules)

# simd_c with many vars
exe := axpb.g++
algo := simd_c
m := 0
c := $(shell seq 2 15)
bs := 1
threads := 1
$(define_rules)

# simd_m 
exe := axpb.g++
algo := simd_m
m := $(shell seq 16 16 320)
c := 1
bs := 1
threads := 1
$(define_rules)

# simd_m_mnm
exe := axpb.g++
algo := simd_m_mnm
m := $(shell seq 16 16 512)
c := $(shell seq 1 16)
bs := 1
threads := 1
$(define_rules)

endif

ifeq ($(processor),gpu)

# cuda cuda_c
exe := axpb.nvcc
algo := cuda
m := 1
c := $(shell seq 1 1 8)
bs := 1
threads := 1
#$(define_rules)

# cuda_c
exe := axpb.nvcc
algo := cuda_c
m := 1
c := $(shell seq 1 1 8)
bs := 1 $(shell seq 32 32 640)
threads := 1
#$(define_rules)

# cuda_c
exe := axpb.nvcc
algo := cuda_c
m := $(shell seq 1024 1024 163840)
c := 4
bs := 256
threads := 1
$(define_rules)

endif

.DELETE_ON_ERROR:
