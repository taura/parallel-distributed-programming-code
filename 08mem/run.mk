
include stuff/psweep.mk

numactl:=numactl -iall --

# ---------------------------------------

out_dir:=output

# ----- parameter and command definitions -----

# from 2^8 to 2^23 elements
#p:=$(shell seq 7 16)
#powers:=$(foreach i,$(p),$(shell echo $$((1 << $(i)))))
# multiply each of 2^8 - 2^23 by 7/9 - 15/9
#n:=$(foreach p,$(powers),$(foreach o,$(shell seq -2 4),$(shell echo $$(($(p) * (9 + $(o)) / 9)))))
# from 2^a to 2^b elements, taking s points between 2^i and 2^(i+1)
a:=7
b:=24
s:=6
#n:=$(shell python3 -c "for i in range($(a)*$(s),$(b)*$(s)): print(int(2.0**(i/$(s))))" | uniq)
n_per_thread:=$(shell echo $$((1024 * 1024 * 10)))

host:=$(shell hostname | tr -d [0-9])
events := l1d.replacement,l2_lines_in.all,longest_lat_cache.miss

#parameters:=host try rec_sz method n n_chains n_threads shuffle payload cpu_node mem_node prefetch
parameters:=host try rec_sz method n_per_thread n_chains n_threads shuffle payload cpu_core mem_opt prefetch

### commands and outputs ###
#cmd=(OMP_NUM_THREADS=$(n_threads) OMP_PROC_BIND=true numactl -N $(cpu_node) -i $(mem_node) -- ./mem -m $(method) -n $(n) -z $(rec_sz) -c $(n_chains) -x $(shuffle) -l $(payload) -p $(prefetch) -r 6 -e $(events)) > $(output)
#output=$(out_dir)/out_$(host)_$(method)_$(n)_$(rec_sz)_$(n_chains)_$(n_threads)_$(shuffle)_$(payload)_$(cpu_node)_$(mem_node)_$(prefetch)_$(try).txt

cmd=(OMP_NUM_THREADS=$(n_threads) OMP_PROC_BIND=true numactl --physcpubind $(cpu_core) -$(mem_opt) -- ./mem -m $(method) -n $$$$(($(n_threads) * $(n_per_thread))) -z $(rec_sz) -c $(n_chains) -x $(shuffle) -l $(payload) -p $(prefetch) -S 20 -r 6 -e $(events)) > $(output)
output=$(out_dir)/out_$(host)_$(method)_$(n_per_thread)_$(rec_sz)_$(n_chains)_$(n_threads)_$(shuffle)_$(payload)_$(cpu_core)_$(mem_opt)_$(prefetch)_$(try).txt

input=$(out_dir)/created

## common parameters ##
#host:=$(shell hostname | tr -d [0-9])
cpu_node:=0
payload:=1
try:=$(shell seq 1 1)
rec_sz:=64

## effect of number of chains ##
method:=p
n_chains:=1 2 4 8 10 12 14
n_threads:=1
shuffle:=1
prefetch:=0
mem_node:=0 1
#$(define_rules)

## effect of access methods ##
method:=s r
n_chains:=1 2 4
n_threads:=1
shuffle:=1
prefetch:=0
mem_node:=0
#$(define_rules)

## effect of prefetch ##
method:=p
n_chains:=1 2 4
n_threads:=1
shuffle:=1
prefetch:=0 10
mem_node:=0
#$(define_rules)

## effect of sorted addresses ##
method:=p
n_chains:=1 2 4
n_threads:=1
shuffle:=0
prefetch:=0
mem_node:=0
#$(define_rules)

## many threads with pointers ##
method:=p
n_chains:=1 5 10
n_threads:=1 2 4 6 8 12 16
shuffle:=1
prefetch:=0
mem_node:=0
#$(define_rules)

## many threads with indexes ##
method:=s r
n_chains:=1
n_threads:=1 2 4 6 8 12 16
shuffle:=1
prefetch:=0
mem_node:=0
#$(define_rules)

method:=s
n_chains:=1
n_threads:=12
shuffle:=0
prefetch:=0
mem_opt:= i0 i1 iall l
cpu_core:=0-11
$(define_rules)

method:=s
n_chains:=1
n_threads:=48
shuffle:=0
prefetch:=0
mem_opt:= i0 iall l
cpu_core:=0-11,16-27,32-43,48-59
$(define_rules)

$(out_dir)/created :
	mkdir -p $@

.DELETE_ON_ERROR:


