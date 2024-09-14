args :=
args += --M 100
args += --N 100
args += --nnz 1000
args += --repeat 3

vargs :=
vargs += --leak-check=full
vargs += --show-leak-kinds=all

val :
	valgrind ./spmv $(args) --format coo --algo serial
	valgrind ./spmv $(args) --format coo --algo parallel
	valgrind ./spmv $(args) --format csr --algo serial
	valgrind ./spmv $(args) --format csr --algo parallel
