c_files := $(wildcard *.c)
exe_files := $(patsubst %.c,%,$(c_files))
asm_files := $(patsubst %.c,%.s,$(c_files))

opts := 
opts += -O3
opts += -Wall -Wextra
opts += -march=native
opts += -funroll-loops
opts += -fopenmp

CC := gcc
CFLAGS := $(opts)
all : $(exe_files) $(asm_files)

$(exe_files) : % : %.c
	$(CC) -o $@ $(CFLAGS) $<

$(asm_files) : %.s : %.c
	$(CC) -o $@ -S $(CFLAGS) $<

clean :
	rm -f comm
