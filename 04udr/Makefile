all : udr_simple udr_varlen_vect

CFLAGS := -fopenmp -Wall -Wextra

udr_simple : udr_simple.c
	gcc $(CFLAGS) -o $@ $< 

udr_varlen_vect : udr_varlen_vect.c
	gcc $(CFLAGS) -o $@ $< 

clean :
	rm -f udr_simple udr_varlen_vect
