
demonstrates how to use user-defined reduction.

```
$ make
gcc -fopenmp -Wall -Wextra -o udr_simple udr_simple.c 
gcc -fopenmp -Wall -Wextra -o udr_varlen_vect udr_varlen_vect.c 
```

```
$ ./udr_simple 
a[0] = 3334.000000
a[1] = 3333.000000
a[2] = 3333.000000
```

```
$ ./udr_simple 
a[0] = 3334.000000
a[1] = 3333.000000
a[2] = 3333.000000
```

```
$ ./udr_varlen_vect 5
a[0] = 200000.000000
a[1] = 200000.000000
a[2] = 200000.000000
a[3] = 200000.000000
a[4] = 200000.000000
```

```
$ ./udr_varlen_vect 7
a[0] = 142858.000000
a[1] = 142857.000000
a[2] = 142857.000000
a[3] = 142857.000000
a[4] = 142857.000000
a[5] = 142857.000000
a[6] = 142857.000000
```
