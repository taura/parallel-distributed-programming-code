#!/bin/bash

db=a.sqlite
rm -f ${db}


sqlite3 ${db} 'select * from a limit 5'

stuff/txt2sql ${db} --table a \
              -f 'out/out_(?P<host>[^_]+)' \
              -e 'algo = (?P<algo>.+)' \
              -e 'bs = (?P<bs>\d+)' \
              -e 'c = (?P<c>\d+)' \
              -e 'm = (?P<m>\d+)' \
              -e 'n = (?P<n>\d+)' \
              -e 'L = (?P<L>\d+)' \
              -e '(?P<nsec>\d+) nsec' \
              -e '(?P<ref_clocks>\d+) ref clocks' \
              -e '(?P<cpu_clocks>\d+) cpu clocks' \
              -e '(?P<nsec_per_iter>\d+\.\d+) nsec       for performing x=ax\+b for \d+ variables once' \
              -e '(?P<ref_clocks_per_iter>\d+\.\d+) ref clocks for performing x=ax\+b for \d+ variables once' \
              -e '(?P<cpu_clocks_per_iter>\d+\.\d+) cpu clocks for performing x=ax\+b for \d+ variables once' \
              -e '(?P<flops_per_nsec>\d+\.\d+) flops/nsec' \
              -e '(?P<flops_per_ref_clock>\d+\.\d+) flops/ref clock' \
              -r '(?P<flops_per_cpu_clock>\d+\.\d+) flops/cpu clock' \
        out/out_*.txt

