#!/bin/bash
set -e
parallel2_dir=$HOME/parallel2
a2sql_dir=stuff
txt2sql=${a2sql_dir}/txt2sql

db=a.sqlite
rm -f ${db}

${txt2sql} ${db} --table a \
    -f 'output/out_(?P<host>.*?)_.*?_(?P<cpu_node>\d+)_(?P<mem_node>\d+)_\d+_(?P<try>\d+).txt' \
    -e '--------- (?P<rep>\d+) ---------' \
    -e '(?P<n>\d+) elements x (?P<nc>\d+) chains x (?P<nscan>\d+) scans x (?P<nthreads>\d+) threads = (?P<nrecords>\d+) record accesses = (?P<nloads>\d+) loads' \
    -e 'record_size: (?P<rec_sz>\d+) bytes' \
    -e 'data: (?P<sz>\d+) bytes' \
    -e 'shuffle: (?P<shuffle>.*)' \
    -e 'payload: (?P<payload>.*)' \
    -e 'stride: (?P<stride>.*)' \
    -e 'prefetch: (?P<prefetch>.*)' \
    -e 'method: (?P<method>[^\s]+)' \
    -e 'metric:l1d.replacement = \d+ -> \d+ = (?P<l1d_replacement>\d+)' \
    -e 'metric:l2_lines_in\.all = \d+ -> \d+ = (?P<l2_lines_in>\d+)' \
    -e 'metric:longest_lat_cache\.miss = \d+ -> \d+ = (?P<longest_lat_cache_miss>\d+)' \
    -e '(?P<cpu_clocks>\d+) CPU clocks' \
    -e '(?P<ref_clocks>\d+) REF clocks' \
    -e '(?P<nano_sec>\d+?) nano sec' \
    -e 'throughput (?P<bytes_per_clock>.*?) bytes/clock' \
    -e 'throughput (?P<gb_per_sec>.*?) GiB/sec' \
    -e 'latency (?P<cpu_clocks_per_rec>.*?) CPU clocks' \
    -e 'latency (?P<ref_clocks_per_rec>.*?) REF clocks' \
    -r 'latency (?P<nano_sec_per_rec>.*?) nano sec' \
    output/out_*.txt



