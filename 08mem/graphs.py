#!/usr/bin/python
import sys,os,types

# ------------- preamble -------------

#import smart_gnuplotter
import lots_plots as lp

def Es(s):
    sys.stderr.write(s)

def get_unique(g, db, f):
    return g.do_sql(db,
                    '''
select distinct %s from a 
order by %s
''' % (f, f))

def get_max(g, db, f):
    return g.do_sql(db,
                    '''
select max(%s) from a 
''' % f)

#g = smart_gnuplotter.smart_gnuplotter()
g = lp.lots_plots()

sqlite_file = sys.argv[1] if len(sys.argv) > 1 else "a.sqlite"
out_dir     = sys.argv[2] if len(sys.argv) > 2 else "graphs"

db = g.open_sql(sqlite_file)


# ------------- contents -------------

def mk_plot_title(b):
    if b["eq"] == "=":
        x = "local"
    else:
        x = "remote"
    rec_sz = b["rec_sz"]
    return x

ws_ranges = [ (0, 2 ** 30),     # all
              #(2 ** 14, 2 ** 16), # around L1 cache
              #(2 ** 18, 2 ** 21), # around L2 cache
              #(2 ** 23, 2 ** 26), # around L3 cache
              (2 ** 25, 2 ** 30) ] # main memory

# -------------- latency with 1 chain --------------

def graph_latency():
    # show latency of link list traversal
    # x : size of the data
    # y : latency per access
    # (1) only local
    # (2) compare local and remote
    for eqs,conf,label in [ ([ "=" ],      "local", "local"), 
                            ([ "=", "<>" ],"local_remote", "local and remote") ]:
        g.graphs((db,
                  '''
select sz,
       avg(cpu_clocks_per_rec),
       cimin(cpu_clocks_per_rec,0.05),
       cimax(cpu_clocks_per_rec,0.05)
from a 
where host="{host}"
  and method="ptrchase"
  and nc=1
  and nthreads=1
  and rep>=1
  and rec_sz={rec_sz}
  and {min_sz}<=sz
  and sz<={max_sz}
  and shuffle=1
  and prefetch=0
  and payload=0
  and cpu_node {eq} mem_node
group by sz 
order by sz;
''',
                  "","",[]),
                 output="{out_dir}/latency_{conf}_{host}_{min_sz}_{max_sz}",
                 graph_vars=[ "out_dir", "conf", "host", "min_sz__max_sz" ],
                 graph_title="latency per load in a random list traversal [{min_sz},{max_sz}]",
                 graph_attr='''
set logscale x 2
set xtics rotate by -20
set key left
#unset key
''',
                 yrange="[0:]",
                 ylabel="latency/load (CPU cycles)",
                 xlabel="size of the region (bytes)",
                 plot_with="yerrorlines",
                 plot_title=mk_plot_title,
                 out_dir=[out_dir],
                 conf=[conf],
                 host=get_unique(g, db, "host"),
                 min_sz__max_sz=ws_ranges,
                 eq=eqs,
                 rec_sz=get_unique(g, db, "rec_sz"),
                 verbose_sql=2,
                 save_gpl=0)

# -------------- bandwidth local vs remote --------------

def graph_bw_ptrchase():
    for eqs,conf,label in [ ([ "=" ], "local", "local"), 
                            ([ "=", "<>" ], "local_remote", "local and remote") ]:
        g.graphs((db,
                  '''
select sz,avg(gb_per_sec) 
from a 
where method="ptrchase"
  and nc=1
  and nthreads=1
  and rep>=1
  and rec_sz={rec_sz}
  and {min_sz}<=sz
  and sz<={max_sz}
  and shuffle=1
  and prefetch=0
  and payload=1
  and cpu_node {eq} mem_node
group by sz 
order by sz
''',
                  "","",[]),
                 output="{out_dir}/bw_{conf}_{host}_{min_sz}_{max_sz}",
                 graph_vars=[ "out_dir", "conf", "host", "min_sz__max_sz" ],
                 graph_title="bandwidth of list traversal [{min_sz},{max_sz}]",
                 graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
                 yrange="[0:]",
                 ylabel="bandwidth (GB/sec)",
                 xlabel="size of the region (bytes)",
                 plot_with="linespoints",
                 plot_title=mk_plot_title,
                 out_dir=[out_dir],
                 conf=[conf],
                 host=get_unique(g, db, "host"),
                 min_sz__max_sz=ws_ranges,
                 rec_sz=get_unique(g, db, "rec_sz"),
                 eq=eqs,
                 verbose_sql=2,
                 save_gpl=0)

# -------------- bandwidth with X chains --------------

def graph_bw_ptrchase_chains():
    for eqs,conf in [ ([ "="  ], "local"), 
                      ([ "<>" ], "remote") ]:
        g.graphs((db,
                  '''
select sz,avg(gb_per_sec) 
from a 
where method="ptrchase"
  and nc={nc}
  and nthreads=1
  and rep>=1
  and rec_sz={rec_sz}
  and {min_sz}<=sz
  and sz<={max_sz}
  and shuffle=1
  and prefetch=0
  and payload=1
  and cpu_node {eq} mem_node
group by sz 
order by sz
''',
                  "","",[]),
                 output="{out_dir}/bw_chains_{conf}_{host}_{min_sz}_{max_sz}",
                 graph_vars=[ "out_dir", "conf", "host", "min_sz__max_sz" ],
                 graph_title="bandwidth with a number of chains [{min_sz},{max_sz}]",
                 graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
                 yrange="[0:]",
                 ylabel="bandwidth (GB/sec)",
                 xlabel="size of the region (bytes)",
                 plot_with="linespoints",
                 plot_title="{nc} chains",
                 out_dir=[out_dir],
                 conf=[conf],
                 host=get_unique(g, db, "host"),
                 min_sz__max_sz=ws_ranges,
                 rec_sz=get_unique(g, db, "rec_sz"),
                 nc=get_unique(g, db, "nc"),
                 eq=eqs,
                 verbose_sql=2,
                 save_gpl=0)

# -------------- bandwidth with prefetch --------------

def graph_bw_prefetch():
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method="ptrchase"
  and nc=1
  and nthreads=1
  and rep>=1
  and rec_sz={rec_sz}
  and {min_sz}<=sz
  and sz<={max_sz}
  and shuffle=1
  and prefetch={prefetch}
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
              "","",[]),
             output="{out_dir}/bw_prefetch_{host}_{min_sz}_{max_sz}",
             graph_vars=[ "out_dir", "host", "min_sz__max_sz" ],
             graph_title="bandwidth w/ and w/o prefetch [{min_sz},{max_sz}]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
             yrange="[0:]",
             ylabel="bandwidth (GB/sec)",
             xlabel="size of the region (bytes)",
             plot_with="linespoints",
             plot_title="prefetch={prefetch}",
             out_dir=[out_dir],
             host=get_unique(g, db, "host"),
             min_sz__max_sz=ws_ranges,
             rec_sz=get_unique(g, db, "rec_sz"),
             prefetch=[0,10],
             verbose_sql=2,
             save_gpl=0)

#
# -------------- compare list, random index, serial --------------
#

def graph_methods():
    # compare link list traversal vs. random index
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method = "{method}"
  and nc=1
  and nthreads=1
  and rep>=1
  and rec_sz={rec_sz}
  and {min_sz}<=sz
  and sz<={max_sz}
  and shuffle=1
  and prefetch=0
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
              "","",[]),
             output="{out_dir}/methods_{host}_{min_sz}_{max_sz}",
             graph_vars=[ "out_dir", "host", "min_sz__max_sz" ],
             graph_title="list traversal vs random access vs sequential access [{min_sz},{max_sz}]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
             yrange="[0:]",
             ylabel="bandwidth (GB/sec)",
             xlabel="size of the region (bytes)",
             plot_with="linespoints",
             plot_title="{method}",
             out_dir=[out_dir],
             host=get_unique(g, db, "host"),
             min_sz__max_sz=ws_ranges,
             rec_sz=get_unique(g, db, "rec_sz"),
             method=get_unique(g, db, "method"),
             verbose_sql=2,
             save_gpl=0)

# -------------- compare sorted vs unsorted --------------

def mk_plot_title_prefetch(b):
    if b["shuffle"] == 0:
        return "address-sorted list"
    else:
        return "random list"

def graph_sort_vs_unsorted():
    # compare two link list traversals
    # randomly ordered list vs address-sorted list
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method="ptrchase"
  and nc=1
  and nthreads=1
  and rep>=1
  and rec_sz={rec_sz}
  and {min_sz}<=sz
  and sz<={max_sz}
  and shuffle={shuffle}
  and prefetch=0
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
              "","",[]),
             output="{out_dir}/sorted_{host}_{min_sz}_{max_sz}",
             graph_vars=[ "out_dir", "host", "min_sz__max_sz" ],
             graph_title="bandwidth of random list traversal vs address-ordered list traversal [{min_sz},{max_sz}]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
             yrange="[0:]",
             ylabel="bandwidth (GB/sec)",
             xlabel="size of the region (bytes)",
             plot_with="linespoints",
             plot_title=mk_plot_title_prefetch,
             out_dir=[out_dir],
             host=get_unique(g, db, "host"),
             min_sz__max_sz=ws_ranges,
             rec_sz=get_unique(g, db, "rec_sz"),
             shuffle=[ 0, 1 ],
             verbose_sql=2,
             save_gpl=0)


# -------------- summary of various settings --------------

def mk_plot_title_all_access(b):
    method   = b["method"]
    shuffle  = ("" if b["shuffle"]  else " (sorted)")
    prefetch = (" (prefetch)" if b["prefetch"] else "")
    nc       = ((" (x %d)" % b["nc"]) if b["nc"] > 1 else "")
    if method == "ptrchase":
        return "%s%s%s%s" % (method, shuffle, prefetch, nc)
    else:
        return "%s" % method

def graph_summary():
    # seq vs random vs ptrchase
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method="{method}"
  and nc={nc}
  and nthreads=1
  and rep>=1
  and rec_sz={rec_sz}
  and {min_sz}<=sz
  and sz<={max_sz}
  and shuffle={shuffle}
  and prefetch={prefetch}
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
              "","",[]),
             output="{out_dir}/summary_{host}_{min_sz}_{max_sz}",
             graph_vars=[ "out_dir", "host", "min_sz__max_sz" ],
             graph_title="summary of various access patterns [{min_sz},{max_sz}]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
             yrange="[0:]",
             ylabel="bandwidth (GB/sec)",
             xlabel="size of the region (bytes)",
             plot_with="linespoints",
             plot_title=mk_plot_title_all_access,
             out_dir=[out_dir],
             host=get_unique(g, db, "host"),
             min_sz__max_sz=ws_ranges,
             rec_sz=get_unique(g, db, "rec_sz"),
             method=[ "ptrchase", "random", "sequential" ],
             shuffle=[ 0, 1 ],
             prefetch=[ 0, 10 ],
             nc=[ 1, 10 ],
             verbose_sql=2,
             save_gpl=0)

# -------------- bandwidth with X threads --------------

def graph_bw_ptrchase_threads():
    # show the effect of increasing number of threads with max chains
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method="ptrchase"
  and nc={nc}
  and nthreads={nthreads}
  and rep>=1
  and rec_sz={rec_sz}
  and {min_sz}<=sz
  and sz<={max_sz}
  and shuffle=1
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
                  "","",[]),
             output="{out_dir}/bw_threads_{host}_{min_sz}_{max_sz}",
             graph_vars=[ "out_dir", "host", "min_sz__max_sz" ],
             graph_title="bandwidth with a number of threads [{min_sz},{max_sz}]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
             yrange="[0:]",
             ylabel="bandwidth (GB/sec)",
             xlabel="size of the region (bytes)",
             plot_with="linespoints",
             plot_title="{nc} chains, {nthreads} threads",
             out_dir=[out_dir],
             host=get_unique(g, db, "host"),
             min_sz__max_sz=ws_ranges,
             rec_sz=get_unique(g, db, "rec_sz"),
             #nc=get_unique(g, db, "nc"),
             nc=[1,10],
             #nthreads=get_unique(g, db, "nthreads"),
             nthreads=[1,4,8,16],
             verbose_sql=2,
             save_gpl=0)

# -------------- bandwidth with X threads --------------

def graph_bw_methods_threads():
    # compare link list traversal vs. random index
    g.graphs((db,
              '''
select sz,avg(gb_per_sec) 
from a 
where method = "{method}"
  and nc=1
  and nthreads={nthreads}
  and rep>=1
  and rec_sz={rec_sz}
  and {min_sz}<=sz
  and sz<={max_sz}
  and shuffle=1
  and prefetch=0
  and payload=1
  and cpu_node=mem_node
group by sz 
order by sz
''',
              "","",[]),
             output="{out_dir}/bw_methods_threads_{host}_{min_sz}_{max_sz}",
             graph_vars=[ "out_dir", "host", "min_sz__max_sz" ],
             graph_title="bandwidth with various methods and number of threads [{min_sz},{max_sz}]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key right
#unset key
''',
             yrange="[0:]",
             ylabel="bandwidth (GB/sec)",
             xlabel="size of the region (bytes)",
             plot_with="linespoints",
             plot_title="{method} {nthreads} threads",
             out_dir=[out_dir],
             host=get_unique(g, db, "host"),
             min_sz__max_sz=ws_ranges,
             rec_sz=get_unique(g, db, "rec_sz"),
             #method=get_unique(g, db, "method"),
             method=["random", "sequential"],
             #nthreads=get_unique(g, db, "nthreads"),
             nthreads=[1,8,12,16],
             verbose_sql=2,
             save_gpl=0)

# -------------- cache miss --------------

def graph_cache(events):
    # show latency of link list traversal
    # x : size of the data
    # y : latency per access
    # (1) only local
    # (2) compare local and remote
    g.graphs((db,
              '''
              select 
              sz,
              avg({event}/(nloads+0.0)),
              cimin({event}/(nloads+0.0),0.05),
              cimax({event}/(nloads+0.0),0.05)
from a 
where host="{host}"
  and method="ptrchase"
  and nc=1
  and nthreads=1
  and rep>=1
  and rec_sz={rec_sz}
  and {min_sz}<=sz
  and sz<={max_sz}
  and shuffle=1
  and prefetch=0
  and payload=0
group by sz 
order by sz;
''',
                  "","",[]),
             output="{out_dir}/cache_miss_{host}_{min_sz}_{max_sz}",
             graph_vars=[ "out_dir", "host", "min_sz__max_sz" ],
             graph_title="cache miss rate of a random list traversal [{min_sz},{max_sz}]",
             graph_attr='''
set logscale x
#set xtics rotate by -20
set key left
#unset key
''',
             yrange="[0:]",
             ylabel="miss rate",
             xlabel="size of the region (bytes)",
             plot_with="yerrorlines",
             plot_title="{event} rec_sz={rec_sz}",
             out_dir=[out_dir],
             host=get_unique(g, db, "host"),
             min_sz__max_sz=ws_ranges,
             rec_sz=get_unique(g, db, "rec_sz"),
             event=events,
             verbose_sql=2,
             save_gpl=0)

#g.default_terminal = 'epslatex color size 9cm,6cm font "" 8'
g.default_terminal = 'epslatex color size 9cm,5.5cm font "" 8'

if 1:
    graph_latency()
    graph_bw_ptrchase()
    graph_bw_ptrchase_chains()
    graph_sort_vs_unsorted()
    graph_methods()
    graph_bw_prefetch()
    graph_summary()
    graph_bw_ptrchase_threads()
    graph_bw_methods_threads()
    graph_cache([ "l1d_replacement", "l2_lines_in", "longest_lat_cache_miss" ])

