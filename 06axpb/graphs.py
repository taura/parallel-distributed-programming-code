#!/usr/bin/python3
import sys,os,math
#sys.path.append("stuff")
#import smart_gnuplotter
import lots_plots as lp

def Es(s):
    sys.stderr.write(s)

g = lp.lots_plots()

sqlite_file = sys.argv[1] if len(sys.argv) > 1 else "a.sqlite"
out_dir     = sys.argv[2] if len(sys.argv) > 2 else "graphs"

db = g.open_sql(sqlite_file)

def get_unique(g, db, f):
    return g.do_sql(db,
                    '''
select distinct %s from a 
order by %s
''' % (f, f))

def mk_graph_title(b):
    algo = b["algo"]
    if algo == "simd_c":
        return "a compile-time constant number of variables"
    elif algo == "simd_m":
        return "a variable number of variables"
    elif algo == "simd_m_mnm":
        return "a compile-time constant number of variables in the innermost loop"
    else:
        assert(0),b

def mk_plot_title(b):
    exp = b["exp"]
    if "cpu_clocks_per_iter" in exp:
        return "latency"
    elif exp == "flops_per_cpu_clock":
        return "throughput"
    else:
        assert(0), b

def mk_axis(b):
    exp = b["exp"]
    if "cpu_clocks_per_iter" in exp:
        return "axis x1y1"
    elif exp == "flops_per_cpu_clock":
        return "axis x1y2"
    else:
        assert(0), b

def plot_simd_c():
    g.graphs((db, 
              r"""
select c,avg({exp}) from a
where algo="{algo}"
and host="{host}"
group by c
order by c
""",
              "",
              "",
              []),
             output='{out_dir}/cpu_latency_throughput_{algo}_{host}',
             graph_vars=[ "out_dir", "algo", "host" ],
             graph_attr='''
set xrange [0:]
set yrange [0:]
set y2range [0:]
set xlabel "variables"
set ylabel "cycles/iter"
set y2label "flops/cycle"
set y2tics 0,8
set ytics nomirror
#set ytics 0,2

#set key outside center top horizontal samplen 3
''',
             graph_title=mk_graph_title, # "{host} {algo}"
             plot_with="linespoints",
             plot_title=mk_plot_title,
             plot_attr=mk_axis,
             out_dir=[ out_dir ],
             exp=[ "cpu_clocks_per_iter", "flops_per_cpu_clock" ],
             algo=[ "simd_c" ],
             host=[ "big" ],
             verbose_sql=1,
             save_gpl=1)

def plot_simd_m():
    g.graphs((db, 
              r"""
select m/16,avg({exp}) from a
where algo="{algo}"
and host="{host}"
group by m
order by m
""",
              "",
              "",
              []),
             output='{out_dir}/cpu_latency_throughput_{algo}_{host}',
             graph_vars=[ "out_dir", "algo", "host" ],
             graph_attr='''
set xrange [0:]
set yrange [0:]
set y2range [0:]
set xlabel "variables"
set ylabel "cycles/iter"
set y2label "flops/cycle"
set y2tics 0,8
set ytics nomirror
#set ytics 0,2

#set key outside center top horizontal samplen 3
''',
             graph_title=mk_graph_title, # "{host} {algo}",
             plot_with="linespoints",
             plot_title=mk_plot_title,
             plot_attr=mk_axis,
             out_dir=[ out_dir ],
             exp=[ "cpu_clocks_per_iter", "flops_per_cpu_clock" ],
             algo=[ "simd_m" ],
             host=[ "big" ],
             verbose_sql=1,
             save_gpl=1)

def plot_simd_m_mnm():
    g.graphs((db, 
              r"""
select c,avg({exp}) from a
where algo="{algo}"
and host="{host}"
group by c
order by c
""",
              "",
              "",
              []),
             output='{out_dir}/cpu_latency_throughput_{algo}_{host}',
             graph_vars=[ "out_dir", "algo", "host" ],
             graph_attr='''
set xrange [0:]
set yrange [0:]
set y2range [0:]
set xlabel "inner variables"
set ylabel "cycles/iter"
set y2label "flops/cycle"
set y2tics 0,8
set ytics nomirror
#set ytics 0,2

#set key outside center top horizontal samplen 3
''',
             graph_title=mk_graph_title, # "{host} {algo}"
             plot_with="linespoints",
             plot_title=mk_plot_title,
             plot_attr=mk_axis,
             out_dir=[ out_dir ],
             exp=[ "cpu_clocks_per_iter * c * 16 / m", "flops_per_cpu_clock" ],
             algo=[ "simd_m_mnm" ],
             host=[ "big" ],
             verbose_sql=1,
             save_gpl=1)
#
#
#

def mk_graph_title_gpu(b):
    if b["host"] == "p":
        return "Pascal"
    elif b["host"] == "v":
        return "Volta"
    else:
        assert(0), b
    
def plot_cuda_single_thread():
    g.graphs((db, 
              r"""
select c,avg(%(exp)s) from a
where algo="%(algo)s"
and host="%(host)s"
and bs = 1
group by c
order by c
""",
              "",
              "",
              []),
             output='%(out_dir)s/gpu_single_thread_%(algo)s_%(host)s',
             terminal='epslatex color size 6cm,3.5cm font "" 6',
             graph_vars=[ "out_dir", "algo", "host" ],
             graph_attr='''
set xrange [0:]
set yrange [0:]
set y2range [0:]
set xlabel "variables"
set ylabel "cycles/iter"
set y2label "flops/cycle"
set y2tics 0,0.1
set ytics nomirror
set key right bottom
''',
             graph_title=mk_graph_title_gpu,
             plot_with="linespoints",
             plot_title=mk_plot_title,
             plot_attr=mk_axis, 
             out_dir=[ out_dir ],
             exp=[ "cpu_clocks_per_iter", "flops_per_cpu_clock" ],
             algo=[ "cuda_c" ],
             host=[ "p", "v" ],
             verbose_sql=2,
             save_gpl=0)

#
# c
#
    
def plot_cuda_c():
    g.graphs((db, 
              r"""
select c,avg(%(exp)s) from a
where algo="%(algo)s"
and host="%(host)s"
and bs = %(bs)s
group by c
order by c
""",
              "",
              "",
              []),
             output='%(out_dir)s/gpu_c_flops_%(algo)s_%(host)s',
             graph_vars=[ "out_dir", "algo", "host" ],
             graph_attr='''
set xrange [0:]
set yrange [0:]
set xlabel "variables"
set ylabel "flops/cycle"
set key left top
''',
             graph_title=mk_graph_title_gpu,
             plot_with="linespoints",
             plot_title="bs=%(bs)s",
             plot_attr="", 
             out_dir=[ out_dir ],
             exp=[ "flops_per_cpu_clock" ],
             algo=[ "cuda_c" ],
             host=get_unique(g, db, "host"),
             bs=[32,64,96,128],
             verbose_sql=2,
             save_gpl=0)

#
# bs 
#
    
def plot_cuda_bs():
    g.graphs((db, 
              r"""
select bs,avg(%(exp)s) from a
where algo="%(algo)s"
and host="%(host)s"
and c = %(c)s
group by bs
order by bs
""",
              "",
              "",
              []),
             output='%(out_dir)s/gpu_bs_flops_%(algo)s_%(host)s',
             graph_vars=[ "out_dir", "algo", "host" ],
             graph_attr='''
set xrange [0:]
set yrange [0:64]
set xlabel "threads per block (bs)"
set ylabel "flops/cycle"
set key right bottom
''',
             graph_title=mk_graph_title_gpu,
             plot_with="linespoints",
             plot_title="c=%(c)s",
             plot_attr="", 
             out_dir=[ out_dir ],
             exp=[ "flops_per_cpu_clock" ],
             algo=[ "cuda_c" ],
             host=[ "p", "v" ],
             #c=get_unique(g, db, "c"),
             c=[1,2,4,8],
             verbose_sql=2,
             save_gpl=0)

g.default_terminal='epslatex color size 6cm,4cm font "" 6'
    
plot_simd_c()
plot_simd_m()
plot_simd_m_mnm()
#plot_cuda_single_thread()
#plot_cuda_bs()
#plot_cuda_c()


