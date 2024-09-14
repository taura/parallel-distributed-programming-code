#!/usr/bin/python
import re

def comm():
    fp = open("a.txt", "rb")
    wp = open("a.gpl", "wb")
    # 0 -> 1 : src_loads=271.56 dst_loads=302.19 src_clocks=1547.15 dst_clocks=1547.12
    p = re.compile("(?P<s>\d+) -> (?P<d>\d+) : src_loads=(?P<sl>.*?) dst_loads=(?P<dl>.*?) src_clocks=(?P<sc>.*?) dst_clocks=(?P<dc>.*?)")
    C = {}
    for line in fp:
        m = p.match(line)
        if m:
            s,d,sc = m.group("s", "d", "sc")
            s = int(s)
            d = int(d)
            sc = float(sc)
            C[s,d] = sc
    n_src = max([ s for s,d in C.keys() ]) + 1
    n_dst = max([ d for s,d in C.keys() ]) + 1
    wp.write("set view map\n")
    wp.write("set size square\n")
    wp.write("set xrange [-0.5:%f]\n" % (n_src - 0.5))
    wp.write("set yrange [-0.5:%f]\n" % (n_dst - 0.5))
    wp.write("set xtics %d\n" % (1 if n_src < 8 else n_src / 8))
    wp.write("set ytics %d\n" % (1 if n_dst < 8 else n_dst / 8))
    wp.write("splot '-' matrix with image\n")
    for s in range(n_src):
        for d in range(n_dst):
            wp.write(" %f" % C[s,d])
        wp.write("\n")
    wp.write("e\n")
    wp.write("e\n")
    wp.close()
    fp.close()

def main():
    comm()

if __name__ == "__main__":
    main()

    
        
        
