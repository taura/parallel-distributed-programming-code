#!/usr/bin/python3
import glob
import os
import re
import sys

def make_index_html(directory):
    tds = []
    for png in sorted(glob.glob("{}/*.png".format(directory))):
        base = os.path.basename(png)
        m = re.match("i\d+_(?P<t>\d).png", base)
        assert(m), base
        t = int(m.group("t"))
        tds.append('<td><img src="{}" /></td><td>{}</td>'.format(base, t))
    trs = []
    width = 20
    trs.append("<tr><td></td>{}</tr>".format("".join(['<td><font color="blue">{}</font></td><td></td>'.format(i) for i in range(width)])))
    for i in range(0, len(tds), width):
        e = min(len(tds), i+width)
        trs.append('<tr>\n<td><font color="blue">{}-{}</font></td>{}\n</tr>'.format(i, e, "\n".join(tds[i:e])))
    table = "<table>\n{}\n</table>".format("\n".join(trs))
    index_html = "{}/index.html".format(directory)
    wp = open(index_html, "w")
    wp.write(table)
    wp.close()

make_index_html(sys.argv[1])

