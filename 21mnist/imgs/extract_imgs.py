#!/usr/bin/python3
import struct
import os

def read_int32(fp):
    s = fp.read(4)
    (p,) = struct.unpack(">I", s)
    return p

def pascal_vincent_to_pgms(image_path, label_path, out_dir):
    fp = open(image_path, "rb")
    fpl = open(label_path, "rb")
    magic = read_int32(fp)
    magicl = read_int32(fpl)
    nd = magic % 256
    ty = magic // 256
    ndl = magicl % 256
    tyl = magicl // 256
    assert(nd in [1, 3, 4]), nd
    assert(ty == 8), ty
    assert(ndl == 1), ndl
    assert(tyl == 8), tyl
    dim = []
    for i in range(nd):
        dim.append(read_int32(fp))
    for i in range(ndl):
        n_labels = read_int32(fpl)
    [n_images, h, w] = dim
    assert(n_images == n_labels), (n_images, n_labels)
    chunk = 500
    for b in range(0, n_images, chunk):
        e = min(b + chunk, n_images)
        os.makedirs("%s/%05d_%05d"
                    % (out_dir, b, e), exist_ok=True)
        for i in range(b, e):
            pixels = fp.read(h * w)
            label, = struct.unpack("B", fpl.read(1))
            wp = open("%s/%05d_%05d/i%05d_%d.pgm"
                      % (out_dir, b, e, i, label), "wb")
            wp.write(b"P5\n")
            wp.write(bytes("{} {} {}\n".format(h, w, 255), "ascii"))
            wp.write(pixels)
            wp.close()

pascal_vincent_to_pgms("../data/train-images-idx3-ubyte",
                       "../data/train-labels-idx1-ubyte",
                       "train-pgms")
pascal_vincent_to_pgms("../data/t10k-images-idx3-ubyte",
                       "../data/t10k-labels-idx1-ubyte",
                       "t10k-pgms")
