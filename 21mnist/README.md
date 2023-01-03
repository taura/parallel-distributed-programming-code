## code

```
make 
```

## data

```
git clone git@github.com:pytorch/examples.git
cd examples/mnist
python3 mnist.py
```

will download data into examples/data/MNIST/raw

make a symlink to raw directory as data

```
cd ../..
ln -s examples/data/MNIST/raw data
```

# Pascal Vincent format

data[3] D (dimension)
data[2] T (element type) {8: torch.uint8, 9: torch.int8, 11: torch.int16, 12: torch.int32, 13: torch.float32, 14: torch.float64}

next D ints (4 bytes x D)
  size of each dimension
  
divide each byte by 255
subtract mean 0.3..
divide by std 0.4..
  
convert cifar10 data to Pascal Vincent format and MNIST can be used for Cifar-10 as well

/usr/local/lib/python3.10/dist-packages/torch/optim/
_single_tensor_adadelta
(Pdb) p lr
1.0
(Pdb) p rho
0.9
(Pdb) p eps
1e-06
(Pdb) p weight_decay
0
(Pdb) p maximize
False

square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)

square_avg -> vt のこと

