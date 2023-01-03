# parallel-distributed

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
