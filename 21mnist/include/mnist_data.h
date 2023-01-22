
/**
   @file mnist_data.h
   @brief mnist dataset handling
 */
#pragma once

#include <err.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "mnist_util.h"
#include "tensor.h"

/**
   @brief read a 32 bit int
  */
static int read_int32(FILE * fp) {
  char a[4];
  union {
    char c[4];
    int x;
  } u;
  size_t r = fread(a, sizeof(a), 1, fp);
  if (r != 1) err(1, "fread");
  for (int i = 0; i < 4; i++) {
    u.c[i] = a[3 - i];
  }
  return u.x;
}

/**
   @brief data structure for pascal vicent format
  */
struct pascal_vincent {
  int n_dims;                   /**< number of dimensions */
  int type;                     /**< type */
  int * dim;                    /**< dimensions */
  long data_sz;                 /**< data size */
  void * data;                  /**< data */
  /** 
      @brief construct pascal_vincent 
  */
  pascal_vincent(int n_dims_, int type_, int * dim_,
                 long data_sz_, void * data_) {
    n_dims = n_dims_;
    type = type_;
    dim = dim_;
    data_sz = data_sz_;
    data = data_;
  }
};

/**
   @brief reada a file of pascal vincent format
   @param (path) file name
*/
static pascal_vincent read_pascal_vincent_format(const char * path) {
  FILE * fp = fopen(path, "rb");
  if (!fp) err(1, "%s", path);
  /* get file size */
  if (fseek(fp, 0, SEEK_END)) err(1, "fseek");
  long file_sz = ftell(fp);
  if (file_sz == -1) err(1, "ftell");
  /* back to start */
  rewind(fp);
  /* first four bytes */
  int32_t magic = read_int32(fp);
  int nd = magic % 256;
  int ty = magic / 256;
  assert(nd == 1 || nd == 3 || nd == 4);
  assert(ty == 8);
  /* next (nd x 4) bytes */
  int * dim = new int[nd];
  long data_sz = 1;
  for (int i = 0; i < nd; i++) {
    dim[i] = read_int32(fp);
    data_sz *= dim[i];
  }
  /* payload follows; get its size */
  long pos = ftell(fp);
  if (pos == -1) err(1, "ftell");
  assert(pos + data_sz == file_sz);
  void * data = malloc(data_sz);
  if (!data) err(1, "malloc");
  size_t r = fread(data, data_sz, 1, fp);
  if (r != 1) err(1, "fread");
  fclose(fp);
  return pascal_vincent(nd, ty, dim, data_sz, data);
}

/**
   @brief a single training datum + label
   @param (IC) channels
   @param (H) height
   @param (W) width
 */

template<idx_t IC,idx_t H,idx_t W>
struct data_item {
  int index;                    /**< index in the original file */
  unsigned char rgb[IC][H][W];  /**< original pixel values (may be grey scale) */
  real w[IC][H][W];             /**< pixels of an image */
  char label;                   /**< true label (0..9) */
  /**
     @brief get the (ic,i,j) pixel of the image
   */
  real& operator()(idx_t ic, idx_t i, idx_t j) {
    range_chk(0, ic, IC);
    range_chk(0, i, H);
    range_chk(0, j, W);
    return w[ic][i][j];
  }
};

/**
   @brief an entire data
*/
template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
struct mnist_dataset {
  long cur;
  long n_data;                  /**< the total number of images  */
  data_item<IC,H,W> * data;     /**< data (n_data items) */
  rnd_gen_t rg; /**< random number generator to pick images for a mini batch  */
  
  /**
     @brief set seed for random number generator
     @sa (sd) seed
   */
  void set_seed(long sd) {
    rg.seed(sd);
  }
  /**
     @brief load training/validation data from the file 
     @param (lgr) logger
     @param (data_dir) directory where training/test data files are in
   */
  int load(logger& lgr, const char * data_dir, long max_data,
           real mean, real std, int train) {
    lgr.log(1, "loading data from %s", data_dir);
    long data_dir_len = strlen(data_dir);
    char images_file[data_dir_len + 100];
    char labels_file[data_dir_len + 100];
    int written = snprintf(images_file, sizeof(images_file),
                           "%s/%s-images-idx3-ubyte", data_dir, (train ? "train" : "t10k"));
    assert(written < (int)sizeof(images_file));
    written = snprintf(labels_file, sizeof(labels_file),
                       "%s/%s-labels-idx1-ubyte", data_dir, (train ? "train" : "t10k"));
    assert(written < (int)sizeof(labels_file));
    pascal_vincent img_pv = read_pascal_vincent_format(images_file);
    pascal_vincent label_pv = read_pascal_vincent_format(labels_file);
    assert(label_pv.n_dims == 1);
    n_data = label_pv.dim[0];
    long n_img_dims = img_pv.n_dims;
    /* 3 : grey scale (n, H, W), 4 : rgb-color (n, 3, H, W) */
    assert(n_img_dims == 3 || n_img_dims == 4);
    assert(img_pv.dim[0] == n_data);
    assert(img_pv.dim[n_img_dims - 2] == H);
    assert(img_pv.dim[n_img_dims - 1] == W);
    assert(img_pv.data_sz == n_data * IC * H * W);

    data = new data_item<IC,H,W>[n_data];
    typedef unsigned char rgb_t[IC][H][W];
    rgb_t * imgs = (rgb_t *)img_pv.data;
    char * labels = (char *)label_pv.data;

    for (int k = 0; k < n_data; k++) {
      data[k].index = k;
      data[k].label = labels[k];
      for (idx_t c = 0; c < IC; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            unsigned char p = imgs[k][c][i][j];
            data[k].rgb[c][i][j] = p;
            data[k](c,i,j) = ((p / 255.0) - mean) / std;
          }
        }
      }
    }
    long n_used_data = (max_data < 0 ? n_data : min_i(n_data, max_data));
    lgr.log(1, "use %ld data items out of %ld", n_used_data, n_data);
    n_data = n_used_data;
    cur = 0;

    free(img_pv.data);
    delete[] img_pv.dim;
    free(label_pv.data);
    delete[] label_pv.dim;
    return 1;
  }

  /**
     @brief close
   */
  void close() {
    delete[] data;
    data = 0;
  }
  
  /**
     @brief get next batch from the beginning
   */
  void rewind() {
    cur = 0;
  }
  
  /**
     @brief load x and t with the a mini batch of B images
     @param (x) array to load images into
     @param (t) array to load true labels into
     @param (B) the number of data to get
     @return the actual number of data returned
   */
  idx_t get_data(tensor<real,maxB,IC,H,W>& x, tensor<idx_t,maxB>& t, tensor<idx_t,maxB>& idxs,
                 idx_t B, int cuda_algo) {
    assert(B <= maxB);
    idx_t actual_B = (n_data - cur < B ? n_data - cur : B);
    x.set_n0(actual_B);
    t.set_n0(actual_B);
    idxs.set_n0(actual_B);
    for (long b = 0; b < actual_B; b++) {
      long idx = cur;
      data_item<IC,H,W>& itm = data[idx];
      idxs(b) = itm.index;
      t(b) = itm.label;
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            x(b,ic,i,j) = itm(ic,i,j);
          }
        }
      }
      cur++;
    }
    to_dev(&x, cuda_algo);
    to_dev(&t, cuda_algo);
    to_dev(&idxs, cuda_algo);
    return actual_B;
  }
};

/**
   @brief entry point of this header file
   @param (argc) the number of command line args
   @param (argv) command line args
*/
int mnist_data_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = opt.batch_size;
  const idx_t C = 1;
  const idx_t H = 28;
  const idx_t W = 28;
  logger lgr;
  lgr.start_log(opt);
  mnist_dataset<maxB,C,H,W> ds;
  real mean = 0.1307;
  real std = 0.3081;
  ds.load(lgr, opt.data_dir, opt.train_data_size, mean, std, 1);
  tensor<real,maxB,C,H,W> x;
  tensor<idx_t,maxB> t;
  tensor<idx_t,maxB> idxs;
  x.init_const(B, 0);
  t.init_const(B, 0);
  idxs.init_const(B, 0);
  ds.get_data(x, t, idxs, B, opt.cuda_algo);
  lgr.end_log();
  delete[] ds.data;
  return 0;
}

