/**
   @file cifar.h
   @brief cifar dataset handling
 */
#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

/**
   @brief an input + true label
 */
template<idx_t IC,idx_t H,idx_t W>
struct cifar10_data_item {
  int index;                    /**< index in the original file */
  unsigned char rgb[IC][H][W];  /**< original pixel values */
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
   @brief an entire cifar10 data
*/
template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
struct cifar10_dataset {
  long n_data;                  /**< the total number of images  */
  long n_validate;              /**< the number of validation images */
  long n_train;                 /**< the number of traininig images  */
  cifar10_data_item<IC,H,W> * dataset; /**< whole dataset (n_data items) */
  cifar10_data_item<IC,H,W> * validate; /**< validation part (first n_validate items) */
  cifar10_data_item<IC,H,W> * train;   /**< training part (remaining items) */
  rnd_gen_t rg;                        /**< random number generator to pick images for a mini batch  */
  /**
     @brief set seed for random number generator
     @sa (sd) seed
   */
  void set_seed(long sd) {
    rg.seed(sd);
  }
  /**
     @brief count the number of images in a file (should be 10000)
     @param (cifar_bin) the name of the file to read data from
   */
  long get_n_data_in_file(const char * cifar_bin) {
    struct stat sb[1];
    if (lstat(cifar_bin, sb) == -1) {
      perror("lstat");
      fprintf(stderr,
              "error: could not find the cifar data file %s. Specify a right file name with -d FILENAME or do this so that it can find the default file (cifar-10-batches-bin/data_batch_1.bin).\n"
              "\n"
              "$ ln -s /home/tau/cifar10/cifar-10-batches-bin\n",
              cifar_bin);
      exit(1);
    }
    long sz = sb->st_size;
    long sz1 = IC * H * W + 1;
    assert(sz % sz1 == 0);
    return sz / sz1;
  }
  long get_decimal_digits(long n) {
    /* n <  10 -> 1 
       n < 100 -> 2
       find minimum d s.t. n < 10^d
*/
    long d = 1;
    long x = 10;                // x = 10^d
    for (d = 1; x < n; d++) {
      x = x * 10;
    }
    assert(n <= x);
    return d;
  }
  
  /**
     @brief dump dataset into files
     @param (dataset) dataset
     @param (n_data) size of dataset
     @param (prefix) prefix of files, like "img/img_" (-> img/img_xxxxx.ppm)
   */
  int dump_cifar_files(cifar10_data_item<IC,H,W>* dataset, long n_data,
                       long n_digits, 
                       const char * prefix) {
    /* chars required for data numbers */
    long nd = get_decimal_digits(n_data);
    if (n_digits < nd) {
      n_digits = nd;
    }
    long len = strlen(prefix) + n_digits + strlen(".ppm");
    char filename[len + 1];
    char fmt[100];
    /* -> "%s%05d.ppm" */
    int w = snprintf(fmt, 100, "%%s%%0%ldld.ppm", n_digits);
    assert(w <= 100);
    for (long d = 0; d < n_data; d++) {
      /* make a filename like "img/img_01234.ppm" */
      int w = snprintf(filename, len + 1, fmt, prefix, d);
      assert(w == len);
      FILE * wp = fopen(filename, "w");
      if (!wp) {
        perror("fopen");
        fprintf(stderr, "%s\n", filename);
        exit(1);
      }
      fprintf(wp, "P3 %d %d 255\n", W, H);
      for (idx_t i = 0; i < H; i++) {
        for (idx_t j = 0; j < W; j++) {
          for (idx_t c = 0; c < IC; c++) {
            fprintf(wp, " %d", dataset[d].rgb[c][i][j]);
          }
          fprintf(wp, "\n");
        }
      }
      fclose(wp);
    }
    return 1;                   /* OK */
  }


  /**
     @brief load training/validation data from the file 
     @param (lgr) logger
     @param (cifar_bin) the name of the file to read data from
     @param (n_samples) the number of data used 
     @param (sample_seed) seed of the random number generator to
     pick training and validation data
     @param (validate_ratio) leave this much for validation (<1.0)
   */
  int load(logger& lgr,
           const char * cifar_bin, long n_samples,
           long sample_seed, double validate_ratio,
           const char * dump_prefix) {
    long n_data_in_file = get_n_data_in_file(cifar_bin);
    if (n_samples == 0) {
      n_samples = n_data_in_file;
    } else if (n_samples > n_data_in_file) {
      lgr.log(1, "specified number of samples (%ld) exeeds data in file (%ld). truncated\n",
              n_samples, n_data_in_file);
      n_samples = n_data_in_file;
    }
    n_data = n_samples;
    n_train = max_i(1, n_data - n_data * validate_ratio);
    n_validate = n_data - n_train;
    lgr.log(1, "loading %ld/%ld training/validation data from %s starts",
            n_train, n_validate, cifar_bin);
    if (n_validate == 0) {
      lgr.log(1, "warning: no data left for validation (validation not performed)");
    }
    
    dataset = new cifar10_data_item<IC,H,W> [n_data_in_file];
    //validate = dataset;
    //train = dataset + n_validate;
    FILE * fp = fopen(cifar_bin, "rb");
    if (!fp) { perror("fopen"); exit(1); }
    for (int k = 0; k < n_data_in_file; k++) {
      unsigned char label[1];
      unsigned char rgb[IC][H][W];
      size_t r = fread(label, sizeof(label), 1, fp);
      if (ferror(fp)) { perror("fread"); exit(1); }
      if (feof(fp)) break;
      r = fread(rgb, sizeof(rgb), 1, fp);
      if (r != 1) { perror("fread"); exit(1); }
      int max_value = 0;
      for (idx_t c = 0; c < IC; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            max_value = max_i(max_value, rgb[c][i][j]);
          }
        }
      }
      float l_max = 1.0 / (float)max_value;
      dataset[k].index = k;
      dataset[k].label = label[0];
      for (idx_t c = 0; c < IC; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            dataset[k].rgb[c][i][j] = rgb[c][i][j];
            dataset[k](c,i,j) = rgb[c][i][j] * l_max;
          }
        }
      }
    }
    fclose(fp);

    if (dump_prefix) {
      dump_cifar_files(dataset, n_data, 0, dump_prefix);
    }
    
    /* shuffle data */
    rnd_gen_t rgv;
    rgv.seed(sample_seed);
    for (long t = 0; t < 15; t++) {
      for (long i = 0; i < n_data_in_file; i++) {
        long j = rgv.randi(i, n_data_in_file);
        cifar10_data_item<IC,H,W> d = dataset[j];
        dataset[j] = dataset[i];
        dataset[i] = d;
      }
    }
    train = dataset;
    validate = dataset + n_train;
    log_dataset(lgr);
    lgr.log(1, "loading data ends");
    return 1;
  }
  /**
     @brief log training dataset
     @param (lgr) logger
   */
  void log_dataset_train(logger& lgr) {
    long l = 0;
    char s[30];
    l += strlen("train:");
    for (long i = 0; i < n_train; i++) {
      sprintf(s, " %d", train[i].index);
      l += strlen(s);
    }
    char * data_str = (char *)malloc(l + 1);
    char * p = data_str;
    p[0] = 0;
    sprintf(p, "train:");
    p += strlen(p);
    for (long i = 0; i < n_train; i++) {
      sprintf(p, " %d", train[i].index);
      p += strlen(p);
    }
    p += strlen(p);
    assert(p == data_str + l);
    lgr.log(3, "%s", data_str);
    free(data_str);
  }
  /**
     @brief log validation dataset
     @param (lgr) logger
   */
  void log_dataset_validate(logger& lgr) {
    long l = 0;
    char s[30];
    l += strlen("validate:");
    for (long i = 0; i < n_validate; i++) {
      sprintf(s, " %d", validate[i].index);
      l += strlen(s);
    }
    char * data_str = (char *)malloc(l + 1);
    char * p = data_str;
    p[0] = 0;
    sprintf(p, "validate:");
    p += strlen(p);
    for (long i = 0; i < n_validate; i++) {
      sprintf(p, " %d", validate[i].index);
      p += strlen(p);
    }
    p += strlen(p);
    assert(p == data_str + l);
    lgr.log(3, "%s", data_str);
    free(data_str);
  }
  /**
     @brief log dataset
     @param (lgr) logger
   */
  void log_dataset(logger& lgr) {
    log_dataset_train(lgr);
    log_dataset_validate(lgr);
  }
  
  /**
     @brief load x and t with the a mini batch of B images
     @param (x) array to load images into
     @param (t) array to load true labels into
     @param (B) the number of images to pick
   */
  int get_data_train(array4<maxB,IC,H,W>& x, ivec<maxB>& t, ivec<maxB>& idxs, idx_t B) {
    assert(B <= maxB);
    x.set_n_rows(B);
    t.set_n(B);
    idxs.set_n(B);
    for (long b = 0; b < B; b++) {
      long idx = rg.randi(0, n_train);
      cifar10_data_item<IC,H,W>& itm = train[idx];
      idxs(b) = itm.index;
      t(b) = itm.label;
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            x(b,ic,i,j) = itm(ic,i,j);
          }
        }
      }
    }
    x.to_dev();
    t.to_dev();
    idxs.to_dev();
    return 1;
  }
  /**
     @brief load x and t with a part of validation data (from:to)
     @param (x) array to load images into
     @param (t) array to load true labels into
     @param (from) the first validation image to return
     @param (to) the last validation image to load + 1
   */
  int get_data_validate(array4<maxB,IC,H,W>& x, ivec<maxB>& t, ivec<maxB>& idxs, idx_t from, idx_t to) {
    idx_t B = to - from;
    assert(B <= maxB);
    x.set_n_rows(B);
    t.set_n(B);
    idxs.set_n(B);
    for (long b = 0; b < B; b++) {
      cifar10_data_item<IC,H,W>& itm = validate[from + b];
      t(b) = itm.label;
      idxs(b) = itm.index;
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            x(b,ic,i,j) = itm(ic,i,j);
          }
        }
      }
    }
    x.to_dev();
    t.to_dev();
    idxs.to_dev();
    return 1;
  }
};

/**
   @brief entry point of this header file
   @param (argc) the number of command line args
   @param (argv) command line args
*/
int cifar_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = opt.batch_sz;
  const idx_t C = 64;
  const idx_t H = 32;
  const idx_t W = 32;
  logger lgr;
  lgr.start_log(opt);
  cifar10_dataset<maxB,C,H,W> ds;
  ds.set_seed(opt.sample_seed);
  ds.get_n_data_in_file(opt.cifar_data);
  ds.load(lgr,
          opt.cifar_data,
          opt.partial_data, opt.partial_data_seed, opt.validate_ratio,
          opt.cifar_data_dump);
  array4<maxB,C,H,W> x;
  ivec<maxB> t;
  ivec<maxB> idxs;
  x.init_const(B, 0);
  t.init_const(B, 0);
  idxs.init_const(B, 0);
  ds.get_data_train(x, t, idxs, opt.batch_sz);
  lgr.end_log();
  return 0;
}

