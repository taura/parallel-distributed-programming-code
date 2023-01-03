#pragma once

#if __NVCC__
/**
   @brief a global CUDA function that implements the baseline 
   forward function for GPU
   @param (dev) the address of the device shadow of the object
   @param (x_dev) the address of the device shadow of the input matrix
   @sa forward_dev
   @sa forward_gpu
  */
template<typename T, typename I>
__global__ void forward_global(T* dev, I* x_dev) {
  /* call the member function */
  dev->forward_dev(*x_dev);
}

template<typename T, typename I0, typename I1>
__global__ void forward_global(T* dev, I0* x_dev, I1* t_dev) {
  /* call the member function */
  dev->forward_dev(*x_dev, *t_dev);
}

/**
   @brief a global CUDA function that implements the baseline 
   backward function for GPU
   @param (dev) the address of the device shadow of the object
   @param (gy_dev) the address of the device shadow of the input matrix
   @sa backward_dev
   @sa backward_gpu
  */
template<typename T, typename O>
__global__ void backward_global(T* dev, O* gy_dev) {
  dev->backward_dev(*gy_dev);
}

template<typename T, typename O, typename I1>
__global__ void backward_global(T* dev, O* gy_dev, I1* t_dev) {
  dev->backward_dev(*gy_dev, *t_dev);
}

/**
   @brief a global CUDA function that implements the baseline 
   update function for GPU
   @param (dev) the address of the device shadow of the object
   @sa update_dev
   @sa update_gpu
  */
template<typename T>
__global__ void update_global(T* dev) {
  dev->update_dev();
}
#endif

/**
   @brief if the algorithm is a gpu algorithm, allocate a device shadow 
   of this object and set dev field of this and all subobjects. otherwise
   it sets all dev fields to null.
   @sa set_dev
   @sa del_dev
*/
template<typename T>
void make_dev(T * layer, int gpu_algo) {
#if __NVCC__
  assert(layer->dev == 0);
  if (gpu_algo) {
    layer->dev = (T*)dev_malloc(sizeof(T));
  }
  layer->set_dev(layer->dev);
#else
  (void)gpu_algo;
  (void)layer;
#endif
}

/**
   @brief make a copy of this 
   @details if this object has a device pointer, the copy will have
   a device pointer too, but its contents are NOT copied
*/
template<typename T>
T* make_copy(T * layer, int gpu_algo) {
  T * c = new T(*layer);
  c->dev = 0;
  make_dev(c, gpu_algo);
  return c;
}
/**
   @brief if the algorithm is a gpu algorithm, dev field must not
   be null and deallocate it.
   @sa make_dev
   @sa set_dev
*/
template<typename T>
void del_dev(T * layer, int gpu_algo) {
#if __NVCC__
  if (gpu_algo) {
    assert(layer->dev);
    dev_free(layer->dev);
    layer->dev = 0;
  }
#else
  (void)gpu_algo;
  (void)layer;
#endif
}
/**
   @brief if the algorithm is a gpu algorithm, dev field must
   not be null and send the host data to the device memory
*/
template<typename T>
void to_dev(T * layer, int gpu_algo) {
#if __NVCC__
  if (gpu_algo) {
    T* dev_ = layer->dev;
    assert(dev_);
    ::to_dev(dev_, layer, sizeof(T));
  }
#else
  (void)gpu_algo;
  (void)layer;
#endif
}
/**
   @brief if the algorithm is a gpu algorithm, dev field must
   not be null and send the device data to the host memory
*/
template<typename T>
void to_host(T * layer, int gpu_algo) {
#if __NVCC__
  if (gpu_algo) {
    T* dev_ = layer->dev;
    assert(dev_);
    ::to_host(layer, dev_, sizeof(T));
  }
#else
  (void)gpu_algo;
  (void)layer;
#endif
}

/**
   @brief check the gradient computation of a convolution layer
   @param (opt) command line option
   @param (lgr) logger 
   @param (rg) random number generator
   @param (B) the number of images
   @sa convolution_main
   @details it first makes a layer object with initial weights W 
   and generates an input (x and t).
   it then creates two layers whose weights are slightly different
   from the original one by dw/2 (i.e., w-dw/2 and w+dw/2), as well as
   two inputs slighly different from the original inputs by dx/2
   (x-dx/2 and x+dx/2).  it then computes L(w,x), L(x-dw/2,x-dx/2) and
   L(w+dw/2,x+dw/2) and check if L(x+dw/2,x+dx/2)-L(x-dw/2,x-dx/2)
   is close to ∂L/∂x dx + ∂L/∂w dw.  ∂L/∂x and ∂L/∂w are obtained
   by backward computation. This is essentially checking if
   the gradients obtained by backward computation correctly approximates
   the diff of the output.
*/

template<typename T, typename I, typename O, typename C>
static double grad_check(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, C cfg, idx_t B) {
  /* make weight and transfer to gpu if working on gpu */
  T * w = new T();
  w->init(opt, lgr, rg, cfg);
  //w->make_dev();
  make_dev(w, opt.gpu_algo);
  //w->to_dev();
  to_dev(w, opt.gpu_algo);
  /* make w - dw/2 and w + dw/2 */
  T * w_minus = make_copy(w, opt.gpu_algo);
  T * w_plus = make_copy(w, opt.gpu_algo);
  /* make coefficients to make the single loss value */
  O * alpha = new O();
  alpha->init_uniform(B, rg, -1.0, 1.0);
  alpha->make_dev(opt.gpu_algo);
  alpha->to_dev();
  /* make input (x) */
  I * x = new I();
  x->init_uniform(B, rg, -1.0, 1.0);
  x->make_dev(opt.gpu_algo);
  x->to_dev();
  /* forward and backward */
  O& y = w->forward(*x);
  I& gx = w->backward(*alpha);
  /* ensure the gradient is back to host */
  to_host(w, opt.gpu_algo);
  
  /* make dx */
  real e = 1.0e-3;
  I * dx = new I();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  I * x_minus = new I(*x);
  I * x_plus  = new I(*x);
  /* make gpu shadow */
  x_minus->make_dev(opt.gpu_algo);
  x_plus->make_dev(opt.gpu_algo);
  /* update on the host and send the to gpu */
  x_minus->add_(-0.5, *dx);
  x_plus->add_(0.5, *dx);
  /* send them to gpu */
  x_minus->to_dev();
  x_plus->to_dev();
  
  /* set gw to a random vector */
  w_minus->rand_grad(rg, -e, e);
  w_plus->copy_grad(*w_minus);
  /* update weights using gw (update runs on gpu) */
  w_minus->add_grad(-0.5);   /* w -= gw/2 */
  w_plus->add_grad(0.5);     /* w += gw/2 */
  /* send them to gpu */
  to_dev(w_minus, opt.gpu_algo);
  to_dev(w_plus, opt.gpu_algo);
  /* make y(w-dw/2,x-dx/2), y(w+dw/2,x+dx/2) */
  O& y_minus = w_minus->forward(*x_minus);
  O& y_plus  = w_plus->forward(*x_plus);
  /* get the result back to host */
  to_host(w_minus, opt.gpu_algo);
  to_host(w_plus, opt.gpu_algo);

  /* get the single loss values */
  double L_minus = alpha->dot(y_minus);
  double L       = alpha->dot(y);
  double L_plus  = alpha->dot(y_plus);
  /* various inner products */
  double gx_gx = gx.dot(gx);                         /* ∂L/∂x・∂L/∂x */
  double dx_dx = dx->dot(*dx);                       /* dx・dx */
  double gx_dx = gx.dot(*dx);                        /* ∂L/∂x・dx */
  double gw_gw = w->grad_dot_grad(*w);             /* ∂L/∂w・∂L/∂w */
  double dw_dw = w_minus->grad_dot_grad(*w_minus); /* dw・dw */
  double gw_dw = w->grad_dot_grad(*w_minus);       /* ∂L/∂w・dw */
  
  double rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  del_dev(w, opt.gpu_algo);
  del_dev(w_minus, opt.gpu_algo);
  del_dev(w_plus, opt.gpu_algo);
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();
  
  delete w;
  delete w_minus;
  delete w_plus;
  delete alpha;
  delete x;
  delete dx;
  delete x_minus;
  delete x_plus;
  return rel_e;
}

template<typename T, typename I0, typename I1, typename O, typename C>
static double grad_check_loss(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, C cfg, idx_t B, idx_t nC) {
  /* make weight and transfer to gpu if working on gpu */
  T * w = new T();
  w->init(opt, lgr, rg, cfg);
  //w->make_dev();
  make_dev(w, opt.gpu_algo);
  //w->to_dev();
  to_dev(w, opt.gpu_algo);
  /* make w - dw/2 and w + dw/2 */
  T * w_minus = make_copy(w, opt.gpu_algo);
  T * w_plus = make_copy(w, opt.gpu_algo);
  /* make coefficients to make the single loss value */
  O * alpha = new O();
  alpha->init_uniform(B, rg, -1.0, 1.0);
  alpha->make_dev(opt.gpu_algo);
  alpha->to_dev();
  /* make input (x) */
  I0 * x = new I0();
  x->init_uniform(B, rg, -1.0, 1.0);
  x->make_dev(opt.gpu_algo);
  x->to_dev();
  /* make input (t) */
  I1 * t = new I1();
  t->make_dev(opt.gpu_algo);
  t->init_uniform(B, rg, 0, nC);
  t->to_dev();
  /* forward and backward */
  O& y = w->forward(*x, *t);
  I0& gx = w->backward(*alpha, *t);
  /* ensure the gradient is back to host */
  to_host(w, opt.gpu_algo);
  
  /* make dx */
  real e = 1.0e-3;
  I0 * dx = new I0();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  I0 * x_minus = new I0(*x);
  x_minus->make_dev(opt.gpu_algo);
  I0 * x_plus  = new I0(*x);
  x_plus->make_dev(opt.gpu_algo);
  /* update on the host and send the to gpu */
  x_minus->add_(-0.5, *dx);
  x_plus->add_(0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
  
  /* set gw to a random vector */
  w_minus->rand_grad(rg, -e, e);
  w_plus->copy_grad(*w_minus);
  /* send them to gpu */
  to_dev(w_minus, opt.gpu_algo);
  to_dev(w_plus, opt.gpu_algo);
  /* update weights using gw (update runs on gpu) */
  w_minus->add_grad(-0.5);   /* w -= gw/2 */
  w_plus->add_grad(0.5);     /* w += gw/2 */
  /* make y(w-dw/2,x-dx/2), y(w+dw/2,x+dx/2) */
  O& y_minus = w_minus->forward(*x_minus, *t);
  O& y_plus  = w_plus->forward(*x_plus, *t);
  /* get the result back to host */
  to_host(w_minus, opt.gpu_algo);
  to_host(w_plus, opt.gpu_algo);

  /* get the single loss values */
  double L_minus = alpha->dot(y_minus);
  double L       = alpha->dot(y);
  double L_plus  = alpha->dot(y_plus);
  /* various inner products */
  double gx_gx = gx.dot(gx);                         /* ∂L/∂x・∂L/∂x */
  double dx_dx = dx->dot(*dx);                       /* dx・dx */
  double gx_dx = gx.dot(*dx);                        /* ∂L/∂x・dx */
  double gw_gw = w->grad_dot_grad(*w);             /* ∂L/∂w・∂L/∂w */
  double dw_dw = w_minus->grad_dot_grad(*w_minus); /* dw・dw */
  double gw_dw = w->grad_dot_grad(*w_minus);       /* ∂L/∂w・dw */
  
  double rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  del_dev(w, opt.gpu_algo);
  del_dev(w_minus, opt.gpu_algo);
  del_dev(w_plus, opt.gpu_algo);
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();
  
  delete w;
  delete w_minus;
  delete w_plus;
  delete alpha;
  delete x;
  delete dx;
  delete x_minus;
  delete x_plus;
  return rel_e;
}

