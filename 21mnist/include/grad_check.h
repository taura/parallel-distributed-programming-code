/**
   @file grad_check.h
   @brief gradient check utility
   @author Kenjiro Taura
   @date Jan. 3, 2023
 */

#pragma once

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
  /* make input (x) */
  I * x = new I();
  x->init_uniform(B, rg, -1.0, 1.0);

  /* make coefficients for backward */
  O * alpha = new O();
  alpha->init_uniform(B, rg, -1.0, 1.0);
  
  real e = 1.0e-3;

  /* make x - dx/2 and x + dx/2 */
  /* dx = random vector */
  I * dx = new I();
  dx->init_uniform(B, rg, -e, e);
  I * x_minus = make_copy(x, opt.gpu_algo);
  I * x_plus  = make_copy(x, opt.gpu_algo);
  x_minus->add_(-0.5, *dx);     /* x+ = x - dx/2 */
  x_plus->add_(0.5, *dx);       /* x- = x + dx/2 */
  
  /* make w - dw/2 and w + dw/2 */
  T * w_minus = make_copy(w, opt.gpu_algo);
  T * w_plus = make_copy(w, opt.gpu_algo);
  /* set dw to a random vector */
  w_minus->rand_grad(rg, -e, e);
  w_plus->copy_grad(*w_minus);
  w_minus->add_grad(-0.5);   /* w- = w - dw/2 */
  w_plus->add_grad(0.5);     /* w+ = w + dw/2 */

  /* make gpu shadow and send data to gpu */
  to_dev(alpha, opt.gpu_algo);
  to_dev(x, opt.gpu_algo);
  to_dev(x_minus, opt.gpu_algo);
  to_dev(x_plus, opt.gpu_algo);
  to_dev(w, opt.gpu_algo);
  to_dev(w_minus, opt.gpu_algo);
  to_dev(w_plus, opt.gpu_algo);
  
  /* forward and backward */
  O& y = w->forward(*x, 1);
  I& gx = w->backward(*alpha);
  /* make y(w-dw/2,x-dx/2), y(w+dw/2,x+dx/2) */
  O& y_minus = w_minus->forward(*x_minus, 1);
  O& y_plus  = w_plus->forward(*x_plus, 1);
  /* get the results back to host */
  to_host(w, opt.gpu_algo);
  to_host(w_minus, opt.gpu_algo);
  to_host(w_plus, opt.gpu_algo);

  /* get the single loss values */
  double L_minus = alpha->dot(y_minus);
  double L       = alpha->dot(y);
  double L_plus  = alpha->dot(y_plus);
  double gx_dx = gx.dot(*dx);                /* ∂L/∂x・dx */
  double gw_dw = w->grad_dot_grad(*w_minus); /* ∂L/∂w・dw */
  
  double rel_e = show_error(gx_dx, gw_dw, L_minus, L, L_plus);
  /* clean up */
  del_dev(w, opt.gpu_algo);
  del_dev(w_minus, opt.gpu_algo);
  del_dev(w_plus, opt.gpu_algo);
  del_dev(alpha, opt.gpu_algo);
  del_dev(x, opt.gpu_algo);
  del_dev(dx, opt.gpu_algo);
  del_dev(x_minus, opt.gpu_algo);
  del_dev(x_plus, opt.gpu_algo);
  
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
  /* make input (x and t) */
  I0 * x = new I0();
  x->init_uniform(B, rg, -1.0, 1.0);
  I1 * t = new I1();
  t->init_uniform_i(B, rg, 0, nC);
  /* make coefficients for backward */
  O * alpha = new O();
  alpha->init_uniform(B, rg, -1.0, 1.0);
  
  real e = 1.0e-3;
  /* make x - dx/2 and x + dx/2 */
  /* dx = random vector */
  I0 * dx = new I0();
  dx->init_uniform(B, rg, -e, e);
  I0 * x_minus = make_copy(x, opt.gpu_algo);
  I0 * x_plus  = make_copy(x, opt.gpu_algo);
  x_minus->add_(-0.5, *dx);     /* x+ = x - dx/2 */
  x_plus->add_(0.5, *dx);       /* x- = x + dx/2 */
  
  /* make w - dw/2 and w + dw/2 */
  T * w_minus = make_copy(w, opt.gpu_algo);
  T * w_plus = make_copy(w, opt.gpu_algo);
  /* set dw to a random vector */
  w_minus->rand_grad(rg, -e, e);
  w_plus->copy_grad(*w_minus);
  w_minus->add_grad(-0.5);   /* w- = w - dw/2 */
  w_plus->add_grad(0.5);     /* w+ = w + dw/2 */

  /* make gpu shadow and send data to gpu */
  to_dev(alpha, opt.gpu_algo);
  to_dev(x, opt.gpu_algo);
  to_dev(x_minus, opt.gpu_algo);
  to_dev(x_plus, opt.gpu_algo);
  to_dev(t, opt.gpu_algo);
  to_dev(w, opt.gpu_algo);
  to_dev(w_minus, opt.gpu_algo);
  to_dev(w_plus, opt.gpu_algo);
  
  /* forward and backward */
  O& y = w->forward(*x, *t, 1);
  I0& gx = w->backward(*alpha, *t);
  /* make y(w-dw/2,x-dx/2), y(w+dw/2,x+dx/2) */
  O& y_minus = w_minus->forward(*x_minus, *t, 1);
  O& y_plus  = w_plus->forward(*x_plus, *t, 1);
  /* get the results back to host */
  to_host(w, opt.gpu_algo);
  to_host(w_minus, opt.gpu_algo);
  to_host(w_plus, opt.gpu_algo);

  /* get the single loss values */
  double L_minus = alpha->dot(y_minus);
  double L       = alpha->dot(y);
  double L_plus  = alpha->dot(y_plus);
  double gx_dx = gx.dot(*dx);                /* ∂L/∂x・dx */
  double gw_dw = w->grad_dot_grad(*w_minus); /* ∂L/∂w・dw */
  
  double rel_e = show_error(gx_dx, gw_dw, L_minus, L, L_plus);
  /* clean up */
  del_dev(alpha, opt.gpu_algo);
  del_dev(x, opt.gpu_algo);
  del_dev(x_minus, opt.gpu_algo);
  del_dev(x_plus, opt.gpu_algo);
  del_dev(w, opt.gpu_algo);
  del_dev(w_minus, opt.gpu_algo);
  del_dev(w_plus, opt.gpu_algo);
  del_dev(t, opt.gpu_algo);
  del_dev(dx, opt.gpu_algo);
  
  delete alpha;
  delete x;
  delete x_minus;
  delete x_plus;
  delete w;
  delete w_minus;
  delete w_plus;
  delete t;
  delete dx;

  return rel_e;
}

