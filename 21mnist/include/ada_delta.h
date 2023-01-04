/**
   @file ada_delta.h
   @brief AdaDelta optimizer
 */
#pragma once

#include "mnist_util.h"
#include "tensor.h"

template<idx_t N0,idx_t N1=1,idx_t N2=1,idx_t N3=1>
struct AdaDelta {
#if __CUDACC__
  AdaDelta<N0,N1,N2,N3> * dev; /**< device shadow */
#endif
  tensor<real,N0,N1,N2,N3> v;
  tensor<real,N0,N1,N2,N3> u;
  tensor<real,N0,N1,N2,N3> std;
  tensor<real,N0,N1,N2,N3> dx;
  real lr;
  real rho;
  real eps;
  void init(real lr_, real rho_=0.9, real eps_=1.0e-6) {
    lr = lr_;
    rho = rho_;
    eps = eps_;
    u.init_const(N0, 0.0);
    v.init_const(N0, 0.0);
    std.set_n0(N0);
    dx.set_n0(N0);
  }
  /**
     @brief set the device pointer for this and all subobjects
     @param (dev) a device memory or null
     @sa make_dev
     @sa del_dev
     @details if dev is not null, dev fields of all subojects 
     point to the corresponding subjects in the device memory.
     if dev is not null, all dev fields become null.
  */
  void set_dev(AdaDelta<N0,N1,N2,N3>* dev) {
#if __CUDACC__
    this->dev = dev;
#else
    (void)dev;
#endif
  }
  __device__ __host__
  void update(tensor<real,N0,N1,N2,N3>& w, tensor<real,N0,N1,N2,N3>& gw) {
    assert(w.n0 == N0);
    v.mul_(rho).addcmul_(1 - rho, gw, gw);     //  v(t) = ρv(t-1) + (1-ρ)g(t)^2
    v.add(eps, std).sqrt_();                   //   std = √v(t)+ε
    u.add(eps, dx).sqrt_().div_(std).mul_(gw); //   Δx = (√u(t)+ε) /(√v(t)+ε) g(t)
    u.mul_(rho).addcmul_(1 - rho, dx, dx);     //  u(t) = ρu(t-1) + (1-ρ)Δx^2
    w.add_(-lr, dx);                           // θ(t) = θ(t-1) - γΔx
  }
};

int ada_delta_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  if (opt.error || opt.help) usage(argv[0]);
  const idx_t N0 = 2;
  tensor<real,N0> x;
  tensor<real,N0> gx;
  AdaDelta<N0> ada_delta;
  x.init_const(N0, 10.0);
  gx.init_const(N0, 0.0);
  ada_delta.init(opt.lr);
  for (int i = 0; i < 100; i++) {
    real a = 10.0;
    real b = 1.0;
    real f = (x(0)/a) * (x(0)/a) + (x(1)/b) * (x(1)/b);
    gx(0) = 2 * x(0) / (a * a);
    gx(1) = 2 * x(1) / (b * b);
    printf("x = (%f,%f), f(x) = %f\n", x(0), x(1), f);
    ada_delta.update(x, gx);
  }
  return 0;
}
