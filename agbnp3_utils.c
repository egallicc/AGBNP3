/* -------------------------------------------------------------------------- *
 *                                   AGBNP3                                   *
 * -------------------------------------------------------------------------- *
 * This file is part of the AGBNP3 implicit solvent model software            *
 * implementation funded by the National Science Foundation under grant:      *
 * NSF SI2 1440665  "SI2-SSE: High-Performance Software for Large-Scale       *
 * Modeling of Binding Equilibria"                                            *
 *                                                                            *
 * copyright (c) 2014-2015 Emilio Gallicchio                                  *
 * Authors: Emilio Gallicchio <egallicchio@brooklyn.cuny.edu>                 *
 * Contributors:                                                              *
 *                                                                            *
 *  AGBNP3 is free software: you can redistribute it and/or modify            *
 *  it under the terms of the GNU Lesser General Public License version 3     *
 *  as published by the Free Software Foundation.                             *
 *                                                                            *
 *  AGBNP3 is distributed in the hope that it will be useful,                 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of            *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
 *  GNU General Public License for more details.                              *
 *                                                                            *
 *  You should have received a copy of the GNU General Public License         *
 *  along with AGBNP3.  If not, see <http://www.gnu.org/licenses/>.           *
 *                                                                            *
 * -------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <float.h>
#ifdef __MINGW32__
#include <malloc.h>
#endif

#include "agbnp3_private.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void agbnp3_errprint(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
#pragma omp critical
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

/*                                                                      *
 * Memory management for arrays. Aligns if using SIMD parallelization.. *
 *                                                                      *
 * Memory allocated by these routines must be free'd by agbnp3_vfree()  *
 *                                                                      */
void agbnp3_vfree(void *x){
#ifdef __MINGW32__
  if(x) __mingw_aligned_free(x);
#else
  if(x) free(x);
#endif
}
 /* Memory allocator for arrays. */
int agbnp3_vmemalloc(void **memptr, const size_t size){
#ifdef INTEL_MIC
  size_t alignment = 64;
#else
  size_t alignment = 16;
#endif
  int retcode = 0;
  void *mem;
  size_t padded_size;
  //make size a multiple of alignment
  padded_size = (size/alignment + 1)*alignment;
#ifdef __MINGW32__
  //  mem = malloc(padded_size);
  mem = __mingw_aligned_malloc(padded_size, alignment);
#else
  retcode =  posix_memalign(&mem, alignment, padded_size);
  if(retcode) mem = NULL;
#endif
  *memptr = mem;
  return retcode;
}
/* calloc() equivalent */
int agbnp3_vcalloc(void **memptr, const size_t size){
  int retcode;
  void *mem;
  retcode = agbnp3_vmemalloc(&mem, size);
  if(retcode==0 && mem){
    memset(mem,0,size);
  }
  *memptr = mem;
  return retcode;
}
/* realloc equivalent */
int agbnp3_vrealloc(void **memptr, const size_t old_size, const size_t new_size){
  int retcode, size;
  void *mem;
  retcode = agbnp3_vmemalloc(&mem, new_size);
  if(*memptr && retcode==0 && mem){
    size = ( new_size > old_size ) ? old_size : new_size;
    memcpy(mem,*memptr,size);
    agbnp3_vfree(*memptr);
  }
  *memptr = mem;
  return retcode;
}


int agbnp3_create_ctablef42d(AGBNPdata *agb,
			     int na, float_a amax, 
			    int nb, float_a bmax, 
			    C1Table2DL **table2d){
  C1Table2DL *tbl2d;
  float_a b, db;

  int i;
  float_a bmax1;


  agbnp3_create_ctablef42d_list(agb, na, amax, &tbl2d);
  *table2d = tbl2d;

  //agbnp3_test_create_ctablef42d_hash(agb, 0.f, tbl2d);

  return AGBNP_OK;
}

int agbnp3_interpolate_ctablef42d(C1Table2DH *table2d, float_a x, float_a y,
				 float_a *f, float_a *fp){
  int iy, key;
  float_a y2;
  float_a dy, dyinv;
  float_a a, b;
  C1Table *table1;
  int nkey = table2d->nkey;

  key = y * nkey;
  iy = agbnp3_h_find(table2d->y2i, key);
  table1 = table2d->table[iy];
  agbnp3_interpolate_ctable(table1, x, f, fp);
  return AGBNP_OK;
}


/* initializes i4p(), the lookup table version of i4 */
int agbnp3_init_i4p(AGBNPdata *agb){
  if(agb->f4c1table2dl != NULL){
    return AGBNP_OK;
  }
  if(agbnp3_create_ctablef42d(agb, 
			      F4LOOKUP_NA, F4LOOKUP_MAXA,
			      F4LOOKUP_NB, F4LOOKUP_MAXB,
			      &(agb->f4c1table2dl)) != AGBNP_OK){
    agbnp3_errprint("agbnp3_init_i4p(): error in agbnp3_create_ctablef42d()\n");
    return AGBNP_ERR;
  }
  return AGBNP_OK;
}

float_a agbnp3_i4p(AGBNPdata *agb, float_a rij, float_a Ri, float_a Rj, 
	    float_a *dr){
  float_a a, b, rj1, f, fp, u, ainv, ainv2;
  static const float_a pf = (4.*pi)/3.;

  a = rij/Rj;
  b = Ri/Rj;
  agbnp3_interpolate_ctablef42d(agb->f4c1table2dh, a, b, &f, &fp);
  *dr = fp/(Rj*Rj);

  return f/Rj;
}

/* cubic spline setup */
/*
This code is based on the cubic spline interpolation code presented in:
Numerical Recipes in C: The Art of Scientific Computing
by
William H. Press,
Brian P. Flannery,
Saul A. Teukolsky, and
William T. Vetterling .
Copyright 1988 (and 1992 for the 2nd edition)

I am assuming zero-offset arrays instead of the unit-offset arrays
suggested by the authors.  You may style me rebel or conformist
depending on your point of view.

Norman Kuring	31-Mar-1999

*/
void agbnp3_cspline_setup(float dx, int n, float* y, 
			 float yp1, float ypn, 
			 float* y2){
  float* u;
  int	i,k;
  float	p,qn,sig,un;

  u = malloc(n*sizeof(float));
    
  if(yp1 > 0.99e30)
    y2[0] = u[0] = 0.0;
  else{
    y2[0] = -0.5;
    u[0] = (3.0/(dx))*((y[1]-y[0])/(dx)-yp1);
  }
  for(i = 1; i < n-1; i++){
    sig = 0.5;
    p = sig*y2[i-1] + 2.0;
    y2[i] = (sig - 1.0)/p;
    u[i] = (y[i+1] - y[i])/(dx) - (y[i] - y[i-1])/(dx);
    u[i] = (6.0*u[i]/(2.*dx) - sig*u[i-1])/p;
  }
  if(ypn > 0.99e30)
    qn = un = 0.0;
  else{
    qn = 0.5;
    un = (3.0/(dx))*(ypn - (y[n-1] - y[n-2])/(2.*dx));
  }
  y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2] + 1.0);
  for(k = n-2; k >= 0; k--){
    y2[k] = y2[k]*y2[k+1] + u[k];
  }

  free(u);
}

void agbnp3_cspline_interpolate(float x, float dx, int n, float* y, float* y2,
				float *f, float *fp){
  int k;
  float xh, b, a, a2, b2, kf, dp1, dp2, dxinv = 1./dx;
  
  dp1 = dx/6.0;
  dp2 = dx*dp1;

  xh = x*dxinv;
  k = xh;
  kf=k;
  if(k > n-2){
    *f = 0.0;
    *fp = 0.0;
    return;
  }
  a = (kf+1.0)-xh;
  a2 = a*a;
  b = xh - kf;
  b2 = b*b;
  *f = a*y[k]+b*y[k+1] +  ((a2*a - a)*y2[k] + (b2*b - b)*y2[k+1])*dp2;
  *fp = (y[k+1] - y[k])*dxinv - 
    ((3.*a2 - 1.)*y2[k] + (3.*b2 - 1.)*y2[k+1])*dp1;
}

/* prepares input for agbnp3_cspline_interpolate_soa() given  


/* applies cspline interpolation to a series of data:
k[]: table look up index
xh[]: x/dx
dx: spacing
m: number of points
yp[], y[]: node ordinates at k+1 and k
y2p[], y2[]: node coefficients at k+1 and k
f[], fp[]: output function values and derivatives
*/
void agbnp3_cspline_interpolate_soa(float *kv, float *xh, float dx, int m, 
				    float* yp, float *y,
				    float* y2p, float *y2,
				    float *f, float *fp){
  int i;
  float b, a, a2, b2, kf, dp1, dp2, yk, ypk, y2k, y2pk, dxinv;
  
  dp1 = dx/6.0f;
  dp2 = dx*dp1;
  dxinv = 1.f/dx;

#pragma vector aligned
#pragma ivdep
  for(i=0;i<m;i++){
    kf=kv[i];
    yk = y[i];
    ypk = yp[i];
    y2k = y2[i];
    y2pk = y2p[i];

    a = (kf+1.0f)-xh[i];
    a2 = a*a;
    b = xh[i] - kf;
    b2 = b*b;

    f[i] = a*yk+b*ypk +  ((a2*a - a)*y2k + (b2*b - b)*y2pk)*dp2;
    fp[i] = (ypk - yk)*dxinv - 
      ((3.0f*a2 - 1.f)*y2k + (3.0f*b2 - 1.f)*y2pk)*dp1;
  }
}

int agbnp3_i4p_soa(AGBNPdata *agb, float* rij, float *Ri, float *Rj, int *btype,
		   int m, float *f, float *fp,
		   float *mbuffera, float *mbufferb,
		   float *qkv, float *qxh, float *qyp, float *qy, float *qy2p, float *qy2,
		   float *qf1, float *qf2, float *qfp1, float *qfp2){
  int i;
  float_a rj1, u, ainv, ainv2;
  float *a = mbuffera;
  float *b = mbufferb;

#pragma vector aligned
#pragma ivdep
  for(i=0;i<m;i++){
    a[i] = rij[i]/Rj[i];
    b[i] = Ri[i]/Rj[i];
    //printf("i4p: %d %f %f\n", i, a[i], b[i]);
  }

  agbnp3_interpolate_ctablef42d_soa(agb->f4c1table2dl, a, b, btype, m, f, fp,
	               qkv, qxh, qyp, qy, qy2p, qy2, qf1, qf2, qfp1, qfp2);

#pragma vector aligned
#pragma ivdep
  for(i=0;i<m;i++){
    f[i] = f[i]/Rj[i];
    fp[i] = fp[i]/(Rj[i]*Rj[i]);
  }

  return AGBNP_OK;
}



/* vectorized form of  agbnp3_interpolate_ctablef42d() */
int agbnp3_interpolate_ctablef42d_soa
(C1Table2DL *table2d, float *x, float *ym, int *btype, int m, float *f, float *fp,
 float *kv, float *xh, float *yp, float *y, float *y2p, float *y2,
 float *f1, float *f2, float *fp1, float *fp2){

  int i, iy, k, slot;
  float dy, dyinv, dx, dxinv, yn;
  float a, b;
  C1Table *table1, *table2;

  for(i=0;i<m;i++){
    
    table1 = table2d->table[btype[i]];

    dx = table1->dx;
    dxinv = table1->dxinv;
    
    xh[i] = x[i]*dxinv;
    k = xh[i];
    kv[i] = k;
    if(k > table1->n-2){
      y[i] = 0.0;
      yp[i] = 0.0;
      y2[i] = 0.0;
      y2p[i] = 0.0;
    }else{
      y[i] = table1->y[k];
      yp[i] = table1->y[k+1];
      y2[i] = table1->y2[k];
      y2p[i] = table1->y2[k+1];
    }
  }

#ifdef USE_SSE
  agbnp3_cspline_interpolate_ps(kv, xh, dx, m, 
				 yp, y,
				 y2p, y2,
				 f, fp);
#else
  agbnp3_cspline_interpolate_soa(kv, xh, dx, m, 
				 yp, y,
				 y2p, y2,
				 f, fp);
#endif

  return AGBNP_OK;
}


int agbnp3_create_ctablef4(int n, float_a amax, float_a b, 
			  C1Table **c1table){
  C1Table *tbl;
  int i;
  float_a da = amax/(n-1);
  float_a a, u1, q1, dr, Rj=1.0, fp, fpp, s;

  float_a *y, *y2, yp1, ypn = 0.0;
  float_a yinf=0.0;

  agbnp3_vmemalloc((void **)&(y), n*sizeof(float));
  agbnp3_vmemalloc((void **)&(y2), n*sizeof(float));
  if(!(y && y2)){
    agbnp3_errprint( "agbnp3_create_ctablef4(): unable to allocate work buffers (%d floats)\n", 3*n);
    return AGBNP_ERR;
  }

  tbl = malloc(sizeof(C1Table));
  if(!tbl){
    agbnp3_errprint( "agbnp3_create_ctablef4(): unable to allocate table structure.\n");
    return AGBNP_ERR;
  }
  tbl->n = n;
  tbl->dx = da;
  tbl->dxinv = 1./da;
  tbl->yinf = yinf;

  a = 0.0;
  for(i=0;i<n-1;i++){
    q1 = agbnp3_i4ov(a,b,Rj,&dr);
    if(i==0) yp1 = dr;
    y[i] = q1;
    a += da;
  }
  y[n-1] = yinf;

  agbnp3_cspline_setup(da, n, y, yp1, ypn, y2);
  tbl->y = y;
  tbl->y2 = y2;

  *c1table = tbl;
  return AGBNP_OK;
}


int agbnp3_interpolate_ctable(C1Table *c1table, float_a x, 
			     float_a *f, float_a *fp){
  int n = c1table->n;
  float dx = c1table->dx;
  float *y = c1table->y;
  float *y2 = c1table->y2;
 
  agbnp3_cspline_interpolate(x, dx, n, y, y2, f, fp);

  return AGBNP_OK;
}



void agbnp3_test_cspline(void){
  float x, dx = 0.2;
  int n = 100;
  float dxi = 0.02;
  int ni = 1000;
  float *y = malloc(n*sizeof(float));
  float *y2 = malloc(n*sizeof(float));
  float yp1, ypn = 0.0;
  float f, fp;
  int i;
  float Rj = 2.0, Ri = 1.0, dq;
  float q;
  float fold, xold;

  x = 0.0;
  for(i=0;i<n;i++){
    y[i] = agbnp3_i4(x, Ri, Rj, &dq);
    if(i==0) yp1 = dq;
    x += dx;
  }
  //  yp1 = 3.0;
  agbnp3_cspline_setup(dx, n, y, yp1, ypn, y2);

  x = 0.0;
  for(i=0;i<ni;i++){
    agbnp3_cspline_interpolate(x, dx, n, y, y2, &f, &fp);
    //q = agbnp3_i4(x, Ri, Rj, &dq);
    if(i>0){
      printf("csp: %f %f %f %f\n", x, f, f-fold, fp*dxi);
    }
    fold = f;
    xold = x;
    x += dxi;
  }
  free(y);
  free(y2);
}

int agbnp3_reset_buffers(AGBNPdata *agb, AGBworkdata *agbw_h){
  
  int natoms = agb->natoms;
  int nheavyat = agb->nheavyat;
  AGBworkdata *agbw = agb->agbw;
  float *r = agb->r;
  float *vols = agbw_h->vols;
  int i, iat;
  float cvdw = AGBNP_RADIUS_INCREMENT;

#ifdef _OPENMP
  memset(agbw_h->volumep,0,natoms*sizeof(float));
  memset(agbw_h->surf_area,0,natoms*sizeof(float));
  memset(agbw_h->br1,0,natoms*sizeof(float));
#endif

#pragma omp single nowait
  {
    memset(agbw->volumep,0,natoms*sizeof(float));
    for(iat=0;iat<nheavyat;iat++){
      agbw->volumep[iat] = vols[iat];
    }
  }
#pragma omp single nowait
  {
    memset(agbw->surf_area,0,natoms*sizeof(float));
    for(iat=0;iat<nheavyat;iat++){
      agbw->surf_area[iat] = 4.*pi*r[iat]*r[iat];
    }
  }
#pragma omp single nowait
  {
    for(iat=0;iat<natoms;iat++){
      agbw->br1[iat] = 1./(r[iat]-cvdw);
    }
  }
#pragma omp single
  {
    agb->ehb = 0.0;
  }

  return AGBNP_OK;
}


/* computes volume scaling factors, also adds surface area corrections to
   self volumes. */
 int agbnp3_scaling_factors(AGBNPdata *agb, AGBworkdata *agbw_h){
  int i,iat;
  int natoms = agb->natoms;
  int nheavyat = agb->nheavyat;
  float_a *r = agb->r;
  AGBworkdata *agbw = agb->agbw;
  float_a *volumep = agbw->volumep;
  float_a *vols = agbw->vols;
  float_a *sp = agbw->sp;
  float_a *spe = agbw->spe;
  float_a *surf_area = agbw->surf_area;
  float_a *surf_area_f = agbw->surf_area_f;

  float_a *volumep_h = agbw_h->volumep;
  float_a *surf_area_h = agbw_h->surf_area;
  float_a *sp_h = agbw_h->sp;
  float_a *spe_h = agbw_h->spe;
  float_a *psvol_h = agbw_h->psvol;
  float_a *vols_h = agbw_h->vols;
  float_a Rw = AGBNP_RADIUS_INCREMENT;
  float_a rvdw, us, pr;
  float_a a, f, fp;

#ifdef _OPENMP
  // threads contributions to master
#pragma omp critical
  for(iat=0;iat<nheavyat;iat++){
      volumep[iat] +=  volumep_h[iat];
  }
#pragma omp critical
  for(iat=0;iat<nheavyat;iat++){
    surf_area[iat] +=  surf_area_h[iat];
  }
#pragma omp barrier
#endif

#pragma omp single
  {
    /* filters surface areas to avoid negative surface areas */
    memset(agb->surf_area,0,natoms*sizeof(float_i));
    memset(agb->agbw->surf_area_f,0,natoms*sizeof(float_a));
    for(iat=0;iat<nheavyat;iat++){
      a = agbw->surf_area[iat];
      f = agbnp3_swf_area(a, &fp);
      agb->surf_area[iat] = agbw->surf_area[iat]*f;
      surf_area_f[iat] = agb->surf_area[iat];
      agbw->gammap[iat] = agbw->gamma[iat]*(f+a*fp);
    }
  }

  /* compute scaled volume factors for enlarged atomic radii, 
     that is before subtracting subtended surface area */
  for(iat=0;iat<nheavyat;iat++){
    spe_h[iat] = volumep[iat]/vols_h[iat];
  }

  /* subtract volume subtended by surface area:
     V = A (1/3) R2 [ 1 - (R1/R2)^3 ] 
     where R2 is the enlarged radius and R1 the vdw radius
  */
  for(iat=0;iat<nheavyat;iat++){
    rvdw = agb->r[iat] - Rw;
    a = surf_area[iat];
    f = agbnp3_swf_area(a, &fp);
    pr = r[iat]*(1. - pow(rvdw/r[iat],3))/3.;
    us = 1.0*pr; //0.8
    psvol_h[iat] = (fp*a+f)*us; /* needed to compute effective gamma's in
				       agbnp3_gb_deruv() */
    volumep_h[iat] = volumep[iat] - surf_area_f[iat]*us;
  }

  /* compute scaled volume factors */
  for(iat=0;iat<nheavyat;iat++){
    sp_h[iat] = volumep_h[iat]/vols_h[iat];
  }

#ifdef _OPENMP
  //sync master copies
  // the barrier is needed because volumep is used above by lagging threads
#pragma omp barrier
#pragma omp single
  {
    memcpy(volumep,  volumep_h,nheavyat*sizeof(float));
    memcpy(spe,      spe_h,    nheavyat*sizeof(float));
    memcpy(sp,       sp_h,     nheavyat*sizeof(float));
  }
#endif

  return AGBNP_OK;
}

 int agbnp3_born_radii(AGBNPdata *agb, AGBworkdata *agbw_h){
  int natoms = agb->natoms;
  int iat;
  float_a fp;
  float_a *r = agb->r;
  AGBworkdata *agbw = agb->agbw;
  float_a *br = agbw->br;
  float_a *br1 = agbw->br1;
  float_a *br1_swf_der = agbw->br1_swf_der;
  float_a *brw = agbw->brw;
  float_a *brw_h = agbw_h->brw;
  float_a *br1_swf_der_h = agbw_h->br1_swf_der;
  float_a *br1_h = agbw_h->br1;
  float_a *br_h = agbw_h->br;
  float_a rw = 1.4;  /* water radius offset for np 
				    energy function */
  float_a _agbnp3_brw1, _agbnp3_brw2, _agbnp3_brw3; 
  float_a cvdw = AGBNP_RADIUS_INCREMENT;
  float_a biat;

#ifdef _OPENMP
  // add thread contributions to master copy
#pragma omp critical
  for(iat=0;iat<natoms;iat++){
    br1[iat] += br1_h[iat];
  }
#pragma omp barrier
#endif

  // now all threads compute born radii etc from master copy
  for(iat = 0; iat < natoms ; iat++){
    /* filters out very large or, worse, negative born radii */
    br1_h[iat] = agbnp3_swf_invbr(br1[iat], &fp);
    /* save derivative of filter function for later use */
    br1_swf_der_h[iat] = fp;
    /* calculates born radius from inverse born radius */
    br_h[iat] = 1./br1_h[iat];
    biat = br_h[iat];
    brw_h[iat] = AGBNP_BRW(biat,rw); /* 3*b^2/(b+rw)^4 for np derivative */
  }

#ifdef _OPENMP
  //barrier is needed because br1 master is used above by lagging threads
#pragma omp barrier
#pragma omp single nowait
  { memcpy(br1,         br1_h,       natoms*sizeof(float)); }
#pragma omp single nowait
  {  memcpy(br,          br_h,        natoms*sizeof(float)); }
#pragma omp single nowait
  {  memcpy(br1_swf_der, br1_swf_der_h, natoms*sizeof(float)); }
#pragma omp single
  {  memcpy(brw,         brw_h,       natoms*sizeof(float)); }
#endif

  return AGBNP_OK;
}


 int agbnp3_reset_derivatives(AGBNPdata *agb, AGBworkdata *agbw_h){
  int natoms = agb->natoms;

#ifdef _OPENMP
  memset(agbw_h->dgbdr_h,0,3*natoms*sizeof(float_a));
  memset(agbw_h->dvwdr_h,0,3*natoms*sizeof(float_a));
  memset(agbw_h->decav_h,0,3*natoms*sizeof(float_a));
  memset(agbw_h->dehb,0,3*natoms*sizeof(float_a));
  memset(agbw_h->dera,0,natoms*sizeof(float_a));
  memset(agbw_h->deru,0,natoms*sizeof(float_a));
  memset(agbw_h->derv,0,natoms*sizeof(float_a));
  memset(agbw_h->derus,0,natoms*sizeof(float_a));
  memset(agbw_h->dervs,0,natoms*sizeof(float_a));
  memset(agbw_h->derh,0,natoms*sizeof(float_a));
#endif

#pragma omp single nowait
  {memset(agb->agbw->dgbdr_h,0,3*natoms*sizeof(float_a)); }
#pragma omp single nowait
  {memset(agb->agbw->dvwdr_h,0,3*natoms*sizeof(float_a));}
#pragma omp single nowait
  {memset(agb->agbw->decav_h,0,3*natoms*sizeof(float_a));}
#pragma omp single nowait
  {memset(agb->agbw->dehb,0,3*natoms*sizeof(float_a));}
#pragma omp single nowait
  {memset(agb->agbw->dera,0,natoms*sizeof(float_a));}
#pragma omp single nowait
  {memset(agb->agbw->deru,0,natoms*sizeof(float_a));}
#pragma omp single nowait
  {memset(agb->agbw->derv,0,natoms*sizeof(float_a));}
#pragma omp single nowait
  {memset(agb->agbw->derus,0,natoms*sizeof(float_a));}
#pragma omp single nowait
  {memset(agb->agbw->dervs,0,natoms*sizeof(float_a));}
#pragma omp single
  {memset(agb->agbw->derh,0,natoms*sizeof(float_a));}

  return AGBNP_OK;
}

/* hash table functions */

unsigned int agbnp3_two2n_size(unsigned int m){
  /* returns smallest power of 2 larger than twice the input */
  unsigned int s = 1;
  unsigned int l = 2*m;
  if(m<=0) return 0;
  while(s<l){
    s = (s<<1);
  }
  return s;
}

HTable *agbnp3_h_create(int nat, int size, int jump){
  HTable *ht = NULL;

#pragma omp critical
  ht = (HTable *)calloc(1,sizeof(HTable));
  if(!ht) return ht;
  if(size <= 0) return ht;

  ht->hsize = size;
  ht->hmask = size - 1;
  ht->hjump = jump;
  ht->nat = nat;
#pragma omp critical
  ht->key =  (int *)malloc(size*sizeof(int));
  if(!ht->key){
    agbnp3_h_delete(ht);
    ht = NULL;
    return ht;
  }
  return ht;
}

void agbnp3_h_delete(HTable *ht){
  if(ht){
    if(ht->key) free(ht->key);
    free(ht);
  }
}

void agbnp3_h_init(HTable *ht){
  int i;
  int *key = ht->key;
  for(i=0;i<ht->hsize;i++){
    key[i] = -1;
  }
}

/* find a slot or return existing slot */
int agbnp3_h_enter(HTable *ht, unsigned int keyij){
  unsigned int hmask;
  unsigned int hjump;
  unsigned int k;
  int *key;
  if(!ht) return -1;
  hmask = ht->hmask;
  hjump = ht->hjump;
  key = ht->key;
  if(!key) return -1;
  k = (keyij & hmask);
  while(key[k] >= 0 && key[k] != keyij){
    k = ( (k + hjump) & hmask);
  }
  key[k] = keyij;
  return k;
}

/* return existing slot, or -1 if not found */
int agbnp3_h_find(HTable *ht, unsigned int keyij){
  unsigned int hmask;
  unsigned int hjump;
  unsigned int k;
  int *key;
  if(!ht) return -1;
  hmask = ht->hmask;
  hjump = ht->hjump;
  key = ht->key;
  if(!key) return -1;
  k = (keyij & hmask);
  while(key[k] >= 0 &&  key[k] !=  keyij){
    k = ( (k+hjump) & hmask);
  }
  if(key[k] < 0) return -1;
  return k;
}

/*                                                          *
 *    Functions to create/manage q4 look-up tables          *
 *                                                          */

/* get the number and list of "radius types"
   it is the responsibility of the caller to free radii array
 */
int agbnp3_list_radius_types(AGBNPdata *agb, float **radii){
  float *radius = (float *)malloc(agb->natoms*sizeof(float));
  int ntypes = 0;
  float riat;
  int i, iat, found;

  for(iat=0;iat<agb->natoms;iat++){
    
    riat = agb->r[iat];

    /* search list of radii */
    found = 0;
    for(i=0;i<ntypes;i++){
      if( fabs(riat-radius[i]) < FLT_MIN ){
	found = 1;
	agb->rtype[iat] = i;
	break;
      }
    }

    /* if not found add it, increment set size */
    if(!found){
      agb->rtype[iat] = ntypes;
      radius[ntypes++] = riat;
    }

  }

  *radii = radius;
  return ntypes;
}

int agbnp3_create_ctablef42d_list(AGBNPdata *agb, int na, float_a amax, 
				  C1Table2DL **table2d){
  C1Table2DL *tbl2d;
  float b;
  int size;
  float *radii;
  int nlookup;
  int i, j;
  int ntypes, key, slot;
  float c = AGBNP_RADIUS_INCREMENT;
  int nkey;

  tbl2d = malloc(sizeof(C1Table2DL));

  /* get list of radii */
  ntypes = agbnp3_list_radius_types(agb, &radii);
  agb->nrtype = ntypes;

  /* number of look-up tables is ~ntypes^2 */
  size = ntypes*ntypes;

  tbl2d->table = (C1Table **)malloc(size*sizeof(C1Table *));

  /* now loop over all possible combinations of radii and constructs look up table for each */
  for(i=0;i<ntypes;i++){
    for(j=0;j<ntypes;j++){

      b = (radii[i]-c)/radii[j];
      slot = i*ntypes + j; //index in list of tables

      //fprintf(stderr,"agbnp3_create_ctablef42d_hash(): creating table for radius ratio: %f (key %d)\n", b, slot);
      if(agbnp3_create_ctablef4(na,amax,b,&(tbl2d->table[slot]))!=AGBNP_OK){
	agbnp3_errprint( "agbnp3_create_ctablef42d_hash(): error in agbnp3_create_ctablef4()\n");
	free(radii);
	return AGBNP_ERR;
      }

    }
  }

  free(radii);
  *table2d = tbl2d;
  return AGBNP_OK;
}


/* for a random pair of atoms, print the Q4 function */
int agbnp3_test_create_ctablef42d_hash(AGBNPdata *agb, float amax, C1Table2DH *table2d){
  int iat = 271;
  int jat = 12;
  int key, slot;
  float b;
  C1Table *table;
  float c = AGBNP_RADIUS_INCREMENT;
  float x, f, fp;
  int nkey = table2d->nkey;

  for(iat=0;iat < agb->natoms;iat++){
    for(jat=iat+1;jat < agb->natoms ; jat++){

      b = (agb->r[iat]-c)/agb->r[jat];
      key = b * nkey;
      slot = agbnp3_h_find(table2d->y2i, key);
      if(slot < 0){
	agbnp3_errprint( "agbnp3_test_create_ctablef42d_hash(): unable to find entry for radii combination (%f,%f)\n", agb->r[iat], agb->r[jat]);
	return AGBNP_ERR;
      }
    }
  }
  
#ifdef NOTNOW
  table = table2d->table[slot];



  {
    int i;
    printf("IntHash: radius: %f %f %f\n", b, agb->r[iat], agb->r[jat]);
    for(i = 0; i < 100; i++){
      x = 0.1*(float)i + 0.0001;
      agbnp3_interpolate_ctable(table, x, &f, &fp);
      printf("IntHash: radius: %f %f\n",x, f);
    }
  }
#endif
  

  return AGBNP_OK;
}


int agbnp3_reallocate_gbuffers(AGBworkdata *agbw, int size){
  int i;
  int err;
  int old_size = agbw->gbuffer_size;
  size_t n = old_size*sizeof(float);
  size_t m = size*sizeof(float);

  agbnp3_vrealloc((void **)&(agbw->a1), n, m);
  agbnp3_vrealloc((void **)&(agbw->p1), n , m);
  agbnp3_vrealloc((void **)&(agbw->c1x), n, m);
  agbnp3_vrealloc((void **)&(agbw->c1y), n, m);
  agbnp3_vrealloc((void **)&(agbw->c1z), n, m);

  agbnp3_vrealloc((void **)&(agbw->a2), n, m);
  agbnp3_vrealloc((void **)&(agbw->p2), n, m);
  agbnp3_vrealloc((void **)&(agbw->c2x), n, m);
  agbnp3_vrealloc((void **)&(agbw->c2y), n, m);
  agbnp3_vrealloc((void **)&(agbw->c2z), n, m);

  agbnp3_vrealloc((void **)&(agbw->v3), n, m);
  agbnp3_vrealloc((void **)&(agbw->v3p), n, m);
  agbnp3_vrealloc((void **)&(agbw->fp3), n, m);
  agbnp3_vrealloc((void **)&(agbw->fpp3), n, m);

  if(!(agbw->a1 && agbw->p1 && agbw->c1x && agbw->c1y && agbw->c1z &&
       agbw->a2 && agbw->p2 && agbw->c2x && agbw->c2y && agbw->c2z &&
       agbw->v3 && agbw->v3p && agbw->fp3 && agbw->fpp3 )){
      agbnp3_errprint( "agbnp3_reallocate_gbuffers(): error allocating memory for Gaussian overlap buffers.\n");
      return AGBNP_ERR;
  }

  agbw->gbuffer_size = size;

  return AGBNP_OK;
}

int agbnp3_reallocate_hbuffers(AGBworkdata *agbw, int size){
  int i;
  int err;
  int old_size = agbw->hbuffer_size;
  size_t n = old_size*sizeof(float);
  size_t m = size*sizeof(float);

  agbnp3_vrealloc((void **)&(agbw->hiat), old_size*sizeof(int), size*sizeof(int));

  agbnp3_vrealloc((void **)&(agbw->ha1), n, m);
  agbnp3_vrealloc((void **)&(agbw->hp1), n , m);
  agbnp3_vrealloc((void **)&(agbw->hc1x), n, m);
  agbnp3_vrealloc((void **)&(agbw->hc1y), n, m);
  agbnp3_vrealloc((void **)&(agbw->hc1z), n, m);

  agbnp3_vrealloc((void **)&(agbw->ha2), n, m);
  agbnp3_vrealloc((void **)&(agbw->hp2), n, m);
  agbnp3_vrealloc((void **)&(agbw->hc2x), n, m);
  agbnp3_vrealloc((void **)&(agbw->hc2y), n, m);
  agbnp3_vrealloc((void **)&(agbw->hc2z), n, m);

  agbnp3_vrealloc((void **)&(agbw->hv3), n, m);
  agbnp3_vrealloc((void **)&(agbw->hv3p), n, m);
  agbnp3_vrealloc((void **)&(agbw->hfp3), n, m);
  agbnp3_vrealloc((void **)&(agbw->hfpp3), n, m);

  if(!(agbw->hiat && agbw->ha1 && agbw->hp1 && agbw->hc1x && agbw->hc1y && agbw->hc1z &&
       agbw->ha2 && agbw->hp2 && agbw->hc2x && agbw->hc2y && agbw->hc2z &&
       agbw->hv3 && agbw->hv3p && agbw->hfp3 && agbw->hfpp3 )){
      agbnp3_errprint( "agbnp3_reallocate_hbuffers(): error allocating memory for Gaussian overlap buffers.\n");
      return AGBNP_ERR;
  }

  agbw->hbuffer_size = size;

  return AGBNP_OK;
}

int agbnp3_reallocate_qbuffers(AGBworkdata *agbw, int size){
  int i;
  int err;
  int old_size = agbw->qbuffer_size;
  size_t n = old_size*sizeof(float);
  size_t m = size*sizeof(float);

  agbnp3_vrealloc((void **)&(agbw->qdv), n, m);
  agbnp3_vrealloc((void **)&(agbw->qR1v), n , m);
  agbnp3_vrealloc((void **)&(agbw->qR2v), n, m);

  agbnp3_vrealloc((void **)&(agbw->qbtype), old_size*sizeof(int), size*sizeof(int));

  agbnp3_vrealloc((void **)&(agbw->qqv), n, m);
  agbnp3_vrealloc((void **)&(agbw->qdqv), n, m);
  agbnp3_vrealloc((void **)&(agbw->qav), n, m);
  agbnp3_vrealloc((void **)&(agbw->qbv), n, m);

  agbnp3_vrealloc((void **)&(agbw->qkv), n, m);
  agbnp3_vrealloc((void **)&(agbw->qxh), n, m);
  agbnp3_vrealloc((void **)&(agbw->qyp), n, m);
  agbnp3_vrealloc((void **)&(agbw->qy), n, m);
  agbnp3_vrealloc((void **)&(agbw->qy2p), n, m);
  agbnp3_vrealloc((void **)&(agbw->qy2), n, m);
  agbnp3_vrealloc((void **)&(agbw->qf1), n, m);
  agbnp3_vrealloc((void **)&(agbw->qf2), n, m);
  agbnp3_vrealloc((void **)&(agbw->qfp1), n, m);
  agbnp3_vrealloc((void **)&(agbw->qfp2), n, m);


  if(!(agbw->qdv && agbw->qR1v && agbw->qR2v && agbw->qqv && agbw->qdqv && agbw->qav && agbw->qav)){
      agbnp3_errprint( "agbnp3_reallocate_qbuffers(): error allocating memory for inverse born radii buffers.\n");
      return AGBNP_ERR;
  }

  agbw->qbuffer_size = size;

  return AGBNP_OK;
}

int agbnp3_reallocate_wbuffers(AGBworkdata *agbw, int size){
  int i;
  int err;
  int old_size = agbw->wbuffer_size;
  size_t n = old_size*sizeof(float);
  size_t m = size*sizeof(float);
  
  agbnp3_vrealloc((void **)&(agbw->wb_iatom), old_size*sizeof(int), size*sizeof(int));

  agbnp3_vrealloc((void **)&(agbw->wb_gvolv), n, m);
  agbnp3_vrealloc((void **)&(agbw->wb_gderwx), n , m);
  agbnp3_vrealloc((void **)&(agbw->wb_gderwy), n, m);
  agbnp3_vrealloc((void **)&(agbw->wb_gderwz), n, m);
  agbnp3_vrealloc((void **)&(agbw->wb_gderix), n, m);
  agbnp3_vrealloc((void **)&(agbw->wb_gderiy), n, m);
  agbnp3_vrealloc((void **)&(agbw->wb_gderiz), n, m);

  if(!(agbw->wb_iatom && agbw->wb_gderwx && agbw->wb_gderwy && agbw->wb_gderwz &&
       agbw->wb_gderix &&  agbw->wb_gderiy && agbw->wb_gderiz)){ 
    agbnp3_errprint( "agbnp3_reallocate_wbuffers(): error allocating memory for water sites buffers.\n");
      return AGBNP_ERR;
  }

  agbw->wbuffer_size = size;

  return AGBNP_OK;
}


int agbnp3_reallocate_overlap_lists(AGBworkdata *agbw, int size){
  int i;

  for(i=0;i<2;i++){

    agbnp3_vrealloc((void **)&(agbw->overlap_lists[i]), 
		   agbw->size_overlap_lists[i]*sizeof(GOverlap),
		   size*sizeof(GOverlap));
    agbw->size_overlap_lists[i] = size;

    agbnp3_vrealloc((void **)&(agbw->root_lists[i]), 
		   agbw->size_overlap_lists[i]*sizeof(int),
		   size*sizeof(GOverlap));
    agbw->size_root_lists[i] = size;

    if(!(agbw->overlap_lists[i] && agbw->root_lists[i])){
      agbnp3_errprint( "agbnp3_reallocate_overlap_lists(): error allocating memory for overlap lists.\n");
      return AGBNP_ERR;
    }

  }

  return AGBNP_OK;
}
