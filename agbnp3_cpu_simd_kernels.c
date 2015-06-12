
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
#include <math.h>
#include <string.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "agbnp3.h"
#include "nblist.h"
#include "agbnp3_private.h"

#ifdef USE_SSE
#include <xmmintrin.h>
#include "sse_mathfun.h"
#endif

/* _PS_CONST(dielin4, dielin);
_PS_CONST(dielout4,dielout);
_PS_CONST(tokcalmol4, tokcalmol);
*/
#ifdef USE_SSE
_PS_CONST(six4, 6.0f);
_PS_CONST(twelve4, 12.0f);
_PS_CONST(pt25, 0.25f);
#endif
#define PI (3.14159265359f)


/* Serial GB pair inner loop. No neighbor list. No exclusions.

iat: pivot i atom
jb: first j atom, je: last atom (excluded)

It computes the interactions between atom i and all jb<=j<=je

*/
int agbnp3_gb_energy_inner_nolist_soa(
		    AGBNPdata *agb, int iat, int natoms, 
		    int jb, int je,
		    float *x, float *y, float *z,
		    float *charge, float *br, float *dera,
		    float *dgbdrx, float *dgbdry, float *dgbdrz, 
		    float *egb_pair, 
		    float dielectric_factor){
   
  int k;
  float xi, yi, zi, qi, bi;
  float d2, dx, dy, dz, qq, bb, etij, fgb, fgb3, mw, atij, egb;
  float gx, gy, gz;
  float valid_mask;
  float *q = charge;

  float en = 0.0f;

  float pt25 = 0.25f;
  float one =  1.0f;

  float vdielf = 2.0f*dielectric_factor;

  float dgbdrix = 0.0f;
  float dgbdriy = 0.0f;
  float dgbdriz = 0.0f;
  float derai = 0.0f;

  xi = x[iat];
  yi = y[iat];
  zi = z[iat];
  qi = q[iat];
  bi = br[iat];

#pragma ivdep
  for(k=jb; k<=je; k++){

    dx = x[k] - xi;
    dy = y[k] - yi;
    dz = z[k] - zi;
    d2 = dx*dx + dy*dy + dz*dz;
    valid_mask = d2 > 0 ? 1.0f : 0.0f;
    qq = valid_mask*qi*q[k];
    bb = bi*br[k];
    etij = expf(-pt25*d2/bb);
    fgb = 1./sqrtf(d2 + bb*etij);
    egb = vdielf*qq*fgb;

    fgb3 = fgb*fgb*fgb;
    mw = qq*(one-pt25*etij)*fgb3;

    // printf("mw= %f %f\n",mw,dielectric_factor);


    gx = mw*dx;
    gy = mw*dy;
    gz = mw*dz;

    /* Ai's */
    atij = qq*(bb+pt25*d2)*etij*fgb3;


    en += egb;

    dera[k]   += atij;
    derai     += atij;


    dgbdrx[k] += gx;
    dgbdry[k] += gy;
    dgbdrz[k] += gz;

    dgbdrix -= gx;
    dgbdriy -= gy;
    dgbdriz -= gz;
  }

  dera[iat] += derai;
  dgbdrx[iat] += dgbdrix;
  dgbdry[iat] += dgbdriy;
  dgbdrz[iat] += dgbdriz;
  *egb_pair += en;

   return AGBNP_OK;
}


#ifdef USE_SSE
/* utility function to print out a quad */
void print4(__m128 v) {
  float *p = (float*)&v; 
  #ifndef USE_SSE2
  _mm_empty();
  #endif
  printf("[%13.8g, %13.8g, %13.8g, %13.8g]", p[0], p[1], p[2], p[3]);
}
#endif

/* Vectorized GB pair inner loop. No neighbor list. No exclusions.

iat: pivot i atom
jb: first j atom, assumed at start of quad boundary

It computes the interactions between atom i and all j>=jb atoms
up to the j<natoms divisible by 4.  

All arrays are assumed to be 16-byte aligned.

*/
#ifdef USE_SSE
int agbnp3_gb_energy_inner_nolist_ps(
		    AGBNPdata *agb, int iat, int natoms, 
		    int jb, 
		    float *x, float *y, float *z,
		    float *charge, float *br, float *dera,
		    float *dgbdrx, float *dgbdry, float *dgbdrz, 
		    float *egb_pair, 
		    float dielectric_factor){
   
  int k4, k4start;
  __m128 xi4, yi4, zi4, qi4, bi4;
  __m128 d2, dx, dy, dz, qq, bb, etij, fgb, fgb3, mw, atij, egb;
  __m128 gx, gy, gz;


  /* accumulators  */
  __m128 en = _mm_setzero_ps();
  float *ent = (float *)&en;

  __m128 derai = _mm_setzero_ps();
  float *derait = (float *)&derai;

  __m128 dgbdrix = _mm_setzero_ps();
  __m128 dgbdriy = _mm_setzero_ps();
  __m128 dgbdriz = _mm_setzero_ps();

  float *dgbdrixt = (float *)&dgbdrix;
  float *dgbdriyt = (float *)&dgbdriy;
  float *dgbdrizt = (float *)&dgbdriz;
  /*               */

  __m128 pt25 = *(__m128*)_ps_pt25;
  __m128 one = *(__m128*)_ps_1;

  __m128 *x4 = (__m128 *)x;
  __m128 *y4 = (__m128 *)y;
  __m128 *z4 = (__m128 *)z;
  __m128 *q4 = (__m128 *)charge;
  __m128 *b4 = (__m128 *)br;
  __m128 *dera4 = (__m128 *)dera;
  __m128 *dgbdrx4 = (__m128 *)dgbdrx;
  __m128 *dgbdry4 = (__m128 *)dgbdry;
  __m128 *dgbdrz4 = (__m128 *)dgbdrz;

  float vdiel = 2.0f*dielectric_factor;
  __m128 vdielf = _mm_set_ps1(vdiel);

  __m128 valid_mask; // to set to zero energies corresponding to zero distance

  xi4 = _mm_set_ps1(x[iat]);
  yi4 = _mm_set_ps1(y[iat]);
  zi4 = _mm_set_ps1(z[iat]);
  qi4 = _mm_set_ps1(charge[iat]);
  bi4 = _mm_set_ps1(br[iat]);

  k4start = jb/4;
  if(k4start*4 != jb){
    agbnp3_errprint("agbnp3_gb_energy_inner_nolist_ps(): expecting jb to be divisible by 4. Got: %d.\n", jb);
    return AGBNP_ERR;
  }

  for(k4=k4start; k4<natoms/4; k4++){

    dx = x4[k4] - xi4;
    dy = y4[k4] - yi4;
    dz = z4[k4] - zi4;
    d2 = dx*dx + dy*dy + dz*dz;
    valid_mask = _mm_cmpgt_ps(d2, _mm_setzero_ps());
    qq = qi4*q4[k4];
    bb = bi4*b4[k4];
    etij = exp_ps(-pt25*d2/bb);
    fgb = rsqrt_ps(d2 + bb*etij);
    egb = vdielf*qq*fgb;

    fgb3 = fgb*fgb*fgb;
    mw = qq*(one-pt25*etij)*fgb3;

    gx = _mm_and_ps(valid_mask,mw*dx);
    gy = _mm_and_ps(valid_mask,mw*dy);
    gz = _mm_and_ps(valid_mask,mw*dz);

    /* Ai's */
    atij = qq*(bb+pt25*d2)*etij*fgb3;

    en += _mm_and_ps(valid_mask, egb);

    dera4[k4] += atij;
    derai += atij;

    dgbdrx4[k4] += gx;
    dgbdry4[k4] += gy;
    dgbdrz4[k4] += gz;

    dgbdrix -= gx;
    dgbdriy -= gy;
    dgbdriz -= gz;
   }

  /* update accumulators for atom iat */
  dera[iat] += derait[0] + derait[1] + derait[2] + derait[3];
  dgbdrx[iat] += dgbdrixt[0] + dgbdrixt[1] + dgbdrixt[2] + dgbdrixt[3];
  dgbdry[iat] += dgbdriyt[0] + dgbdriyt[1] + dgbdriyt[2] + dgbdriyt[3];
  dgbdrz[iat] += dgbdrizt[0] + dgbdrizt[1] + dgbdrizt[2] + dgbdrizt[3];
  /* update pair energy */
  *egb_pair += ent[0] + ent[1] + ent[2] + ent[3];
  return AGBNP_OK;
}

/* returns loop indexes to find leading, trailing, and quad sections */
void agbnp3_qindex(int jat, int n, int *beglead, int *endlead, int* begquad, int* endquad, int* begtrail, int* endtrail){
  int lead1, lead2, s4, e4, trail1, trail2;

  *beglead = *endlead = *begquad = *endquad = *begtrail = *endtrail = -1;
  if(jat > n - 1) return;

  s4 = jat%4 == 0 ? jat : ((jat/4)+1)*4;
  e4 = (n/4)*4 - 1;
  if(e4 - s4 + 1 < 4) {// no quads
    *beglead = jat;
    *endlead = n - 1;
    return;
  }

  lead1 = jat%4 == 0 ? -1 : jat;
  lead2 = s4 - 1;
  if(lead1 < 0 || lead1 > lead2) lead2 = lead1 = -1;

  trail1 = e4+1;
  trail2 = n-1;
  if(trail1 > trail2) trail1 = trail2 = -1;

  *beglead = lead1;
  *endlead = lead2;

  *begquad = s4;
  *endquad = e4;

  *begtrail = trail1;
  *endtrail = trail2;
} 
#endif


/* 

GB pair and self-energy 
          + derivatives (at constant Born radii).
No neighbor list. No exclusions.

All arrays are assumed to be 16-byte aligned.
*/
 int agbnp3_gb_energy_nolist_ps(AGBNPdata *agb, AGBworkdata *agbw_h,
				float *x, float *y, float *z,
				float *egb_self, float *egb_pair){
  int iat, jstart, jend;
  float qiat,biat;
  float dielectric_factor = 
    -0.5*(1./agb->dielectric_in - 1./agb->dielectric_out);
  float vdielf = dielectric_factor;
  int natoms = agb->natoms;
  float *charge = (float *)agb->charge;
  float *br = (float *)agbw_h->br;
  float *dera = (float *)agbw_h->dera;
  float *dgbdrx = (float *)agbw_h->dgbdrx;
  float *dgbdry = (float *)agbw_h->dgbdry;
  float *dgbdrz = (float *)agbw_h->dgbdrz;
  float egb_self_h = 0.0;
  float egb_pair_h = 0.0;

  int beglead, endlead, begquad, endquad, begtrail, endtrail;

  float *dera_m = agb->agbw->dera;

  memset(dgbdrx,0,natoms*sizeof(float));
  memset(dgbdry,0,natoms*sizeof(float));
  memset(dgbdrz,0,natoms*sizeof(float));

#pragma omp for schedule(static,1)
  for(iat=0;iat<natoms;iat++){
    qiat = charge[iat];
    biat = br[iat];
    egb_self_h += vdielf*qiat*qiat/biat;

#ifdef USE_SSE
    //SSE inner loop
    agbnp3_qindex(iat+1, natoms, &beglead, &endlead, &begquad, &endquad, &begtrail, &endtrail);
    // leading j-particles before the start of the quads
    if(beglead >= 0){// first j-particle not at quad boundary
      agbnp3_gb_energy_inner_nolist_soa(agb, iat, natoms, 
					beglead, endlead,
					x, y, z, charge, br, dera,
					dgbdrx, dgbdry, dgbdrz, 
					&egb_pair_h, dielectric_factor);
    }

    /* compute the bulk in group of quads */
    if(begquad >= 0){
      agbnp3_gb_energy_inner_nolist_ps(
				       agb, iat, natoms, 
				       begquad, 
				       x, y, z,
				       charge, br, dera,
				       dgbdrx, dgbdry, dgbdrz, 
				       &egb_pair_h, dielectric_factor);
    }

    // add the trailing left-overs
    if(begtrail >= 0){
      agbnp3_gb_energy_inner_nolist_soa(agb, iat, natoms, 
					begtrail, endtrail,
					x, y, z, charge, br, dera,
					dgbdrx, dgbdry, dgbdrz, 
					&egb_pair_h, dielectric_factor);
    }
#else // USE_SSE

    //regular inner loop
    jstart = iat + 1;
    jend = natoms - 1;
    if(jend>=jstart){
      agbnp3_gb_energy_inner_nolist_soa(agb, iat, natoms, 
					jstart, jend,
					x, y, z, charge, br, dera,
					dgbdrx, dgbdry, dgbdrz, 
					&egb_pair_h, dielectric_factor);
    }
#endif
  }


  //TBF copy the derivatives to old format
  {
    int iat;
    for(iat=0;iat<natoms;iat++){
      agbw_h->dgbdr_h[iat][0] += dgbdrx[iat];
      agbw_h->dgbdr_h[iat][1] += dgbdry[iat];
      agbw_h->dgbdr_h[iat][2] += dgbdrz[iat];
    }
  }

  // add energies to total
#pragma omp atomic
  *egb_self += egb_self_h;
#pragma omp atomic
  *egb_pair += egb_pair_h;

#ifdef _OPENMP
  //reduce dera
#pragma omp critical
  for(iat=0;iat<natoms;iat++){
    dera_m[iat] += dera[iat];
  }
#endif

#pragma omp barrier
#pragma omp single
  /* auxiliary quantities */
  {
    float_a *q2ab = agb->agbw->q2ab;
    float_a *br1_swf_der = agb->agbw->br1_swf_der;
    float_a *alpha = agb->agbw->alpha;
    float_a *brw = agb->agbw->brw;
    float_a *abrw = agb->agbw->abrw;
    float_a *br_m = agb->agbw->br;
    /* q2ab[], brw[], abrw[], br1_swf_der[], and dera_m[] 
       are master copies */
    for(iat=0;iat<natoms;iat++){
      q2ab[iat] = charge[iat]*charge[iat]+dera_m[iat]*br_m[iat];
      q2ab[iat] *= br1_swf_der[iat];
      abrw[iat] = alpha[iat]*brw[iat];
      abrw[iat] *= br1_swf_der[iat];
    }
  }

#ifdef _OPENMP
  /* copy dera and auxiliary arrays to threads */
  memcpy(agbw_h->dera, agb->agbw->dera, natoms*sizeof(float));
  memcpy(agbw_h->q2ab, agb->agbw->q2ab, natoms*sizeof(float));
  memcpy(agbw_h->abrw, agb->agbw->abrw, natoms*sizeof(float));
#endif

  return AGBNP_OK;
}

#ifdef USE_SSE
int agbnp3_gb_energy_nolist_ps_testders(AGBNPdata *agb, AGBworkdata *agbw_h,
			       float *x, float *y, float *z,
			       float *egb_self, float *egb_pair){

  float en, en_old, dx = 0.01, de;;
  float eself, epair;
  int iat = 34;
  int iter, niter = 100;
  float *dgbdrx = agbw_h->dgbdrx;
  float *dgbdry = agbw_h->dgbdry;
  float *dgbdrz = agbw_h->dgbdrz;

  agbnp3_gb_energy_nolist_ps(agb, agbw_h,
			     x, y, z,
			     &eself, &epair);
  en_old = eself + epair;

  x[iat] += dx;
  for(iter = 0; iter<niter; iter++){
    agbnp3_gb_energy_nolist_ps(agb, agbw_h,
			       x, y, z,
			       &eself, &epair);
    en = eself + epair;
    de = dgbdrx[iat]*dx;
    printf("en: %f %f %f\n", en, en-en_old, de);
    en_old = en;
    x[iat] += dx;
  }

  return AGBNP_OK;

}
#endif

 

/* calculates inverse Born radii */
int agbnp3_inverse_born_radii_nolist_soa(AGBNPdata *agb, AGBworkdata *agbw_h,
					 float *x, float *y, float *z,
					 int init_frozen){
  float fourpi1 = 1./(4.*pi);
  int iq4cache=0; /* cache counters */
  int i, iat, j, jat;
  float spiat, spjat, dx, dy, dz, d, q, dr4;
  float riat, rjat;

  int natoms = agb->natoms;
  int nheavyat = agb->nheavyat;
  float *r = agb->r;
  float *sp = agbw_h->sp;
  float *vols = agbw_h->vols;
  float *br1 = agbw_h->br1;
  float *q4cache = agbw_h->q4cache;
  float vol2i;

  float cvdw = AGBNP_RADIUS_INCREMENT;

  float *dv = agbw_h->qdv;
  float *R1v = agbw_h->qR1v;
  float *R2v = agbw_h->qR2v;
  float *qv = agbw_h->qqv;
  float *dqv = agbw_h->qdqv;

  float *av = agbw_h->qav;
  float *bv = agbw_h->qbv;
  
  float *qkv = agbw_h->qkv;
  float *qxh= agbw_h->qxh;
  float *qyp= agbw_h->qyp;
  float *qy= agbw_h->qy;
  float *qy2p= agbw_h->qy2p;
  float *qy2= agbw_h->qy2;
  float *qf1= agbw_h->qf1;
  float *qf2= agbw_h->qf2;
  float *qfp1= agbw_h->qfp1;
  float *qfp2= agbw_h->qfp2;


  float xiat, yiat, ziat;

  int iv;

  iq4cache = 0;
  /* Loop over heavy atom pairs, these need scaled volume correction */
#pragma omp for schedule(static,1)
  for(iat=0;iat<nheavyat;iat++){

    iv = 0;

    xiat = x[iat];
    yiat = y[iat];
    ziat = z[iat];
    riat = r[iat];

    for(jat=iat+1; jat < nheavyat; jat++){
      rjat = r[jat];
      dx = x[jat] - xiat;
      dy = y[jat] - yiat;
      dz = z[jat] - ziat;
      d = mysqrt(dx*dx + dy*dy + dz*dz);
      dv[iv] = d;
      R1v[iv] = rjat - cvdw;
      R2v[iv] = riat;
      iv += 1;
      dv[iv] = d;
      R1v[iv] = riat - cvdw;
      R2v[iv] = rjat;
      iv += 1;
    }

#ifdef USE_SSE
    agbnp3_i4p_ps(agb, dv, R1v, R2v, iv, qv, dqv, av, bv,
		  qkv, qxh, qyp, qy, qy2p, qy2, qf1, qf2, qfp1, qfp2);
#else
    agbnp3_i4p_soa(agb, dv, R1v, R2v, iv, qv, dqv, av, bv,
		  qkv, qxh, qyp, qy, qy2p, qy2, qf1, qf2, qfp1, qfp2);
#endif
    
    iv = 0;
    for(jat=iat+1; jat < nheavyat; jat++){

      spiat = sp[iat];
      spjat = sp[jat];

      q = qv[iv];
      dr4 = dqv[iv];
      iv += 1;
      br1[jat] -= fourpi1*q*spiat;
      q4cache[iq4cache++] = q;
      q4cache[iq4cache++] = dr4;

      q = qv[iv];
      dr4 = dqv[iv];
      iv += 1;
      br1[iat] -= fourpi1*q*spjat;
      q4cache[iq4cache++] = q;
      q4cache[iq4cache++] = dr4;

    }


  }

  /* born radii of hydrogens.
     Assumes that hydrogens are listed after heavy atoms. */
#pragma omp for schedule(static,1)
  for(iat = 0; iat < nheavyat ; iat++){ //heavy atoms

    iv = 0;

    xiat = x[iat];
    yiat = y[iat];
    ziat = z[iat];
    riat = r[iat];

    for(jat = nheavyat; jat < natoms; jat++){ //hydrogens
      dx = x[jat] - xiat;
      dy = y[jat] - yiat;
      dz = z[jat] - ziat;
      d = mysqrt(dx*dx + dy*dy + dz*dz);
      dv[iv] = d;
      R1v[iv] = r[jat]-cvdw;
      R2v[iv] = riat;
      iv += 1;
    }

#ifdef USE_SSE
    agbnp3_i4p_ps(agb, dv, R1v, R2v, iv, qv, dqv, av, bv,
		  qkv, qxh, qyp, qy, qy2p, qy2, qf1, qf2, qfp1, qfp2);
#else
    agbnp3_i4p_soa(agb, dv, R1v, R2v, iv, qv, dqv, av, bv,
		  qkv, qxh, qyp, qy, qy2p, qy2, qf1, qf2, qfp1, qfp2);
#endif
    
    iv = 0;

    spiat = sp[iat];

    for(jat = nheavyat; jat < natoms; jat++){ //hydrogens
      q = qv[iv];
      dr4 = dqv[iv];
      iv += 1;
      br1[jat] -= fourpi1*q*spiat;
      q4cache[iq4cache++] = q;
      q4cache[iq4cache++] = dr4;
    }

  }

  return AGBNP_OK;
}

/* GB and vdw derivatives contribution at constant self volumes */
 int agbnp3_gb_ders_constvp_nolist_ps(AGBNPdata *agb, AGBworkdata *agbw_h,
				      float_a *x, float_a *y, float_a *z,
				      int init_frozen){

  int natoms = agb->natoms;
  float_a dielectric_factor = 
    -0.5*(1./agb->dielectric_in - 1./agb->dielectric_out);
  float_a vdielf = 1.0;
  int iq4cache=0; /* cache counters */
  int i, iat, j, jat, hk;
  float_a dx, dy, dz, d2, d, volume2, q, dr4, spiat, htij, utij, spjat;
  float_a fourpi1 = 1./(4.*pi);
  
  int nheavyat = agb->nheavyat;
  int *iheavyat = agb->iheavyat;
  int *isheavy = agbw_h->isheavy;
  int nhydrogen = agb->nhydrogen;
  int *ihydrogen = agb->ihydrogen;
  NeighList *near_nl = agbw_h->near_nl;
  NeighList *far_nl = agbw_h->far_nl;
  float_a *sp = agbw_h->sp;
  float_a *vols = agbw_h->vols;
  float_a *q2ab = agbw_h->q2ab;
  float_a *abrw = agbw_h->abrw;
  float_a (*dgbdr)[3] = agbw_h->dgbdr_h;
  float_a (*dvwdr)[3] = agbw_h->dvwdr_h;
  float *q4cache = agbw_h->q4cache;
  float_a u[3], ur[3];
  float_a w[3], wr[3];

  float *dgbdrx = agbw_h->dgbdrx;
  float *dgbdry = agbw_h->dgbdry;
  float *dgbdrz = agbw_h->dgbdrz;

  /* loop over near heavy-heavy interactions */
  iq4cache = 0;
#pragma omp for schedule(static,1)
  for(iat=0;iat<nheavyat;iat++){
    for(jat=iat+1;jat<nheavyat;jat++){
      dx = x[jat] - x[iat];
      dy = y[jat] - y[iat];
      dz = z[jat] - z[iat];
      d2 = dx*dx + dy*dy + dz*dz;
      d = mysqrt(d2);
      q = q4cache[iq4cache++];
      dr4 = q4cache[iq4cache++];
      spiat = sp[iat];
      htij = q2ab[jat]*dr4*spiat;
      utij = abrw[jat]*dr4*spiat;
      q = q4cache[iq4cache++];
      dr4 = q4cache[iq4cache++];
      spjat = sp[jat];
      htij += q2ab[iat]*dr4*spjat;
      utij += abrw[iat]*dr4*spjat;
      htij = fourpi1*vdielf*dielectric_factor*htij/d;	
      u[0] = htij*dx;
      u[1] = htij*dy;
      u[2] = htij*dz;

      dgbdr[iat][0] +=  u[0]; 
      dgbdr[iat][1] +=  u[1]; 
      dgbdr[iat][2] +=  u[2]; 
      dgbdr[jat][0] -=  u[0];
      dgbdr[jat][1] -=  u[1];
      dgbdr[jat][2] -=  u[2];
      
      utij = fourpi1*utij/d;
      w[0] = utij*dx;
      w[1] = utij*dy;
      w[2] = utij*dz;

      dvwdr[iat][0] +=  w[0]; 
      dvwdr[iat][1] +=  w[1]; 
      dvwdr[iat][2] +=  w[2]; 
      dvwdr[jat][0] -=  w[0];
      dvwdr[jat][1] -=  w[1];
      dvwdr[jat][2] -=  w[2];

    }
  }

  /* loop for hydrogen-heavy interactions */
#pragma omp for schedule(static,1)
   for(iat = 0; iat < nheavyat ; iat++){//heavy atoms
     for(jat = nheavyat; jat < natoms; jat++){ //hydrogens

      dx = x[jat] - x[iat];
      dy = y[jat] - y[iat];
      dz = z[jat] - z[iat];
      d2 = dx*dx + dy*dy + dz*dz;
      d = mysqrt(d2);
      iq4cache += 1;
      dr4 = q4cache[iq4cache++];
      htij = q2ab[jat]*dr4*sp[iat];
      utij = abrw[jat]*dr4*sp[iat];
      
      htij = fourpi1*vdielf*dielectric_factor*htij/d;	
      u[0] = htij*dx;
      u[1] = htij*dy;
      u[2] = htij*dz;
      
      dgbdr[iat][0] +=  u[0]; 
      dgbdr[iat][1] +=  u[1]; 
      dgbdr[iat][2] +=  u[2]; 
      dgbdr[jat][0] -=  u[0];
      dgbdr[jat][1] -=  u[1];
      dgbdr[jat][2] -=  u[2];
      
      utij = fourpi1*utij/d;
      w[0] = utij*dx;
      w[1] = utij*dy;
      w[2] = utij*dz;
      
      dvwdr[iat][0] +=  w[0]; 
      dvwdr[iat][1] +=  w[1]; 
      dvwdr[iat][2] +=  w[2]; 
      dvwdr[jat][0] -=  w[0];
      dvwdr[jat][1] -=  w[1];
      dvwdr[jat][2] -=  w[2];
      
    }
  }

  return AGBNP_OK;
}


/* Evaluates Ui's and Vi's */
int agbnp3_gb_deruv_nolist_ps(AGBNPdata *agb, AGBworkdata *agbw_h, 
			  int init_frozen){
  int iq4cache=0;
  int i, iat, j, jat;
  float_a q;

  int natoms = agb->natoms;
  int nheavyat = agb->nheavyat;
  float *q2ab = agbw_h->q2ab;
  float *abrw = agbw_h->abrw;
  float *deru = agbw_h->deru;
  float *derv = agbw_h->derv;
  float *derus = agbw_h->derus;
  float *dervs = agbw_h->dervs;
  float *psvol = agbw_h->psvol;
  float *q4cache = agbw_h->q4cache;
  float *deru_m = agb->agbw->deru;
  float *derv_m = agb->agbw->derv;
  float *derus_m = agb->agbw->derus;
  float *dervs_m = agb->agbw->dervs;

  float dielectric_factor = 
    -0.5*(1./agb->dielectric_in - 1./agb->dielectric_out);
  float *vols = agbw_h->vols;

  /* heavy atoms loop */
  iq4cache = 0;
#pragma omp for schedule(static,1)
  for(iat=0;iat<nheavyat;iat++){
    for(jat=iat+1;jat<nheavyat;jat++){
      /* get from cache */
      q = q4cache[iq4cache++];
      iq4cache += 1;
      deru[iat] += q2ab[jat]*q;
      derv[iat] += abrw[jat]*q;
      q = q4cache[iq4cache++];
      iq4cache += 1;
      deru[jat] += q2ab[iat]*q;
      derv[jat] += abrw[iat]*q;
    }
  }

#pragma omp for schedule(static,1)
  for(iat = 0; iat < nheavyat ; iat++){ //heavy atoms
   for(jat = nheavyat; jat < natoms; jat++){ //hydrogens
	q = q4cache[iq4cache++];
	iq4cache += 1;
	deru[iat] += q2ab[jat]*q;
	derv[iat] += abrw[jat]*q;
    }
  }

  for(iat=0;iat<nheavyat;iat++){
    deru[iat] /= vols[iat];
    derv[iat] /= vols[iat];
  }
  /* compute effective gammas for GB and vdW derivatives due to surface
     area corrections */
  q = 1./(4.*pi);
  for(iat=0;iat<nheavyat;iat++){
    dervs[iat] = q*psvol[iat]*derv[iat];
  }
  q = dielectric_factor/(4.*pi);
  for(iat=0;iat<nheavyat;iat++){
    derus[iat] = q*psvol[iat]*deru[iat];
  }


#ifdef _OPENMP
  //reduce deru/derv
#pragma omp critical
  for(iat = 0; iat < nheavyat ; iat++){
    deru_m[iat] += deru[iat];
  }
#pragma omp critical
  for(iat = 0; iat < nheavyat ; iat++){
    derv_m[iat] += derv[iat];
  }
#pragma omp critical
  for(iat = 0; iat < nheavyat ; iat++){
    derus_m[iat] += derus[iat];
  }
#pragma omp critical
  for(iat = 0; iat < nheavyat ; iat++){
    dervs_m[iat] += dervs[iat];
  }
#pragma omp barrier
#endif
  // copy to threads
  memcpy(derv, derv_m, nheavyat*sizeof(float));
  memcpy(deru, deru_m, nheavyat*sizeof(float));
  memcpy(dervs, dervs_m, nheavyat*sizeof(float));
  memcpy(derus, derus_m, nheavyat*sizeof(float));

  return AGBNP_OK;
}

/* applies cspline interpolation to a series of data:
k[]: table look up index
xh[]: x/dx
dx: spacing
m: number of points
yp[], y[]: node ordinates at k+1 and k
y2p[], y2[]: node coefficients at k+1 and k
f[], fp[]: output function values and derivatives
*/
#ifdef USE_SSE
void agbnp3_cspline_interpolate_ps(float *kvf, float *xhf, float dx, int m, 
				   float* ypf, float *yf,
				   float* y2pf, float *y2f,
				   float *ff, float *fpf){
  int i, end;
  __m128 b, a, a2, b2, kf, dp1, dp2, xhk, yk, ypk, y2k, y2pk, dxinv;
  __m128 dx4 = _mm_set_ps1(dx);
  __m128 six = *(__m128*)_ps_six4;
  __m128 one = *(__m128*)_ps_1;
  __m128 three = *(__m128*)_ps_3;

  __m128 *kv  = (__m128*)kvf;
  __m128 *xh  = (__m128*)xhf;
  __m128 *yp  = (__m128*)ypf;
  __m128 *y   = (__m128*)yf;
  __m128 *y2p = (__m128*)y2pf;
  __m128 *y2  = (__m128*)y2f;
  __m128 *f   = (__m128*)ff;
  __m128 *fp  = (__m128*)fpf;



  if(!m) return;

  if(m%4 == 0){
    end = m/4;
  }else{
    end = m/4 + 1;
  }

  dp1 = dx4/six;
  dp2 = dx4*dp1;
  dxinv = one/dx4;

  for(i=0;i<end;i++){
    kf=kv[i];
    xhk = xh[i];
    yk = y[i];
    ypk = yp[i];
    y2k = y2[i];
    y2pk = y2p[i];



    a = (kf+one)-xhk;
    a2 = a*a;
    b = xhk - kf;
    b2 = b*b;

    f[i] = a*yk+b*ypk +  ((a2*a - a)*y2k + (b2*b - b)*y2pk)*dp2;
    fp[i] = (ypk - yk)*dxinv - 
      ((three*a2 - one)*y2k + (three*b2 - one)*y2pk)*dp1;
  }
}
#endif

#ifdef USE_SSE
int agbnp3_i4p_ps(AGBNPdata *agb, float* rijf, float *Rif, float *Rjf, 
      int m, float *ff, float *fpf,
      float *mbuffera, float *mbufferb,
      float *qkv, float *qxh, float *qyp, float *qy, float *qy2p, float *qy2,
      float *qf1, float *qf2, float *qfp1, float *qfp2){

  int i, end;
  int msize = 0;
  __m128 *a = (__m128 *)mbuffera;
  __m128 *b = (__m128 *)mbufferb;
  __m128 one = *(__m128*)_ps_1;
  __m128 Rjinv;
  __m128 *rij = (__m128 *)rijf;
  __m128 *Ri = (__m128 *)Rif;
  __m128 *Rj = (__m128 *)Rjf;
  __m128 *f = (__m128 *)ff;
  __m128 *fp = (__m128 *)fpf;

  if(!m) return AGBNP_OK;
  
  if(m%4 == 0){
    end = m/4;
  }else{
    end = m/4 + 1;
  }

  for(i=0;i<end;i++){
    Rjinv = one/Rj[i];
    a[i] = rij[i]*Rjinv;
    b[i] = Ri[i]*Rjinv;
  }

  agbnp3_interpolate_ctablef42d_soa(agb->f4c1table2dh, 
		       (float *)a, (float *)b, m, ff, fpf,
	               qkv, qxh, qyp, qy, qy2p, qy2, qf1, qf2, qfp1, qfp2);

  for(i=0;i<end;i++){
    Rjinv = one/Rj[i];
    f[i] = f[i]*Rjinv;
    fp[i] = fp[i]*Rjinv*Rjinv;
  }

  return AGBNP_OK;
}
#endif

/* print overlaps for debugging */
int agbnp3_print_overlap_buffer(int order, int noverlaps, GOverlap *overlap, 
				int nroots, int *roots){
  int iroot, begoverlap, endoverlap;
  int ioverlap,ig;

  printf("List of Gaussian overlap buffers\n");

  for(iroot=0;iroot<nroots;iroot++){

    begoverlap = roots[iroot];
    endoverlap = roots[iroot+1];

    if(endoverlap - begoverlap <= 0) continue;

    if(order == 1){
      printf("Root = NULL\n");
    }else{
      //read root from first overlap
      printf("Root = (");
      for(ig=0;ig<order-1;ig++){
	printf("%d ", overlap[begoverlap].parents[ig]);
      }
      printf(")\n");
    }

    printf("Overlaps = ");
    for(ioverlap=begoverlap;ioverlap<endoverlap;ioverlap++){
      printf("(");
      for(ig=0;ig<order;ig++){
	printf("%d ", overlap[ioverlap].parents[ig]);
      }
      printf(") ");
    }

    printf("\n");

  }

  return AGBNP_OK;
}


/* Computes self-volumes, corrected self-volumes for pair
    descreening, and surface areas */ 
int agbnp3_self_volumes_rooti(AGBNPdata *agb, AGBworkdata *agbw,
			       float_a *x, float_a *y, float_a *z){
  /* coordinate buffer for Gaussian overlap calculation */
  float_a gx[AGBNP_MAX_OVERLAP_LEVEL][3];
  /* radius buffer for  Gaussian overlap calculation */
  float_a gr[AGBNP_MAX_OVERLAP_LEVEL];
  /* Gaussian parameters buffers for overlap calculation */
  float_a ga[AGBNP_MAX_OVERLAP_LEVEL];
  float_a gp[AGBNP_MAX_OVERLAP_LEVEL];
  /* derivatives buffers for Gaussian overlaps */
  float_a gdr[AGBNP_MAX_OVERLAP_LEVEL][3];
  float_a gdR[AGBNP_MAX_OVERLAP_LEVEL];
  float_a gd2rR[AGBNP_MAX_OVERLAP_LEVEL][AGBNP_MAX_OVERLAP_LEVEL][3];
  /* holds the atom indexes being overlapped */
  int gatlist[AGBNP_MAX_OVERLAP_LEVEL];
  float_a gvol; /* gaussian overlap volume */
  int order; /* order of the overlap */
  /* holds overlap Gaussian parameters at each level */
  GParm gparams[AGBNP_MAX_OVERLAP_LEVEL];
  float_a an, pn, cn[3];
  /* coefficients for scaled volume */
  float volpcoeff[AGBNP_MAX_OVERLAP_LEVEL];
  float sign;

  int i, iat, jat, kat, ii,j;
  int ii1, ii2, iats, jats, ia, ja;  
  float sr,u,v,w,altw=1.0;

  int natoms = agb->natoms;
  int nheavyat = agb->nheavyat;
  NeighList *near_nl = agbw->near_nl;
  float *r = agb->r;
  float *galpha = agbw->galpha;
  float *gprefac = agbw->gprefac;
  float *volumep = agbw->volumep;
  float *surf_area = agbw->surf_area;
  float *vols = agbw->vols;

  GParm *atm_gs = agbw->atm_gs;
  GParm *gsi, *gsj, gsij;

#ifdef _OPENMP
  float *volumep_m = agb->agbw->volumep;
  float *surf_area_m = agb->agbw->surf_area;
#endif

  int *root, *root_next;
  GOverlap *overlap; // current overlap list
  GOverlap *overlap_next; // next overlap list
  int iovl, iovl_next, iovt;; // switches between 0 and 1 for above
  int iroot, new_root;
  GOverlap *ov;
  int nov_next, nroot_next, ip;

  int nov, nov_beg, nov_end;

  int _nov_ = 0;

  int nadd;

  float *a1 = agbw->a1;
  float *p1 = agbw->p1;
  float *c1x = agbw->c1x;
  float *c1y = agbw->c1y;
  float *c1z = agbw->c1z;

  float *a2 = agbw->a2;
  float *p2 = agbw->p2;
  float *c2x = agbw->c2x;
  float *c2y = agbw->c2y;
  float *c2z = agbw->c2z;

  float *v3 = agbw->v3;
  float *v3p = agbw->v3p;
  float *fp3 = agbw->fp3;
  float *fpp3 = agbw->fpp3;

  float h3, h4, u1, u2, v0, v1, v2, gvolp, fp, deltai, d2;

  float volmina = AGBNP_MIN_VOLA;
  float volminb = AGBNP_MIN_VOLB;

  float *xx;

  /* verbose = 1; */


  for(iat=0;iat<nheavyat;iat++){
    agbw->atm_gs[iat].a = agbw->galpha[iat];
    agbw->atm_gs[iat].p = agbw->gprefac[iat];
    agbw->atm_gs[iat].c[0] = agb->x[iat];
    agbw->atm_gs[iat].c[1] = agb->y[iat];
    agbw->atm_gs[iat].c[2] = agb->z[iat];
  }

  /* set scaled volume coefficients */
  sign = 1.0;
  for(order=1;order<=AGBNP_MAX_OVERLAP_LEVEL;order++){
    volpcoeff[order-1] = sign/((float_a)order);
    sign *= -1.0;
  }

  //order 1 overlap buffer
  iovl = 0;
  iovl_next = 1;
  order = 1;  

  //constructs order=2 overlaps based on neighbor list
  nov_next = 0;
  nroot_next = 0;
  overlap_next = agbw->overlap_lists[iovl_next];
  root_next =  agbw->root_lists[iovl_next];
  order += 1;

  nov_beg = 0;

  nov = nov_beg;
  for(iat=0;iat<nheavyat;iat++){ //assume heavy atoms are on top

    gsi = &(atm_gs[iat]);

    //fprintf(stderr,"ov: order= %d rootsize= %d\n", order,  near_nl->nne[iat]);
	//check to resize Gaussian overlap buffer
    nadd = near_nl->nne[iat]; //number of Gaussians j>i in the same root
    
    //fprintf(stderr,"nadd: %d %d %d\n", nov, nadd, agbw->gbuffer_size);

    if(nov + nadd > agbw->gbuffer_size-2){
      // reallocate overlap lists
      int new_size = agbw->gbuffer_size + natoms*AGBNP_OVERLAPS_INCREMENT/10;
      if (new_size - agbw->gbuffer_size < nadd){
	new_size = agbw->gbuffer_size + nadd;
      }
      if(agbnp3_reallocate_gbuffers(agbw, new_size) != AGBNP_OK){
	agbnp3_errprint("agbnp3_self_volumes_rooti(): Unable to expand Gaussian overlap buffers. Requested size = %d\n", new_size);
	return AGBNP_ERR;
      }else{
	a1 = agbw->a1;
	p1 = agbw->p1;
	c1x = agbw->c1x;
	c1y = agbw->c1y;
	c1z = agbw->c1z;
	a2 = agbw->a2;
	p2 = agbw->p2;
	c2x = agbw->c2x;
	c2y = agbw->c2y;
	c2z = agbw->c2z;
	v3 = agbw->v3;
	v3p = agbw->v3p;
	fp3 = agbw->fp3;
	fpp3 = agbw->fpp3;
      }
    }


    for(j=0; j < near_nl->nne[iat]; j++){

      /* buffer 1 */
      a1[nov] = gsi->a;
      p1[nov] = gsi->p;
      c1x[nov] = gsi->c[0];
      c1y[nov] = gsi->c[1];
      c1z[nov] = gsi->c[2];
      
      /* buffer 2 */
      jat = near_nl->neighl[iat][j];
      gsj = &(atm_gs[jat]);
      a2[nov] = gsj->a;
      p2[nov] = gsj->p;
      c2x[nov] = gsj->c[0];
      c2y[nov] = gsj->c[1];
      c2z[nov] = gsj->c[2];

      nov += 1;

    }
  }
  nov_end = nov;

#ifdef USE_SSE
  agbnp3_ogauss_ps(nov_beg, nov_end,
		    c1x, c1y, c1z, a1, p1,
		    c2x, c2y, c2z, a2, p2,
		    volmina, volminb,
		    v3, v3p, fp3, fpp3);
#else
  agbnp3_ogauss_soa(nov_beg, nov_end,
		    c1x, c1y, c1z, a1, p1,
		    c2x, c2y, c2z, a2, p2,
		    volmina, volminb,
		    v3, v3p, fp3, fpp3);
#endif
  
  nov = nov_beg;
  for(iat=0;iat<nheavyat;iat++){ //assume heavy atoms are on top

    gsi = &(atm_gs[iat]);
    gx[0][0] = gsi->c[0];
    gx[0][1] = gsi->c[1];
    gx[0][2] = gsi->c[2];
    ga[0] = gsi->a;
    gp[0] = gsi->p;
    gr[0] = agb->r[iat];

    gatlist[0] = iat;

    root_next[nroot_next] = nov_next; //starts a new root, maybe emtpty 

    for(j=0; j < near_nl->nne[iat]; j++, nov++){

      gvolp = v3p[nov]; //switched volume
      
      _nov_ += 1;
      
      if(gvolp>FLT_MIN){
      //if(sr > SURF_MIN){
	jat = near_nl->neighl[iat][j];
	gatlist[1] = jat;
	gsj=&(atm_gs[jat]);
	gx[1][0] = gsj->c[0];
	gx[1][1] = gsj->c[1];
	gx[1][2] = gsj->c[2];
	ga[1] = gsj->a;
	gp[1] = gsj->p;
	gr[1] = agb->r[jat];

	gsij.a = a2[nov];
	gsij.p = p2[nov];
	gsij.c[0] = c2x[nov];
	gsij.c[1] = c2y[nov];
	gsij.c[2] = c2z[nov];
	gvol = v3[nov]; //raw volume
	fp = fp3[nov];

	//constructs new overlap

	// check that buffer is large enough
	if(nov_next > agbw->size_overlap_lists[iovl_next]-1){
	  // reallocate overlap lists
	  int new_size = agbw->size_overlap_lists[iovl_next] + natoms*AGBNP_OVERLAPS_INCREMENT;
	  if(agbnp3_reallocate_overlap_lists(agbw, new_size) != AGBNP_OK){
	    agbnp3_errprint("agbnp3_self_volumes_rooti(): Unable to expand overlap list. Requested size = %d\n", new_size);
	    return AGBNP_ERR;
	  }
	  overlap = agbw->overlap_lists[iovl];
	  root =  agbw->root_lists[iovl];	  
	  overlap_next = agbw->overlap_lists[iovl_next];
	  root_next =  agbw->root_lists[iovl_next];
	}
	
	ov = &(overlap_next[nov_next]);
	ov->order = order;
	for(ip=0;ip<order;ip++){
	  ov->parents[ip] = gatlist[ip];
	}
	ov->gs = gsij;
	nov_next += 1;

	/* collect scaled volumes */
	u = volpcoeff[order-1];
	v = order*volpcoeff[order-1];
	w = u*gvolp;
	for(ii=0;ii<order;ii++){
	  kat = gatlist[ii];
	  volumep[kat] += w;
	}
	deltai = 0.0;
	for(ii=0;ii<order;ii++){
	  deltai += ga[ii];
	}
	h3 = v*fp*3.0f/deltai;
	h4 = 0.5f*v*fp;
	for(ii=0;ii<order;ii++){
	  kat = gatlist[ii];
	  u1 = ga[ii]*gvol;
	  u2 = -2.*u1;
	  v0 = u2*(gx[ii][0]-gsij.c[0]);
	  v1 = u2*(gx[ii][1]-gsij.c[1]);
	  v2 = u2*(gx[ii][2]-gsij.c[2]);
	  d2 = v0*v0 + v1*v1 + v2*v2;
	  surf_area[kat] += (h3*u1 + h4*d2/u1)/gr[ii];
	}
	
      }

    }
    nroot_next += 1; //new root
  }
  root_next[nroot_next] = nov_next; //termination of Verlet list

  agbw->n_overlap_lists[iovl_next] = nov_next;
  agbw->n_root_lists[iovl_next] = nroot_next;
  //  agbnp3_setup_second_order_overlap_buffer(agb,agbw,iovl);
  //agbnp3_print_overlap_buffer(order, agbw->n_overlap_lists[iovl_next], agbw->overlap_lists[iovl_next],agbw->n_root_lists[iovl_next], agbw->root_lists[iovl_next]);

  // swap overlap buffers, increment order
  iovt = iovl_next;
  iovl_next = iovl;
  iovl = iovt;
  order += 1;

  // go through all the other orders swapping overlap buffers each time
  // swap overlap buffers, increment order
  while(agbw->n_overlap_lists[iovl]>0 && order < AGBNP_MAX_OVERLAP_LEVEL){
    
    overlap = agbw->overlap_lists[iovl];
    root =  agbw->root_lists[iovl];
    
    nov_next = 0;
    nroot_next = 0;
    overlap_next = agbw->overlap_lists[iovl_next];
    root_next =  agbw->root_lists[iovl_next];
    

    /* first time through, do not do any actual overlap calculation, just assemble the buffers */
    nov_beg = nov_end;
    nov = nov_beg;
    for(iroot=0;iroot<agbw->n_root_lists[iovl];iroot++){
      
      //tests number of gaussians in this root
      if(root[iroot+1] - root[iroot] <= 0) continue;

      //fprintf(stderr,"ov: order= %d rootsize= %d\n", order, root[iroot+1] - root[iroot]);
      
      for(i=root[iroot];i<root[iroot+1]; i++){ //loop over overlaps
	
	//root Gaussian
	gsi = &(overlap[i].gs);
	
	//check to resize Gaussian overlap buffer
	nadd = root[iroot+1] - (i+1); //number of Gaussians j>i in the same root

	//fprintf(stderr,"nadd: %d %d %d\n", nov, nadd, agbw->gbuffer_size);

	if(nov + nadd > agbw->gbuffer_size-1000){
	  // reallocate overlap lists
	  int new_size = agbw->gbuffer_size + natoms*AGBNP_OVERLAPS_INCREMENT;
	  if (new_size - agbw->gbuffer_size < nadd){
	    new_size = agbw->gbuffer_size + nadd;
	  }
	  if(agbnp3_reallocate_gbuffers(agbw, new_size) != AGBNP_OK){
	    agbnp3_errprint("agbnp3_self_volumes_rooti(): Unable to expand Gaussian overlap buffers. Requested size = %d\n", new_size);
	    return AGBNP_ERR;
	  }else{
	    a1 = agbw->a1;
	    p1 = agbw->p1;
	    c1x = agbw->c1x;
	    c1y = agbw->c1y;
	    c1z = agbw->c1z;
	    a2 = agbw->a2;
	    p2 = agbw->p2;
	    c2x = agbw->c2x;
	    c2y = agbw->c2y;
	    c2z = agbw->c2z;
	    v3 = agbw->v3;
	    v3p = agbw->v3p;
	    fp3 = agbw->fp3;
	    fpp3 = agbw->fpp3;
	  }
	}
	
	for(j=i+1;j<root[iroot+1];j++){//gaussians in the same root

	  //	  printf("iroot = %d root[iroot] = %d root[iroot+1] = %d  i = %d  j = %d\n", iroot, root[iroot], root[iroot+1], i, j);
	  /* buffer 1 */

	  a1[nov] = gsi->a;
	  p1[nov] = gsi->p;
	  c1x[nov] = gsi->c[0];
	  c1y[nov] = gsi->c[1];
	  c1z[nov] = gsi->c[2];

	  /* buffer 2 */
	  ov = &(overlap[j]);
	  jat = ov->parents[order-2];
	  gsj = &(atm_gs[jat]);
	  a2[nov] = gsj->a;
	  p2[nov] = gsj->p;
	  c2x[nov] = gsj->c[0];
	  c2y[nov] = gsj->c[1];
	  c2z[nov] = gsj->c[2];

	  nov += 1;

	}
      }
    }
    nov_end =  nov;

#ifdef USE_SSE
    agbnp3_ogauss_ps(nov_beg, nov_end,
		      c1x, c1y, c1z, a1, p1,
		      c2x, c2y, c2z, a2, p2,
		      volmina, volminb,
		      v3, v3p, fp3, fpp3);
#else
    agbnp3_ogauss_soa(nov_beg, nov_end,
		      c1x, c1y, c1z, a1, p1,
		      c2x, c2y, c2z, a2, p2,
		      volmina, volminb,
		      v3, v3p, fp3, fpp3);
#endif

    /* second time through, assemble next overlap list and scatter to atoms */
    nov = nov_beg;
    for(iroot=0;iroot<agbw->n_root_lists[iovl];iroot++){
      
      //tests number of gaussians in this root
      if(root[iroot+1] - root[iroot] <= 0) continue;

      //fprintf(stderr,"ov: order= %d rootsize= %d\n", order, root[iroot+1] - root[iroot]);
      
      for(i=root[iroot];i<root[iroot+1]; i++){ //loop over overlaps
	
	//extract parents etc. of this overlap
	ov = &(overlap[i]);
	for(ip=0;ip<order-1;ip++){
	  iat = ov->parents[ip];
	  gatlist[ip] = iat;
	  gx[ip][0] = atm_gs[iat].c[0];
	  gx[ip][1] = atm_gs[iat].c[1];
	  gx[ip][2] = atm_gs[iat].c[2];
	  ga[ip] = atm_gs[iat].a;
	  gp[ip] = atm_gs[iat].p;
	  gr[ip] = agb->r[iat];
	}
	
	root_next[nroot_next] = nov_next; //starts a new root, maybe empty 
	
	for(j=i+1;j<root[iroot+1];j++,nov++){//gaussians in the same root

	  //	  printf("iroot = %d root[iroot] = %d root[iroot+1] = %d  i = %d  j = %d\n", iroot, root[iroot], root[iroot+1], i, j);

	  gvolp = v3p[nov]; //switched volume

	  _nov_ += 1;

	  if(gvolp>FLT_MIN){
	    //if(sr > SURF_MIN){
	    ov = &(overlap[j]);
	    jat = ov->parents[order-2];
	    ip = order-1;
	    gatlist[ip] = jat;
	    gx[ip][0] = atm_gs[jat].c[0];
	    gx[ip][1] = atm_gs[jat].c[1];
	    gx[ip][2] = atm_gs[jat].c[2];
	    ga[ip] = atm_gs[jat].a;
	    gp[ip] = atm_gs[jat].p;
	    gr[ip] = agb->r[jat];

	    gsij.a = a2[nov];
	    gsij.p = p2[nov];
	    gsij.c[0] = c2x[nov];
	    gsij.c[1] = c2y[nov];
	    gsij.c[2] = c2z[nov];
	    gvol = v3[nov]; //raw volume
	    fp = fp3[nov];

	    // constructs new overlap

	    // check that buffer is large enough
	    if(nov_next > agbw->size_overlap_lists[iovl_next]-1){
	      // reallocate overlap lists
	      int new_size = agbw->size_overlap_lists[iovl_next] + natoms*AGBNP_OVERLAPS_INCREMENT;
	      if(agbnp3_reallocate_overlap_lists(agbw, new_size) != AGBNP_OK){
		agbnp3_errprint("agbnp3_self_volumes_rooti(): Unable to expand overlap list. Requested size = %d\n", new_size);
		return AGBNP_ERR;
	      }
	      overlap = agbw->overlap_lists[iovl];
	      root =  agbw->root_lists[iovl];	  
	      overlap_next = agbw->overlap_lists[iovl_next];
	      root_next =  agbw->root_lists[iovl_next];
	    }
	    
	    ov = &(overlap_next[nov_next]);
	    ov->order = order;
	    for(ip=0;ip<order;ip++){
	      ov->parents[ip] = gatlist[ip];
	    }
	    ov->gs = gsij;
	    nov_next += 1;
	    
	    /* collect scaled volumes */
	    u = volpcoeff[order-1];
	    v = order*volpcoeff[order-1];
	    w = u*gvolp;
	    for(ii=0;ii<order;ii++){
	      kat = gatlist[ii];
	      volumep[kat] += w;
	    }
	    deltai = 0.0;
	    for(ii=0;ii<order;ii++){
	      deltai += ga[ii];
	    }
	    h3 = v*fp*3.0f/deltai;
	    h4 = 0.5f*v*fp;
	    for(ii=0;ii<order;ii++){
	      kat = gatlist[ii];
	      u1 = ga[ii]*gvol;
	      u2 = -2.*u1;
	      v0 = u2*(gx[ii][0]-gsij.c[0]);
	      v1 = u2*(gx[ii][1]-gsij.c[1]);
	      v2 = u2*(gx[ii][2]-gsij.c[2]);
	      d2 = v0*v0 + v1*v1 + v2*v2;
	      surf_area[kat] += (h3*u1 + h4*d2/u1)/gr[ii];
	    }

	  }
	    
	}//joverlap in root
	nroot_next += 1; //new root
      } //ioverlap in root
    } //iroot loop
    


    root_next[nroot_next] = nov_next; //termination of Verlet list
      
    agbw->n_overlap_lists[iovl_next] = nov_next;
    agbw->n_root_lists[iovl_next] = nroot_next;
    //  agbnp3_setup_second_order_overlap_buffer(agb,agbw,iovl);
    //agbnp3_print_overlap_buffer(order, agbw->n_overlap_lists[iovl_next], agbw->overlap_lists[iovl_next],  agbw->n_root_lists[iovl_next], agbw->root_lists[iovl_next]);
    
    // swap overlap buffers
    iovt = iovl_next;
    iovl_next = iovl;
    iovl = iovt;
    
    order += 1;
  }


  //fprintf(stderr,"agbnp3_self_volumes_rooti(): Number of gaussian overlaps: %d\n",_nov_);
  return AGBNP_OK;
}


int agbnp3_der_vp_rooti(AGBNPdata *agb, AGBworkdata *agbw,
			       float_a *x, float_a *y, float_a *z){


  NeighList *near_nl = agbw->near_nl;
  float_a *r = agb->r;
  float_a *galpha = agbw->galpha;
  float_a *gprefac = agbw->gprefac;
  float_a *vols = agbw->vols;
  float_a *q2ab = agbw->q2ab;
  float_a *abrw = agbw->abrw;
  float_a *deru = agbw->deru;
  float_a *derv = agbw->derv;
  float_a *derh = agbw->derh;
  float_a (*dgbdr)[3] = agbw->dgbdr_h;
  float_a (*dvwdr)[3] = agbw->dvwdr_h;
  float_a (*dehb)[3] = agbw->dehb;


  float_a dielectric_factor = 
    -0.5*(1./agb->dielectric_in - 1./agb->dielectric_out);
  float_a fourpi1 = 1./(4.*pi);

  /* coordinate buffer for Gaussian overlap calculation */
  float_a gx[AGBNP_MAX_OVERLAP_LEVEL][3];
  /* radius buffer for  Gaussian overlap calculation */
  float_a gr[AGBNP_MAX_OVERLAP_LEVEL];
  /* Gaussian parameters buffers for overlap calculation */
  float_a ga[AGBNP_MAX_OVERLAP_LEVEL];
  float_a gp[AGBNP_MAX_OVERLAP_LEVEL];
  /* derivatives buffers for Gaussian overlaps */
  float_a gdr[AGBNP_MAX_OVERLAP_LEVEL][3];
  float_a gdR[AGBNP_MAX_OVERLAP_LEVEL];
  /* holds the atom indexes being overlapped */
  int gatlist[AGBNP_MAX_OVERLAP_LEVEL];
  float_a gvol; /* gaussian overlap volume */
  int order; /* order of the overlap */
  /* holds overlap Gaussian parameters at each level */
  GParm gparams[AGBNP_MAX_OVERLAP_LEVEL];
  float_a an, pn, cn[3];
  /* coefficients for scaled volume */
  float_a volpcoeff[AGBNP_MAX_OVERLAP_LEVEL];
  float_a sign;

  int i, iat, jat, kat, ii,j;
  int ii1, ii2, iats, jats, ia, ja;  
  float_a sr,u,v,w,ur[3];

  float_a deruij,deruji,dervij,dervji,q;

  int natoms = agb->natoms;
  int nheavyat = agb->nheavyat;

  GParm *atm_gs = agbw->atm_gs;
  GParm *gsi, gsij, *gsj;

  int *root, *root_next;
  GOverlap *overlap; // current overlap list
  GOverlap *overlap_next; // next overlap list
  int iovl, iovl_next, iovt;; // switches between 0 and 1 for above
  int iroot, new_root;
  GOverlap *ov;
  int nov, nov_beg, nov_end, nov_next, nroot_next, ip;

  float *a1 = agbw->a1;
  float *p1 = agbw->p1;
  float *c1x = agbw->c1x;
  float *c1y = agbw->c1y;
  float *c1z = agbw->c1z;

  float *a2 = agbw->a2;
  float *p2 = agbw->p2;
  float *c2x = agbw->c2x;
  float *c2y = agbw->c2y;
  float *c2z = agbw->c2z;

  float *v3 = agbw->v3;
  float *v3p = agbw->v3p;
  float *fp3 = agbw->fp3;
  float *fpp3 = agbw->fpp3;

  float u1, u2, gvolp, fp, h;

  float volmina = AGBNP_MIN_VOLA;
  float volminb = AGBNP_MIN_VOLB;

  /* verbose = 1; */

  /* set scaled volume coefficients */
  sign = 1.0;
  for(order=1;order<=AGBNP_MAX_OVERLAP_LEVEL;order++){
    volpcoeff[order-1] = sign/((float_a)order);
    sign *= -1.0;
  }

  //order 1 overlap buffer
  iovl = 0;
  iovl_next = 1;
  order = 1;  

  //agbnp3_setup_first_order_overlap_buffer(agb,agbw,iovl);
  //agbnp3_print_overlap_buffer(order, agbw->n_overlap_lists[iovl], agbw->overlap_lists[iovl], 
  //			      agbw->n_root_lists[iovl], agbw->root_lists[iovl]);

  //constructs order=2 overlaps based on neighbor list
  nov_next = 0;
  nroot_next = 0;
  overlap_next = agbw->overlap_lists[iovl_next];
  root_next =  agbw->root_lists[iovl_next];
  order += 1;

  nov = 0;
  for(iat=0;iat<nheavyat;iat++){ //assume heavy atoms are on top

    gsi = &(atm_gs[iat]);
    gx[0][0] = gsi->c[0];
    gx[0][1] = gsi->c[1];
    gx[0][2] = gsi->c[2];
    ga[0] = gsi->a;
    gp[0] = gsi->p;
    gr[0] = agb->r[iat];

    gatlist[0] = iat;

    root_next[nroot_next] = nov_next; //starts a new root, maybe emtpty 

    for(j=0; j < near_nl->nne[iat]; j++, nov++){

      gvolp = v3p[nov]; //switched volume
      
      if(gvolp>FLT_MIN){
      //if(sr > SURF_MIN){
	jat = near_nl->neighl[iat][j];
	gatlist[1] = jat;
	gsj=&(atm_gs[jat]);
	gx[1][0] = gsj->c[0];
	gx[1][1] = gsj->c[1];
	gx[1][2] = gsj->c[2];
	ga[1] = gsj->a;
	gp[1] = gsj->p;
	gr[1] = agb->r[jat];

	gsij.a = a2[nov];
	gsij.p = p2[nov];
	gsij.c[0] = c2x[nov];
	gsij.c[1] = c2y[nov];
	gsij.c[2] = c2z[nov];
	gvol = v3[nov]; //raw volume
	fp = fp3[nov];

	/* new overlap */
	ov = &(overlap_next[nov_next]);
	ov->order = order;
	for(ip=0;ip<order;ip++){
	  ov->parents[ip] = gatlist[ip];
	}
	ov->gs = gsij;
	nov_next += 1;

	/* computes pair descreening corrections */
	u = 0.0;
	v = 0.0;
	h = 0.0;
	for(ii=0;ii<order;ii++){
	  kat = gatlist[ii];
	  u += deru[kat];
	  v += derv[kat];
	  h += derh[kat];
	}
	  
	u *= -dielectric_factor*fourpi1*volpcoeff[order-1];
	v *= -fourpi1*volpcoeff[order-1];
	h *= -volpcoeff[order-1];
	/* contribution to GB energy derivatives due to the change in scaled
	   volumes */
	for(ii=0;ii<order;ii++){
	  kat = gatlist[ii];
	  u1 = ga[ii]*gvol;
	  u2 = -2.*u1*fp;
	  ur[0] = u2*(gx[ii][0]-gsij.c[0]);
	  ur[1] = u2*(gx[ii][1]-gsij.c[1]);
	  ur[2] = u2*(gx[ii][2]-gsij.c[2]);
	  
	  dgbdr[kat][0] += u*ur[0];
	  dgbdr[kat][1] += u*ur[1];
	  dgbdr[kat][2] += u*ur[2];
	  
	  dvwdr[kat][0] += v*ur[0];
	  dvwdr[kat][1] += v*ur[1];
	  dvwdr[kat][2] += v*ur[2];

	  dehb[kat][0] += h*ur[0];
	  dehb[kat][1] += h*ur[1];
	  dehb[kat][2] += h*ur[2];

	  
	}

      }

    }

    nroot_next += 1; //new root
  }
  root_next[nroot_next] = nov_next; //termination of Verlet list

  agbw->n_overlap_lists[iovl_next] = nov_next;
  agbw->n_root_lists[iovl_next] = nroot_next;
  //  agbnp3_setup_second_order_overlap_buffer(agb,agbw,iovl);
  //agbnp3_print_overlap_buffer(order, agbw->n_overlap_lists[iovl_next], agbw->overlap_lists[iovl_next], 
  //			      agbw->n_root_lists[iovl_next], agbw->root_lists[iovl_next]);

  //now go through all the other orders swapping overlap buffers each time

  // swap overlap buffers, increment order
  iovt = iovl_next;
  iovl_next = iovl;
  iovl = iovt;
  order += 1;
  
  while(agbw->n_overlap_lists[iovl]>0 && order < AGBNP_MAX_OVERLAP_LEVEL){
    
    overlap = agbw->overlap_lists[iovl];
    root =  agbw->root_lists[iovl];
    
    nov_next = 0;
    nroot_next = 0;
    overlap_next = agbw->overlap_lists[iovl_next];
    root_next =  agbw->root_lists[iovl_next];

    for(iroot=0;iroot<agbw->n_root_lists[iovl];iroot++){
      
      //tests number of gaussians in this root
      if(root[iroot+1] - root[iroot] <= 0) continue;

      //fprintf(stderr,"ov: order= %d rootsize= %d\n", order, root[iroot+1] - root[iroot]);
      
      for(i=root[iroot];i<root[iroot+1]; i++){ //loop over overlaps
	
	//extract parents etc. of this overlap
	ov = &(overlap[i]);
	for(ip=0;ip<order-1;ip++){
	  iat = ov->parents[ip];
	  gatlist[ip] = iat;
	  gx[ip][0] = atm_gs[iat].c[0];
	  gx[ip][1] = atm_gs[iat].c[1];
	  gx[ip][2] = atm_gs[iat].c[2];
	  ga[ip] = atm_gs[iat].a;
	  gp[ip] = atm_gs[iat].p;
	  gr[ip] = agb->r[iat];
	}
	
	root_next[nroot_next] = nov_next; //starts a new root, maybe empty 
	
	for(j=i+1;j<root[iroot+1];j++, nov++){//gaussians in the same root

	  gvolp = v3p[nov]; //switched volume

	  if(gvolp>FLT_MIN){
	    //if(sr > SURF_MIN){
	    ov = &(overlap[j]);
	    jat = ov->parents[order-2];
	    ip = order-1;
	    gatlist[ip] = jat;
	    gx[ip][0] = atm_gs[jat].c[0];
	    gx[ip][1] = atm_gs[jat].c[1];
	    gx[ip][2] = atm_gs[jat].c[2];
	    ga[ip] = atm_gs[jat].a;
	    gp[ip] = atm_gs[jat].p;
	    gr[ip] = agb->r[jat];

	    gsij.a = a2[nov];
	    gsij.p = p2[nov];
	    gsij.c[0] = c2x[nov];
	    gsij.c[1] = c2y[nov];
	    gsij.c[2] = c2z[nov];
	    gvol = v3[nov]; //raw volume
	    fp = fp3[nov];

	    // constructs new overlap
	    ov = &(overlap_next[nov_next]);
	    ov->order = order;
	    for(ip=0;ip<order;ip++){
	      ov->parents[ip] = gatlist[ip];
	    }
	    ov->gs = gsij;
	    nov_next += 1;

	    /* computes pair descreening corrections */
	    u = 0.0;
	    v = 0.0;
	    h = 0.0;

	    /* these are standard coefficients */
	    for(ii=0;ii<order;ii++){
	      kat = gatlist[ii];
	      u += deru[kat];
	      v += derv[kat];
	      h += derh[kat];
	    }
	  
	    u *= -dielectric_factor*fourpi1*volpcoeff[order-1];
	    v *= -fourpi1*volpcoeff[order-1];
	    h *= -volpcoeff[order-1];
	    /* contribution to GB energy derivatives due to the change in scaled
	       volumes */
	    for(ii=0;ii<order;ii++){
	      kat = gatlist[ii];
	      u1 = ga[ii]*gvol;
	      u2 = -2.*u1*fp;
	      ur[0] = u2*(gx[ii][0]-gsij.c[0]);
	      ur[1] = u2*(gx[ii][1]-gsij.c[1]);
	      ur[2] = u2*(gx[ii][2]-gsij.c[2]);
	      
	      dgbdr[kat][0] += u*ur[0];
	      dgbdr[kat][1] += u*ur[1];
	      dgbdr[kat][2] += u*ur[2];
	      
	      dvwdr[kat][0] += v*ur[0];
	      dvwdr[kat][1] += v*ur[1];
	      dvwdr[kat][2] += v*ur[2];
	      
	      dehb[kat][0] += h*ur[0];
	      dehb[kat][1] += h*ur[1];
	      dehb[kat][2] += h*ur[2];

	    }
	    
	  }
	  
	}
	nroot_next += 1; //new root
      } //ioverlap in root
    } //iroot loop
    root_next[nroot_next] = nov_next; //termination of Verlet list
      
    agbw->n_overlap_lists[iovl_next] = nov_next;
    agbw->n_root_lists[iovl_next] = nroot_next;
    //  agbnp3_setup_second_order_overlap_buffer(agb,agbw,iovl);
    //agbnp3_print_overlap_buffer(order, agbw->n_overlap_lists[iovl_next], agbw->overlap_lists[iovl_next],  agbw->n_root_lists[iovl_next], agbw->root_lists[iovl_next]);
    
    // swap overlap buffers
    iovt = iovl_next;
    iovl_next = iovl;
    iovl = iovt;
    
    order += 1;
  }

  //for(iat=0;iat<nheavyat;iat++){
  //  printf("dehb(2): %d (%f, %f, %f)\n",iat,dehb[iat][0],dehb[iat][1],dehb[iat][2]);
  //}


  return AGBNP_OK;
}


int agbnp3_cavity_dersgb_rooti(AGBNPdata *agb, AGBworkdata *agbw,
			       float_a *x, float_a *y, float_a *z){


  int nheavyat = agb->nheavyat;
  NeighList *near_nl = agbw->near_nl;
  float_a *r = agb->r;
  float_a *galpha = agbw->galpha;
  float_a *gprefac = agbw->gprefac;
  float_a *gammap = agb->agbw->gammap;
  float_a *derus = agbw->derus;
  float_a *dervs = agbw->dervs;
  float_a *vols = agbw->vols;
  float_a *q2ab = agbw->q2ab;
  float_a *abrw = agbw->abrw;
  float_a *deru = agbw->deru;
  float_a *derv = agbw->derv;
  float_a (*dgbdr)[3] = agbw->dgbdr_h;
  float_a (*dvwdr)[3] = agbw->dvwdr_h;
  float_a (*decav)[3] = agbw->decav_h;

  float_a dielectric_factor = 
    -0.5*(1./agb->dielectric_in - 1./agb->dielectric_out);
  float_a fourpi1 = 1./(4.*pi);

  /* coordinate buffer for Gaussian overlap calculation */
  float_a gx[AGBNP_MAX_OVERLAP_LEVEL][3];
  /* radius buffer for  Gaussian overlap calculation */
  float_a gr[AGBNP_MAX_OVERLAP_LEVEL];
  /* Gaussian parameters buffers for overlap calculation */
  float_a ga[AGBNP_MAX_OVERLAP_LEVEL];
  float_a gp[AGBNP_MAX_OVERLAP_LEVEL];
  /* derivatives buffers for Gaussian overlaps */
  float_a gdr[AGBNP_MAX_OVERLAP_LEVEL][3];
  float_a gdR[AGBNP_MAX_OVERLAP_LEVEL];
  float_a gd2rR[AGBNP_MAX_OVERLAP_LEVEL][AGBNP_MAX_OVERLAP_LEVEL][3];
  /* holds the atom indexes being overlapped */
  int gnlist[AGBNP_MAX_OVERLAP_LEVEL];
  int gatlist[AGBNP_MAX_OVERLAP_LEVEL];
  float_a gvol; /* gaussian overlap volume */
  int order; /* order of the overlap */
  /* holds overlap Gaussian parameters at each level */
  GParm gparams[AGBNP_MAX_OVERLAP_LEVEL];
  float_a an, pn, cn[3];
  /* coefficients for scaled volume */
  float_a volpcoeff[AGBNP_MAX_OVERLAP_LEVEL];
  float_a sign;

  int i, iat, jat, kat, ii,j;
  int ii1, ii2, iats, jats, ia, ja;
  float_a sr,u,v,w,ur[3];
  int jj,lat;

  float_a deruij,deruji,dervij,dervji,q;

  int natoms = agb->natoms;
  GParm *atm_gs = agbw->atm_gs;
  GParm *gsi, gsij, *gsj;

  int *root, *root_next;
  GOverlap *overlap; // current overlap list
  GOverlap *overlap_next; // next overlap list
  int iovl, iovl_next, iovt;; // switches between 0 and 1 for above
  int iroot, new_root;
  GOverlap *ov;
  int nov, nov_next, nroot_next, ip;

  int _nov_ = 0;

  float *a1 = agbw->a1;
  float *p1 = agbw->p1;
  float *c1x = agbw->c1x;
  float *c1y = agbw->c1y;
  float *c1z = agbw->c1z;

  float *a2 = agbw->a2;
  float *p2 = agbw->p2;
  float *c2x = agbw->c2x;
  float *c2y = agbw->c2y;
  float *c2z = agbw->c2z;

  float *v3 = agbw->v3;
  float *v3p = agbw->v3p;
  float *fp3 = agbw->fp3;
  float *fpp3 = agbw->fpp3;

  float h3, h4, u1, u2, v0, v1, v2, gvolp, fp, fpp, deltai, d2;

  float volmina = AGBNP_MIN_VOLA;
  float volminb = AGBNP_MIN_VOLB;

  /* verbose = 1; */

  /* set scaled volume coefficients */
  sign = 1.0;
  for(order=1;order<=AGBNP_MAX_OVERLAP_LEVEL;order++){
    volpcoeff[order-1] = sign/((float_a)order);
    sign *= -1.0;
  }

  //order 1 overlap buffer
  iovl = 0;
  iovl_next = 1;
  order = 1;  

  //agbnp3_setup_first_order_overlap_buffer(agb,agbw,iovl);
  //agbnp3_print_overlap_buffer(order, agbw->n_overlap_lists[iovl], agbw->overlap_lists[iovl], 
  //			      agbw->n_root_lists[iovl], agbw->root_lists[iovl]);

  //constructs order=2 overlaps based on neighbor list
  nov_next = 0;
  nroot_next = 0;
  overlap_next = agbw->overlap_lists[iovl_next];
  root_next =  agbw->root_lists[iovl_next];
  order += 1;

  nov = 0;
  for(iat=0;iat<nheavyat;iat++){ //assume heavy atoms are on top

    gsi = &(atm_gs[iat]);
    gx[0][0] = gsi->c[0];
    gx[0][1] = gsi->c[1];
    gx[0][2] = gsi->c[2];
    ga[0] = gsi->a;
    gp[0] = gsi->p;
    gr[0] = agb->r[iat];

    gatlist[0] = iat;

    root_next[nroot_next] = nov_next; //starts a new root, maybe emtpty 

    for(j=0; j < near_nl->nne[iat]; j++, nov++){


      gvolp = v3p[nov]; //switched volume
      
      _nov_ += 1;
      
      if(gvolp>FLT_MIN){
      //if(sr > SURF_MIN){
	jat = near_nl->neighl[iat][j];
	gatlist[1] = jat;
	gsj=&(atm_gs[jat]);
	gx[1][0] = gsj->c[0];
	gx[1][1] = gsj->c[1];
	gx[1][2] = gsj->c[2];
	ga[1] = gsj->a;
	gp[1] = gsj->p;
	gr[1] = agb->r[jat];

	gsij.a = a2[nov];
	gsij.p = p2[nov];
	gsij.c[0] = c2x[nov];
	gsij.c[1] = c2y[nov];
	gsij.c[2] = c2z[nov];
	gvol = v3[nov]; //raw volume
	fp = fp3[nov];
	fpp = fpp3[nov];

	//constructs new overlap
	ov = &(overlap_next[nov_next]);
	ov->order = order;
	for(ip=0;ip<order;ip++){
	  ov->parents[ip] = gatlist[ip];
	}
	ov->gs = gsij;
	nov_next += 1;

	/* derivatives of cavity energy*/
	u = volpcoeff[order-1];
	v = order*volpcoeff[order-1];
	deltai = 0.0;
	for(ii=0;ii<order;ii++){
	  deltai += ga[ii];
	}
	deltai = 1.f/deltai;
	h3 = 3.0f*deltai;
	h4 = 0.5f;
	for(ii=0;ii<order;ii++){
	  u1 = ga[ii]*gvol;
	  u2 = -2.*u1;
	  v0 = gdr[ii][0] = u2*(gx[ii][0]-gsij.c[0]);
	  v1 = gdr[ii][1] = u2*(gx[ii][1]-gsij.c[1]);
	  v2 = gdr[ii][2] = u2*(gx[ii][2]-gsij.c[2]);
	  d2 = v0*v0 + v1*v1 + v2*v2;
	  gdR[ii] =  (h3*u1 + h4*d2/u1)/gr[ii];
	}
	/* diagonal terms */
	h3 = fpp+fp/gvol;
	for(ii=0;ii<order;ii++){
	  kat = gatlist[ii];
	  u1 = -2.f*fp*(1.f-ga[ii]*deltai)/gr[ii] + h3*gdR[ii];
	  ur[0] = u1*gdr[ii][0];
	  ur[1] = u1*gdr[ii][1];
	  ur[2] = u1*gdr[ii][2];

	  w = v*gammap[kat];
	  decav[kat][0] += w*ur[0];
	  decav[kat][1] += w*ur[1];
	  decav[kat][2] += w*ur[2];

	  w = v*dervs[kat];
	  dvwdr[kat][0] += w*ur[0];
	  dvwdr[kat][1] += w*ur[1];
	  dvwdr[kat][2] += w*ur[2];
	  
	  w = v*derus[kat];
	  dgbdr[kat][0] += w*ur[0];
	  dgbdr[kat][1] += w*ur[1];
	  dgbdr[kat][2] += w*ur[2];	  
	}
	/* off diagonal terms */
	h4 = 2.*fp*deltai;
	for(ii=0;ii<order;ii++){
	  kat = gatlist[ii];
	  
	  for(jj=0;jj<order;jj++){
	    if(jj==ii) continue;

	    lat = gatlist[jj];
	    
	    u1 = h4*ga[ii]/gr[jj];
	    u2 = h3*gdR[jj];
	    
	    ur[0] = u1*gdr[jj][0]+u2*gdr[ii][0];
	    ur[1] = u1*gdr[jj][1]+u2*gdr[ii][1];
	    ur[2] = u1*gdr[jj][2]+u2*gdr[ii][2];
	      
	    w = v*gammap[lat];
	    decav[kat][0] += w*ur[0];
	    decav[kat][1] += w*ur[1];
	    decav[kat][2] += w*ur[2];
 
	    w = v*dervs[lat];
	    dvwdr[kat][0] += w*ur[0];
	    dvwdr[kat][1] += w*ur[1];
	    dvwdr[kat][2] += w*ur[2];
	    
	    w = v*derus[lat];
	    dgbdr[kat][0] += w*ur[0];
	    dgbdr[kat][1] += w*ur[1];
	    dgbdr[kat][2] += w*ur[2];
	 }
	}

      }
    }
    nroot_next += 1; //new root
  }
  root_next[nroot_next] = nov_next; //termination of Verlet list

  agbw->n_overlap_lists[iovl_next] = nov_next;
  agbw->n_root_lists[iovl_next] = nroot_next;
  //  agbnp3_setup_second_order_overlap_buffer(agb,agbw,iovl);
  //agbnp3_print_overlap_buffer(order, agbw->n_overlap_lists[iovl_next], agbw->overlap_lists[iovl_next], 
  //			      agbw->n_root_lists[iovl_next], agbw->root_lists[iovl_next]);

  //now go through all the other orders swapping overlap buffers each time
  // swap overlap buffers, increment order
  iovt = iovl_next;
  iovl_next = iovl;
  iovl = iovt;
  order += 1;

  // go through all the other orders swapping overlap buffers each time
  // swap overlap buffers, increment order
  while(agbw->n_overlap_lists[iovl]>0 && order < AGBNP_MAX_OVERLAP_LEVEL){
    
    overlap = agbw->overlap_lists[iovl];
    root =  agbw->root_lists[iovl];
    
    nov_next = 0;
    nroot_next = 0;
    overlap_next = agbw->overlap_lists[iovl_next];
    root_next =  agbw->root_lists[iovl_next];
    

    for(iroot=0;iroot<agbw->n_root_lists[iovl];iroot++){
      
      //tests number of gaussians in this root
      if(root[iroot+1] - root[iroot] <= 0) continue;

      for(i=root[iroot];i<root[iroot+1]; i++){ //loop over overlaps
	
	//extract parents etc. of this overlap
	ov = &(overlap[i]);
	for(ip=0;ip<order-1;ip++){
	  iat = ov->parents[ip];
	  gatlist[ip] = iat;
	  gx[ip][0] = atm_gs[iat].c[0];
	  gx[ip][1] = atm_gs[iat].c[1];
	  gx[ip][2] = atm_gs[iat].c[2];
	  ga[ip] = atm_gs[iat].a;
	  gp[ip] = atm_gs[iat].p;
	  gr[ip] = agb->r[iat];
	}
	
	root_next[nroot_next] = nov_next; //starts a new root, maybe empty 
	
	for(j=i+1;j<root[iroot+1];j++,nov++){//gaussians in the same root

	  //	  printf("iroot = %d root[iroot] = %d root[iroot+1] = %d  i = %d  j = %d\n", iroot, root[iroot], root[iroot+1], i, j);

	  gvolp = v3p[nov]; //switched volume

	  _nov_ += 1;

	  if(gvolp>FLT_MIN){
	    //if(sr > SURF_MIN){
	    ov = &(overlap[j]);
	    jat = ov->parents[order-2];
	    ip = order-1;
	    gatlist[ip] = jat;
	    gx[ip][0] = atm_gs[jat].c[0];
	    gx[ip][1] = atm_gs[jat].c[1];
	    gx[ip][2] = atm_gs[jat].c[2];
	    ga[ip] = atm_gs[jat].a;
	    gp[ip] = atm_gs[jat].p;
	    gr[ip] = agb->r[jat];

	    gsij.a = a2[nov];
	    gsij.p = p2[nov];
	    gsij.c[0] = c2x[nov];
	    gsij.c[1] = c2y[nov];
	    gsij.c[2] = c2z[nov];
	    gvol = v3[nov]; //raw volume
	    fp = fp3[nov];
	    fpp = fpp3[nov];

	    // constructs new overlap
	    ov = &(overlap_next[nov_next]);
	    ov->order = order;
	    for(ip=0;ip<order;ip++){
	      ov->parents[ip] = gatlist[ip];
	    }
	    ov->gs = gsij;
	    nov_next += 1;

	    /* derivatives of cavity energy*/
	    u = volpcoeff[order-1];
	    v = order*volpcoeff[order-1];
	    deltai = 0.0;
	    for(ii=0;ii<order;ii++){
	      deltai += ga[ii];
	    }
	    deltai = 1.f/deltai;
	    //h3 = v*fp*3.0f*deltai;
	    //h4 = 0.5f*v*fp;
	    h3 = 3.0f*deltai;
	    h4 = 0.5f;
	    for(ii=0;ii<order;ii++){
	      u1 = ga[ii]*gvol;
	      u2 = -2.*u1;
	      v0 = gdr[ii][0] = u2*(gx[ii][0]-gsij.c[0]);
	      v1 = gdr[ii][1] = u2*(gx[ii][1]-gsij.c[1]);
	      v2 = gdr[ii][2] = u2*(gx[ii][2]-gsij.c[2]);
	      d2 = v0*v0 + v1*v1 + v2*v2;
	      gdR[ii] =  (h3*u1 + h4*d2/u1)/gr[ii];
	    }
	    /* diagonal terms */
	    h3 = fpp+fp/gvol;
	    for(ii=0;ii<order;ii++){
	      kat = gatlist[ii];
	      u1 = -2.*fp*(1.-ga[ii]*deltai)/gr[ii] + h3*gdR[ii];
	      ur[0] = u1*gdr[ii][0];
	      ur[1] = u1*gdr[ii][1];
	      ur[2] = u1*gdr[ii][2];
	      
	      w = v*gammap[kat];
	      decav[kat][0] += w*ur[0];
	      decav[kat][1] += w*ur[1];
	      decav[kat][2] += w*ur[2];
	      
	      w = v*dervs[kat];
	      dvwdr[kat][0] += w*ur[0];
	      dvwdr[kat][1] += w*ur[1];
	      dvwdr[kat][2] += w*ur[2];
	      
	      w = v*derus[kat];
	      dgbdr[kat][0] += w*ur[0];
	      dgbdr[kat][1] += w*ur[1];
	      dgbdr[kat][2] += w*ur[2];	  
	    }
	    /* off diagonal terms */
	    h4 = 2.*fp*deltai;
	    for(ii=0;ii<order;ii++){
	      kat = gatlist[ii];
	      
	      for(jj=0;jj<order;jj++){
		if(jj==ii) continue;
		
		lat = gatlist[jj];
		
		u1 = h4*ga[ii]/gr[jj];
		u2 = h3*gdR[jj];
		
		ur[0] = u1*gdr[jj][0]+u2*gdr[ii][0];
		ur[1] = u1*gdr[jj][1]+u2*gdr[ii][1];
		ur[2] = u1*gdr[jj][2]+u2*gdr[ii][2];
		
		w = v*gammap[lat];
		decav[kat][0] += w*ur[0];
		decav[kat][1] += w*ur[1];
		decav[kat][2] += w*ur[2];

		w = v*dervs[lat];
		dvwdr[kat][0] += w*ur[0];
		dvwdr[kat][1] += w*ur[1];
		dvwdr[kat][2] += w*ur[2];
		
		w = v*derus[lat];
		dgbdr[kat][0] += w*ur[0];
		dgbdr[kat][1] += w*ur[1];
		dgbdr[kat][2] += w*ur[2];
	      }
	    }
	    
	  }
	  
	}
	nroot_next += 1; //new root
      } //ioverlap in root
    } //iroot loop
    root_next[nroot_next] = nov_next; //termination of Verlet list
      
    agbw->n_overlap_lists[iovl_next] = nov_next;
    agbw->n_root_lists[iovl_next] = nroot_next;
    //  agbnp3_setup_second_order_overlap_buffer(agb,agbw,iovl);
    //agbnp3_print_overlap_buffer(order, agbw->n_overlap_lists[iovl_next], agbw->overlap_lists[iovl_next],  agbw->n_root_lists[iovl_next], agbw->root_lists[iovl_next]);
    
    // swap overlap buffers
    iovt = iovl_next;
    iovl_next = iovl;
    iovl = iovt;
    
    order += 1;
  }
  
  //for(iat=0;iat<agb->natoms;iat++){
  //  printf("decavf: %d %f %f %f\n",iat, decav[iat][0], decav[iat][1], decav[iat][2]);
  //}


  //fprintf(stderr,"agbnp3_cavity_dersgb_rooti(): Number of overlaps recorded: %d\n",_nov_);

  return AGBNP_OK;
}

/* Gaussian overlaps among n pairs of gaussians:
input:
 n1, n2: limits, n1 to n2 excluded
 c1x, etc.: center, exponent, prefactor of 1st gaussian
 c2x, etc.: center, exponent, prefactor of 2nd gaussian
 volmina, volminb: parameters for switching function.
output:
 gvol: raw overlap volume
 gvolp: switched overlap volume
 fp: derivative from switching function
*/
int agbnp3_ogauss_soa(int n1, int n2,
		      float *c1x, float *c1y, float *c1z, float *a1, float *p1,
		      float *c2x, float *c2y, float *c2z, float *a2, float *p2,
		      float volmina, float volminb,
		      float *gvol, float *gvolp, float *fp, float *fpp){

  float d2, dx, dy, dz, a12t, p12t;
  float deltai;
  int i;
  float s, sp, spp;
  float swu,swd,swu2,swu3,swf,swfp;

  //#pragma vector aligned
#pragma ivdep
  for (i=n1;i<n2;i++){

    dx = c2x[i] - c1x[i];
    dy = c2y[i] - c1y[i];
    dz = c2z[i] - c1z[i];
    d2 = dx*dx + dy*dy + dz*dz;

    a12t = a1[i] + a2[i];
    deltai = 1.f/(a12t);

    p12t = p1[i]*p2[i]*exp(-a1[i]*a2[i]*d2*deltai);
    gvol[i] = p12t*pow(PI*deltai,1.5f);
    
    c2x[i] = deltai*(a1[i]*c1x[i] + a2[i]*c2x[i]);
    c2y[i] = deltai*(a1[i]*c1y[i] + a2[i]*c2y[i]);
    c2z[i] = deltai*(a1[i]*c1z[i] + a2[i]*c2z[i]);
    
    a2[i] = a12t;
    p2[i] = p12t;

    //s = agbnp3_pol_switchfunc_ps(gvol_4[k4], volmina_4, volminb_4, &sp, &spp);    
    swf = 0.0f;
    swfp = 1.0f;
    if(gvol[i] > volminb) {
      swf = 1.0f;
      swfp = 0.0f;
    }else if(gvol[i] < volmina){
      swf = 0.0f;
      swfp = 0.0f;
    }
    swd = 1.f/(volminb - volmina);
    swu = (gvol[i] - volmina)*swd;
    swu2 = swu*swu;
    swu3 = swu*swu2;
    s = swf + swfp*swu3*(10.f-15.f*swu+6.f*swu2);
    sp = swfp*swd*30.f*swu2*(1.f - 2.f*swu + swu2);
    spp = swfp*swd*swd*60.f*swu*(1.f - 3.f*swu + 2.f*swu2);

    //gvolp[i] = agbnp3_swf_vol3(gvol[i], &(fp[i]), &(fpp[i]), volmina, volminb);
    gvolp[i] = s*gvol[i];
    fp[i] = s + gvol[i]*sp;
    fpp[i] = 2.f*sp + gvol[i]*spp;
  }

  return n2-n1;
}

#ifdef USE_SSE
_PS_CONST(2, 2.0f);
_PS_CONST(6, 6.0f);
_PS_CONST(10, 10.0f);
_PS_CONST(15, 15.0f);
_PS_CONST(30, 30.0f);
_PS_CONST(60, 60.0f);
/* goes smoothly from 0 at xa to 1 at xb */
__m128 agbnp3_pol_switchfunc_ps(__m128 x, __m128 xa, __m128 xb,
				__m128 *fp, __m128 *fpp){
  __m128 u,d,u2,u3,f;
  __m128 maska, maskb, maskab, masknb;
  __m128 one = *(__m128 *)_ps_1;
  __m128 two = *(__m128 *)_ps_2;
  __m128 three = *(__m128 *)_ps_3;
  __m128 six = *(__m128 *)_ps_6;
  __m128 ten = *(__m128 *)_ps_10;
  __m128 fifteen = *(__m128 *)_ps_15;
  __m128 thirty = *(__m128 *)_ps_30;
  __m128 sixty = *(__m128 *)_ps_60;
  
  /*
  if(x > xb) {
    if(fp) *fp = 0.0;
    if(fpp) *fpp = 0.0;
    return 1.0;
  }
  if(x < xa) {
    if(fp) *fp = 0.0;
    if(fpp) *fpp = 0.0;
    return 0.0;
  }
  // else ... polynomial function
  */

  maskb = _mm_cmple_ps(x, xb);
  maska = _mm_cmpgt_ps(x, xa);
  maskab = _mm_and_ps(maska, maskb);

  d = one/(xb - xa);
  u = (x - xa)*d;
  u2 = u*u;
  u3 = u*u2;

  f = _mm_and_ps(maskab, u3*(ten-fifteen*u+six*u2));
  f += _mm_andnot_ps(maskb,one);
  *fp = _mm_and_ps(maskab, d*thirty*u2*(one - two*u + u2));
  *fpp = _mm_and_ps(maskab, d*d*sixty*u*(one - three*u + two*u2));

  return f;
}
#endif

#ifdef USE_SSE
int agbnp3_ogauss_ps(int n1, int n2,
		      float *c1x, float *c1y, float *c1z, float *a1, float *p1,
		      float *c2x, float *c2y, float *c2z, float *a2, float *p2,
		      float volmina, float volminb,
		      float *gvol, float *gvolp, float *fp, float *fpp){

  float d2, dx, dy, dz, a12t, p12t;
  float deltai;
  int i, n2p, n1p, kl1, kt2, k0, k4;

  __m128 d2_4, dx_4, dy_4, dz_4, a12t_4, p12t_4;
  __m128 deltai_4, u_4;
  __m128 one = *(__m128 *)_ps_1;
  __m128 two = *(__m128 *)_ps_2;
  __m128 PI_4 = *(__m128*)_ps_PI;

  __m128 *c1x_4 =(__m128 *)c1x;
  __m128 *c1y_4 =(__m128 *)c1y;
  __m128 *c1z_4 =(__m128 *)c1z;
  __m128 *a1_4 =(__m128 *)a1;  
  __m128 *p1_4 =(__m128 *)p1;  
	                             
  __m128 *c2x_4 =(__m128 *)c2x;
  __m128 *c2y_4 =(__m128 *)c2y;
  __m128 *c2z_4 =(__m128 *)c2z;
  __m128 *a2_4 =(__m128 *)a2;  
  __m128 *p2_4 =(__m128 *)p2;  

  __m128 volmina_4 = _mm_set_ps1(volmina);
  __m128 volminb_4 = _mm_set_ps1(volminb);

  __m128 *gvol_4   =(__m128 *)gvol;   
  __m128 *gvolp_4  =(__m128 *)gvolp; 
  __m128 *fp_4     =(__m128 *)fp;	   
  __m128 *fpp_4    =(__m128 *)fpp;   

  __m128 s, sp, spp;

  kl1 = n1/4; /* starting quad */
  kt2 = n2/4; /* ending quad */

  if( kl1 == kt2 ){
    /* if n1 and n2 are in the same quad do only SOA loop */
    agbnp3_ogauss_soa(n1, n2, 
		      c1x, c1y, c1z, a1, p1,
		      c2x, c2y, c2z, a2, p2,
		      volmina, volminb,
		      gvol, gvolp, fp, fpp);
    return n2-n1;
  }

  n1p = (n1%4 == 0) ? n1 : (kl1+1)*4; // position at next leading quad
  /* leading term in first quad */
  agbnp3_ogauss_soa(n1, n1p, 
		    c1x, c1y, c1z, a1, p1,
		    c2x, c2y, c2z, a2, p2,
		    volmina, volminb,
		    gvol, gvolp, fp, fpp);

  k0 = n1p/4; // first full quad
  /* quads */
  for(k4=k0; k4 < kt2; k4++){

    dx_4 = c2x_4[k4] - c1x_4[k4];
    dy_4 = c2y_4[k4] - c1y_4[k4];
    dz_4 = c2z_4[k4] - c1z_4[k4];
    d2_4 = dx_4*dx_4 + dy_4*dy_4 + dz_4*dz_4;

    a12t_4 = a1_4[k4] + a2_4[k4];
    deltai_4 = one/(a12t_4);

    p12t_4 = p1_4[k4]*p2_4[k4]*exp_ps(-a1_4[k4]*a2_4[k4]*d2_4*deltai_4);
    u_4 = PI_4*deltai_4;
    gvol_4[k4] = p12t_4*u_4*u_4*rsqrt_ps(u_4);
    
    c2x_4[k4] = deltai_4*(a1_4[k4]*c1x_4[k4] + a2_4[k4]*c2x_4[k4]);
    c2y_4[k4] = deltai_4*(a1_4[k4]*c1y_4[k4] + a2_4[k4]*c2y_4[k4]);
    c2z_4[k4] = deltai_4*(a1_4[k4]*c1z_4[k4] + a2_4[k4]*c2z_4[k4]);
    
    a2_4[k4] = a12t_4;
    p2_4[k4] = p12t_4;

    //gvolp[i] = agbnp3_swf_vol3(gvol[i], &(fp[i]), &(fpp[i]), volmina, volminb)
    s = agbnp3_pol_switchfunc_ps(gvol_4[k4], volmina_4, volminb_4, &sp, &spp);
    gvolp_4[k4] = s*gvol_4[k4];
    fp_4[k4] = s + gvol_4[k4]*sp;
    fpp_4[k4] = two*sp + gvol_4[k4]*spp;

  }


  n2p = (n2%4 == 0) ? n2 : kt2*4; // position at beg. of last quad
  /* trailing terms in last quad */
  agbnp3_ogauss_soa(n2p, n2, 
		    c1x, c1y, c1z, a1, p1,
		    c2x, c2y, c2z, a2, p2,
		    volmina, volminb,
		    gvol, gvolp, fp, fpp);

  return n2-n1;
}
#endif


int agbnp3_ws_free_volumes_scalev_ps(AGBNPdata *agb, AGBworkdata *agbw){

  WSat *wsat;

  float *x = agb->x;
  float *y = agb->y;
  float *z = agb->z;

  /* coordinate buffer for Gaussian overlap calculation */
  float gx[AGBNP_MAX_OVERLAP_LEVEL][3];
  /* radius buffer for  Gaussian overlap calculation */
  float gr[AGBNP_MAX_OVERLAP_LEVEL];
  /* Gaussian parameters buffers for overlap calculation */
  float ga[AGBNP_MAX_OVERLAP_LEVEL];
  float gp[AGBNP_MAX_OVERLAP_LEVEL];
  /* derivatives buffers for Gaussian overlaps */
  float_a gdr[AGBNP_MAX_OVERLAP_LEVEL][3];
  float_a gdR[AGBNP_MAX_OVERLAP_LEVEL];
  /* holds overlap Gaussian parameters at each level */
  GParm gparams[AGBNP_MAX_OVERLAP_LEVEL];

  int natoms = agb->natoms;
  int nheavyat = agb->nheavyat;
  float *r = agb->r;  
  float *galpha = agbw->galpha;
  float *gprefac = agbw->gprefac;
  float *spe = agbw->spe;

  float* derh = agbw->derh;
  float* derh_m = agb->agbw->derh;
  float* vols = agbw->vols;

  float_a an, pn, cn[3], sr;
  float gvol; /* raw gaussian overlap volume */
  float gvolp; /* switched gaussian overlap volume */
  int order = 2; /* order of the overlap */
  const float_a kf_ws = KFC;
  const float_a pf_ws = PFC;
  int iws, iat;

  int *w_iov = agbw->w_iov; // pointer in "nov" buffers for each water site
  int *w_nov = agbw->w_nov; // number of overlaps for water site

  int nc, ic;
  int m;

  float ce = 0.0;
  float xa = AGBNP_HB_SWA;
  float xb = AGBNP_HB_SWB;
  float fp, s;
  float ehb = 0.0; 

  float (*dehb)[3] = agbw->dehb; // derivatives of WS energy

  int ip;
  float u, v, ur[3], jw[3];

  int nov;
  int _nov_ = 0;

  int i;

  int nadd;

  int *hiat = agbw->hiat;
  float *ha1 = agbw->ha1;
  float *hp1 = agbw->hp1;
  float *hc1x = agbw->hc1x;
  float *hc1y = agbw->hc1y;
  float *hc1z = agbw->hc1z;

  float *ha2 = agbw->ha2;
  float *hp2 = agbw->hp2;
  float *hc2x = agbw->hc2x;
  float *hc2y = agbw->hc2y;
  float *hc2z = agbw->hc2z;

  float *hv3 = agbw->hv3;
  float *hv3p = agbw->hv3p;
  float *hfp3 = agbw->hfp3;
  float *hfpp3 = agbw->hfpp3;

  int   *b_iatom =  agbw->wb_iatom;
  float *b_gvolv =  agbw->wb_gvolv;
  float *b_gderwx = agbw->wb_gderwx;
  float *b_gderwy = agbw->wb_gderwy;
  float *b_gderwz = agbw->wb_gderwz;
  float *b_gderix = agbw->wb_gderix;
  float *b_gderiy = agbw->wb_gderiy;
  float *b_gderiz = agbw->wb_gderiz;

  float volmina = AGBNP_MIN_VOLA;
  float volminb = AGBNP_MIN_VOLB;

  float u1,u2;
  float xw, yw, zw, aw, rw, dx, dy, dz, d2;
  float cutoff;
  float nboffset = AGBNP_NBOFFSET; 


  if(agbw->nwsat > agbw->wsize){
    agbw->wsize += agbw->nwsat;
    agbnp3_vfree(agbw->w_iov);
    agbnp3_vfree(agbw->w_nov);
    agbnp3_vcalloc((void **)&(agbw->w_iov), agbw->wsize*sizeof(int));
    agbnp3_vcalloc((void **)&(agbw->w_nov), agbw->wsize*sizeof(int));
    w_iov = agbw->w_iov;
    w_nov = agbw->w_nov;
  }

  nadd = agbw->nwsat*nheavyat;
  if(nadd > agbw->hbuffer_size){
      // reallocate overlap lists
    int new_size = nadd;
    if(agbnp3_reallocate_hbuffers(agbw, new_size) != AGBNP_OK){
      agbnp3_errprint("agbnp3_ws_free_volumes_scalev_ps(): Unable to expand Gaussian overlap buffers. Requested size = %d\n", new_size);
      return AGBNP_ERR;
    }else{
      hiat = agbw->hiat;
      ha1 = agbw->ha1;
      hp1 = agbw->hp1;
      hc1x = agbw->hc1x;
      hc1y = agbw->hc1y;
      hc1z = agbw->hc1z;
      ha2 = agbw->ha2;
      hp2 = agbw->hp2;
      hc2x = agbw->hc2x;
      hc2y = agbw->hc2y;
      hc2z = agbw->hc2z;
      hv3 = agbw->hv3;
      hv3p = agbw->hv3p;
      hfp3 = agbw->hfp3;
      hfpp3 = agbw->hfpp3;
    }
  }

  //phase1 collect interactions, place them in buffers 1 and 2
  nov = 0;
  for(iws = 0 ; iws<agbw->nwsat;iws++){ 

     wsat = &(agbw->wsat[iws]);
     xw = wsat->pos[0];
     yw = wsat->pos[1];
     zw = wsat->pos[2];
     rw = wsat->r;
     aw = kf_ws/(rw*rw);

     w_iov[iws] = nov;
     w_nov[iws] = 0;
     for(iat=0;iat<nheavyat;iat++){
       
       dx = x[iat] - xw;
       dy = y[iat] - yw;
       dz = z[iat] - zw;
       d2 = dx*dx + dy*dy + dz*dz;
       u = (r[iat]+rw)*nboffset;
       if(d2 < u*u){

	 hiat[nov] = iat;
	 /* buffer 1 */
	 ha1[nov] = aw;
	 hp1[nov] = pf_ws;
	 hc1x[nov] = xw;
	 hc1y[nov] = yw;
	 hc1z[nov] = zw;
	 /* buffer 2 */
	 ha2[nov] = galpha[iat];
	 hp2[nov] = gprefac[iat];
	 hc2x[nov] = x[iat];
	 hc2y[nov] = y[iat];
	 hc2z[nov] = z[iat];
	 
	 w_nov[iws] += 1;
	 nov += 1;
       }
     }
  }
  
  /* evaluate gaussian overlaps and derivatives */
#ifdef USE_SSE
  agbnp3_ogauss_ps(0, nov,
		   hc1x, hc1y, hc1z, ha1, hp1,
		   hc2x, hc2y, hc2z, ha2, hp2,
		   volmina, volminb,
		   hv3, hv3p, hfp3, hfpp3);
#else
  agbnp3_ogauss_soa(0, nov,
		    hc1x, hc1y, hc1z, ha1, hp1,
		    hc2x, hc2y, hc2z, ha2, hp2,
		    volmina, volminb,
		    hv3, hv3p, hfp3, hfpp3);
#endif

  //first pass, free volumes
  for(iws = 0 ; iws<agbw->nwsat;iws++){ 

    wsat = &(agbw->wsat[iws]);
    gx[0][0] = wsat->pos[0];
    gx[0][1] = wsat->pos[1];
    gx[0][2] = wsat->pos[2];
    ga[0] = kf_ws/(wsat->r*wsat->r);
    gp[0] = pf_ws;
    
    wsat->free_volume = wsat->volume;
    cutoff = AGBNP_HB_SWA0*wsat->volume;

    nc = 0;
    for(i=0;i<w_nov[iws];i++){

      if(wsat->free_volume > cutoff){

	nov = w_iov[iws] + i;

	gvolp = hv3p[nov];  //switched volume
	
	if(gvolp > 0.0f){
	  
	  wsat->free_volume -= spe[iat]*gvolp;
	  
	  gvol = hv3[nov]; //raw volume
	  fp = hfp3[nov];

	  iat = hiat[nov];
	  
	  gx[1][0] = x[iat];
	  gx[1][1] = y[iat];
	  gx[1][2] = z[iat];
	  ga[1] = galpha[iat];
	  
	  // dV12/dr w.r.t water site and real atom
	  
	  u1 = ga[0]*gvol;
	  u2 = -2.*u1*fp;
	  // c2x, etc now hold the coalescence center of the two gaussians
	  gdr[0][0] = u2*(gx[0][0]-hc2x[nov]);
	  gdr[0][1] = u2*(gx[0][1]-hc2y[nov]);
	  gdr[0][2] = u2*(gx[0][2]-hc2z[nov]);
	  
	  u1 = ga[1]*gvol;
	  u2 = -2.*u1*fp;
	  gdr[1][0] = u2*(gx[1][0]-hc2x[nov]);
	  gdr[1][1] = u2*(gx[1][1]-hc2y[nov]);
	  gdr[1][2] = u2*(gx[1][2]-hc2z[nov]);
	  
	  b_iatom[nc] = iat;
	  b_gvolv[nc] = gvolp;
	  b_gderwx[nc] = gdr[0][0];
	  b_gderwy[nc] = gdr[0][1];
	  b_gderwz[nc] = gdr[0][2];
	  b_gderix[nc] = gdr[1][0];
	  b_gderiy[nc] = gdr[1][1];
	  b_gderiz[nc] = gdr[1][2];
	  nc += 1;
		
	}
      }
    }

    wsat->sp = wsat->free_volume/wsat->volume;
    
    // contribution to WS energy
    s = agbnp3_pol_switchfunc(wsat->sp, xa, xb, &fp, NULL);
    wsat->dhw = wsat->khb*fp/wsat->volume;
    ehb += wsat->khb*s;

    //printf("ws(%d): %f %f %f\n", iws, wsat->khb*s, s, wsat->sp);

    // second pass:  
    // (w,s) derivatives, Jw's, Hi's
    jw[0] = 0.0;
    jw[1] = 0.0;
    jw[2] = 0.0;
    for(ic=0;ic<nc;ic++){
      iat = b_iatom[ic];
      // (w,s) contribution to derivatives
      u = -spe[iat]*wsat->dhw;
      dehb[iat][0] += u*b_gderix[ic];
      dehb[iat][1] += u*b_gderiy[ic];
      dehb[iat][2] += u*b_gderiz[ic];
      // Hi's
      u = wsat->dhw/vols[iat];
      derh[iat] += u*b_gvolv[ic];
      // Jw's
      v = -wsat->dhw*spe[iat];
      jw[0] += v*b_gderwx[ic];
      jw[1] += v*b_gderwy[ic];
      jw[2] += v*b_gderwz[ic];
    }
    
    //project Jw's forces onto real atoms
    for(ip=0;ip<wsat->nparents;ip++){
      iat=wsat->parent[ip];
      agbnp3_rtvec(ur,wsat->dpos[ip],jw);
      dehb[iat][0] += ur[0];
      dehb[iat][1] += ur[1];
      dehb[iat][2] += ur[2];
    } 
  }


#pragma omp critical
  {
    agb->ehb += ehb;
  }
#pragma omp critical
  for(iat=0;iat<nheavyat;iat++){
    derh_m[iat] += derh[iat];
  }
#pragma omp barrier
  memcpy(derh,derh_m,nheavyat*sizeof(float));

  //for(iat=0;iat<nheavyat;iat++){
  //  printf("dehb(1): %d (%f, %f, %f)\n",iat,dehb[iat][0],dehb[iat][1],dehb[iat][2]);
  //  }


  return AGBNP_OK;
}
