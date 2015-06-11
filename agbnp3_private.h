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

#ifndef AGBNP3_PRIV_H
#define AGBNP3_PRIV_H

#include "agbnp3.h"

#ifdef USE_SSE
#include <xmmintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define SINGLE_PREC
#ifdef SINGLE_PREC
typedef float float_a;
#else
typedef double float_a;
#endif

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif 

#define mysqrt(x)   (sqrt(x))
#define mypow(x,y)  (pow(x,y))

#ifndef AGBNP_NEARNEIGHBORS
#define AGBNP_NEARNEIGHBORS (32)
#endif

#ifndef AGBNP_FARNEIGHBORS
#define AGBNP_FARNEIGHBORS (256)
#endif

/* average number of overlaps per (heavy) atom */
//#define AGBNP3_OVCACHE
#ifndef AGBNP_OVERLAPS
#define AGBNP_OVERLAPS (20)
#endif
#ifndef AGBNP_OVERLAPS_INCREMENT
#define AGBNP_OVERLAPS_INCREMENT (10)
#endif

// #define AGBNP3_SRM

#ifdef AGBNP3_SRM

#ifndef AGBNP_MIN_VOLA
#define AGBNP_MIN_VOLA (1.0)
#endif
#ifndef AGBNP_MIN_VOLB
#define AGBNP_MIN_VOLB (2.0)
#endif

#else

#ifndef AGBNP_MIN_VOLA
#define AGBNP_MIN_VOLA (0.01)
#endif
#ifndef AGBNP_MIN_VOLB
#define AGBNP_MIN_VOLB (0.1)
#endif

#endif


#define SURF_MIN (0.8*AGBNP_MIN_VOLA)

#ifndef AGBNP_MAX_OVERLAP_LEVEL
/* #define AGBNP_MAX_OVERLAP_LEVEL (32) */
#define AGBNP_MAX_OVERLAP_LEVEL (6)
#endif

#ifndef AGBNP_RADIUS_INCREMENT
#define AGBNP_RADIUS_INCREMENT (0.5)
#endif

#define pi (3.14159265359)

/* conversion factors from spheres to Gaussians */
#define KFC (2.5)
#define PFC (2.97354)


/* bond length for water sites pseudo-atoms */
#define AGBNP_HB_LENGTH (2.5) 

/* radius of water site pseudo-atom */
#define AGBNP_HB_RADIUS (1.4) 

/* a multiplicative offset to search for neighbors */
#define AGBNP_NBOFFSET (1.6)

/* cutoff for searching neighbors of a water site */
#define AGBNP_WS_CUTOFF (5.0)

/* lower and upper limits of switching function for HB correction */
#ifndef AGBNP_HB_SWA
#define AGBNP_HB_SWA (0.50)
#endif
#ifndef AGBNP_HB_SWB 
#define AGBNP_HB_SWB (0.85)
#endif
/* limit below which a water site is neglected */
#ifndef AGBNP_HB_SWA0
#define AGBNP_HB_SWA0 (0.4)
#endif

/* "jump" parameter of hash table */
//#define AGBNP_HT_JUMP (501)
#define AGBNP_HT_JUMP (1)

// #define HACK_MAXN (512)
#define MAX_OVERLAP_LIST (52000)

/* a structure that holds 3D Gaussian parameters */
typedef struct gparm_ {
  float_a a; /* Gaussian exponent */
  float_a p; /* Gaussian prefactor */
  float_a c[3]; /* center */
} GParm;

/* a structure that holds info for a gaussian overlap */
typedef struct goverlap_ {
  int order;
  int parents[AGBNP_MAX_OVERLAP_LEVEL];
  GParm gs;
} GOverlap;

/* a pseudo atom for water sites */
typedef struct wat_ {
  float_a pos[3]; /* position */
  float_a r;    /* radius */
  int type;
  int nparents; /* number of parents */
  int parent[4]; /* parents */
  int iseq;      /* sequence index for each ws for this parent */ 
  float_a dpos[4][3][3]; /* gradient of pos with respect to parents
			    dpos[0][1][2]: derivative of the y component of pos
			    with respect to the z coordinate of the first parent */
  float_a volume;   /* the volume of the ms sphere (4/3) pi r^3 */
  float_a sp;       /* free_volume/volume where the free volume calculated 
		      using only heavy atoms */
  float_a dhw;        /* used in the calculation of derivatives */
  float_a free_volume; /* the volume that doesn't overlap with real atoms */
  float_a self_volume; /* the free_volume minus overlaps with other 
			  water sites */
  float_a sps;         /* self_volume/volume */
  float_a khb;         /* HB correction factor */
  int nneigh, nlist_size, *nlist;    /* neighbor list for water site */  
} WSat;


typedef struct twofloats {
  float_a q4ij;
  float_a q4ji;
} TwoFloats;

typedef struct htable {
  unsigned int hsize; /* size of hash table (power of 2) */
  unsigned int hmask; /* hmask=size-1 */
  unsigned int hjump; /* used for trying new entries */
  unsigned int nat;   /* used to compute key=i*nat+j */
  int *key;           /* list of keys (less than 0 means unassigned) */
} HTable; /* hash table  */

/* a striped hash table in which each atom has its own htable */
typedef struct ghtable {
  unsigned int nat;
  HTable **ht;       /* list of 1D hash tables */
} GHTable;

/* 1D and 2D cubic splines lookup tables */
typedef struct c1table_  {
  int n;  // number of nodes
  float_a dx;  // spacing
  float_a dxinv; // inverse spacing
  float_a yinf;  // limiting value for largest x 
  float_a *y;    // cubic spline node values
  float_a *y2;   // cubic spline coefficients
} C1Table;


typedef struct c1table2d_  {
  int ny;           /* number of nodes */
  float_a dy;       /* spacing */
  float_a dyinv;    /* inverse of spacing */
  C1Table **table;  /* array of tables along x */
} C1Table2D;

/* a striped hash table in which each atom has its own htable */
typedef struct c1table2dh_ {
  unsigned int size;   /* number of look-up tables */
  int nkey;            /* y*nkey is the key */
  HTable *y2i;          /* hash table from y values to pointer into list of HTables */
  C1Table **table;     /* list of look-up tables */
} C1Table2DH;



typedef struct AGBworkdata_ {
  
  int natoms;
  float_a *vols; /* unscaled atomic volumes */
  float_a *volumep; /* scaled volumes */
  float_a *dera, *deru; /* used in the calculation of derivatives of 
			  AGB energy */
  float_a *q2ab; /* q^2+dera*b */
  float_a *derv; /* used in the calculation of the vdw part of
		   the np energy */
  float_a *abrw;   /* alpha*brw */
  float_a *br1_swf_der; /* derivative of inverse Born radius switching 
			  function */

  float_a *derh; /* used in the calculation of the gradients of the water 
		    site energy */

  float_a *psvol;         /* "p" paramter is p*A volume correction */

  float_a *derus, *dervs; /* "gamma" parameters for gb and vdw derivatives
			     related to surface area volume correction */
  int *isheavy;       /* 0 if hydrogen */
  int *nbiat; /* neighbors of atom iat */
  float_a *br1; /* inverse Born radii */
  float_a *br;  /* Born radii */
  float_a *brw; /* 3b^2/(b+rw)^4 for np derivative */
  float_a *alpha; /* ideal alpha + correction alpha */
  float_a *delta; /* ideal delta + correction delta */
  float_a *galpha; /* Gaussian exponents (= kfc/r[i]^2) */
  float_a *gprefac; /* Gaussian prefactors (= pfc) */

  GParm *atm_gs; // Gaussians representing each atom

  float_a *sp; /* scaling volume factors */
  float_a *spe; /* scaling volume factors w/o surface corrections */
  void **nbdata; /* a buffer to hold pointers to PBC translation vectors */

  int nq4cache;    /* i4() memory cache size */
  float  *q4cache; /* a memory cache to store i4() stuff */

  int nnl, nnlrc;    /* size of near_nl and far_nl neighbor lists */
  NeighList *near_nl; /* near (d<Ri+Rj) neighbor list for heavy atoms */
  NeighList *far_nl;  /* far Ri+Rj<d<cutoff neigh.list */

  float_a *dgbdrx;
  float_a *dgbdry;
  float_a *dgbdrz;

  float_a (*dgbdr_h)[3]; /* gradient of GB energy */
  float_a (*dvwdr_h)[3]; /* gradient of vdW energy */
  float_a (*dehb)[3];    /* gradient of HB energy */

  /* from cavity work data */
  float_a *surf_area; /* surface areas (unfiltered) */
  float_a *surf_area_f; /* filtered surface areas */
  float_a *gamma; /* ideal gamma + correction gamma */
  float_a *gammap; /* gamma corrected by derivative of surface area
			    switching function (for derivative calculation) */
  float_a (*decav_h)[3];

  int *nlist;  /* temporary arrays of length 'natoms' for sorting, etc. */
  int *js;
  NeighVector *pbcs;
  void **datas;

  int *nl_indx;   /* index buffer used in neigh. list reordering */ 
  float_a *nl_r2v;/* distance buffer used in neigh. list reordering */ 

  int nwsat;            /* number of water sites pseudo atoms */
  int wsat_size;        /* size of ws atom list (wsat) */
  WSat *wsat;           /* list of ws atoms in the system */

  /* list of overlaps and overlap roots for iterative volumetric algorithm */
  int size_overlap_lists[2]; //allocated size of overlap lists
  int n_overlap_lists[2]; // number of overlaps in overlap lists
  GOverlap *overlap_lists[2]; //overlap lists

  int size_root_lists[2]; //allocated size of root lists
  int n_root_lists[2]; // number of roots
  int *root_lists[2]; //roots, point to entries in overlap_lists

  /* buffers for vectorized Gaussian overlaps */
  int gbuffer_size;
  /* buffer 1 contains the overlap Gaussians (R,i) */
  float *a1, *p1, *c1x, *c1y, *c1z;
  /* input: buffer 2 contains the overlap Gaussians   (j) 
     output: contains the resulting overlap Gaussian (R,i,j) parameters */
  float *a2, *p2, *c2x, *c2y, *c2z;
  /* overlap volume of (R,i,j) */
  float *v3;  //raw volume
  float *v3p; //switched volume
  float *fp3; //derivative factor due to switching function
  float *fpp3; //derivative factor due to switching function

  /* buffers for vectorized Gaussian overlaps for water sites */
  int hbuffer_size;
  int *hiat; // real atom index
  /* buffer 1 contains the overlap Gaussians (R,i) */
  float *ha1, *hp1, *hc1x, *hc1y, *hc1z;
  /* input: buffer 2 contains the overlap Gaussians   (j) 
     output: contains the resulting overlap Gaussian (R,i,j) parameters */
  float *ha2, *hp2, *hc2x, *hc2y, *hc2z;
  /* overlap volume of (R,i,j) */
  float *hv3;  //raw volume
  float *hv3p; //switched volume
  float *hfp3; //derivative factor due to switching function
  float *hfpp3; //derivative factor due to switching function
  
  int qbuffer_size;
  float *qdv;
  float *qR1v;
  float *qR2v;
  float *qqv;
  float *qdqv;
  float *qav;
  float *qbv;
  float *qkv;
  float *qxh;
  float *qyp;
  float *qy;
  float *qy2p;
  float *qy2;
  float *qf1;
  float *qf2;
  float *qfp1;
  float *qfp2;



  int wbuffer_size;
  int   *wb_iatom;
  float *wb_gvolv;
  float *wb_gderwx;
  float *wb_gderwy;
  float *wb_gderwz;
  float *wb_gderix;
  float *wb_gderiy;
  float *wb_gderiz;

  int wsize;
  int *w_iov;
  int *w_nov;

  float dtv0;

  GHTable q4_btables;

} AGBworkdata;

typedef struct AGBNPdata_ {
  int in_use; /* TRUE if in use */

  int natoms; /* Numb. of atoms. Atoms are numbered from 0 to natoms-1 */

  float_a *x, *y, *z; /* coordinates */
  float_a *r;  /* atomic radii (input + offset) */
  float_a *charge; /* the partial charge of each atom */

  float_a *igamma, *sgamma; /* np parameters */
  float_a *ialpha, *salpha;
  float_a *idelta, *sdelta;

  int do_w;             /* turns on HB correction */
  int *hbtype;          /* HB atom type 0=inactive, 1=hydrogen donor,
			   10=sp acceptor, 20=sp2 acceptor, 30=sp3 acceptor */
  float_a *hbcorr;       /* HB to solvent correction strength */

  int nheavyat, *iheavyat; /* number of and list of heavy atoms */
  int nhydrogen, *ihydrogen; /* number of and list of hydrogen atoms */
  int ndummy, *idummy; /* number of and list of dummy atoms */

  int do_frozen; /* 1 if it should turn on optimizations relating to
		    frozen atoms */
  int *isfrozen; /* 1 if a frozen atom, 0 otherwise */

  float_a dielectric_in, dielectric_out; /* default dielectric constants */
  
  float_a *vdiel_in;

  NeighList *neigh_list; /* Verlet neighbor list (NULL if not using) */
  NeighList *excl_neigh_list; /* Verlet neighbor list of excluded neighbors
				 (NULL if not using) */

  int dopbc;            /* > 0 if doing PBC's */
  int nsym;             /* number of space group operations */
  int docryst;          /* > 0 if doing crystal PBC */
  int ssize;            /* size of coordinate buffers xs, ys, zs */
  float_i *xs, *ys, *zs; /* coordinates of symmetric images (for crystal
			   PBC's) */
  float_i (*rot)[3][3];  /* cell rotation matrices */

  NeighList *conntbl; /* atomic connection table */

  int *int2ext; /* mapping from internal to external indexes */
  int *ext2int; /* mapping from external to internal indexes */

  float_i *br; /* Born radii */
  float_i *sp; /* volume scaling factors */
  float_i *surf_area; /* surface areas */

  float_i (*dgbdr)[3]; /* gradient of GB energy */
  float_i (*dvwdr)[3]; /* gradient of vdW energy */
  float_i (*dehb)[3];    /* gradient of HB energy */
  float_i (*decav)[3]; /* gradient of CAVITY energy */

  float_a ehb; // water site energy


  AGBworkdata *agbw; /* AGB work data */
  
  int nprocs;  /* number of OPENMP threads (>1 if multithreading) */
  int maxprocs;  /* max number of OPENMP threads */
  AGBworkdata **agbw_p; /* store pointers to thread local copies of 
			   agb work data */
#ifdef _OPENMP
  omp_lock_t *omplock; /* OpenMP locks for multithreading, one for each atom */
#endif
  
  int verbose;

  C1Table2D *f4c1table2d;//lookup table for i4 function
  C1Table2DH *f4c1table2dh;//lookup table for i4 function
} AGBNPdata;




/*                                            *
 * macros                                     *
 *                                            */
#ifdef I4_FAR_MACRO
#define I4_FAR( rij, Ri, Rj) ( \
  _i4rij2 = rij*rij, \
  _i4u1 = rij+Rj,  \
  _i4u2 = rij-Rj,  \
  _i4u3 = _i4u1*_i4u2,  \
  _i4u4 = 0.5*log(_i4u1/_i4u2),  \
  _i4dr4 = _i4twopi*( (Rj/(rij*_i4u3))*(1. - 2.*_i4rij2/_i4u3 ) + _i4u4/_i4rij2 ), \
  _i4twopi*(Rj/_i4u3 - _i4u4/rij) )
#endif
#define I4_FAR( agb, rij, Ri, Rj) ( agbnp3_i4p((agb),(rij),(Ri),(Rj),&_i4dr4))
//#define AI4(agb, rij, Ri, Rj) ( agbnp3_i4((rij),(Ri),(Rj),&_i4dr4)) )
#define AI4(agb, rij, Ri, Rj) ( agbnp3_i4p((agb),(rij),(Ri),(Rj),&_i4dr4))



/*                                            *
 * function prototypes of local functions     *
 *                                            */

int agbnp3_reset(AGBNPdata *data);
int agbnp3_reset_agbworkdata(AGBworkdata *agbw);
int agbnp3_allocate_agbworkdata(int natoms, AGBNPdata *agb, AGBworkdata *agbw);
int agbnp3_delete_agbworkdata(AGBworkdata *agbw);
int agbnp3_init_agbworkdata(AGBNPdata *agbdata, AGBworkdata *agbw);
int agbnp3_tag_ok(int tag);

float_a agbnp3_i4(float_a rij, float_a Ri, float_a Rj, float_a *dr);
float_a agbnp3_i4p(AGBNPdata *data, float_a rij, float_a Ri, float_a Rj, float_a *dr);
float_a agbnp3_swf_area(float_a x, float_a *fp);
float_a agbnp3_swf_vol3(float_a x, float_a *fp, float_a *fpp, float_a a, float_a b);
int agbnp3_ogauss_soa(int n1, int n2,
		 float *c1x, float *c1y, float *c1z, float *a1, float *p1,
		 float *c2x, float *c2y, float *c2z, float *a2, float *p2,
		 float volmina, float volminb,
		      float *gvol, float *gvolp, float *fp, float *fpp);

int agbnp3_ogauss_ps(int n1, int n2,
		      float *c1x, float *c1y, float *c1z, float *a1, float *p1,
		      float *c2x, float *c2y, float *c2z, float *a2, float *p2,
		      float volmina, float volminb,
		     float *gvol, float *gvolp, float *fp, float *fpp);


float_a agbnp3_swf_invbr(float_a beta, float_a *fp);

int agbnp3_neighbor_lists(AGBNPdata *agb, AGBworkdata *agbw,
			 float_a *x, float_a *y, float_a *z);
int agbnp3_self_volumes_rooti(AGBNPdata *agb, AGBworkdata *agbw,
			      float_a *x, float_a *y, float_a *z);
int agbnp3_scaling_factors(AGBNPdata *agb, AGBworkdata *agbw_h);
int agbnp3_inverse_born_radii_nolist_soa(AGBNPdata *agb, AGBworkdata *agbw_h,
					float_a *x, float_a *y, float_a *z,
					int init_frozen);
 int agbnp3_born_radii(AGBNPdata *agb, AGBworkdata *agbw_h);
 int agbnp3_reset_derivatives(AGBNPdata *agb, AGBworkdata *agbw_h);

#ifdef USE_SSE
int agbnp3_gb_energy_inner_nolist_ps(
		    AGBNPdata *agb, int iat, int natoms, 
		    int jb, 
		    float *x, float *y, float *z,
		    float *charge, float *br, float *dera,
		    float *dgbdrx, float *dgbdry, float *dgbdrz, 
		    float *egb_pair, 
		    float dielectric_factor);
#endif
int agbnp3_gb_energy_inner_nolist_soa(
		    AGBNPdata *agb, int iat, int natoms, 
		    int jb, int je,
		    float *x, float *y, float *z,
		    float *charge, float *br, float *dera,
		    float *dgbdrx, float *dgbdry, float *dgbdrz, 
		    float *egb_pair, 
		    float dielectric_factor);

int agbnp3_gb_energy_nolist_ps(AGBNPdata *agb, AGBworkdata *agbw_h,
			       float *x, float *y, float *z,
			       float *egb_self, float *egb_pair);
 int agbnp3_gb_ders_constvp_nolist_ps(AGBNPdata *agb, AGBworkdata *agbw_h,
				 float_a *x, float_a *y, float_a *z,
				 int init_frozen);
 int agbnp3_der_vp_rooti(AGBNPdata *agb, AGBworkdata *agbw_h,
			 float_a *x, float_a *y, float_a *z);
 int agbnp3_gb_deruv(AGBNPdata *agb, AGBworkdata *agbw_h, 
			  int init_frozen);
 int agbnp3_gb_deruv_nolist_ps(AGBNPdata *agb, AGBworkdata *agbw_h, 
			       int init_frozen);
int agbnp3_total_energy(AGBNPdata *agb, int init,
		    float_i *mol_volume,
		    float_i *egb, 
		    float_i *evdw, float_i *ecorr_vdw, 
		    float_i *ecav, float_i *ecorr_cav, 
			float_i *ehb);

int agbnp3_surface_areas(AGBNPdata *agb, AGBworkdata *agbw,
			float_a *x, float_a *y, float_a *z);
float_a agbnp3_pol_switchfunc(float_a x, float_a xa, float_a xb,
			      float_a *fp, float_a *fpp);

int agbnp3_mymax(int a, int b);
void agbnp3_rtvec(float_a y[3], float_a rot[3][3], float_a x[3]);

int agbnp3_cpy_wsat(WSat *wsat1, WSat *wsat2);
int agbnp3_clr_wsat(WSat *wsat);
int agbnp3_create_wsatoms(AGBNPdata *agb, AGBworkdata *agbw);
int agbnp3_update_wsatoms(AGBNPdata *agb, AGBworkdata *agbw);
int agbnp3_create_ws_ofatom(AGBNPdata *agb, int iat, int *nws, WSat *twsatb);
int agbnp3_create_ws_atoms_ph(AGBNPdata *agb, int iat, 
			      int *nws, WSat *twsatb);
int agbnp3_update_ws_atoms_ph(AGBNPdata *agb, WSat *wsat);
void agbnp3_place_wat_hydrogen(float_a xd, float_a yd, float_a zd,
			      float_a xh, float_a yh, float_a zh, 
			      float_a d, 
			      float_a *xw, float_a *yw, float_a *zw,
			      float_a der1[3][3], float_a der2[3][3]);
void agbnp3_cpyd(float_a *local, float_i *caller, int n);


int agbnp3_create_ws_atoms_trigonal1(AGBNPdata *agb, int iat, 
				     int *nws, WSat *twsatb);
int agbnp3_update_ws_atoms_trigonal1(AGBNPdata *agb, WSat *wsat, WSat *wsat1, WSat *wsat2);
void agbnp3_place_wat_trigonal1(float_a xa, float_a ya, float_a za, 
				float_a xr, float_a yr, float_a zr, 
				float_a xr1, float_a yr1, float_a zr1, 
				float_a xr2, float_a yr2, float_a zr2,
				float_a d,
				float_a *xw1, float_a *yw1, float_a *zw1,
				float_a *xw2, float_a *yw2, float_a *zw2,
				float_a der1[4][3][3], float_a der2[4][3][3]);

int agbnp3_create_ws_atoms_trigonal_s(AGBNPdata *agb, int iat, 
				      int *nws, WSat *twsatb);
int agbnp3_update_ws_atoms_trigonal_s(AGBNPdata *agb, WSat *wsat, 
				      WSat *wsat1, WSat *wsat2,
				      WSat *wsat3, WSat *wsat4);
void agbnp3_place_wat_trigonal_s(float_a xa, float_a ya, float_a za, 
				 float_a xr, float_a yr, float_a zr, 
				 float_a xr1, float_a yr1, float_a zr1, 
				 float_a xr2, float_a yr2, float_a zr2,
				 float_a d,
				 float_a *xw1, float_a *yw1, float_a *zw1,
				 float_a *xw2, float_a *yw2, float_a *zw2,
				 float_a der1[4][3][3], float_a der2[4][3][3]);

int agbnp3_create_ws_atoms_trigonal_oop(AGBNPdata *agb, int iat, 
					int *nws, WSat *twsatb);
int agbnp3_update_ws_atoms_trigonal_oop(AGBNPdata *agb, WSat *wsat,
					WSat *wsat1, WSat *wsat2);
void agbnp3_place_wat_trigonal_oop(float_a xa, float_a ya, float_a za, 
				float_a xr1, float_a yr1, float_a zr1, 
				float_a xr2, float_a yr2, float_a zr2, 
				float_a xr3, float_a yr3, float_a zr3,
				float_a d1,
				float_a *xw1, float_a *yw1, float_a *zw1,
				float_a *xw2, float_a *yw2, float_a *zw2,
             		        float_a der1[4][3][3], float_a der2[4][3][3]);


int agbnp3_create_ws_atoms_trigonal2(AGBNPdata *agb, int iat, 
				     int *nws, WSat *twsatb);
int agbnp3_update_ws_atoms_trigonal2(AGBNPdata *agb, WSat *wsat);
void agbnp3_place_wat_trigonal2(float_a xa, float_a ya, float_a za, 
				float_a xr1, float_a yr1, float_a zr1, 
				float_a xr2, float_a yr2, float_a zr2,
				float_a d,
				float_a *xw, float_a *yw, float_a *zw,
				float_a der[3][3][3]);
int agbnp3_create_ws_atoms_tetrahedral2(AGBNPdata *agb, int iat, 
					int *nws,  WSat *twsatb);
int agbnp3_update_ws_atoms_tetrahedral2(AGBNPdata *agb, WSat *wsat, WSat *wsat1, WSat *wsat2);
void agbnp3_cross_product(float_a a[3], float_a b[3], float_a c[3],
			  float_a dera[3][3], float_a derb[3][3]);
void agbnp3_der_unitvector(float_a u[3], float_a invr, float_a der[3][3]);
void agbnp3_matmul(float_a a[3][3], float_a b[3][3], float_a c[3][3]);
void agbnp3_place_wat_tetrahedral2(float_a xa, float_a ya, float_a za, 
				   float_a xr1, float_a yr1, float_a zr1, 
				   float_a xr2, float_a yr2, float_a zr2,
				   float_a d,
				   float_a *xw1, float_a *yw1, float_a *zw1,
				   float_a *xw2, float_a *yw2, float_a *zw2,
				   float_a der1[3][3][3],float_a der2[3][3][3]);
int agbnp3_create_ws_atoms_tetrahedral3(AGBNPdata *agb, int iat, 
					int *nws, WSat *twsatb);
int agbnp3_update_ws_atoms_tetrahedral3(AGBNPdata *agb, WSat *wsat);
void agbnp3_place_wat_tetrahedral3(float_a xa, float_a ya, float_a za, 
				   float_a xr1, float_a yr1, float_a zr1, 
				   float_a xr2, float_a yr2, float_a zr2,
				   float_a xr3, float_a yr3, float_a zr3,
				   float_a d,
				   float_a *xw, float_a *yw, float_a *zw,
				   float_a der[4][3][3]);


int agbnp3_create_ws_atoms_tetrahedral1(AGBNPdata *agb, int iat, 
					int *nws, WSat *twsatb);
int agbnp3_update_ws_atoms_tetrahedral1(AGBNPdata *agb,  WSat *wsat);
void agbnp3_place_wat_tetrahedral1_one(float_a xa, float_a ya, float_a za, 
				       float_a xr, float_a yr, float_a zr, 
				       float_a xr1, float_a yr1, float_a zr1, 
				       float_a d,
				       float_a xw[3], 
				       float_a der[4][3][3]
				       );


/* functions to create and evaluate lookup tables for i4() */
#define F4LOOKUP_MAXA (20.0)
#define F4LOOKUP_SMOOTHA (10.0)
#define F4LOOKUP_NA (512)
#define F4LOOKUP_MAXB (8.0)
#define F4LOOKUP_NB (64)
int agbnp3_fill_ctable(int n, float_a *x, float_a *y, float_a *yp,
		      C1Table *c1table);
int agbnp3_create_ctablef4(int n, float_a amax, float_a b, 
			  C1Table **c1table);
int agbnp3_create_ctablef42d(AGBNPdata *agb,
			     int na, float_a amax, 
			     int nb, float_a bmax, 
			     C1Table2DH **table2d);
int agbnp3_interpolate_ctable(C1Table *c1table, float_a x, 
			     float_a *f, float_a *fp);
int agbnp3_interpolate_ctablef42d(C1Table2DH *table2d, float_a x, float_a y,
				 float_a *f, float_a *fp);
int agbnp3_init_i4p(AGBNPdata *agb);


int agbnp3_cavity_dersgb_rooti(AGBNPdata *agb, AGBworkdata *agbw,
			float_a *x, float_a *y, float_a *z);

unsigned int two2n_size(unsigned int m);
HTable *agbnp3_h_create(int nat, int size, int jump);
void agbnp3_h_delete(HTable *ht);
void agbnp3_h_init(HTable *ht);
int agbnp3_h_enter(HTable *ht, unsigned int key);
int agbnp3_h_find(HTable *ht, unsigned int key);

int agbnp3_fcompare( const void *val1, const void *val2 );
void agbnp3_fsortindx( int pnval, float_a *val, int *indx );
int agbnp3_nblist_reorder(AGBworkdata *agbw, NeighList *nl, int iat, int *indx);
int agbnp3_int_reorder(AGBworkdata *agbw, int n, int *nl, int *indx);
void agbnp3_errprint(const char *fmt, ...);

#define agbnp3_mymin(a,b) ((a) < (b) ? (a) : (b))

int agbnp3_vmemalloc(void **memptr, const size_t size);
int agbnp3_vcalloc(void **memptr, const size_t size);
int agbnp3_vralloc(void **memptr, const size_t old_size, const size_t new_size);
void agbnp3_vfree(void *x);




#ifdef USE_SSE
void print4(__m128 v);
#endif

void agbnp3_cspline_setup(float dx, int n, float* y, 
			 float yp1, float ypn, 
			  float* y2);
void agbnp3_cspline_interpolate(float x, float dx, int n, float* y, float* y2,
				float *f, float *fp);
void agbnp3_cspline_interpolate_soa(float *kv, float *xh, float dx, int m, 
				    float* yp, float *y,
				    float* y2p, float *y2,
				    float *f, float *fp);
#ifdef USE_SSE
void agbnp3_cspline_interpolate_ps(float *kv, float *xh, float dx, int m, 
				   float* yp, float *y,
				   float* y2p, float *y2,
				   float *f, float *fp);
#endif
int agbnp3_interpolate_ctablef42d_soa(C1Table2DH *table2d, float *x, float *ym, 
			 int m, float *f, float *fp,
			 float *kv, float *xh, float *yp, float *y, float *y2p, float *y2,
                         float *f1, float *f2, float *fp1, float *fp2);

int agbnp3_i4p_soa(AGBNPdata *agb, float* rij, float *Ri, float *Rj, 
		   int m, float *f, float *fp,
		   float *a, float *b,
		   float *qkv, float *qxh, float *qyp, float *qy, float *qy2p, float *qy2,
		  float *qf1, float *qf2, float *qfp1, float *qfp2);


#ifdef USE_SSE
int agbnp3_i4p_ps(AGBNPdata *agb, float* rij, float *Ri, float *Rj, 
		  int m, float *f, float *fp,
		  float *a, float *b,
		  float *qkv, float *qxh, float *qyp, float *qy, float *qy2p, float *qy2,
		  float *qf1, float *qf2, float *qfp1, float *qfp2);
#endif

void agbnp3_test_cspline(void);

int agbnp3_ws_free_volumes_scalev_ps(AGBNPdata *agb, AGBworkdata *agbw);

int agbnp3_reset_buffers(AGBNPdata *agb, AGBworkdata *agbw_h);

/* a macro to calculate 3 b^2/(b+r)^4 */
#define AGBNP_BRW(b,r) ( _agbnp3_brw1 = (b) + (r) , _agbnp3_brw2 = _agbnp3_brw1 * _agbnp3_brw1 , _agbnp3_brw3 = _agbnp3_brw2 * _agbnp3_brw2 ,  3.*((b)*(b))/_agbnp3_brw3 )

/* get the number of "radius types" */
int agbnp3_list_radius_types(AGBNPdata *agb, float **radii);
int agbnp3_create_ctablef42d_hash(AGBNPdata *agb, int na, float_a amax, 
				  C1Table2DH **table2d);
int agbnp3_test_create_ctablef42d_hash(AGBNPdata *agb, float amax, C1Table2DH *table2d);

int agbnp3_reallocate_gbuffers(AGBworkdata *agbw, int size);
int agbnp3_reallocate_hbuffers(AGBworkdata *agbw, int size);
int agbnp3_reallocate_qbuffers(AGBworkdata *agbw, int size);
int agbnp3_reallocate_wbuffers(AGBworkdata *agbw, int size);
int agbnp3_reallocate_overlap_lists(AGBworkdata *agbw, int size);

#endif
