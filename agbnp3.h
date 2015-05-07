#ifndef AGBNP3_H
#define AGBNP3_H

#include "libnblist.h"

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

#ifndef FLOAT_I_H
#define FLOAT_I_H
#ifdef SINGLE_PREC_INTERFACE
typedef float float_i;
#else
typedef double float_i;
#endif
#endif

#ifndef AGBNP_OK
#define AGBNP_OK (2)
#endif
#ifndef AGBNP_ERR
#define AGBNP_ERR (-1)
#endif



/* Initializes libagbnp library.*/
int agbnp3_initialize( void );

/* Terminate libagbnp library. */
void agbnp3_terminate( void );

/* creates a new public instance of an agbnp structure */
int agbnp3_new(int *tag, int natoms, 
	      float_i *x, float_i *y, float_i *z, float_i *r, 
	      float_i *charge, float_i dielectric_in, float_i dielectric_out,
	      float_i *igamma, float_i *sgamma,
	      float_i *ialpha, float_i *salpha,
	      float_i *idelta, float_i *sdelta,
	      int *hbtype, float_i *hbcorr,
	      int nhydrogen, int *ihydrogen, 
	      int ndummy, int *idummy,
	      int *isfrozen,
	      int dopbc, int nsym, int ssize,
              float_i *xs, float_i *ys, float_i *zs,
	      float_i (*rot)[3][3], NeighList *conntbl,
	       float_i *vdiel_in, int verbose);

/* deletes a AGBNP object */
int agbnp3_delete(int tag);

/* returns AGBNP total energy and derivatives:
   generalized born, cavity, van der Waals, and hb correction energies */
int agbnp3_ener(int tag, int init,
		float_i *x, float_i *y, float_i *z,
		float_i *sp, float_i *br, 
		float_i *mol_volume, float_i *surf_area, 
		float_i *egb, float_i (*dgbdr)[3],
		float_i *evdw, float_i *ecorr_vdw, float_i (*dvwdr)[3], 
		float_i *ecav, float_i *ecorr_cav, float_i (*decav)[3],
		float_i *ehb,  float_i (*dehb)[3]);

/* return born radii and scaled radii only (no energies, no derivatives) */
int agbnp3_bornr(int tag, float_i *x, float_i *y, float_i *z,
		float_i *sp, float_i *br);

#endif
