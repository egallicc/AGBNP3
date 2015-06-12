
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
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

//#define ATIMER
#ifdef ATIMER
#include <time.h>
#endif

#include "agbnp3.h"
#include "nblist.h"
#include "agbnp3_private.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* static int verbose = 0; */


/*                                                     *
 * Data structures that holds list of AGBNP structures *
 *                                                     */
static int agbnp3_initialized = FALSE;
/* a pointer to all the agbnp strctures which have been allocated */
static AGBNPdata *agbdata3_list = NULL;
/* the number of allocated agbnp structures */
int agbdata3_allocated = 0;
/* the number of atom lists currently in use */
int agbdata3_used = 0;
/* the initial number of agbnp structures allocated */
static const int AGBDATA_INITIAL_NUMBER = 1;
/* the increment of agbnp structures slots for reallocation */
static const int AGBDATA_INCREMENT = 1;

/* Initializes libagbnp library.*/
int agbnp3_initialize( void ){
  int i;

  if(agbnp3_initialized){
    return AGBNP_OK;
  }

  /* creates list of pointers of agbnp structures managed by library */
  agbdata3_list = (AGBNPdata * ) 
    calloc(AGBDATA_INITIAL_NUMBER, sizeof(AGBNPdata) );
  if(!agbdata3_list) {
    agbnp3_errprint("agbnp3_initialize(): error allocating memory for %d agbnp objects.\n", AGBDATA_INITIAL_NUMBER);
     return AGBNP_ERR;
  }
  agbdata3_allocated = AGBDATA_INITIAL_NUMBER;
  agbdata3_used = 0; 

  /* reset lists values */
  for(i=0; i<agbdata3_allocated; i++){
    /* set in_use=0 among other things */
    agbnp3_reset(&(agbdata3_list[i]));
  }

  agbnp3_initialized = TRUE;
  
  return AGBNP_OK;
}

/* Terminate libagbnp library. */
void agbnp3_terminate( void ){
  int i;

  if(!agbnp3_initialized){
    agbnp3_errprint("agbnp3_terminate(): agbnp library is not initialized.\n");
    return;
  }

  /* deletes allocated structures */
  for(i = 0; i<agbdata3_allocated ; i++){
    if(agbdata3_list[i].in_use){
      if(agbnp3_delete(i) != AGBNP_OK){
	agbnp3_errprint( "agbnp3_terminate(): error in deleting agbnp structure %d.\n", i);
      }
    }
  }

  /* free list of allocated lists */
  free(agbdata3_list);
  agbdata3_list = NULL;
  agbdata3_allocated = 0;
  agbdata3_used = 0;
}

/* creates a new public instance of an agbnp structure */
int agbnp3_new(int *tag, int natoms, 
	      float_i *x, float_i *y, float_i *z, float_i *r, 
	      float_i *charge, float_i dielectric_in, float_i dielectric_out,
	      float_i *igamma, float_i *sgamma,
	      float_i *ialpha, float_i *salpha,
	      int *hbtype, float_i *hbcorr,
	      int nhydrogen, int *ihydrogen, 
	      NeighList *conntbl, int verbose){

  int slot, j, il, i, iat, indx;
  int *iswhat;
  int *int2ext, *ext2int;
  AGBNPdata *agbdata;
#ifdef _OPENMP
  int iproc, error = 0;
#endif

  if(!agbnp3_initialized){
    agbnp3_errprint("agbnp3_new(): agbnp library is not initialized.\n");
    return AGBNP_ERR;
  }

  /* find an allocated structure not in use */
  slot = 0;
  while(agbdata3_list[slot].in_use){

    if(slot+1 >= agbdata3_allocated){
      /* can't find it, expand list */
      agbdata3_list = (AGBNPdata * ) 
	realloc(agbdata3_list, 
	   (agbdata3_allocated+AGBDATA_INCREMENT)*sizeof(AGBNPdata) );
      if(!agbdata3_list){
	agbnp3_errprint("agbnp3_new(): error reallocating memory for %d agbnp objects.\n", agbdata3_allocated+AGBDATA_INCREMENT);
	return AGBNP_ERR;
      }
      /* reset new lists */
      for(il=0; il < AGBDATA_INCREMENT ; il++){
	agbnp3_reset(&(agbdata3_list[agbdata3_allocated+il]));
      }
      /* increase number of allocated lists */
      agbdata3_allocated += AGBDATA_INCREMENT;
    }

    /* next slot */
    slot += 1;
  }

  /* this is the index of the available structure */
  *tag = slot;
  agbdata = &(agbdata3_list[slot]);

  /* reset new structure */
  agbnp3_reset(agbdata);

  /* set natoms */
  agbdata->natoms = natoms;

  /* sets int2ext and ext2int atomic index mapping arrays */
  /* atomic indexes mapping */
  agbnp3_vcalloc((void **)&(agbdata->int2ext), natoms*sizeof(int));
  agbnp3_vcalloc((void **)&(agbdata->ext2int), natoms*sizeof(int));
  agbnp3_atom_reorder(agbdata, nhydrogen, ihydrogen); 
  int2ext = agbdata->int2ext;
  ext2int = agbdata->ext2int;

  /* allocates and set coordinates */
  agbnp3_vcalloc((void **)&(agbdata->x), natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbdata->y), natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbdata->z), natoms*sizeof(float_a));
  if(!(agbdata->x && agbdata->y && agbdata->z)){
    agbnp3_errprint("agbnp3_new(): error allocating memory for coordinates (%d doubles)\n",3*natoms);
    return AGBNP_ERR;
  }
  for(iat = 0; iat < natoms; iat++){
    agbdata->x[iat] = x[int2ext[iat]];
    agbdata->y[iat] = y[int2ext[iat]];
    agbdata->z[iat] = z[int2ext[iat]];
  }

  /* verbose level */
  agbdata->verbose = verbose;

  /* allocates and set radii */
  agbnp3_vcalloc((void **)&(agbdata->r),natoms*sizeof(float_a));
  if(!(agbdata->r)){
     agbnp3_errprint("agbnp3_new(): error allocating memory for atomic radii (%d doubles)\n",2*natoms);
    return AGBNP_ERR;
  }
  for(iat = 0; iat < natoms; iat++){
    agbdata->r[iat] = r[int2ext[iat]] + AGBNP_RADIUS_INCREMENT;
  }

  /* allocates and set charges */
  agbnp3_vcalloc((void **)&(agbdata->charge), natoms*sizeof(float_a));
  if(!agbdata->charge){
     agbnp3_errprint("agbnp3_new(): error allocating memory for atomic partial charges (%d doubles)\n",natoms);
    return AGBNP_ERR;
  }
  for(iat = 0; iat < natoms; iat++){
    agbdata->charge[iat] = charge[int2ext[iat]];
  }

  /* allocates and set np parameters */
  agbnp3_vcalloc((void **)&(agbdata->igamma), natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbdata->sgamma), natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbdata->ialpha), natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbdata->salpha), natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbdata->idelta), natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbdata->sdelta), natoms*sizeof(float_a));
  if(!(agbdata->igamma && agbdata->sgamma && agbdata->ialpha &&
       agbdata->salpha && agbdata->idelta && agbdata->sdelta)){
    agbnp3_errprint("agbnp3_new(): error allocating memory for non-polar parameters (%d doubles)\n",6*natoms);
    return AGBNP_ERR;
  }

  agbnp3_vcalloc((void **)&(agbdata->hbtype), natoms*sizeof(int));
  agbnp3_vcalloc((void **)&(agbdata->hbcorr), natoms*sizeof(float_a));
  if(!(agbdata->hbtype && agbdata->hbcorr)){
    agbnp3_errprint("agbnp3_new(): error allocating memory for non-polar parameters\n");
    return AGBNP_ERR;
  }

  for(iat=0;iat<natoms;iat++){
    agbdata->igamma[iat] = igamma[int2ext[iat]];
    agbdata->sgamma[iat] = sgamma[int2ext[iat]];
    agbdata->ialpha[iat] = ialpha[int2ext[iat]];
    agbdata->salpha[iat] = salpha[int2ext[iat]];
    agbdata->idelta[iat] = 0.0;
    agbdata->sdelta[iat] = 0.0;
    agbdata->hbtype[iat] = hbtype[int2ext[iat]];
    agbdata->hbcorr[iat] = hbcorr[int2ext[iat]];
  }

  /* allocates and set list of hydrogens, heavy atoms and dummy atoms */
  /* iswhat: 1 = heavy atom, 2 = hydrogen,  3 = dummy */
  agbnp3_vcalloc((void **)&(iswhat), natoms*sizeof(int));
  agbnp3_vcalloc((void **)&(agbdata->iheavyat), natoms*sizeof(int));
  agbnp3_vcalloc((void **)&(agbdata->ihydrogen), natoms*sizeof(int));
  agbnp3_vcalloc((void **)&(agbdata->idummy), natoms*sizeof(int));
  if(!(iswhat && agbdata->iheavyat && agbdata->ihydrogen && agbdata->idummy)){
    agbnp3_errprint("agbnp3_new(): error allocating memory for lists of hydrogens, heavy atoms and dummy atoms (%d integers)\n",4*natoms);
    return AGBNP_ERR;
  }

  for(i=0;i<nhydrogen;i++){
    iat = ext2int[ihydrogen[i]];
    iswhat[iat] = 2;
    agbdata->ihydrogen[i] = iat;
  }
  agbdata->nhydrogen = nhydrogen;

  agbdata->ndummy = 0;

  /* an atom that is not an hydrogen atom or a dummy atom is a heavy atom */
  agbdata->nheavyat = 0;
  for(iat=0;iat<natoms;iat++){
    if(!iswhat[iat]){
      agbdata->iheavyat[agbdata->nheavyat++] = iat;
    }
  }

  agbnp3_vfree(iswhat);

  /* set dielectric contstants */
  agbdata->dielectric_in = dielectric_in;
  agbdata->dielectric_out = dielectric_out;

  /* no neighbor list in agbnp3 for now */
  agbdata->neigh_list = NULL;
  agbdata->excl_neigh_list = NULL;

  /* copy connection table to libagbnp internal structure */
  if(!conntbl){
    agbnp3_errprint( "agbnp3_new(): error: NULL connection table. AGBNP v.3 requires a connection table.\n");
    return AGBNP_ERR;
  }
  agbdata->do_w = 1;
  agbdata->conntbl = (NeighList *)calloc(1, sizeof(NeighList));
  nblist_reset_neighbor_list(agbdata->conntbl);
  if(nblist_reallocate_neighbor_list(agbdata->conntbl,conntbl->natoms,
				     conntbl->neighl_size) != NBLIST_OK){
    agbnp3_errprint( "agbnp3_new(): unable to allocate connection table (size = %d ints).\n",conntbl->neighl_size );
    return AGBNP_ERR;
  }
  /* copy connection table "neighbor" list */
  indx = 0;
  for(iat=0; iat < natoms; iat++){
    agbdata->conntbl->nne[iat] = conntbl->nne[int2ext[iat]];
    agbdata->conntbl->neighl[iat] = &(agbdata->conntbl->neighl1[indx]);
    for(j=0 ; j<agbdata->conntbl->nne[iat] ; j++){
      agbdata->conntbl->neighl[iat][j] = ext2int[conntbl->neighl[int2ext[iat]][j]];
    }
    indx += agbdata->conntbl->nne[iat];
  }
  
#ifdef AGBNP_WRITE_DATA
  printf("%d\n",natoms);
  /*  printf("i x y z  r rcav igamma sgamma ialpha salpha idelta sdelta ab hbtype hbcorr\n"); */
  for(iat=0;iat<natoms;iat++){
    printf("%d %f %f %f %f %f %f %f %f %f %f %f %d %f\n",iat,
	   agbdata->x[iat],agbdata->y[iat],agbdata->z[iat],
	   agbdata->r[iat],
	   agbdata->charge[iat], 
	   agbdata->igamma[iat],agbdata->sgamma[iat],
	   agbdata->ialpha[iat], agbdata->salpha[iat], 
	   agbdata->idelta[iat] , agbdata->sdelta[iat],  
	   agbdata->hbtype[iat], agbdata->hbcorr[iat]);
  }
  printf("%d\n",nhydrogen);
  for(i=0;i<nhydrogen;i++){
    printf("%d\n",ihydrogen[i]);
  }
#endif

  /* allocates Born radii, etc, buffers */
  agbnp3_vcalloc((void **)&(agbdata->br),natoms*sizeof(float_i));
  agbnp3_vcalloc((void **)&(agbdata->sp),natoms*sizeof(float_i));
  agbnp3_vcalloc((void **)&(agbdata->surf_area),natoms*sizeof(float_i));

  /* allocates gradient buffers */
  agbnp3_vcalloc((void **)&(agbdata->dgbdr),natoms*sizeof(float_i [3]));
  agbnp3_vcalloc((void **)&(agbdata->dvwdr),natoms*sizeof(float_i [3]));
  agbnp3_vcalloc((void **)&(agbdata->dehb),natoms*sizeof(float_i [3]));
  agbnp3_vcalloc((void **)&(agbdata->decav),natoms*sizeof(float_i [3]));

  /* initializes lookup table version of i4 */
  if(agbnp3_init_i4p(agbdata) != AGBNP_OK){
    agbnp3_errprint("agbnp3_initialize(): error in agbnp3_init_i4p()\n");
    return AGBNP_ERR;
  }

  /* allocates work arrays */
  agbdata->agbw = (AGBworkdata *)calloc(1, sizeof(AGBworkdata));
  if(!agbdata->agbw){
    agbnp3_errprint("agbnp3_new(): error allocating memory for AGB work data structure.\n");
    return AGBNP_ERR;
  }
  agbnp3_reset_agbworkdata(agbdata->agbw);
  if(agbnp3_allocate_agbworkdata(natoms, agbdata, agbdata->agbw) != AGBNP_OK){
    agbnp3_errprint("agbnp3_new(): error in agbnp3_allocate_agbworkdata()\n");
    return AGBNP_ERR;
  }
  agbnp3_init_agbworkdata(agbdata,agbdata->agbw);

#ifdef _OPENMP
  error = 0;
  /* here each thread allocates its own work space. Apparently a better
     strategy than having one thread do it for all 
     (see http://stephen-tu.blogspot.com/2013/04/on-importance-of-numa-aware-memory.html) */
#pragma omp parallel private(iproc)
  {
    iproc = omp_get_thread_num();
#pragma omp single
    {
      agbdata->nprocs = omp_get_num_threads();    
      printf("\n agbnp3_new(): info: using %5d OpenMP thread(s).\n\n",
	     agbdata->nprocs);  
      /* array of pointers to thread memory work spaces */ 
      agbdata->agbw_p = (AGBworkdata **)calloc(agbdata->nprocs, sizeof(AGBworkdata *));
      /* creates and initializes atomic locks */
      agbdata->omplock = (omp_lock_t *)malloc(agbdata->natoms*sizeof(omp_lock_t));
      for(iat=0;iat<natoms;iat++){
	omp_init_lock(&(agbdata->omplock[iat]));
      }
    }
#pragma omp critical
    {
      /* allocates and initializes work space for this thread */
      agbdata->agbw_p[iproc] = (AGBworkdata *)calloc(1, sizeof(AGBworkdata));
      if(!agbdata->agbw_p[iproc]){
	agbnp3_errprint("agbnp3_new(): error allocating memory for AGB work data structure.\n");
	error = 1;
      }
      agbnp3_reset_agbworkdata(agbdata->agbw_p[iproc]);
      if(agbnp3_allocate_agbworkdata(natoms,agbdata,agbdata->agbw_p[iproc]) != AGBNP_OK){
	agbnp3_errprint("agbnp3_new(): error in agbnp3_allocate_agbworkdata()\n");
	error = 1;
      }
      agbnp3_init_agbworkdata(agbdata,agbdata->agbw_p[iproc]);
    }
  }
#pragma omp barrier
  /* return if any of the above went wrong */
  if(error){
    return AGBNP_ERR;
  }
#endif /* _OPENMP */

  /* set in_use=TRUE */
  agbdata->in_use = TRUE;

  /* update number of active structures */
  agbdata3_used += 1;

  return AGBNP_OK;
}

/* deletes a public instance of an agbnp structure */
int agbnp3_delete(int tag){
    AGBNPdata *agb;

  if(!agbnp3_initialized){
    agbnp3_errprint("agbnp3_delete(): agbnp library is not initialized.\n");
    return AGBNP_ERR;
  }
  if(!agbnp3_tag_ok(tag)){
    agbnp3_errprint("agbnp3_delete(): invalid tag %d.\n",tag);
    return AGBNP_ERR;
  }
  /* pointer to agb data structure */
  agb = &(agbdata3_list[tag]);

  if(agb->x){ agbnp3_vfree(agb->x); agb->x = NULL;}
  if(agb->y){ agbnp3_vfree(agb->y); agb->y = NULL;}
  if(agb->z){ agbnp3_vfree(agb->z); agb->z = NULL;}
  if(agb->r){ agbnp3_vfree(agb->r); agb->r = NULL;}
  if(agb->charge){ agbnp3_vfree(agb->charge); agb->charge = NULL;}
  if(agb->igamma){ agbnp3_vfree(agb->igamma); agb->igamma = NULL;}
  if(agb->sgamma){ agbnp3_vfree(agb->sgamma); agb->sgamma = NULL;}
  if(agb->ialpha){ agbnp3_vfree(agb->ialpha); agb->ialpha = NULL;}
  if(agb->salpha){ agbnp3_vfree(agb->salpha); agb->salpha = NULL;}
  if(agb->idelta){ agbnp3_vfree(agb->idelta); agb->idelta = NULL;}
  if(agb->sdelta){ agbnp3_vfree(agb->sdelta); agb->sdelta = NULL;}
  if(agb->hbtype){ agbnp3_vfree(agb->hbtype); agb->hbtype = NULL;}
  if(agb->hbcorr){ agbnp3_vfree(agb->hbcorr); agb->hbcorr = NULL;}
  if(agb->iheavyat){ agbnp3_vfree(agb->iheavyat); agb->iheavyat = NULL;}
  if(agb->ihydrogen){ agbnp3_vfree(agb->ihydrogen); agb->ihydrogen = NULL;}
  if(agb->idummy){ agbnp3_vfree(agb->idummy); agb->idummy = NULL;}
  if(agb->int2ext){ agbnp3_vfree(agb->int2ext) ; agb->int2ext = NULL;}
  if(agb->ext2int){ agbnp3_vfree(agb->ext2int) ; agb->ext2int = NULL;}
  if(agb->rot){ agbnp3_vfree(agb->rot) ; agb->rot = NULL; }
  if(agb->vdiel_in){ agbnp3_vfree(agb->vdiel_in) ; agb->vdiel_in = NULL;}

  if(agb->agbw){
    agbnp3_delete_agbworkdata(agb->agbw); free(agb->agbw) ; agb->agbw = NULL;}
  if(agb->conntbl){
    nblist_delete_neighbor_list(agb->conntbl); free(agb->conntbl) ; agb->conntbl = NULL; }

#ifdef _OPENMP
  if(agb->nprocs > 0){
    int iproc;
    if(agb->agbw_p){
      for(iproc=0;iproc<agb->nprocs;iproc++){
	if(agb->agbw_p[iproc]){
	  agbnp3_delete_agbworkdata(agb->agbw_p[iproc]);
	}
	agb->agbw_p[iproc] = NULL;
      }
      free(agb->agbw_p); agb->agbw_p = NULL;
    }
  }
  if(agb->omplock) { 
    int iat;
    for(iat=0;iat<agb->natoms;iat++){
      omp_destroy_lock(&(agb->omplock[iat]));
    }
    free(agb->omplock); agb->omplock = NULL; 
  }
#endif

  agbnp3_reset(agb);
  return AGBNP_OK;
}


/* returns AGBNP total energy and derivatives:
   generalized born, cavity, van der Waals, and hb correction energies */
int agbnp3_ener(int tag, int init,
               float_i *x, float_i *y, float_i *z,
	       float_i *sp, float_i *br, 
	       float_i *mol_volume, float_i *surf_area, 
	       float_i *egb, float_i (*dgbdr)[3],
	       float_i *evdw, float_i *ecorr_vdw, float_i (*dvwdr)[3], 
	       float_i *ecav, float_i *ecorr_cav, float_i (*decav)[3],
	       float_i *ehb,  float_i (*dehb)[3]){
  AGBNPdata *agb;
  int iat, iatext;
  int *int2ext, *ext2int;

  if(!agbnp3_initialized){
    agbnp3_errprint("agbnp3_agb_energy(): agbnp library is not initialized.\n");
    return AGBNP_ERR;
  }
  if(!agbnp3_tag_ok(tag)){
    agbnp3_errprint("agbnp3_agb_energy(): invalid tag %d.\n",tag);
    return AGBNP_ERR;
  }

  /* pointer to agb data structure */
  agb = &(agbdata3_list[tag]);

  int2ext = agb->int2ext;
  ext2int = agb->ext2int;

  /* copy to coordinates to local arrays */
  for(iat=0; iat < agb->natoms; iat++){
    iatext = int2ext[iat];
    agb->x[iat] = x[iatext];
    agb->y[iat] = y[iatext];
    agb->z[iat] = z[iatext];
  }

  if(agbnp3_total_energy(agb, init,
			mol_volume,  
			egb, 
			evdw, ecorr_vdw,  
			ecav, ecorr_cav, 
			ehb) != AGBNP_OK){
    agbnp3_errprint("agbnp3_ener(): error in agbnp3_total_energy().\n");
    return AGBNP_ERR;
  }

  /* copy to output buffers */
  for(iat=0; iat < agb->natoms; iat++){
    iatext = int2ext[iat];
    br[iatext] = agb->br[iat];
    sp[iatext] = agb->sp[iat];
    surf_area[iatext] = agb->surf_area[iat];
  }

  for(iat=0; iat < agb->natoms; iat++){
    iatext = int2ext[iat];

    dgbdr[iatext][0] = agb->dgbdr[iat][0];
    dgbdr[iatext][1] = agb->dgbdr[iat][1];
    dgbdr[iatext][2] = agb->dgbdr[iat][2];

    dvwdr[iatext][0] = agb->dvwdr[iat][0];
    dvwdr[iatext][1] = agb->dvwdr[iat][1];
    dvwdr[iatext][2] = agb->dvwdr[iat][2];

    dehb[iatext][0] = agb->dehb[iat][0];
    dehb[iatext][1] = agb->dehb[iat][1];
    dehb[iatext][2] = agb->dehb[iat][2];

    decav[iatext][0] = agb->decav[iat][0];
    decav[iatext][1] = agb->decav[iat][1];
    decav[iatext][2] = agb->decav[iat][2];
  }

  return AGBNP_OK;
}

/* check if it is a valid tag */
 int agbnp3_tag_ok(int tag){
  /* check ranges */
  if(tag < 0) return FALSE;
  if(tag >= agbdata3_allocated) return FALSE;
  
  /* check if list is in use */
  if(!agbdata3_list[tag].in_use) return FALSE;

  return TRUE;
}

/* reset an agbnp structure */
 int agbnp3_reset(AGBNPdata *data){
  data->in_use = FALSE;
  data->natoms = 0;
  data->x = data->y = data->z = NULL;
  data->r = NULL;
  data->charge = NULL;
  data->igamma = data->sgamma = NULL;
  data->ialpha = data->salpha = NULL;
  data->idelta = data->sdelta = NULL;
  data->hbtype = NULL;
  data->hbcorr = NULL;
  data->nheavyat = 0;
  data->iheavyat = NULL;
  data->nhydrogen = 0;
  data->ihydrogen = NULL;
  data->ndummy = 0;
  data->idummy = NULL;
  data->int2ext = NULL;
  data->ext2int = NULL;
  data->vdiel_in = NULL;
  data->dielectric_in = data->dielectric_out = -1.0;
  data->neigh_list = data->excl_neigh_list = NULL;
  data->nsym = 1;
  data->docryst = 0;
  data->ssize = 0;
  data->xs = data->ys = data->zs = NULL;
  data->rot = NULL;
  data->conntbl = NULL;
  data->do_w = 1;
  data->agbw = NULL;
  data->nprocs = data->maxprocs = 0;
  data->agbw_p = NULL;
  data->f4c1table2d = NULL;
  data->f4c1table2dh = NULL;
  return AGBNP_OK;
}

 int agbnp3_reset_agbworkdata(AGBworkdata *agbw){
   int i;
  agbw->natoms = 0;
  agbw->vols = agbw->volumep = NULL;
  agbw->dera = agbw->deru = agbw->derv = agbw->derh = NULL;
  agbw->psvol = agbw->derus = agbw->dervs = NULL;
  agbw->q2ab = agbw->abrw = NULL;
  agbw->br1_swf_der = NULL;
  agbw->isheavy = NULL;
  agbw->nbiat = NULL;
  agbw->nbdata = NULL;
  agbw->br = agbw->br1 = agbw->brw = NULL;
  agbw->alpha = agbw->delta = NULL;
  agbw->galpha = agbw->gprefac = NULL;
  agbw->atm_gs = NULL;
  agbw->sp = NULL;

  agbw->nq4cache = 0;
  agbw->q4cache = NULL;
  agbw->near_nl = NULL;
  agbw->far_nl = NULL;

  agbw->dgbdrx = NULL;
  agbw->dgbdry = NULL;
  agbw->dgbdrz = NULL;

  agbw->dgbdr_h = NULL;
  agbw->dvwdr_h = NULL;
  agbw->dehb = NULL;

  agbw->surf_area = agbw->surf_area_f = NULL;
  agbw->gamma = agbw->gammap = NULL;
  agbw->decav_h = NULL;

  agbw->nlist = NULL;
  agbw->js = NULL;
  agbw->pbcs = NULL;
  agbw->datas = NULL;

  agbw->nwsat = 0;
  agbw->wsat_size = 0;
  agbw->wsat = NULL;

  for(i=0;i<2;i++){
    agbw->overlap_lists[i] = NULL;
    agbw->size_overlap_lists[i] = 0;
    agbw->root_lists[i] = NULL;
    agbw->size_root_lists[i] = 0;
  }

  agbw->gbuffer_size = 0;
  agbw->a1 = NULL;
  agbw->p1 = NULL;
  agbw->c1x = NULL;
  agbw->c1y = NULL;
  agbw->c1z = NULL;
  agbw->a2 = NULL;
  agbw->p2 = NULL;
  agbw->c2x = NULL;
  agbw->c2y = NULL;
  agbw->c2z = NULL;
  agbw->v3 = NULL;
  agbw->v3p = NULL;
  agbw->fp3 = NULL;
  agbw->fp3 = NULL;

  agbw->hbuffer_size = 0;
  agbw->hiat = NULL;
  agbw->ha1 = NULL;
  agbw->hp1 = NULL;
  agbw->hc1x = NULL;
  agbw->hc1y = NULL;
  agbw->hc1z = NULL;
  agbw->ha2 = NULL;
  agbw->hp2 = NULL;
  agbw->hc2x = NULL;
  agbw->hc2y = NULL;
  agbw->hc2z = NULL;
  agbw->hv3 = NULL;
  agbw->hv3p = NULL;
  agbw->hfp3 = NULL;
  agbw->hfp3 = NULL;

  agbw->qbuffer_size = 0;
  agbw->qdv = NULL;
  agbw->qR1v = NULL;
  agbw->qR2v = NULL;
  agbw->qqv = NULL;
  agbw->qdqv = NULL;
  agbw->qav = NULL;
  agbw->qbv = NULL;
  agbw->qkv = NULL;
  agbw->qxh= NULL;
  agbw->qyp= NULL;
  agbw->qy= NULL;
  agbw->qy2p= NULL;
  agbw->qy2= NULL;
  agbw->qf1= NULL;
  agbw->qf2= NULL;
  agbw->qfp1= NULL;
  agbw->qfp2= NULL;

  agbw->wbuffer_size = 0;
  agbw->wb_iatom = NULL;
  agbw->wb_gvolv = NULL;
  agbw->wb_gderwx = NULL;
  agbw->wb_gderwy = NULL;
  agbw->wb_gderwz = NULL;
  agbw->wb_gderix = NULL;
  agbw->wb_gderiy = NULL;
  agbw->wb_gderiz = NULL;

  agbw->wsize = 0;
  agbw->w_iov = NULL;
  agbw->w_nov = NULL;

  return AGBNP_OK;
}

/* reorder atoms so that hydrogens are at the end */
int agbnp3_atom_reorder(AGBNPdata *agb, int nhydrogen, int *ihydrogen){
  int *int2ext = agb->int2ext;
  int *ext2int = agb->ext2int;
  int natoms = agb->natoms;
  int *ishydrogen = (int *)calloc(natoms,sizeof(int));
  int nc, nh, i, ih, nheavy;

  nheavy =  natoms - nhydrogen;

  for(i = 0; i < natoms; i++){
    int2ext[i] = i;
    ext2int[i] = i;
  }

  for(i = 0; i < nhydrogen; i++){
    ih = ihydrogen[i];
    ishydrogen[ih] = 1;
  }

  nc = 0;
  nh = 0;
  for(i = 0; i < natoms; i++){
    if(ishydrogen[i]){
      int2ext[nheavy + nh] = i;
      ext2int[i] = nheavy + nh;
      nh += 1;
    }else{
      int2ext[nc] = i;
      ext2int[i] = nc;
      nc += 1;
    } 
  }

  //  for(i=0;i<natoms;i++){
  //  printf("%d %d %d %d\n", i, ishydrogen[i], int2ext[i], ext2int[i]);
  // }
  //exit(1);

  free(ishydrogen);
  return AGBNP_OK;
} 


 int agbnp3_allocate_agbworkdata(int natoms, AGBNPdata *agb, 
		 		      AGBworkdata *agbw){
   int i, n;
  agbw->natoms = natoms;

  agbnp3_vcalloc((void **)&(agbw->vols),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->volumep),natoms*sizeof(float_a));

  agbnp3_vcalloc((void **)&(agbw->dera),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->deru),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->derv),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->derh),natoms*sizeof(float_a));

  agbnp3_vcalloc((void **)&(agbw->derus),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->dervs),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->psvol),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->q2ab),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->abrw),natoms*sizeof(float_a));

  agbnp3_vcalloc((void **)&(agbw->br1_swf_der),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->isheavy),natoms*sizeof(int));
  agbnp3_vcalloc((void **)&(agbw->nbiat),natoms*sizeof(int));

  agbnp3_vcalloc((void **)&(agbw->br),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->br1),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->brw),natoms*sizeof(float_a));

  agbnp3_vcalloc((void **)&(agbw->alpha),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->delta),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->galpha),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->gprefac),natoms*sizeof(float_a));

  agbnp3_vcalloc((void **)&(agbw->atm_gs),natoms*sizeof(GParm));

  agbnp3_vcalloc((void **)&(agbw->sp),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->spe),natoms*sizeof(float_a));

  agbnp3_vcalloc((void **)&(agbw->dgbdrx),natoms*sizeof(float));
  agbnp3_vcalloc((void **)&(agbw->dgbdry),natoms*sizeof(float));
  agbnp3_vcalloc((void **)&(agbw->dgbdrz),natoms*sizeof(float));

  agbnp3_vcalloc((void **)&(agbw->dgbdr_h),natoms*sizeof(float_a [3]));
  agbnp3_vcalloc((void **)&(agbw->dvwdr_h),natoms*sizeof(float_a [3]));
  agbnp3_vcalloc((void **)&(agbw->dehb),natoms*sizeof(float_a [3]));
  agbnp3_vcalloc((void **)&(agbw->decav_h),natoms*sizeof(float_a [3]));

  agbnp3_vcalloc((void **)&(agbw->nlist),natoms*sizeof(int));
  agbnp3_vcalloc((void **)&(agbw->js),natoms*sizeof(int));

  agbnp3_vcalloc((void **)&(agbw->datas),natoms*sizeof(void *));

  agbnp3_vcalloc((void **)&(agbw->surf_area),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->surf_area_f),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->gamma),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->gammap),natoms*sizeof(float_a));

  agbnp3_vcalloc((void **)&(agbw->nl_r2v),natoms*sizeof(float_a));
  agbnp3_vcalloc((void **)&(agbw->nl_indx),natoms*sizeof(int));

  if(!(agbw->vols && agbw->volumep && agbw->dera && agbw->deru && 
       agbw->derv && agbw->derh &&
       agbw->derus && agbw->dervs && agbw->psvol && 
       agbw->q2ab  && agbw->abrw  && agbw->br1_swf_der && 
       agbw->isheavy && agbw->nbiat && 
       agbw->br && agbw->br1 && agbw->brw  && 
       agbw->alpha && agbw->delta && agbw->galpha && agbw->gprefac &&
       agbw->atm_gs &&
       agbw->sp && agbw->spe &&
       agbw->dgbdrx && agbw->dgbdry && agbw->dgbdrz && 
       agbw->dgbdr_h && agbw->dvwdr_h && 
       agbw->dehb &&
       agbw->surf_area && agbw->surf_area_f &&
       agbw->gamma && agbw->gammap && agbw->decav_h && 
       agbw->nlist && agbw->js && agbw->datas && 
       agbw->nl_r2v && agbw->nl_indx 
       )){
    agbnp3_errprint( "agbnp3_allocate_agbworkdata(): unable to allocate memory for AGB work data structure.\n");
    return AGBNP_ERR;
  }

  agbw->near_nl = (NeighList *)malloc(1*sizeof(NeighList));
  if(!agbw->near_nl){
    agbnp3_errprint( "agbnp3_allocate_agbworkdata(): unable to allocate memory for near_nl (%d NeighList).\n", 1);
    return AGBNP_ERR;
  }
  nblist_reset_neighbor_list(agbw->near_nl);
  if(nblist_reallocate_neighbor_list(agbw->near_nl, natoms,
				     natoms*AGBNP_NEARNEIGHBORS) != NBLIST_OK){
      agbnp3_errprint("agbnp3_allocate_agbworkdata(): unable to allocate near_nl neighbor list (natoms=%d, size=%d)\n", natoms, natoms*AGBNP_NEARNEIGHBORS);
      return AGBNP_ERR;
  }

  agbw->far_nl = (NeighList *)malloc(1*sizeof(NeighList));
  if(!agbw->far_nl){
    agbnp3_errprint( "agbnp3_allocate_agbworkdata(): unable to allocate memory for far_nl (%d NeighList).\n", 1);
    return AGBNP_ERR;
  }
  nblist_reset_neighbor_list(agbw->far_nl);
  if(nblist_reallocate_neighbor_list(agbw->far_nl, natoms,
				     natoms*AGBNP_FARNEIGHBORS) != NBLIST_OK){
      agbnp3_errprint("agbnp3_allocate_agbworkdata(): unable to allocate far_nl neighbor list (natoms=%d, size=%d)\n", natoms, natoms*AGBNP_FARNEIGHBORS);
      return AGBNP_ERR;
  }

  n = natoms*AGBNP_OVERLAPS; //initial size of overlap lists
  if(agbnp3_reallocate_overlap_lists(agbw, n) != AGBNP_OK){
      agbnp3_errprint( "agbnp3_allocate_agbworkdata(): unable to allocate memory for overlap lists.\n");
      return AGBNP_ERR;
  }

  n = natoms*AGBNP_OVERLAPS/10; //initial size of Gaussian overlap buffers
  if(agbnp3_reallocate_gbuffers(agbw, n) != AGBNP_OK){
      agbnp3_errprint( "agbnp3_allocate_agbworkdata(): unable to allocate memory for Gaussian overlap buffers.\n");
      return AGBNP_ERR;
  }

  n = natoms*natoms/4; //initial size of water site Gaussian overlap buffers
  if(agbnp3_reallocate_hbuffers(agbw, n) != AGBNP_OK){
      agbnp3_errprint( "agbnp3_allocate_agbworkdata(): unable to allocate memory for Gaussian overlap buffers.\n");
      return AGBNP_ERR;
  }

  n = 2*natoms + 4; //initial size of buffers for inverse born radii
  if(agbnp3_reallocate_qbuffers(agbw, n) != AGBNP_OK){
    agbnp3_errprint( "agbnp3_allocate_agbworkdata(): unable to allocate memory for inverse born radii buffers.\n");
      return AGBNP_ERR;
  }

  n = natoms; //initial size of buffers for water sites
  if(agbnp3_reallocate_wbuffers(agbw, n) != AGBNP_OK){
    agbnp3_errprint( "agbnp3_allocate_agbworkdata(): unable to allocate memory for water sites.\n");
      return AGBNP_ERR;
  }
  
  return AGBNP_OK;
}

 int agbnp3_delete_agbworkdata(AGBworkdata *agbw){
  int i;
  if(agbw->vols){ agbnp3_vfree(agbw->vols); agbw->vols = NULL;}
  if(agbw->volumep){agbnp3_vfree(agbw->volumep); agbw->volumep = NULL;}
  if(agbw->dera){ agbnp3_vfree(agbw->dera); agbw->dera = NULL;}
  if(agbw->deru){ agbnp3_vfree(agbw->deru); agbw->deru = NULL;}
  if(agbw->derv){ agbnp3_vfree(agbw->derv); agbw->derv = NULL;}
  if(agbw->derh){ agbnp3_vfree(agbw->derh); agbw->derh = NULL;}
  if(agbw->psvol){ agbnp3_vfree(agbw->psvol); agbw->psvol = NULL;}
  if(agbw->derus){ agbnp3_vfree(agbw->derus); agbw->derus = NULL;}
  if(agbw->dervs){ agbnp3_vfree(agbw->dervs); agbw->dervs = NULL;}
  if(agbw->q2ab){ agbnp3_vfree(agbw->q2ab); agbw->q2ab = NULL;}
  if(agbw->abrw){ agbnp3_vfree(agbw->abrw); agbw->abrw = NULL;}
  if(agbw->br1_swf_der){agbnp3_vfree(agbw->br1_swf_der); agbw->br1_swf_der = NULL;}
  if(agbw->isheavy){agbnp3_vfree(agbw->isheavy); agbw->isheavy = NULL;}
  if(agbw->nbiat){agbnp3_vfree(agbw->nbiat); agbw->nbiat = NULL;}
  if(agbw->br){agbnp3_vfree(agbw->br); agbw->br = NULL;}
  if(agbw->br1){agbnp3_vfree(agbw->br1); agbw->br1 = NULL;}
  if(agbw->brw){agbnp3_vfree(agbw->brw); agbw->brw = NULL;}
  if(agbw->alpha){agbnp3_vfree(agbw->alpha); agbw->alpha = NULL;}
  if(agbw->delta){agbnp3_vfree(agbw->delta); agbw->delta = NULL;}
  if(agbw->galpha){agbnp3_vfree(agbw->galpha); agbw->galpha = NULL;}
  if(agbw->gprefac){agbnp3_vfree(agbw->gprefac); agbw->gprefac = NULL;}
  if(agbw->atm_gs){agbnp3_vfree(agbw->atm_gs); agbw->atm_gs = NULL;}
  if(agbw->sp){agbnp3_vfree(agbw->sp); agbw->sp = NULL;}
  if(agbw->spe){agbnp3_vfree(agbw->spe); agbw->spe = NULL;}
  if(agbw->q4cache){agbnp3_vfree(agbw->q4cache); agbw->q4cache = NULL;}

  if(agbw->dgbdrx){agbnp3_vfree(agbw->dgbdrx); agbw->dgbdrx = NULL;}
  if(agbw->dgbdry){agbnp3_vfree(agbw->dgbdry); agbw->dgbdry = NULL;}
  if(agbw->dgbdrz){agbnp3_vfree(agbw->dgbdrz); agbw->dgbdrz = NULL;}

  if(agbw->dgbdr_h){agbnp3_vfree(agbw->dgbdr_h); agbw->dgbdr_h = NULL;}
  if(agbw->dvwdr_h){agbnp3_vfree(agbw->dvwdr_h); agbw->dvwdr_h = NULL;}
  if(agbw->dehb){agbnp3_vfree(agbw->dehb); agbw->dehb = NULL;}


  if(agbw->surf_area)  {agbnp3_vfree(agbw->surf_area);  agbw->surf_area = NULL;}
  if(agbw->surf_area_f) {agbnp3_vfree(agbw->surf_area_f); agbw->surf_area_f = NULL;}
  if(agbw->gamma)  {agbnp3_vfree(agbw->gamma); agbw->gamma = NULL;}
  if(agbw->gammap) {agbnp3_vfree(agbw->gammap);  agbw->gammap = NULL;}
  if(agbw->decav_h)  {agbnp3_vfree(agbw->decav_h);  agbw->decav_h = NULL;}

  if(agbw->nlist)  {agbnp3_vfree(agbw->nlist); agbw->nlist  = NULL;}
  if(agbw->js)  {agbnp3_vfree(agbw->js); agbw->js  = NULL;}
  if(agbw->pbcs)  {agbnp3_vfree(agbw->pbcs); agbw->pbcs  = NULL;}
  if(agbw->datas)  {agbnp3_vfree(agbw->datas); agbw->datas  = NULL;}

  if(agbw->near_nl){
    nblist_delete_neighbor_list(agbw->near_nl);
    agbw->near_nl = NULL;
  }
  if(agbw->far_nl){
    nblist_delete_neighbor_list(agbw->far_nl);
    agbw->far_nl = NULL;
  }  

  if(agbw->nl_r2v)  {agbnp3_vfree(agbw->nl_r2v); agbw->nl_r2v = NULL;}
  if(agbw->nl_indx) {agbnp3_vfree(agbw->nl_indx); agbw->nl_indx = NULL;}

  if(agbw->wsat){ 
    int iw;
    WSat *wsat;
    for(iw=0;iw<agbw->wsat_size;iw++){
      wsat = &(agbw->wsat[iw]);
      agbnp3_clr_wsat(wsat);
    }
    free(agbw->wsat); 
    agbw->wsat = NULL;
  }

  for(i=0;i<2;i++){
    if(agbw->overlap_lists[i]){agbnp3_vfree(agbw->overlap_lists[i]); agbw->overlap_lists[i] = NULL;}
    if(agbw->root_lists[i]){agbnp3_vfree(agbw->root_lists[i]); agbw->root_lists[i] = NULL;}
  }

  if(agbw->a1){agbnp3_vfree(agbw->a1); agbw->a1 = NULL;}
  if(agbw->p1){agbnp3_vfree(agbw->p1); agbw->p1 = NULL;}
  if(agbw->c1x){agbnp3_vfree(agbw->c1x); agbw->c1x = NULL;}
  if(agbw->c1y){agbnp3_vfree(agbw->c1y); agbw->c1y = NULL;}
  if(agbw->c1z){agbnp3_vfree(agbw->c1z); agbw->c1z = NULL;}

  if(agbw->a2){agbnp3_vfree(agbw->a2); agbw->a2 = NULL;}
  if(agbw->p2){agbnp3_vfree(agbw->p2); agbw->p2 = NULL;}
  if(agbw->c2x){agbnp3_vfree(agbw->c2x); agbw->c2x = NULL;}
  if(agbw->c2y){agbnp3_vfree(agbw->c2y); agbw->c2y = NULL;}
  if(agbw->c2z){agbnp3_vfree(agbw->c2z); agbw->c2z = NULL;}

  if(agbw->v3){agbnp3_vfree(agbw->hv3); agbw->v3 = NULL;}
  if(agbw->v3p){agbnp3_vfree(agbw->hv3p); agbw->v3p = NULL;}
  if(agbw->fp3){agbnp3_vfree(agbw->hfp3); agbw->fp3 = NULL;}
  if(agbw->fpp3){agbnp3_vfree(agbw->hfpp3); agbw->fpp3 = NULL;}

  if(agbw->hiat){agbnp3_vfree(agbw->hiat); agbw->hiat = NULL;}
  if(agbw->ha1){agbnp3_vfree(agbw->ha1); agbw->ha1 = NULL;}
  if(agbw->hp1){agbnp3_vfree(agbw->hp1); agbw->hp1 = NULL;}
  if(agbw->hc1x){agbnp3_vfree(agbw->hc1x); agbw->hc1x = NULL;}
  if(agbw->hc1y){agbnp3_vfree(agbw->hc1y); agbw->hc1y = NULL;}
  if(agbw->hc1z){agbnp3_vfree(agbw->hc1z); agbw->hc1z = NULL;}

  if(agbw->ha2){agbnp3_vfree(agbw->ha2); agbw->ha2 = NULL;}
  if(agbw->hp2){agbnp3_vfree(agbw->hp2); agbw->hp2 = NULL;}
  if(agbw->hc2x){agbnp3_vfree(agbw->hc2x); agbw->hc2x = NULL;}
  if(agbw->hc2y){agbnp3_vfree(agbw->hc2y); agbw->hc2y = NULL;}
  if(agbw->hc2z){agbnp3_vfree(agbw->hc2z); agbw->hc2z = NULL;}

  if(agbw->hv3){agbnp3_vfree(agbw->hv3); agbw->hv3 = NULL;}
  if(agbw->hv3p){agbnp3_vfree(agbw->hv3p); agbw->hv3p = NULL;}
  if(agbw->hfp3){agbnp3_vfree(agbw->hfp3); agbw->hfp3 = NULL;}
  if(agbw->hfpp3){agbnp3_vfree(agbw->hfpp3); agbw->hfpp3 = NULL;}

  if(agbw->qdv){agbnp3_vfree(agbw->qdv); agbw->qdv = NULL;}
  if(agbw->qR1v){agbnp3_vfree(agbw->qR1v); agbw->qR1v = NULL;}
  if(agbw->qR2v){agbnp3_vfree(agbw->qR2v); agbw->qdv = NULL;}
  if(agbw->qqv){agbnp3_vfree(agbw->qqv); agbw->qqv = NULL;}
  if(agbw->qdqv){agbnp3_vfree(agbw->qdqv); agbw->qdqv = NULL;}
  if(agbw->qav){agbnp3_vfree(agbw->qav); agbw->qav = NULL;}
  if(agbw->qbv){agbnp3_vfree(agbw->qbv); agbw->qbv = NULL;}

  if(agbw->qkv){agbnp3_vfree(agbw->qkv); agbw->qkv = NULL;}
  if(agbw->qxh){agbnp3_vfree(agbw->qxh); agbw->qxh = NULL;}
  if(agbw->qyp){agbnp3_vfree(agbw->qyp); agbw->qyp = NULL;}
  if(agbw->qy){agbnp3_vfree(agbw->qy); agbw->qy = NULL;}
  if(agbw->qy2p){agbnp3_vfree(agbw->qy2p); agbw->qy2p = NULL;}
  if(agbw->qy2){agbnp3_vfree(agbw->qy2); agbw->qy2 = NULL;}
  if(agbw->qf1){agbnp3_vfree(agbw->qf1); agbw->qf1 = NULL;}
  if(agbw->qf2){agbnp3_vfree(agbw->qf2); agbw->qf2 = NULL;}
  if(agbw->qfp1){agbnp3_vfree(agbw->qfp1); agbw->qfp1 = NULL;}
  if(agbw->qfp2){agbnp3_vfree(agbw->qfp2); agbw->qfp2 = NULL;}

  if(agbw->wb_iatom){agbnp3_vfree(agbw->wb_iatom); agbw->wb_iatom = NULL;}
  if(agbw->wb_gvolv){agbnp3_vfree(agbw->wb_gvolv); agbw->wb_gvolv = NULL;}
  if(agbw->wb_gderwx){agbnp3_vfree(agbw->wb_gderwx); agbw->wb_gderwx = NULL;}
  if(agbw->wb_gderwy){agbnp3_vfree(agbw->wb_gderwy); agbw->wb_gderwy = NULL;}
  if(agbw->wb_gderwz){agbnp3_vfree(agbw->wb_gderwz); agbw->wb_gderwz = NULL;}
  if(agbw->wb_gderix){agbnp3_vfree(agbw->wb_gderix); agbw->wb_gderix = NULL;}
  if(agbw->wb_gderiy){agbnp3_vfree(agbw->wb_gderiy); agbw->wb_gderiy = NULL;}
  if(agbw->wb_gderiz){agbnp3_vfree(agbw->wb_gderiz); agbw->wb_gderiz = NULL;}

  return AGBNP_OK;
}


int agbnp3_init_agbworkdata(AGBNPdata *agb, AGBworkdata *agbw){
  int i, iat;
  float_a c4;

  /* set isheavy array */
  memset(agbw->isheavy,0,agbw->natoms*sizeof(int));
  for(i=0;i<agb->nheavyat;i++){
    agbw->isheavy[agb->iheavyat[i]] = 1;
  }

  /* set np parameters */
  for(iat=0;iat<agbw->natoms;iat++){
    agbw->alpha[iat] = agb->ialpha[iat] + agb->salpha[iat];
    agbw->gamma[iat] = agb->igamma[iat] + agb->sgamma[iat];
    agbw->delta[iat] = agb->idelta[iat] + agb->sdelta[iat];
  }

  /* initialize atomic volumes */
  c4 = 4.0*pi/3.0;
  for(iat=0;iat<agbw->natoms;iat++){
    agbw->vols[iat] = c4*pow(agb->r[iat],3);
  }
  
  /* Gaussian parameters initialization */
  for(i=0;i<agb->nheavyat;i++){
    iat = agb->iheavyat[i];
    agbw->galpha[iat] = KFC/(agb->r[iat]*agb->r[iat]);
    agbw->gprefac[iat] = PFC;
  }
  for(i=0;i<agb->nheavyat;i++){
    iat = agb->iheavyat[i];
    agbw->atm_gs[iat].a = agbw->galpha[iat];
    agbw->atm_gs[iat].p = agbw->gprefac[iat];
    agbw->atm_gs[iat].c[0] = agb->x[iat];
    agbw->atm_gs[iat].c[1] = agb->y[iat];
    agbw->atm_gs[iat].c[2] = agb->z[iat];
  }

  return AGBNP_OK;
}

float_a agbnp3_swf_area(float_a x, float_a *fp){
  static const float_a a2 = 5.f*5.f;
  float_a t, f, x2;
 
  if(x<0.0){
    *fp = 0.0;
    return (float_a)0.0;
  }
  x2 = x*x;
  t = x/(a2 + x2);
  f = x*t;
  *fp  = (2.*t)*(1. - f);
  return f;
}

/* goes smoothly from 0 at xa to 1 at xb */
 float_a agbnp3_pol_switchfunc(float_a x, float_a xa, float_a xb,
				   float_a *fp, float_a *fpp){
  float_a u,d,u2,u3,f;
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
  d = 1./(xb - xa);
  u = (x - xa)*d;
  u2 = u*u;
  u3 = u*u2;
  f = u3*(10.-15.*u+6.*u2);
  if(fp){   /* first derivative */
    *fp = d*30.*u2*(1. - 2.*u + u2);
  }
  if(fpp){ /* second derivative */
    *fpp = d*d*60.*u*(1. - 3.*u + 2.*u2);
  }
  return f;
}


 float_a agbnp3_swf_vol3(float_a x, float_a *fp, float_a *fpp, 
		       float_a a, float_a b){
  float_a f, s, sp, spp;

  if(x>b){
    *fp = 1.0;
    *fpp = 0.0;
    return x;
  }
  if(x<a){
    *fp = 0.0;
    *fpp = 0.0;
    return (float_a)0.0;
  }
  s = agbnp3_pol_switchfunc(x, a, b, &sp, &spp);
  f = s*x;
  *fp = s + x*sp;
  *fpp = 2.*sp + x*spp;

  return f;
}

/* a switching function for the inverse born radius (beta)
   so that if beta is negative -> beta' = minbeta
   and otherwise beta' = beta^3/(beta^2+a^2) + minbeta
*/ 
 float_a agbnp3_swf_invbr(float_a beta, float_a *fp){
  /* the maximum born radius is 50.0 Ang. */
  static const float_a a  = 0.02;
  static const float_a a2 = 0.02*0.02;
  float_a t;

  if(beta<0.0){
    *fp = 0.0;
    return a;
  }
  t = sqrt(a2 + beta*beta);

  *fp  = beta/t;
  return t;
}

int agbnp3_mymax(int a, int b){
  return a > b ? a : b;
}

/* transpose x vector matrix multiplication. To rotate gradients */
 void agbnp3_rtvec(float_a y[3], float_a rot[3][3], float_a x[3]){
  int i,j;
  for(i=0;i<3;i++){
    y[i] = 0.0;
    for(j=0;j<3;j++){
      y[i] += x[j]*rot[j][i];
    }
  }
}

void agbnp3_matmul(float_a a[3][3], float_a b[3][3], float_a c[3][3]){
  int i,j,k;
  
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      c[i][j] = 0.;
      for(k=0;k<3;k++){
	c[i][j] += a[i][k]*b[k][j];
      }
    }
  }

}


/* computes all energy components */
int agbnp3_total_energy(AGBNPdata *agb, int init,
		    float_i *mol_volume,
		    float_i *egb, 
		    float_i *evdw, float_i *ecorr_vdw, 
		    float_i *ecav, float_i *ecorr_cav, 
			float_i *ehb){

  int verbose = 0;

  static const float_a tokcalmol = 332.0; /* conversion to kcal/mol */
  int i, ki; /* counters */
  float_a a, f, fp;
  int iat; /* atomic counter */
  static const float_a rw = 1.4;  /* water radius offset for np 
				    energy function */
  int no_init_frozen = 0;
  float_a dielectric_factor = 
    -0.5*(1./agb->dielectric_in - 1./agb->dielectric_out);

  int natoms = agb->natoms;
  int nheavyat = agb->nheavyat;
  int *iheavyat = agb->iheavyat;
  float_a *ialpha = agb->ialpha;
  float_a *salpha = agb->salpha;
  float_a *idelta = agb->idelta;
  float_a *sdelta = agb->sdelta;
  float_a egb_self = 0.0, egb_pair = 0.0;
  float_a hbe;
  
  AGBworkdata *agbw = agb->agbw; /* shared work space */
  AGBworkdata *agbw_h;           /* work space for this thread */

  float_a *volumep = agbw->volumep; /* self-volumes */
  float_a *surf_area_f = agbw->surf_area_f; /* filtered surface areas */
  float_a *igamma = agb->igamma; /* ideal gamma */
  float_a *gamma = agbw->gamma; /* ideal gamma + corrction gamma */
  float_a *sgamma = agb->sgamma; /* correction gamma */
  float_a *gammap = agbw->gammap; /* gamma corrected by derivative of 
				     surface area switching function 
				     (for derivative calculation) */


  int iproc = 0;
  int res, error = 0, nop = 0;

#ifdef ATIMER
  static float timer_nblist  = 0.0f;
  static float timer_volumes = 0.0f;
  static float timer_invb    = 0.0f;
  static float timer_agb     = 0.0f;
  static float timer_gbdersconstvp = 0.0f;
  static float timer_deruv = 0.0f;
  static float timer_dervp = 0.0f;
  static float timer_dersgb = 0.0f;
  static float timer_ws = 0.0f;
  static float timer_wsder = 0.0f;
  static int timecounter = 0;
  float startime, endtime, fproc;
#endif

#ifdef _OPENMP
#pragma omp parallel private(agbw_h, iproc, res)
#endif
  {

#ifdef _OPENMP
  /* get local work structure */
  iproc = omp_get_thread_num();
  agbw_h = agb->agbw_p[iproc];
#else
  agbw_h = agbw;
#endif

#ifdef ATIMER
  if(!iproc) startime = (float)clock()/CLOCKS_PER_SEC;
#pragma omp barrier
#endif


  /*                                                */
  /*        reset buffers                           */
  /*  (self volumes, surface areas, etc.            */

  res = agbnp3_reset_buffers(agb, agbw_h);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_reset_buffers()\n");
#pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;


  /*                                                    */
  /*              reset derivatives and related         */
  /*                                                    */
  //printf("%d: agbnp3_reset_derivatives()\n",iproc);

  res = AGBNP_OK;
  res = agbnp3_reset_derivatives(agb,agbw_h);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_reset_derivatives\n");
 #pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;

  /*                                             */
  /*           neighbor lists                    */
  /*                                             */
  ///printf("%d: agbnp3_neighbor_lists()\n",iproc);

  if(verbose) printf("agbnp3_neighbor_lists() ...\n");

  res = agbnp3_neighbor_lists(agb, agbw_h, agb->x, agb->y, agb->z);
  if(res != AGBNP_OK){
    agbnp3_errprint("agbnp3_total_energy(): error in agbnp3_neighbor_lists()\n");
 #pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;

#ifdef ATIMER
#pragma omp barrier
  if(!iproc) {
    endtime = (float)clock()/CLOCKS_PER_SEC;
    timer_nblist += endtime - startime;
  }
#endif

  /*                                                */
  /*         self volumes and surface areas         */
  /*                                                */
  //printf("%d: agbnp3_self_volumes_rooti()\n",iproc);

  if(verbose) printf("agbnp3_self_volumes_rooti() ...\n");

#ifdef ATIMER
  if(!iproc) startime = (float)clock()/CLOCKS_PER_SEC;
#pragma omp barrier
#endif

  res = agbnp3_self_volumes_rooti(agb, agbw_h, agb->x, agb->y, agb->z);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_self_volumes()\n");
#pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;

  /*                                                                     */
  /* calculate volume scaling factors and surface-corrected self volumes */
  /* does gather for volumes, surface areas                              */
  /*                                                                     */
  //printf("%d: agbnp3_scaling_factors()\n",iproc);


  res = agbnp3_scaling_factors(agb, agbw_h);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_scaling_factors()\n");
 #pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;    

#pragma omp barrier
#pragma omp single nowait
  {
    int i;
    /* calculates cavity energy */
    *ecav = 0.0;
    *ecorr_cav = 0.0;
    for(i=0;i<agb->nheavyat;i++){
      *ecav += igamma[i]*agb->surf_area[i];
      *ecorr_cav += sgamma[i]*agb->surf_area[i];
    }
  }
#pragma omp single
  {
    int iat, i;
    /* volume of molecule */
    *mol_volume = 0.0;
    for(i = 0; i < nheavyat; i++){
      iat = iheavyat[i];
      *mol_volume += volumep[iat];
    }
  }

#ifdef ATIMER
#pragma omp barrier
  if(!iproc) {
    endtime = (float)clock()/CLOCKS_PER_SEC;
    timer_volumes += endtime - startime;  
  }
#endif

 /*                                                     */
 /*              inverse Born radii                     */
 /*                                                     */
 //printf("%d: agbnp3_inverse_born_radii_nolist_soa()\n",iproc);

  if(verbose) printf("agbnp3_inverse_born_radii_nolist_soa() ...\n");

#ifdef ATIMER
 if(!iproc) startime = (float)clock()/CLOCKS_PER_SEC;
#pragma omp barrier
#endif

 res = agbnp3_inverse_born_radii_nolist_soa(agb, agbw_h, agb->x, agb->y, agb->z, 
					      no_init_frozen);
 if(res != AGBNP_OK){
   agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_inverse_born_radii()\n");
 #pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;
 
  /*                                                            */
  /*            Born radii and related quantities (brw, etc.)   */
  /*                                                            */
  //printf("%d: agbnp3_born_radii()\n",iproc);


  res = agbnp3_born_radii(agb, agbw_h);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_born_radii()\n");
#pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;
  
#ifdef ATIMER
#pragma omp barrier
  if(!iproc) {
    endtime = (float)clock()/CLOCKS_PER_SEC;
    timer_invb += endtime - startime;  
  }
#endif

#pragma omp barrier
#pragma omp single
  /* calculates van der waals energy */
  {
    for(iat=0;iat<natoms;iat++){
      agb->br[iat] = agb->agbw->br[iat];
    }
    *evdw = 0.0;
    *ecorr_vdw = 0.0;
#ifdef AGBNP_VDW_PRINT
      agbnp3_errprint("Id Bradius alpha*a 1/(B+Rw)^3 alpha*a/(B+Rw)^3\n");
#endif
    for(iat=0;iat<natoms;iat++){
      a = 1.0/(agb->br[iat]+rw);
      a = pow(a,3);
      *evdw += (ialpha[iat]*a + idelta[iat]);
      *ecorr_vdw += (salpha[iat]*a + sdelta[iat]);
#ifdef AGBNP_VDW_PRINT
      agbnp3_errprint("VDW: %d %f %f %f %f\n",iat+1,agb->br[iat],ialpha[iat],a,ialpha[iat]*a);
#endif
    }
  }


  /*                                                                 */
  /* Evaluates solvation energy, Ai's and derivatives of GB energy   */
  /* at constant Born radii                                          */
  /*                                                                 */
  //printf("%d: agbnp3_gb_energy_nolist_ps()\n",iproc);

  if(verbose) printf("agbnp3_gb_energy_nolist_ps() ...\n");

#ifdef ATIMER
  if(!iproc) startime = (float)clock()/CLOCKS_PER_SEC;
#pragma omp barrier
#endif

  res = agbnp3_gb_energy_nolist_ps(agb, agbw_h, agb->x, agb->y, agb->z, 
				     &egb_self, &egb_pair);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_gb_energy()\n");
#pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;

#pragma omp barrier
#pragma omp single
  {
    /* calculates GB energy */
    egb_self *=  tokcalmol;
    egb_pair *= tokcalmol;
    *egb = (egb_self + egb_pair);
  }

#ifdef ATIMER
#pragma omp barrier
  if(!iproc) {
    endtime = (float)clock()/CLOCKS_PER_SEC;
    timer_agb += endtime - startime;
  }
#endif


  /*                                                          */
  /* GB derivatives contribution at constant self volumes     */
  /*                                                          */

   if(verbose) printf("agbnp3_gb_ders_constvp_nolist_ps() ...\n");
  
#ifdef ATIMER
  if(!iproc) startime = (float)clock()/CLOCKS_PER_SEC;
#pragma omp barrier
#endif

  res = agbnp3_gb_ders_constvp_nolist_ps(agb, agbw_h, agb->x, agb->y, agb->z, 
					   no_init_frozen);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_gb_ders_constvp()\n");
 #pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;

#ifdef ATIMER
#pragma omp barrier
  if(!iproc) {
    endtime = (float)clock()/CLOCKS_PER_SEC;
    timer_gbdersconstvp += endtime - startime;
  }
#endif

  /*                                                                          */
  /*                 creates water sites                                      */
  /*                                                                          */
  //printf("%d: agbnp3_create_wsatoms()\n",iproc);

  if(verbose) printf("agbnp3_create_wsatoms() ...\n");

#ifdef ATIMER
  if(!iproc) startime = (float)clock()/CLOCKS_PER_SEC;
#pragma omp barrier 
#endif

  res = agbnp3_create_wsatoms(agb, agbw_h);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_create_wsatoms()\n");
 #pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;

  /*                                                                          */
  /*                       evaluates ehb energy                               */
  /*                                                                          */
  //printf("%d: agbnp3_ws_free_volumes_scalev_ps()\n",iproc);  
  
  if(verbose) printf("agbnp3_ws_free_volumes_scalev_ps() ...\n");

  res = agbnp3_ws_free_volumes_scalev_ps(agb, agbw_h);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_ws_free_volumes_scalev(agb)\n");
 #pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;

#pragma omp critical
  if(agb->verbose)
  {
    int iws;
    float xa = AGBNP_HB_SWA;
    float xb = AGBNP_HB_SWB;
    float fp, s;
    printf("WSphere   x   y   z  Parent   Type    Khb   FreeVol    FilterVol   Energy\n");
    for(iws=0;iws<agbw_h->nwsat;iws++){
      s = agbnp3_pol_switchfunc(agbw_h->wsat[iws].sp, xa, xb, &fp, NULL);    
      printf("WS %d %f %f %f %d %d %f  %f  %f  %f\n",iws, 
	     agbw_h->wsat[iws].pos[0], agbw_h->wsat[iws].pos[1], agbw_h->wsat[iws].pos[2],
	     agbw_h->wsat[iws].parent[0], agbw_h->wsat[iws].type, agbw_h->wsat[iws].khb, agbw_h->wsat[iws].sp, s, agbw_h->wsat[iws].khb*s); 
    }    
    printf("WST: Nws = %d\n", agbw_h->nwsat);
  }



#ifdef ATIMER
#pragma omp barrier
  if(!iproc) {
    endtime = (float)clock()/CLOCKS_PER_SEC;
    timer_ws += endtime - startime;
  }
#endif

  /*                                                   */
  /*             evaluation of Ui's and Vi's           */
  /*                                                   */

#ifdef ATIMER
  if(!iproc) startime = (float)clock()/CLOCKS_PER_SEC;
#pragma omp barrier 
#endif

  res = agbnp3_gb_deruv_nolist_ps(agb, agbw_h, no_init_frozen);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_gb_deruv_nolist_ps()\n");
#pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;

#ifdef ATIMER
#pragma omp barrier
  if(!iproc) {
    endtime = (float)clock()/CLOCKS_PER_SEC;
    timer_deruv += endtime - startime;
  }
#endif

  /*                                                                 */
  /* derivatives due to changes in self volumes                      */
  /*                                                                 */
#ifdef ATIMER
  if(!iproc) startime = (float)clock()/CLOCKS_PER_SEC;
#pragma omp barrier 
#endif

  res = agbnp3_der_vp_rooti(agb, agbw_h, agb->x, agb->y, agb->z);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_der_vp()\n");
#pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;
  
#ifdef ATIMER
#pragma omp barrier
  if(!iproc) {
    endtime = (float)clock()/CLOCKS_PER_SEC;
    timer_dervp += endtime - startime;
  }
#endif

  /*                                                                         */
  /*  derivatives of cavity energy and quantities related to surface areas   */
  /*                                                                         */

#ifdef ATIMER
  if(!iproc) startime = (float)clock()/CLOCKS_PER_SEC;
#pragma omp barrier 
#endif

  res = agbnp3_cavity_dersgb_rooti(agb, agbw_h, agb->x, agb->y, agb->z);
  if(res != AGBNP_OK){
    agbnp3_errprint( "agbnp3_total_energy(): error in agbnp3_cavity_dersgb()\n");
#pragma omp atomic
    error += 1; 
  }
#pragma omp flush(error)
  if(error) goto ERROR;

#ifdef ATIMER
#pragma omp barrier
  if(!iproc) {
    endtime = (float)clock()/CLOCKS_PER_SEC;
    timer_dersgb += endtime - startime;
  }
#endif

  /*                                                                         */
  /* ----------------------------------------------------------------------  */
  /*                                                                         */

  /*
  {
    int iws;
    agbnp3_errprint("%d\n\n",agb->nwsat);
    for(iws = 0; iws < agb->nwsat ; iws++){
      agbnp3_errprint("X %lf %lf %lf   %lf %d\n",
	      agb->wsat[iws].pos[0],agb->wsat[iws].pos[1],agb->wsat[iws].pos[2],
	      agb->wsat[iws].sp, agb->wsat[iws].parent[0]);
    }
    agbnp3_errprint("HB correction energy: %lf\n",hbe);
  }
  */

#ifdef _OPENMP
  /* reduction of derivatives */
#pragma omp critical
  for(iat=0;iat<natoms;iat++){
    for(i=0;i<3;i++){
      agbw->dgbdr_h[iat][i] += agbw_h->dgbdr_h[iat][i];
    }
  }
#pragma omp critical
  for(iat=0;iat<natoms;iat++){
    for(i=0;i<3;i++){
      agbw->dvwdr_h[iat][i] += agbw_h->dvwdr_h[iat][i];
    }
  }
#pragma omp critical
  for(iat=0;iat<natoms;iat++){
    for(i=0;i<3;i++){
      agbw->decav_h[iat][i] += agbw_h->decav_h[iat][i];
    }
  }
#pragma omp critical
  for(iat=0;iat<natoms;iat++){
    for(i=0;i<3;i++){
      agbw->dehb[iat][i]    += agbw_h->dehb[iat][i];
    }
  }
#pragma omp barrier
#endif

  //printf("%d: done()\n",iproc);

  ERROR:
  nop;

  DONE:
  nop;

  }/* #pragma omp parallel */

  if(error){
    return AGBNP_ERR;
  }

  /* return derivatives */
  for(iat=0;iat<natoms;iat++){
    for(ki=0;ki<3;ki++){
      agb->dgbdr[iat][ki] = tokcalmol*agbw->dgbdr_h[iat][ki];
      agb->dvwdr[iat][ki] = agbw->dvwdr_h[iat][ki];
      agb->decav[iat][ki] = agbw->decav_h[iat][ki];
      agb->dehb[iat][ki]  = agbw->dehb[iat][ki];
    }
  }

  /* returns scaled volume factors */
  for(iat=0;iat<natoms;iat++){
    agb->sp[iat] = 1.0;
  }
  for(i=0;i<nheavyat;i++){
    iat = iheavyat[i];
    agb->sp[iat] = agbw->sp[iat];
  }

  /* return born radii */
  for(iat=0;iat<natoms;iat++){
    agb->br[iat] = agbw->br[iat];
  }
  
  // HB energy
  *ehb = agb->ehb;

#ifdef ATIMER
  timecounter += 1;
  if(timecounter%100==0){
    float total = timer_nblist + timer_volumes + timer_invb + timer_agb + 
      timer_gbdersconstvp + timer_deruv + timer_dervp + timer_dersgb + 
      timer_ws + timer_wsder;
    fproc = 1.0;
#ifdef _OPENMP
    fproc = 1./(float)agb->nprocs;
#endif
    printf("Timing:\n");
    printf("%40s: %f\n", "nblist:", timer_nblist*fproc);
    printf("%40s: %f\n", "volumes:", timer_volumes*fproc);
    printf("%40s: %f\n", "invb:", timer_invb*fproc);
    printf("%40s: %f\n", "agb:", timer_agb*fproc);
    printf("%40s: %f\n", "gbdersconstvp:", timer_gbdersconstvp*fproc);
    printf("%40s: %f\n", "deruv:", timer_deruv*fproc);
    printf("%40s: %f\n", "dervp:", timer_dervp*fproc);
    printf("%40s: %f\n", "dersgb:", timer_dersgb*fproc);
    printf("%40s: %f\n", "ws:", timer_ws*fproc);
    printf("%40s: %f\n", "wsder:", timer_wsder*fproc);
    printf("--------------------\n");
    printf("%40s: %f\n", "Total:", total*fproc);
    printf("%40s: %f\n", "Time x step:", total*fproc/timecounter);
  }
#endif

  return AGBNP_OK;
}

/* clear contents of water site, frees associated neighbor list */
int agbnp3_clr_wsat(WSat *wsat){
  if(wsat->nlist){
    free(wsat->nlist);
  }
  memset(wsat,0,sizeof(WSat));
  return AGBNP_OK;
}

/* copy contents of wsat1 into wsat2 */
int agbnp3_cpy_wsat(WSat *wsat1, WSat *wsat2){
  agbnp3_clr_wsat(wsat1);
  memcpy(wsat1, wsat2, sizeof(WSat));
  /* duplicate neighbor list */
  wsat1->nlist = (int *)malloc(wsat1->nlist_size*sizeof(int));
  memcpy(wsat1->nlist,wsat2->nlist,wsat1->nneigh*sizeof(int));
  return AGBNP_OK;
}

int agbnp3_create_wsatoms(AGBNPdata *agb, AGBworkdata *agbw){
  int iat, iws, nws;
  int incr, init;
  static const int min_incr = 20;
  WSat *wsat, *twsat;
  WSat twsatb[4];  /* temporary buffer for ws atoms */
  int error = 0;

  /* check that connection table exists */
  if(!agb->conntbl){
    agbnp3_errprint( "agbnp3_create_watoms(): unable to retrieve connection table.\n");
    return AGBNP_ERR;
  }

  /* allocate master list of ws atoms */
  if(!agbw->wsat){
    nws = 1*agb->nheavyat; /* initial number of ws atoms */
#pragma omp critical
    agbw->wsat = (WSat *)calloc(nws,sizeof(WSat));
    if(!agbw->wsat){
      agbnp3_errprint( "agbnp3_create_watoms(): unable to allocate list of ws atoms (%d WSat)\n",nws);
      return AGBNP_ERR;
    }
    agbw->wsat_size = nws;
  }

  /* reset ws buffer */
  memset(twsatb,0,4*sizeof(WSat));

  /* create ws atoms for each HB active atom */
  agbw->nwsat = 0;
  incr = agb->natoms/10;
  if(incr<min_incr) incr = min_incr;

#pragma omp for schedule(static,1)
  for(iat = 0; iat < agb->natoms ; iat++){
    if(error) continue;
    if(agb->hbtype[iat] == AGBNP_HB_INACTIVE) continue;
    if(agbw->wsat_size - agbw->nwsat < min_incr){
      nws = agbw->wsat_size + incr;
      agbw->wsat = (WSat *)realloc(agbw->wsat,nws*sizeof(WSat));
      if(!agbw->wsat){
	agbnp3_errprint( "agbnp3_create_watoms(): unable to re-allocate list of ws atoms (%d WSat)\n",nws);
	error = 1;
	continue;
      }
      /* init newly allocated memory area */ 
      memset(&(agbw->wsat[agbw->wsat_size]),0,
	     (nws - agbw->wsat_size)*sizeof(WSat));
      agbw->wsat_size = nws;
    }

    if(agbnp3_create_ws_ofatom(agb, iat, &nws, twsatb) != AGBNP_OK){
      agbnp3_errprint( "agbnp3_create_watoms(): error in agbnp3_create_ws_ofatom()\n");
      error = 1;
      continue;
    }
    for(iws = 0; iws < nws; iws++){
      twsat = &(twsatb[iws]);
      wsat = &(agbw->wsat[agbw->nwsat]);      
      agbnp3_cpy_wsat(wsat, twsat);
      agbw->nwsat += 1;
    }
  }

  /* clean up temporary ws buffer */
  for(iws = 0; iws < 4; iws++){
    twsat = &(twsatb[iws]);
    agbnp3_clr_wsat(twsat);
  }

  if(error){
    return AGBNP_ERR;
  }else{
    return AGBNP_OK;
  }

}


/* create water sites pseudo atoms for atom iat, stores ws atoms
   starting at location iws and returns the number of added ws atoms
   in nws */
int agbnp3_create_ws_ofatom(AGBNPdata *agb, int iat, int *nws, WSat *twsatb){
  *nws = 0;
  switch (agb->hbtype[iat]){
  case AGBNP_HB_INACTIVE:
    return AGBNP_OK;
  case AGBNP_HB_POLARH:
    if(agbnp3_create_ws_atoms_ph(agb, iat, nws, twsatb) != AGBNP_OK){
      /*
#pragma omp critical
      agbnp3_errprint( "agbnp3_create_ws_atoms(): error in agbnp3_create_ws_atoms_ph()\n");
      return AGBNP_ERR;
      */
      *nws = 0;
    }
    return AGBNP_OK;
  case AGBNP_HB_TRIGONAL:
    if(agb->conntbl->nne[iat] == 1){
      /* like O in carboxy group */
      if(agbnp3_create_ws_atoms_trigonal1(agb, iat, nws, twsatb) != AGBNP_OK){
        /*
#pragma omp critical
	agbnp3_errprint( "agbnp3_create_ws_atoms(): error in agbnp3_create_ws_atoms_sp2()\n");
	return AGBNP_ERR;
        */
        *nws = 0;
      }
    }else if(agb->conntbl->nne[iat] == 2){
      /* like aromatic N with a in-plane lone-pair */
      if(agbnp3_create_ws_atoms_trigonal2(agb, iat, nws, twsatb) != AGBNP_OK){
        /*
#pragma omp critical
	agbnp3_errprint( "agbnp3_create_ws_atoms(): error in agbnp3_create_ws_atoms_trigonal2()\n");
	return AGBNP_ERR;
        */
        *nws = 0;
      }
    }
    return AGBNP_OK;
  case AGBNP_HB_TRIGONAL_S:
    /* like O in carboxy group with 4 water sites */
    if(agbnp3_create_ws_atoms_trigonal_s(agb, iat, nws, twsatb) != AGBNP_OK){
      /*
#pragma omp critical
      agbnp3_errprint( "agbnp3_create_ws_atoms(): error in agbnp3_create_ws_atoms_trigonal_s()\n");
      return AGBNP_ERR;
      */
      *nws = 0;
    }
    return AGBNP_OK;
  case AGBNP_HB_TRIGONAL_OOP:
    /* two out-of-plane water sites on top of a trigonal atom */
    if(agbnp3_create_ws_atoms_trigonal_oop(agb, iat, nws, twsatb) != AGBNP_OK){
#pragma omp critical
      fprintf(stderr, "agbnp3_create_ws_atoms(): error in agbnp3_create_ws_atoms_trigonal_oop()\n");
      return AGBNP_ERR;
    }
    return AGBNP_OK;
  case AGBNP_HB_TETRAHEDRAL:
    if(agb->conntbl->nne[iat] == 2){
      /* like sp3 O and S */
      if(agbnp3_create_ws_atoms_tetrahedral2(agb, iat, nws, twsatb) != AGBNP_OK){
        /*
#pragma omp critical
	agbnp3_errprint( "agbnp3_create_ws_atoms(): error in agbnp3_create_ws_atoms_sp3_o()\n");
	return AGBNP_ERR;
        */
        *nws = 0;
      }
    }else if(agb->conntbl->nne[iat] == 3){
      /* like sp3 N */
      if(agbnp3_create_ws_atoms_tetrahedral3(agb, iat, nws, twsatb) != AGBNP_OK){
        /*
#pragma omp critical
	agbnp3_errprint( "agbnp3_create_ws_atoms(): error in agbnp3_create_ws_atoms_sp3_o()\n");
	return AGBNP_ERR;
        */
        *nws = 0;
      }
    }else if(agb->conntbl->nne[iat] == 1){
      if(agbnp3_create_ws_atoms_tetrahedral1(agb, iat, nws, twsatb) != AGBNP_OK){
        /*
#pragma omp critical
	agbnp3_errprint( "agbnp3_create_ws_atoms(): error in agbnp3_create_ws_atoms_sp3_o()\n");
	return AGBNP_ERR;
        */
        *nws = 0;
      }
    }
    return AGBNP_OK;
  default:
    return AGBNP_OK;
  }
}

int agbnp3_create_ws_atoms_ph(AGBNPdata *agb, int iat, 
			      int *nws, WSat *twsatb){
  int jat = -1; /* heavy atom the hydrogen is attached to */
  int i, j;
  float_a xw, yw, zw;
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  NeighList *conntbl = agb->conntbl;
  WSat *wsat;
  float_a der1[3][3], der2[3][3];

  if(conntbl->nne[iat] <= 0){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_ph(): atom %d is isolated, I can't do isolated atoms yet.\n", iat);
    return AGBNP_ERR;
  }
  /* find parent */
  for(i=0;i<agb->conntbl->nne[iat];i++){
    if(agb->agbw->isheavy[conntbl->neighl[iat][i]]){
      jat = conntbl->neighl[iat][i];
      break;
    }
  }
  if(jat<0){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_ph(): unable to find parent heavy atom of atom %d.\n", iat);
    return AGBNP_ERR;
  }
  agbnp3_place_wat_hydrogen(x[jat],y[jat],z[jat],
			   x[iat],y[iat],z[iat],
			   (float_a)AGBNP_HB_LENGTH,
			   &xw, &yw, &zw,
			   der1, der2);
  /* stores water sites */
  wsat = &(twsatb[0]);
  wsat->pos[0] = xw;
  wsat->pos[1] = yw;
  wsat->pos[2] = zw;  
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_POLARH;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 2;
  wsat->parent[0] = iat;
  wsat->parent[1] = jat;
  wsat->iseq = 0; 
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      wsat->dpos[0][i][j] = der1[i][j];
      wsat->dpos[1][i][j] = der2[i][j];
    }
  }
  wsat->khb = agb->hbcorr[iat];

  *nws = 1;
  return AGBNP_OK;
}

int agbnp3_update_ws_atoms_ph(AGBNPdata *agb, WSat *wsat){
  int iat = wsat->parent[0]; /* hydrogen */
  int jat = wsat->parent[1]; /* heavy atom */
  float_a xw, yw, zw;
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  float_a der1[3][3], der2[3][3];
  int i, j;

  agbnp3_place_wat_hydrogen(x[jat],y[jat],z[jat],
			   x[iat],y[iat],z[iat],
			   (float_a)AGBNP_HB_LENGTH,
			   &xw, &yw, &zw,
			   der1, der2);
  wsat->pos[0] = xw;
  wsat->pos[1] = yw;
  wsat->pos[2] = zw;  
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      wsat->dpos[0][i][j] = der1[i][j];
      wsat->dpos[1][i][j] = der2[i][j];
    }
  }

  return AGBNP_OK;
}




/* places a water site at distance d from donor along hydrogen-donor bond 
   Input:
    xd: position of heavy atom
    xh: position of hydrogen atom
    d: distance between donor and water site
   Output:
    xw: position of water site
*/
void agbnp3_place_wat_hydrogen(float_a xd, float_a yd, float_a zd,
			      float_a xh, float_a yh, float_a zh, 
			      float_a d, 
			      float_a *xw, float_a *yw, float_a *zw,
			      float_a der1[3][3], float_a der2[3][3]){
  float_a dx[3], w, w1, w2, w3, dinv, d2inv;
  int i,j;

  dx[0] = xh - xd;
  dx[1] = yh - yd;
  dx[2] = zh - zd;
  d2inv =  1./(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
  dinv = sqrt(d2inv);
  w = d*dinv;
  *xw = xd + w*dx[0];
  *yw = yd + w*dx[1];
  *zw = zd + w*dx[2];

  if(der1 && der2){
    w2 = w*d2inv;
    for(i=0;i<3;i++){
      w3 = w2*dx[i];
      for(j=0;j<3;j++){
	w1 = w3*dx[j]; 
	der1[i][j] = -w1;
	der2[i][j] =  w1;
      }
    }
    w1 = 1. - w;
    for(i=0;i<3;i++){
      der1[i][i] += w;
      der2[i][i] += w1;
    }
  }

}

int agbnp3_create_ws_atoms_trigonal1(AGBNPdata *agb, int iat, 
				     int *nws, WSat *twsatb){
  int i;
  int ir, ir1, ir2;
  float_a xw1, yw1, zw1;
  float_a xw2, yw2, zw2;
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  NeighList *conntbl = agb->conntbl;
  WSat *wsat;
  float_a der1[4][3][3], der2[4][3][3];
  /* assumed trigonal topology:

       R1
         \
          R-A 
         /
       R2
  */

  if(conntbl->nne[iat] != 1){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_trigonal1(): acceptor atom (%d) should have one connecting atoms. Found %d.\n", iat,conntbl->nne[iat]);
    return AGBNP_ERR;
  }
  /* R parent */
  ir = conntbl->neighl[iat][0];
  /* R1 and R2 parents */
  if(conntbl->nne[ir] != 3){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_trigonal1(): atom bound to acceptor atom (%d) should have 3 connecting atoms. Found %d.\n", ir,conntbl->nne[ir]);
    return AGBNP_ERR;
  }
  ir1 = iat;
  i = 0;
  while(ir1 == iat){
    if(i >= conntbl->nne[ir]){
      /*
#pragma omp critical
      */
      agbnp3_errprint( "agbnp3_create_ws_atoms_trigonal1(): error parsing connection table of R atom (atom %d).\n", ir);
      return AGBNP_ERR;
    }
    ir1 = conntbl->neighl[ir][i++];
  }
  ir2 = iat;
  i = 0;
  while(ir2 == iat || ir2 == ir1){
    if(i >= conntbl->nne[ir]){
      /*
#pragma omp critical
      */
      agbnp3_errprint( "agbnp3_create_ws_atoms_trigonal1(): error parsing connection table of R atom (atom %d).\n", ir);
      return AGBNP_ERR;
    }
    ir2 = conntbl->neighl[ir][i++];
  } 
  agbnp3_place_wat_trigonal1(x[iat],y[iat],z[iat],
			     x[ir],y[ir],z[ir],
			     x[ir1],y[ir1],z[ir1],
			     x[ir2],y[ir2],z[ir2],
			     (float_a)AGBNP_HB_LENGTH,
			     &xw1, &yw1, &zw1,
			     &xw2, &yw2, &zw2,
			     der1, der2);
  /* stores water sites */
  wsat = &(twsatb[0]);
  wsat->pos[0] = xw1;
  wsat->pos[1] = yw1;
  wsat->pos[2] = zw1;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TRIGONAL1;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 4;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir;
  wsat->parent[2] = ir1;
  wsat->parent[3] = ir2;
  wsat->iseq = 0;
  memcpy(wsat->dpos,der1,36*sizeof(float_a));
  wsat->khb = agb->hbcorr[iat];

  wsat = &(twsatb[1]);
  wsat->pos[0] = xw2;
  wsat->pos[1] = yw2;
  wsat->pos[2] = zw2;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TRIGONAL1;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 4;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir;
  wsat->parent[2] = ir1;
  wsat->parent[3] = ir2;  
  wsat->iseq = 1;
  memcpy(wsat->dpos,der2,36*sizeof(float_a));
  wsat->khb = agb->hbcorr[iat];

  *nws = 2;
  return AGBNP_OK;
}


int agbnp3_update_ws_atoms_trigonal1(AGBNPdata *agb, WSat *wsat,
				     WSat *wsat1, WSat *wsat2){
  int iat = wsat->parent[0];
  int ir =  wsat->parent[1];
  int ir1 = wsat->parent[2];
  int ir2 = wsat->parent[3];
  float_a xw1, yw1, zw1;
  float_a xw2, yw2, zw2;
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  float_a der1[4][3][3], der2[4][3][3];

  agbnp3_place_wat_trigonal1(x[iat],y[iat],z[iat],
			     x[ir],y[ir],z[ir],
			     x[ir1],y[ir1],z[ir1],
			     x[ir2],y[ir2],z[ir2],
			     (float_a)AGBNP_HB_LENGTH,
			     &xw1, &yw1, &zw1,
			     &xw2, &yw2, &zw2,
			     der1, der2);
  if(wsat1){
    wsat1->pos[0] = xw1;
    wsat1->pos[1] = yw1;
    wsat1->pos[2] = zw1;
    memcpy(wsat1->dpos,der1,36*sizeof(float_a));
  }

  if(wsat2){
    wsat2->pos[0] = xw2;
    wsat2->pos[1] = yw2;
    wsat2->pos[2] = zw2;
    memcpy(wsat2->dpos,der2,36*sizeof(float_a));
  }

  return AGBNP_OK;
}


/* places water sites for sp2 acceptors such as the carboxy oxygen:
   Input:
    xa: position of acceptor
    xr: position of atom bound to acceptor
    xr1, xr2: positions of atoms flanking atom r
    d: distance between acceptor and water site
   Output:
    xw1,xw2: positions of water sites 
    derx[p][i][j]: gradient of ith coordinate of water with respect to
                   jth coordinate of parent p 
*/
void agbnp3_place_wat_trigonal1(float_a xa, float_a ya, float_a za, 
				float_a xr, float_a yr, float_a zr, 
				float_a xr1, float_a yr1, float_a zr1, 
				float_a xr2, float_a yr2, float_a zr2,
				float_a d,
				float_a *xw1, float_a *yw1, float_a *zw1,
				float_a *xw2, float_a *yw2, float_a *zw2,
				float_a der1[4][3][3], float_a der2[4][3][3]){

  int i, j;
  float_a dx1[3], dx2[3], d1i, d2i, w1, w2;

  dx1[0] = xr1 - xr;
  dx1[1] = yr1 - yr;
  dx1[2] = zr1 - zr;
  d1i = 1./sqrt(dx1[0]*dx1[0]+dx1[1]*dx1[1]+dx1[2]*dx1[2]);
  w1 = -d*d1i;
  *xw1 = xa + w1*dx1[0];
  *yw1 = ya + w1*dx1[1];
  *zw1 = za + w1*dx1[2];

  dx2[0] = xr2 - xr;
  dx2[1] = yr2 - yr;
  dx2[2] = zr2 - zr;
  d2i = 1./sqrt(dx2[0]*dx2[0]+dx2[1]*dx2[1]+dx2[2]*dx2[2]);
  w2 = -d*d2i;
  *xw2 = xa + w2*dx2[0];
  *yw2 = ya + w2*dx2[1];
  *zw2 = za + w2*dx2[2];

  if(der1 && der2){
    /* unit vectors */
    dx1[0] *= d1i;
    dx1[1] *= d1i;
    dx1[2] *= d1i;

    dx2[0] *= d2i;
    dx2[1] *= d2i;
    dx2[2] *= d2i;

    memset(der1,0,36*sizeof(float_a));
    memset(der2,0,36*sizeof(float_a));

    /* acceptor */
    for(i=0;i<3;i++){
      der1[0][i][i] = 1.;
      der2[0][i][i] = 1.;
    }

    /* R */
    for(i=0;i<3;i++){
      der1[1][i][i] = -w1;
      der2[1][i][i] = -w2;
    }
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der1[1][i][j] += w1*dx1[i]*dx1[j];
	der2[1][i][j] += w2*dx2[i]*dx2[j];
      }
    }

    /* R1/R2 */
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der1[2][i][j] = -der1[1][i][j];
	der2[3][i][j] = -der2[1][i][j];
      }
    }
  }

}

int agbnp3_create_ws_atoms_trigonal_s(AGBNPdata *agb, int iat, 
				      int *nws, WSat *twsatb){
  int i;
  int ir, ir1, ir2;
  float_a xw1, yw1, zw1;
  float_a xw2, yw2, zw2;
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  NeighList *conntbl = agb->conntbl;
  WSat *wsat;
  float_a der1[4][3][3], der2[4][3][3];
  /* assumed trigonal topology:

       R1
         \
          R-A 
         /
       R2
  */

  if(conntbl->nne[iat] != 1){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_trigonal_s(): acceptor atom (%d) should have one connecting atoms. Found %d.\n", iat,conntbl->nne[iat]);
    return AGBNP_ERR;
  }
  /* R parent */
  ir = conntbl->neighl[iat][0];
  /* R1 and R2 parents */
  if(conntbl->nne[ir] != 3){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_trigonal_s(): atom bound to acceptor atom (%d) should have 3 connecting atoms. Found %d.\n", ir,conntbl->nne[ir]);
    return AGBNP_ERR;
  }
  ir1 = iat;
  i = 0;
  while(ir1 == iat){
    if(i >= conntbl->nne[ir]){
      /*
#pragma omp critical
      */
      agbnp3_errprint( "agbnp3_create_ws_atoms_trigonal_s(): error parsing connection table of R atom (atom %d).\n", ir);
      return AGBNP_ERR;
    }
    ir1 = conntbl->neighl[ir][i++];
  }
  ir2 = iat;
  i = 0;
  while(ir2 == iat || ir2 == ir1){
    if(i >= conntbl->nne[ir]){
      /*
#pragma omp critical
      */
      agbnp3_errprint( "agbnp3_create_ws_atoms_trigonal_s(): error parsing connection table of R atom (atom %d).\n", ir);
      return AGBNP_ERR;
    }
    ir2 = conntbl->neighl[ir][i++];
  } 

  /* place first two water sites (in plane) */
  agbnp3_place_wat_trigonal1(x[iat],y[iat],z[iat],
			     x[ir],y[ir],z[ir],
			     x[ir1],y[ir1],z[ir1],
			     x[ir2],y[ir2],z[ir2],
			     (float_a)AGBNP_HB_LENGTH,
			     &xw1, &yw1, &zw1,
			     &xw2, &yw2, &zw2,
			     der1, der2);
  /* stores first two water sites */
  wsat = &(twsatb[0]);
  wsat->pos[0] = xw1;
  wsat->pos[1] = yw1;
  wsat->pos[2] = zw1;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TRIGONAL_S;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 4;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir;
  wsat->parent[2] = ir1;
  wsat->parent[3] = ir2;
  wsat->iseq = 0;
  memcpy(wsat->dpos,der1,36*sizeof(float_a));
  wsat->khb = agb->hbcorr[iat];

  wsat = &(twsatb[1]);
  wsat->pos[0] = xw2;
  wsat->pos[1] = yw2;
  wsat->pos[2] = zw2;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TRIGONAL_S;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 4;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir;
  wsat->parent[2] = ir1;
  wsat->parent[3] = ir2;  
  wsat->iseq = 1;
  memcpy(wsat->dpos,der2,36*sizeof(float_a));
  wsat->khb = agb->hbcorr[iat];

  /* places second set of water sites (out of plane) */
  agbnp3_place_wat_trigonal_s(x[iat],y[iat],z[iat],
			     x[ir],y[ir],z[ir],
			     x[ir1],y[ir1],z[ir1],
			     x[ir2],y[ir2],z[ir2],
			     (float_a)AGBNP_HB_LENGTH,
			     &xw1, &yw1, &zw1,
			     &xw2, &yw2, &zw2,
			     der1, der2);
  /* stores second set of water sites */
  wsat = &(twsatb[2]);
  wsat->pos[0] = xw1;
  wsat->pos[1] = yw1;
  wsat->pos[2] = zw1;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TRIGONAL_S;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 4;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir;
  wsat->parent[2] = ir1;
  wsat->parent[3] = ir2;
  wsat->iseq = 2;
  memcpy(wsat->dpos,der1,36*sizeof(float_a));
  wsat->khb = agb->hbcorr[iat];

  wsat = &(twsatb[3]);
  wsat->pos[0] = xw2;
  wsat->pos[1] = yw2;
  wsat->pos[2] = zw2;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TRIGONAL_S;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 4;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir;
  wsat->parent[2] = ir1;
  wsat->parent[3] = ir2;  
  wsat->iseq = 3;
  memcpy(wsat->dpos,der2,36*sizeof(float_a));
  wsat->khb = agb->hbcorr[iat];

  *nws = 4;
  return AGBNP_OK;
}


int agbnp3_update_ws_atoms_trigonal_s(AGBNPdata *agb, WSat *wsat,
				      WSat *wsat1, WSat *wsat2,
				      WSat *wsat3, WSat *wsat4){ 
  int iat = wsat->parent[0];
  int ir =  wsat->parent[1];
  int ir1 = wsat->parent[2];
  int ir2 = wsat->parent[3];
  float_a xw1, yw1, zw1;
  float_a xw2, yw2, zw2;
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  float_a der1[4][3][3], der2[4][3][3];

  /* place first two water sites (in plane) */
  agbnp3_place_wat_trigonal1(x[iat],y[iat],z[iat],
			     x[ir],y[ir],z[ir],
			     x[ir1],y[ir1],z[ir1],
			     x[ir2],y[ir2],z[ir2],
			     (float_a)AGBNP_HB_LENGTH,
			     &xw1, &yw1, &zw1,
			     &xw2, &yw2, &zw2,
			     der1, der2);
  /* stores first two water sites */
  if(wsat1){
    wsat1->pos[0] = xw1;
    wsat1->pos[1] = yw1;
    wsat1->pos[2] = zw1;
    memcpy(wsat1->dpos,der1,36*sizeof(float_a));
  }

  if(wsat2){
    wsat2->pos[0] = xw2;
    wsat2->pos[1] = yw2;
    wsat2->pos[2] = zw2;
    memcpy(wsat2->dpos,der2,36*sizeof(float_a));
  }

  /* places second set of water sites (out of plane) */
  agbnp3_place_wat_trigonal_s(x[iat],y[iat],z[iat],
			     x[ir],y[ir],z[ir],
			     x[ir1],y[ir1],z[ir1],
			     x[ir2],y[ir2],z[ir2],
			     (float_a)AGBNP_HB_LENGTH,
			     &xw1, &yw1, &zw1,
			     &xw2, &yw2, &zw2,
			     der1, der2);
  /* stores second set of water sites */
  if(wsat3){
    wsat3->pos[0] = xw1;
    wsat3->pos[1] = yw1;
    wsat3->pos[2] = zw1;
    memcpy(wsat3->dpos,der1,36*sizeof(float_a));
  }

  if(wsat4){
    wsat4->pos[0] = xw2;
    wsat4->pos[1] = yw2;
    wsat4->pos[2] = zw2;
    memcpy(wsat4->dpos,der2,36*sizeof(float_a));
  }

  return AGBNP_OK;
}


void agbnp3_place_wat_trigonal_s(float_a xa, float_a ya, float_a za, 
				float_a xr, float_a yr, float_a zr, 
				float_a xr1, float_a yr1, float_a zr1, 
				float_a xr2, float_a yr2, float_a zr2,
				float_a d,
				float_a *xw1, float_a *yw1, float_a *zw1,
				float_a *xw2, float_a *yw2, float_a *zw2,
				float_a der1[4][3][3], float_a der2[4][3][3]){

  int i, j;
  float_a dx1[3], dx0[3], d2i, d0i, w, w1, w2;
  float_a uin[3], uout[3];
  float_a drprp[3][3];
  float_a drt1[3][3], drt2[3][3];
  float_a costh =  0.5;      /* cos(60) */
  float_a sinth =  0.866025; /* sin(60) */

  dx0[0] = xa - xr;
  dx0[1] = ya - yr;
  dx0[2] = za - zr;

  dx1[0] = xr1 - xr;
  dx1[1] = yr1 - yr;
  dx1[2] = zr1 - zr;

  /* in plane direction given by ar */
  d0i = 1./sqrt(dx0[0]*dx0[0]+dx0[1]*dx0[1]+dx0[2]*dx0[2]);
  uin[0] = dx0[0]*d0i;
  uin[1] = dx0[1]*d0i;
  uin[2] = dx0[2]*d0i;

  /* out of plane given by (r1r x ar ) */
  agbnp3_cross_product(dx1, dx0, uout, NULL, NULL);
  d2i = 1./sqrt(uout[0]*uout[0]+uout[1]*uout[1]+uout[2]*uout[2]);
  uout[0] *= d2i;
  uout[1] *= d2i;
  uout[2] *= d2i;
  
  *xw1 = xa + d*(costh*uin[0]+sinth*uout[0]);
  *yw1 = ya + d*(costh*uin[1]+sinth*uout[1]);
  *zw1 = za + d*(costh*uin[2]+sinth*uout[2]);
  
  *xw2 = xa + d*(costh*uin[0]-sinth*uout[0]);
  *yw2 = ya + d*(costh*uin[1]-sinth*uout[1]);
  *zw2 = za + d*(costh*uin[2]-sinth*uout[2]);

  if(der1 && der2){
    /* derx[p][i][j]: gradient of i-th coordinate of water with respect to
       j-th coordinate of parent p */

    memset(der1,0,36*sizeof(float_a));
    memset(der2,0,36*sizeof(float_a));

    /* gredient from uin */
    agbnp3_der_unitvector(uin, d0i, drprp);
    w = d*costh;
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	w1 = w*drprp[i][j];
	der1[0][i][j] = der2[0][i][j] =  w1; 
	der1[1][i][j] = der2[1][i][j] = -w1;
      }
    }

    /* gradient from uout */
    agbnp3_der_unitvector(uout, d2i, drprp);
    for(i=0;i<3;i++){
      agbnp3_cross_product(dx1, drprp[i], drt1[i],NULL,NULL);
      agbnp3_cross_product(dx0, drprp[i], drt2[i],NULL,NULL);
    }

    w = d*sinth;
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	w1 = -w*drt1[i][j];
	w2 =  w*drt2[i][j];

	der1[0][i][j] += w1;
	der1[1][i][j] -= (w1+w2);
	der1[2][i][j] += w2;

	der2[0][i][j] -= w1;
	der2[1][i][j] += (w1+w2);
	der2[2][i][j] -= w2;

	/* printf("%f %f %f\n",w1,w2, der1[2][1][1]); */

      }
    }

    for(i=0;i<3;i++){
	der1[0][i][i] += 1.;
	der2[0][i][i] += 1.;
    }

  }

}



int agbnp3_create_ws_atoms_trigonal_oop(AGBNPdata *agb, int iat, 
					int *nws, WSat *twsatb){
  int i, p, ii, jj;
  int ia, ir1, ir2, ir3;
  float_a xw1, yw1, zw1;
  float_a xw2, yw2, zw2;
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  NeighList *conntbl = agb->conntbl;
  WSat *wsat;
  float_a der1[4][3][3], der2[4][3][3];
  /* assumed trigonal topology:

       R1
         \
          A-R3       water sites are placed across A along the normal    
         /           of the R1, R2, R3 plane
       R2
  */

  if(conntbl->nne[iat] != 3){
#pragma omp critical
    fprintf(stderr, "agbnp3_create_ws_atoms_trigonal_oop(): acceptor atom (%d) should have one connecting atoms. Found %d.\n", iat,conntbl->nne[iat]);
    return AGBNP_ERR;
  }
  /* R1, R2 and R3 parents */
  ir1 = conntbl->neighl[iat][0];
  ir2 = conntbl->neighl[iat][1];
  ir3 = conntbl->neighl[iat][2];
  agbnp3_place_wat_trigonal_oop(x[iat],y[iat],z[iat],
				x[ir1],y[ir1],z[ir1],
				x[ir2],y[ir2],z[ir2],
				x[ir3],y[ir3],z[ir3],
				(float_a)AGBNP_HB_LENGTH,
				&xw1, &yw1, &zw1,
				&xw2, &yw2, &zw2,
				der1, der2);
  /* stores water sites */
  wsat = &(twsatb[0]);
  wsat->pos[0] = xw1;
  wsat->pos[1] = yw1;
  wsat->pos[2] = zw1;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TRIGONAL_OOP;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 4;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir1;
  wsat->parent[2] = ir2;
  wsat->parent[3] = ir3;
  wsat->iseq = 0;
  memcpy(wsat->dpos,der1,36*sizeof(float_a));
  wsat->khb = agb->hbcorr[iat];

  wsat = &(twsatb[1]);
  wsat->pos[0] = xw2;
  wsat->pos[1] = yw2;
  wsat->pos[2] = zw2;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TRIGONAL_OOP;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 4;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir1;
  wsat->parent[2] = ir2;
  wsat->parent[3] = ir3;  
  wsat->iseq = 1;
  memcpy(wsat->dpos,der2,36*sizeof(float_a));
  wsat->khb = agb->hbcorr[iat];

  *nws = 2;
  return AGBNP_OK;
}

int agbnp3_update_ws_atoms_trigonal_oop(AGBNPdata *agb, WSat *wsat,
				     WSat *wsat1, WSat *wsat2){
  int iat = wsat->parent[0];
  int ir1 =  wsat->parent[1];
  int ir2 = wsat->parent[2];
  int ir3 = wsat->parent[3];
  float_a xw1, yw1, zw1;
  float_a xw2, yw2, zw2;
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  float_a der1[4][3][3], der2[4][3][3];

  agbnp3_place_wat_trigonal_oop(x[iat],y[iat],z[iat],
				x[ir1],y[ir1],z[ir1],
				x[ir2],y[ir2],z[ir2],
				x[ir3],y[ir3],z[ir3],
				(float_a)AGBNP_HB_LENGTH,
				&xw1, &yw1, &zw1,
				&xw2, &yw2, &zw2,
				der1, der2);
  if(wsat1){
    wsat1->pos[0] = xw1;
    wsat1->pos[1] = yw1;
    wsat1->pos[2] = zw1;
    memcpy(wsat1->dpos,der1,36*sizeof(float_a));
  }

  if(wsat2){
    wsat2->pos[0] = xw2;
    wsat2->pos[1] = yw2;
    wsat2->pos[2] = zw2;
    memcpy(wsat2->dpos,der2,36*sizeof(float_a));
  }

  return AGBNP_OK;
}


void agbnp3_place_wat_trigonal_oop(float_a xa, float_a ya, float_a za, 
				float_a xr1, float_a yr1, float_a zr1, 
				float_a xr2, float_a yr2, float_a zr2, 
				float_a xr3, float_a yr3, float_a zr3,
				float_a d1,
				float_a *xw1, float_a *yw1, float_a *zw1,
				float_a *xw2, float_a *yw2, float_a *zw2,
				float_a der1[4][3][3], float_a der2[4][3][3]){

  int i, j;
  float_a fxw1[3], fxw2[3];
  float_a x0[3] = {xa, ya, za};
  float_a x1[3] = {xr1, yr1, zr1};
  float_a x2[3] = {xr2, yr2, zr2};
  float_a x3[3] = {xr3, yr3, zr3};
  float_a d = d1 + 0.2;
  float_a v1[3], v2[3], nu[3], nun, dn;
  float_a nu_outer[3][3];
  float_a dernu1[3][3], dernu2[3][3];
  float_a derxv1[3][3], derxv2[3][3];


  /* displacements relative to x1 */
  for(i=0;i<3;i++){
    v1[i] = x2[i] - x1[i];
  }
  for(i=0;i<3;i++){
    v2[i] = x3[i] - x1[i];
  }

  /* unit vector normal to 1,2,3 plane */
  agbnp3_cross_product(v2, v1, nu, dernu2, dernu1);
  nun = sqrt(nu[0]*nu[0]+nu[1]*nu[1]+nu[2]*nu[2]);
  for(i=0;i<3;i++){
    nu[i] /= nun;
  }

  /* water site positions */
  for(i=0;i<3;i++){
    fxw1[i] = x0[i] + d*nu[i];
  }
  for(i=0;i<3;i++){
    fxw2[i] = x0[i] - d*nu[i];
  }
  *xw1 = fxw1[0];
  *yw1 = fxw1[1];
  *zw1 = fxw1[2];
  *xw2 = fxw2[0];
  *yw2 = fxw2[1];
  *zw2 = fxw2[2];

  if(der1 && der2){

    /* I - nu x nu ; x=outer product */
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	nu_outer[i][j] = 0.;
      }
    }
    for(i=0;i<3;i++){
      nu_outer[i][i] = 1.;
    }
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	nu_outer[i][j] -= nu[i]*nu[j];
      }
    }
    
    dn = d/nun;

    /* derivatives w.r.t x0 */
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der1[0][i][j] = 0.;
      }
    }
    for(i=0;i<3;i++){
      der1[0][i][i] = 1.;
    }
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der2[0][i][j] = der1[0][i][j];
      }
    }

    agbnp3_matmul(nu_outer, dernu1, derxv1);
    agbnp3_matmul(nu_outer, dernu2, derxv2);

    /* derivatives w.r.t x2 */
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der1[2][i][j] = dn*derxv1[i][j];
      }
    }
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der2[2][i][j] = -dn*derxv1[i][j];
      }
    }

    /* derivatives w.r.t x3 */
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der1[3][i][j] = dn*derxv2[i][j];
      }
    }
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der2[3][i][j] = -dn*derxv2[i][j];
      }
    }

    /* derivatives w.r.t x1 */
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der1[1][i][j] = -(der1[2][i][j]+der1[3][i][j]);
      }
    }
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der2[1][i][j] = -(der2[2][i][j]+der2[3][i][j]);
      }
    }
    

  }

}




int agbnp3_create_ws_atoms_trigonal2(AGBNPdata *agb, int iat, 
				     int *nws, WSat *twsatb){
  int ir1, ir2;
  float_a xw, yw, zw;
  float_a der[3][3][3];
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  NeighList *conntbl = agb->conntbl;
  WSat *wsat;
  /* assumed trigonal topology:

       R1
         \
          A 
         /
       R2
  */

  if(conntbl->nne[iat] != 2){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_trigonal2(): acceptor atom (%d) should have two connected atoms. Found %d.\n", iat,conntbl->nne[iat]);
    return AGBNP_ERR;
  }
  /* parents */
  ir1 = conntbl->neighl[iat][0];
  ir2 = conntbl->neighl[iat][1];
  agbnp3_place_wat_trigonal2(x[iat],y[iat],z[iat],
			     x[ir1],y[ir1],z[ir1],
			     x[ir2],y[ir2],z[ir2],
			     (float_a)AGBNP_HB_LENGTH,
			     &xw, &yw, &zw,
			     der);
  /* stores water sites */
  wsat = &(twsatb[0]);
  wsat->pos[0] = xw;
  wsat->pos[1] = yw;
  wsat->pos[2] = zw;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TRIGONAL2;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 3;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir1;
  wsat->parent[2] = ir2;
  wsat->iseq = 0;
  wsat->khb = agb->hbcorr[iat];
  memcpy(wsat->dpos,der,27*sizeof(float_a));

  *nws = 1;
  return AGBNP_OK;
}


int agbnp3_update_ws_atoms_trigonal2(AGBNPdata *agb, WSat *wsat){
  int iat = wsat->parent[0];
  int ir1 = wsat->parent[1];
  int ir2 = wsat->parent[2];
  float_a xw, yw, zw;
  float_a der[3][3][3];
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;

  agbnp3_place_wat_trigonal2(x[iat],y[iat],z[iat],
			     x[ir1],y[ir1],z[ir1],
			     x[ir2],y[ir2],z[ir2],
			     (float_a)AGBNP_HB_LENGTH,
			     &xw, &yw, &zw,
			     der);
  wsat->pos[0] = xw;
  wsat->pos[1] = yw;
  wsat->pos[2] = zw;
  memcpy(wsat->dpos,der,27*sizeof(float_a));

  return AGBNP_OK;
}

void agbnp3_place_wat_trigonal2(float_a xa, float_a ya, float_a za, 
				float_a xr1, float_a yr1, float_a zr1, 
				float_a xr2, float_a yr2, float_a zr2,
				float_a d,
				float_a *xw, float_a *yw, float_a *zw,
				float_a der[3][3][3]){
  int i,j;
  float_a dx, dy, dz, w, w1, w2;
  float_a d1[3], d2[3], dw[3], idist1, idist2, idistw;
  float_a drprp[3][3], dr1r1[3][3], dr2r2[3][3], drp1[3][3], drp2[3][3];

  /* unit vectors of bonds */
  dx = xr1 - xa;
  dy = yr1 - ya;
  dz = zr1 - za;
  idist1 =  1./sqrt(dx*dx+dy*dy+dz*dz);
  d1[0] = idist1*dx;
  d1[1] = idist1*dy;
  d1[2] = idist1*dz;

  dx = xr2 - xa;
  dy = yr2 - ya;
  dz = zr2 - za;
  idist2 =  1./sqrt(dx*dx+dy*dy+dz*dz);
  d2[0] = idist2*dx;
  d2[1] = idist2*dy;
  d2[2] = idist2*dz;

  /* water site along the negative of the sum of the bond vectors */
  dw[0] = d1[0] + d2[0];
  dw[1] = d1[1] + d2[1];
  dw[2] = d1[2] + d2[2];
  idistw = 1./sqrt(dw[0]*dw[0]+dw[1]*dw[1]+dw[2]*dw[2]);
  dw[0] *= idistw;
  dw[1] *= idistw;
  dw[2] *= idistw;

  *xw = xa - d*dw[0]; 
  *yw = ya - d*dw[1]; 
  *zw = za - d*dw[2]; 

  if(der){
    agbnp3_der_unitvector(dw, idistw, drprp);
    agbnp3_der_unitvector(d1, idist1, dr1r1);
    agbnp3_der_unitvector(d2, idist2, dr2r2);
    agbnp3_matmul(drprp, dr1r1, drp1);
    agbnp3_matmul(drprp, dr2r2, drp2);

    w = -d;
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	w1 = w*drp1[i][j];
	w2 = w*drp2[i][j];
	der[1][i][j] = w1; 
	der[2][i][j] = w2;
	der[0][i][j] = -(w1+w2);
      }
    }
    for(i=0;i<3;i++){
      der[0][i][i] += 1.;
    }
  }

}

int agbnp3_create_ws_atoms_tetrahedral2(AGBNPdata *agb, int iat, 
					int *nws,  WSat *twsatb){
  int ir1, ir2;
  float_a xw1, yw1, zw1;
  float_a xw2, yw2, zw2;
  float_a der1[3][3][3], der2[3][3][3];
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  NeighList *conntbl = agb->conntbl;
  WSat *wsat;

  /* assumed tetrahedral topology:

       R1
         \
          A 
         /
       R2
  */

  if(conntbl->nne[iat] != 2){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_tetrahedral2(): acceptor atom (%d) should have two connecting atoms. Found %d.\n", iat,conntbl->nne[iat]);
    return AGBNP_ERR;
  }
  /* R1 and R2 parents */
  ir1 = conntbl->neighl[iat][0];
  ir2 = conntbl->neighl[iat][1];
  agbnp3_place_wat_tetrahedral2(x[iat],y[iat],z[iat],
				x[ir1],y[ir1],z[ir1],
				x[ir2],y[ir2],z[ir2],
				(float_a)AGBNP_HB_LENGTH,
				&xw1, &yw1, &zw1,
				&xw2, &yw2, &zw2,
				der1, der2);
  /* stores water sites */
  wsat = &(twsatb[0]);
  wsat->pos[0] = xw1;
  wsat->pos[1] = yw1;
  wsat->pos[2] = zw1;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TETRAHEDRAL2;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 3;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir1;
  wsat->parent[2] = ir2;
  wsat->iseq = 0;
  wsat->khb = agb->hbcorr[iat];
  memcpy(wsat->dpos,der1,27*sizeof(float_a));

  wsat = &(twsatb[1]);
  wsat->pos[0] = xw2;
  wsat->pos[1] = yw2;
  wsat->pos[2] = zw2;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TETRAHEDRAL2;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 3;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir1;
  wsat->parent[2] = ir2;
  wsat->iseq = 1;
  wsat->khb = agb->hbcorr[iat];
  memcpy(wsat->dpos,der2,27*sizeof(float_a));

  *nws = 2;
  return AGBNP_OK;
}

int agbnp3_update_ws_atoms_tetrahedral2(AGBNPdata *agb, WSat *wsat,
					WSat *wsat1, WSat *wsat2){
  int iat = wsat->parent[0];
  int ir1 = wsat->parent[1];
  int ir2 = wsat->parent[2];
  float_a xw1, yw1, zw1;
  float_a xw2, yw2, zw2;
  float_a der1[3][3][3], der2[3][3][3];
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;

  agbnp3_place_wat_tetrahedral2(x[iat],y[iat],z[iat],
				x[ir1],y[ir1],z[ir1],
				x[ir2],y[ir2],z[ir2],
				(float_a)AGBNP_HB_LENGTH,
				&xw1, &yw1, &zw1,
				&xw2, &yw2, &zw2,
				der1, der2);
  if(wsat1){
    wsat1->pos[0] = xw1;
    wsat1->pos[1] = yw1;
    wsat1->pos[2] = zw1;
    memcpy(wsat1->dpos,der1,27*sizeof(float_a));
  }

  if(wsat2){
    wsat2->pos[0] = xw2;
    wsat2->pos[1] = yw2;
    wsat2->pos[2] = zw2;
    memcpy(wsat2->dpos,der2,27*sizeof(float_a));
  }

  return AGBNP_OK;
}


/* cross product between vectors a and b */
void agbnp3_cross_product(float_a a[3], float_a b[3], float_a c[3],
			  float_a dera[3][3], float_a derb[3][3]){
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
  if(dera && derb){
    /* derivatives follow directly from the properties of the Levi-Civita
       symbols epsilon_ijk */
    dera[0][0] =  0.;
    dera[0][1] =  b[2];
    dera[0][2] = -b[1];
    
    dera[1][0] = -b[2];
    dera[1][1] =  0.;
    dera[1][2] =  b[0];
    
    dera[2][0] =  b[1];
    dera[2][1] = -b[0];
    dera[2][2] =  0.;

    derb[0][0] =  0.;
    derb[0][1] = -a[2];
    derb[0][2] =  a[1];
    
    derb[1][0] =  a[2];
    derb[1][1] =  0.;
    derb[1][2] = -a[0];
    
    derb[2][0] = -a[1];
    derb[2][1] =  a[0];
    derb[2][2] =  0.;

  }

}

/* computes derivative matrix of a unit vector 'u' of a vector 'r' 
   with respect to 'r'.

   invr (supplied) is the inverse of the length of 'r'

   der[i][j] is the derivative of u[i] with respect to r[j]

   the formula is:

   du/dr = (1/r) { I - u x u }

*/
void agbnp3_der_unitvector(float_a u[3], float_a invr, 
			   float_a der[3][3]){
  int i,j;
  float_a s = -invr;

  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      der[i][j] = s*u[i]*u[j];
    }
  }
  for(i=0;i<3;i++){
    der[i][i] += invr;
  }
}

void agbnp3_place_wat_tetrahedral2(float_a xa, float_a ya, float_a za, 
				   float_a xr1, float_a yr1, float_a zr1, 
				   float_a xr2, float_a yr2, float_a zr2,
				   float_a d,
				   float_a *xw1, float_a *yw1, float_a *zw1,
				   float_a *xw2, float_a *yw2, float_a *zw2,
				   float_a der1[3][3][3], float_a der2[3][3][3]){
  int i,j;
  float_a costh = -0.577350269; /* -1/sqrt(3) */
  float_a sinth =  0.816496581; /* sqrt(2/3) */
  float_a dx, dy, dz, d1[3], d2[3], dpin[3], dout[3], w, w1, w2;
  float_a r1[3], r2[3];
  float_a dist1, dist2, distin, distout, u[3];
  float_a drprp[3][3], dr1r1[3][3], dr2r2[3][3];
  float_a drtrt[3][3], drp1[3][3], drp2[3][3];
  float_a drt1[3][3], drt2[3][3];


  /* w1 = A + d costh uin + d sinth uout
     w2 = A + d costh uin - d sinth uout

     where: A is pos of central atom, d is HB bond length, and uin and uout
     are the in plane and out of plane unit vectors.
  */ 

  /* unit vectors of bonds */
  r1[0] = dx = xr1 - xa;
  r1[1] = dy = yr1 - ya;
  r1[2] = dz = zr1 - za;
  dist1 =  1./sqrt(dx*dx+dy*dy+dz*dz);
  d1[0] = dist1*dx;
  d1[1] = dist1*dy;
  d1[2] = dist1*dz;

  r2[0] = dx = xr2 - xa;
  r2[1] = dy = yr2 - ya;
  r2[2] = dz = zr2 - za;
  dist2 =  1./sqrt(dx*dx+dy*dy+dz*dz);
  d2[0] = dist2*dx;
  d2[1] = dist2*dy;
  d2[2] = dist2*dz;

  /* in plane direction (sum of bond unit vectors) */
  dx = d1[0] + d2[0];
  dy = d1[1] + d2[1];
  dz = d1[2] + d2[2];
  distin =  1./sqrt(dx*dx+dy*dy+dz*dz);
  dpin[0] = distin*dx;
  dpin[1] = distin*dy;
  dpin[2] = distin*dz;

  /* out of plane projection (cross product between bond vectors) */
  dx = r2[1]*r1[2] - r2[2]*r1[1];
  dy = r2[2]*r1[0] - r2[0]*r1[2];
  dz = r2[0]*r1[1] - r2[1]*r1[0];
  distout =  1./sqrt(dx*dx+dy*dy+dz*dz);
  dout[0] = distout*dx;
  dout[1] = distout*dy;
  dout[2] = distout*dz;

  /* first LP position */
  u[0] = costh*dpin[0] + sinth*dout[0];
  u[1] = costh*dpin[1] + sinth*dout[1];
  u[2] = costh*dpin[2] + sinth*dout[2];
  *xw1 = xa + d*u[0];
  *yw1 = ya + d*u[1];
  *zw1 = za + d*u[2];
  /* second LP position */
  u[0] = costh*dpin[0] - sinth*dout[0];
  u[1] = costh*dpin[1] - sinth*dout[1];
  u[2] = costh*dpin[2] - sinth*dout[2];
  *xw2 = xa + d*u[0];
  *yw2 = ya + d*u[1];
  *zw2 = za + d*u[2];

  if(der1 && der2){
    /* derx[p][i][j]: gradient of i-th coordinate of water with respect to
       j-th coordinate of parent p */
    agbnp3_der_unitvector(dpin, distin, drprp);
    agbnp3_der_unitvector(d1, dist1, dr1r1);
    agbnp3_der_unitvector(d2, dist2, dr2r2);
    agbnp3_matmul(drprp, dr1r1, drp1);
    agbnp3_matmul(drprp, dr2r2, drp2);

    w = d*costh;
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	w1 = w*drp1[i][j];
	w2 = w*drp2[i][j];
	der1[1][i][j] = der2[1][i][j] = w1; 
	der1[2][i][j] = der2[2][i][j] = w2;
	der1[0][i][j] = der2[0][i][j] = -(w1+w2);
      }
    }

    agbnp3_der_unitvector(dout, distout, drtrt);
    for(i=0;i<3;i++){
      agbnp3_cross_product(drtrt[i], r2, drt1[i],NULL,NULL);
      agbnp3_cross_product(r1, drtrt[i], drt2[i],NULL,NULL);
    }

    w = d*sinth;
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	w1 = w*drt1[i][j];
	w2 = w*drt2[i][j];
	der1[1][i][j] += w1;
	der1[2][i][j] += w2;
	der2[1][i][j] -= w1;
	der2[2][i][j] -= w2;

	der1[0][i][j] -= (w1+w2);
	der2[0][i][j] += (w1+w2);
      }
    }

    for(i=0;i<3;i++){
	der1[0][i][i] += 1.;
	der2[0][i][i] += 1.;
    }


  }

}


int agbnp3_create_ws_atoms_tetrahedral3(AGBNPdata *agb, int iat, 
					int *nws, WSat *twsatb){
  int ir1, ir2, ir3;
  float_a xw, yw, zw;
  float_a der[4][3][3];
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  NeighList *conntbl = agb->conntbl;
  WSat *wsat;

  /* assumed tetrahedral topology:

       R1
         \
          A--R3 
         /
       R2
  */

  if(conntbl->nne[iat] != 3){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_tetrahedral3(): acceptor atom (%d) should have two connecting atoms. Found %d.\n", iat,conntbl->nne[iat]);
    return AGBNP_ERR;
  }
  /* R1, R2 and R3 parents */
  ir1 = conntbl->neighl[iat][0];
  ir2 = conntbl->neighl[iat][1];
  ir3 = conntbl->neighl[iat][2];
  agbnp3_place_wat_tetrahedral3(x[iat],y[iat],z[iat],
				x[ir1],y[ir1],z[ir1],
				x[ir2],y[ir2],z[ir2],
				x[ir3],y[ir3],z[ir3],
				(float_a)AGBNP_HB_LENGTH,
				&xw, &yw, &zw,
				der);
  /* stores water site */
  wsat = &(twsatb[0]);
  wsat->pos[0] = xw;
  wsat->pos[1] = yw;
  wsat->pos[2] = zw;
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TETRAHEDRAL3;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 4;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir1;
  wsat->parent[2] = ir2;
  wsat->parent[3] = ir3;
  wsat->khb = agb->hbcorr[iat];
  memcpy(wsat->dpos,der,36*sizeof(float_a));

  *nws = 1;
  return AGBNP_OK;
}

int agbnp3_update_ws_atoms_tetrahedral3(AGBNPdata *agb, WSat *wsat){
  int iat = wsat->parent[0];
  int ir1 = wsat->parent[1];
  int ir2 = wsat->parent[2];
  int ir3 = wsat->parent[3];
  float_a xw, yw, zw;
  float_a der[4][3][3];
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;

  agbnp3_place_wat_tetrahedral3(x[iat],y[iat],z[iat],
				x[ir1],y[ir1],z[ir1],
				x[ir2],y[ir2],z[ir2],
				x[ir3],y[ir3],z[ir3],
				(float_a)AGBNP_HB_LENGTH,
				&xw, &yw, &zw,
				der);
  wsat->pos[0] = xw;
  wsat->pos[1] = yw;
  wsat->pos[2] = zw;
  memcpy(wsat->dpos,der,36*sizeof(float_a));

  return AGBNP_OK;
}

void agbnp3_place_wat_tetrahedral3(float_a xa, float_a ya, float_a za, 
				   float_a xr1, float_a yr1, float_a zr1, 
				   float_a xr2, float_a yr2, float_a zr2,
				   float_a xr3, float_a yr3, float_a zr3,
				   float_a d,
				   float_a *xw, float_a *yw, float_a *zw,
				   float_a der[4][3][3]){
  float_a dx, dy, dz, dw[3], d1[3], d2[3], d3[3];
  float_a idist1, idist2, idist3, idistw;
  int i,j;
  float_a w, w1, w2, w3;
  float_a drprp[3][3], dr1r1[3][3], dr2r2[3][3], dr3r3[3][3];
  float_a drp1[3][3], drp2[3][3], drp3[3][3];
  

  /* unit vectors of bonds */
  dx = xr1 - xa;
  dy = yr1 - ya;
  dz = zr1 - za;
  idist1 =  1./sqrt(dx*dx+dy*dy+dz*dz);
  d1[0] = idist1*dx;
  d1[1] = idist1*dy;
  d1[2] = idist1*dz;

  dx = xr2 - xa;
  dy = yr2 - ya;
  dz = zr2 - za;
  idist2 =  1./sqrt(dx*dx+dy*dy+dz*dz);
  d2[0] = idist2*dx;
  d2[1] = idist2*dy;
  d2[2] = idist2*dz;

  dx = xr3 - xa;
  dy = yr3 - ya;
  dz = zr3 - za;
  idist3 =  1./sqrt(dx*dx+dy*dy+dz*dz);
  d3[0] = idist3*dx;
  d3[1] = idist3*dy;
  d3[2] = idist3*dz;

  /* water site along the negative of the sum of the bond vectors */
  dw[0] = d1[0] + d2[0] + d3[0];
  dw[1] = d1[1] + d2[1] + d3[1];
  dw[2] = d1[2] + d2[2] + d3[2];
  idistw = 1./sqrt(dw[0]*dw[0]+dw[1]*dw[1]+dw[2]*dw[2]);
  dw[0] *= idistw;
  dw[1] *= idistw;
  dw[2] *= idistw;

  *xw = xa - d*dw[0]; *yw = ya - d*dw[1]; *zw = za - d*dw[2];

  if(der){
    agbnp3_der_unitvector(dw, idistw, drprp);
    agbnp3_der_unitvector(d1, idist1, dr1r1);
    agbnp3_der_unitvector(d2, idist2, dr2r2);
    agbnp3_der_unitvector(d3, idist3, dr3r3);
    agbnp3_matmul(drprp, dr1r1, drp1);
    agbnp3_matmul(drprp, dr2r2, drp2);
    agbnp3_matmul(drprp, dr3r3, drp3);
    
    w = -d;
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	w1 = w*drp1[i][j];
	w2 = w*drp2[i][j];
	w3 = w*drp3[i][j];
	der[0][i][j] = -(w1+w2+w3);
	der[1][i][j] = w1; 
	der[2][i][j] = w2;
	der[3][i][j] = w3;
      }
    }
    for(i=0;i<3;i++){
      der[0][i][i] += 1.;
    }

  }

 
}

int agbnp3_create_ws_atoms_tetrahedral1(AGBNPdata *agb, int iat, 
					int *nws, WSat *twsatb){
  int i, nr, ir, irr[3], ir1, ir2, ir3;
  float_a xw[3];
  float_a der[4][3][3];
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;
  NeighList *conntbl = agb->conntbl;
  WSat *wsat;

  /* assumed tetrahedral topology:

       R1
         \
     R3--R--A 
         /
       R2

     for example A=O in sulphones and sulphates
  */

  if(conntbl->nne[iat] != 1){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_tetrahedral1(): acceptor atom (%d) should have 1 connecting atoms. Found %d.\n", iat,conntbl->nne[iat]);
    return AGBNP_ERR;
  }
  /* R parent */
  ir = conntbl->neighl[iat][0];
  if(conntbl->nne[ir] != 4){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_create_ws_atoms_tetrahedral1(): central atom (%d) should have 4 connecting atoms. Found %d.\n", ir, conntbl->nne[ir]);
    return AGBNP_ERR;
  }

  nr = 0;
  for(i=0;i<4;i++){
    if(conntbl->neighl[ir][i] != iat){
      irr[nr++] = conntbl->neighl[ir][i];
    }
  }
  if(nr != 3){
    agbnp3_errprint( "agbnp3_create_ws_atoms_tetrahedral1(): internal error: problems finding central atom substituents.\n");
    return AGBNP_ERR;
  }

  /* water site 1 */
  ir1 = irr[0];
  agbnp3_place_wat_tetrahedral1_one(x[iat],y[iat],z[iat],
				    x[ir],y[ir],z[ir],
				    x[ir1],y[ir1],z[ir1],
				    (float_a)AGBNP_HB_LENGTH,
				    xw, der);
  wsat = &(twsatb[0]);
  wsat->pos[0] = xw[0];
  wsat->pos[1] = xw[1];
  wsat->pos[2] = xw[2];
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TETRAHEDRAL1;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 3;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir;
  wsat->parent[2] = ir1;
  wsat->iseq = 0;
  wsat->khb = agb->hbcorr[iat];
  memcpy(wsat->dpos,der,36*sizeof(float_a));


  /* water site 2 */
  ir2 = irr[1];
  agbnp3_place_wat_tetrahedral1_one(x[iat],y[iat],z[iat],
				    x[ir],y[ir],z[ir],
				    x[ir2],y[ir2],z[ir2],
				    (float_a)AGBNP_HB_LENGTH,
				    xw, der);
  wsat = &(twsatb[1]);
  wsat->pos[0] = xw[0];
  wsat->pos[1] = xw[1];
  wsat->pos[2] = xw[2];
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TETRAHEDRAL1;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 3;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir;
  wsat->parent[2] = ir2;
  wsat->iseq = 0;
  wsat->khb = agb->hbcorr[iat];
  memcpy(wsat->dpos,der,36*sizeof(float_a));

  /* water site 3 */
  ir3 = irr[2];
  agbnp3_place_wat_tetrahedral1_one(x[iat],y[iat],z[iat],
				    x[ir],y[ir],z[ir],
				    x[ir3],y[ir3],z[ir3],
				    (float_a)AGBNP_HB_LENGTH,
				    xw, der);
  wsat = &(twsatb[2]);
  wsat->pos[0] = xw[0];
  wsat->pos[1] = xw[1];
  wsat->pos[2] = xw[2];
  wsat->r = AGBNP_HB_RADIUS;
  wsat->type = AGBNP_HB_TETRAHEDRAL1;
  wsat->volume = (4./3.)*pi*pow(wsat->r,3);
  wsat->nparents = 3;
  wsat->parent[0] = iat;
  wsat->parent[1] = ir;
  wsat->parent[2] = ir3;
  wsat->iseq = 0;
  wsat->khb = agb->hbcorr[iat];
  memcpy(wsat->dpos,der,36*sizeof(float_a));

  *nws = 3;
  return AGBNP_OK;
}

int agbnp3_update_ws_atoms_tetrahedral1(AGBNPdata *agb,  WSat *wsat){

  int iat = wsat->parent[0];
  int ir  = wsat->parent[1];
  int irr = wsat->parent[2];
  float_a xw[3];
  float_a der[4][3][3];
  float_a *x = agb->x;
  float_a *y = agb->y;
  float_a *z = agb->z;

  agbnp3_place_wat_tetrahedral1_one(x[iat],y[iat],z[iat],
				    x[ir],y[ir],z[ir],
				    x[irr],y[irr],z[irr],
				    (float_a)AGBNP_HB_LENGTH,
				    xw, der);

  wsat->pos[0] = xw[0];
  wsat->pos[1] = xw[1];
  wsat->pos[2] = xw[2];
  memcpy(wsat->dpos,der,36*sizeof(float_a));

  return AGBNP_OK;
}

void agbnp3_place_wat_tetrahedral1_one(float_a xa, float_a ya, float_a za, 
				       float_a xr, float_a yr, float_a zr, 
				       float_a xr1, float_a yr1, float_a zr1, 
				       float_a d,
				       float_a xw[3], 
				       float_a der[4][3][3]
				       ){

  float_a x0[3] = {xa, ya, za};
  float_a x1[3] = {xr, yr, zr};
  float_a x2[3] = {xr1, yr1, zr1};
  int i,j;
  float_a rs1[3], rsu1[3], rsn1;
  float_a u;

  /* R-sulfur vectors and unit vectors */
  rsn1 = 0.;
  for(i=0;i<3;i++){
    rs1[i] = x2[i] - x1[i];
    rsn1 += rs1[i]*rs1[i];
  }
  rsn1 = sqrt(rsn1);
  for(i=0;i<3;i++){
    rsu1[i] = rs1[i]/rsn1;
  }
  /* center on oxygen */
  for(i=0;i<3;i++){
    xw[i] = x0[i] - d*rsu1[i];
  }

  if( der ){

    u = -d/rsn1;
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der[0][i][j] = 0.;
      }
    }
    for(i=0;i<3;i++){
      der[0][i][i] = 1.;
    }
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der[1][i][j] = u*rsu1[i]*rsu1[j]; 
      }
    }
    for(i=0;i<3;i++){
      der[1][i][i] -= u; 
    }
    for(i=0;i<3;i++){
      for(j=0;j<3;j++){
	der[2][i][j] = -der[1][i][j];
      }
    }

  }

}

static float_a *fextval;
#pragma omp threadprivate(fextval)
/* float comparison function for qsort */
int agbnp3_fcompare( const void *val1, const void *val2 ) {
  int indx1 = *(int*)val1;
  int indx2 = *(int*)val2;

  /* primary key, sort by fextval */
  if( fextval[indx1] < fextval[indx2] ) {
    return -1;
  } else if( fextval[indx1] > fextval[indx2] ) {
    return 1;
  }

  /* secondary key, if fextvals identical, sort by index number */
  if( indx1 < indx2 )
    return -1;
  else if( indx1 > indx2 )
    return 1;

  return 0;
}

/* index sort of a float array */
void agbnp3_fsortindx( int pnval, float_a *val, int *indx ) {
  int i;
  for (i=0; i< pnval; i++) indx[i] = i;
  fextval = val;
  qsort( indx, pnval, sizeof(int), agbnp3_fcompare );
}

/* reorder neighbor list of atom iat based on mapping index indx */
int agbnp3_nblist_reorder(AGBworkdata *agbw, NeighList *nl, int iat, int *indx){
  int j, nn = nl->nne[iat];
  int *js = agbw->js;
  NeighVector  *pbcs = agbw->pbcs;
  void **datas = agbw->datas;
  /* int error = 0; */

  /* reorder neighbors */
  /*
#pragma omp critical
  if(!(js = malloc(nn*sizeof(int)))){
    agbnp3_errprint( "agbnp3_nblist_reorder(): unable to allocate neighbor buffer (%d ints).\n", nn);
    error = 1;
  }
  if (error) return AGBNP_ERR;
  */

  for(j=0;j<nn;j++){
    js[j] = nl->neighl[iat][indx[j]];
  }
  for(j=0;j<nn;j++){
    nl->neighl[iat][j] = js[j];
  }

  /* reorder pbc_trans */
  if(nl->pbc_trans){
    /*
#pragma omp critical
    if(!(pbcs = malloc(nn*sizeof(NeighVector)))){
      agbnp3_errprint( "agbnp3_nblist_reorder(): unable to allocate pbc_trans buffer (%d NeighVector's).\n", nn);
      error = 1;      
    }
    if (error) return AGBNP_ERR;
    */

    for(j=0;j<nn;j++){
      pbcs[j] = nl->pbc_trans[iat][indx[j]];
    }
    for(j=0;j<nn;j++){
      nl->pbc_trans[iat][j] = pbcs[j];
    }
  }

  /* reorder data */
  if(nl->data){
    /*
#pragma omp critical
    if(!(datas = malloc(nn*sizeof(void *)))){
      agbnp3_errprint( "agbnp3_nblist_reorder(): unable to allocate data buffer (%d void*'s).\n", nn);
      error = 1;
    }
    if (error) return AGBNP_ERR;
    */

    for(j=0;j<nn;j++){
      datas[j] = nl->data_index[iat][indx[j]];
    }
    for(j=0;j<nn;j++){
      nl->data_index[iat][j] = datas[j];
    }
  }


  return AGBNP_OK;
}

/* reorder a int list based on mapping index indx */
int agbnp3_int_reorder(AGBworkdata *agbw, int n, int *nl, int *indx){
  int j;
  int *js = agbw->js;
  /* int error = 0;  */

  /* reorder neighbors */
  /*
#pragma omp critical
  if(!(js = malloc(n*sizeof(int)))){
    agbnp3_errprint( "agbnp3_int_reorder(): unable to allocate neighbor buffer (%d ints).\n", n);
    error = 1;
  }
  if(error) return AGBNP_ERR;
  */

  for(j=0;j<n;j++){
    js[j] = nl[indx[j]];
  }
  memcpy(nl,js,n*sizeof(int));

 return AGBNP_OK;
}


float_a agbnp3_i4(float_a rij, float_a Ri, float_a Rj, 
	  float_a *dr){
  float_a u1, u2, u3,u4,u5,u6,a, ad;
  float_a u4sq,u5sq;
  float_a rij2 = rij*rij;
  float_a q;
  static const float_a twopi = 2.0*pi;
  static const float_a twothirds = 2.0/3.0;
  

  if(rij>(Ri+Rj)){
    u1 = rij+Rj;
    u2 = rij-Rj;
    u3 = u1*u2;
    u4 = 0.5*log(u1/u2);
    q = twopi*(Rj/u3 - u4/rij);
    *dr =  twopi*( (Rj/(rij*u3))*(1. - 2.*rij2/u3 ) + u4/rij2 );
  }else{
    u1 = Rj-Ri;
    if (rij2 > u1*u1){
      /* overlap */
      u1 = rij+Rj;
      u2 = rij-Rj;
      u3 = u1*u2;
      u4 = 1./u1;
      u4sq = u4*u4;
      u5 = 1./Ri;
      u5sq = u5*u5;
      u6 = 0.5*log(u1/Ri);
      q = twopi*(-(u4-u5) + (0.25*u3*(u4sq-u5sq) - u6)/rij);
      *dr = twopi*(0.5*(1.-0.5*u3/rij2)*(u4sq-u5sq)+u6/rij2);
    }else{
      /* inclusion */
      if(Ri>Rj){
	q = 0.0;
	*dr = 0.0;
      }else{
	u1 = rij+Rj;
	u2 = Rj - rij;
	u3 = -u1*u2; /* rij^2 - Rj^2 */
	if(rij < .001*Rj){
	  // removable singularity of (1/2a)*log((1+a)/(1-a)) at a=0.
	  a = rij/Rj;
	  ad = a*a - 1;
	  u6 = (1. + twothirds*a*a)/Rj;
	  q = twopi*(2./Ri + Rj/u3 - u6); 
	  *dr = -(2.*twopi*a/(Rj*Rj))*(1./ad + twothirds);
	}else{
	  u6 =  0.5*log(u1/u2);
	  q = twopi*(2./Ri + Rj/u3 - u6/(rij)); 
	  *dr = twopi*(-(Rj/u3)*(2.*rij/u3 - 1./(rij)) + u6/(rij2));
	}
      }
    }

  }

  return q;
}


 int agbnp3_neighbor_lists(AGBNPdata *agb, AGBworkdata *agbw,
			  float_a *x, float_a *y, float_a *z){

  int nnl; /* neighbor list counter for near heavy-heavy (d<r1+r2)*/
  int nnlrc; /* neighbor list counter for far heavy-heavy (r1+r2<d<rc) */
  int iat,jat,j,nlsize=0,nbnum,hsize;
  float_a dx, dy, dz, d2, u;
  float_a nboffset = AGBNP_NBOFFSET; /* offset for neighbor list distance test */
  float_a nlsize_increment = 1.2;
  int error = 0;

  int natoms = agb->natoms;
  int nheavyat = agb->nheavyat;
  float_a *r = agb->r;
  NeighList *near_nl = agbw->near_nl;
  NeighList *far_nl = agbw->far_nl;
  int *nbiat = agbw->nbiat;
  int *isheavy = agbw->isheavy;
  float_a *galpha = agb->agbw->galpha;
  float_a *gprefac = agb->agbw->gprefac;
  int nsym = agb->nsym;
  int nearflag;
  /* dbg */
  float_a gvol;
  float_a *nl_r2v = agbw->nl_r2v;
  int *nl_indx = agbw->nl_indx;


  /* coordinate buffer for Gaussian overlap calculation */
  float_a gx[2][3];
  /* radius buffer for  Gaussian overlap calculation */
  float_a gr[2];
  /* Gaussian parameters buffers for overlap calculation */
  float_a ga[2];
  float_a gp[2];
  int order; /* order of the overlap */
  /* holds overlap Gaussian parameters at each level */
  GParm gparams[2];
  float_a sr, an, pn, cn[3];
  AGBworkdata *agbwm = agb->agbw;
  int hk;

  /* reset neighbor lists */
  memset(near_nl->nne, 0, natoms*sizeof(int));
  memset(far_nl->nne, 0, natoms*sizeof(int));

  /* constructs  near and far neighbor lists */
  nnl = 0;
  nnlrc = 0;
#pragma omp for schedule(static,1) nowait
  for(iat=0;iat<nheavyat;iat++){
    if(error) continue;
    while(nnl + nsym*natoms >= near_nl->neighl_size){
      nlsize = agbnp3_mymax(nlsize_increment*near_nl->neighl_size, nnl + nsym*natoms);
      if(nblist_reallocate_neighbor_list(near_nl,natoms,nlsize) != NBLIST_OK){
	error = 2;
      }
      if (error) continue;
    }
    /* constructs  neighbor lists for atom iat */
    near_nl->nne[iat] = 0;  /* reset number of near neighbors for atom iat */
    /*set pntr to beg.of near neigh.list of atm iat*/
    near_nl->neighl[iat] = &(near_nl->neighl1[nnl]); 
    for(jat=iat+1;jat<nheavyat;jat++){
      dx = x[jat] - x[iat];
      dy = y[jat] - y[iat];
      dz = z[jat] - z[iat];
      d2 = dx*dx + dy*dy + dz*dz;
      u = (r[iat]+r[jat])*nboffset;
      /* Include only near neighbors based on sum of radii */
      if(d2<u*u){
	/* jat is a near neighbor */
	near_nl->neighl1[nnl] = jat;    /* place jat in neighbor list */
	nnl += 1;                       /* updates neighbor list counter */
	nl_r2v[near_nl->nne[iat]] = d2; /* store distance for reordering */
	near_nl->nne[iat] += 1;         /* updates counter for iat neighbors */
      }else{
	/* jat is a far neighbor */
	nnlrc += 1;                     /* keeps also track of number of far neighbors to allocate q4cache */
      }
    }
    //add hydrogens to total for q4cache allocation
    nnlrc += natoms - nheavyat + 1;

    if(near_nl->nne[iat] > 0){
      /* order near_nl in ascending order of distance */
      agbnp3_fsortindx(near_nl->nne[iat], nl_r2v, nl_indx );
      agbnp3_nblist_reorder(agbw, near_nl, iat, nl_indx);
    }

  }

  /* printf("nnl = %d nnl/nat = %f\n",nnl, nnl/(float)agb->nheavyat); */

  if(error==1){
    /*
#pragma omp critical
    */
    agbnp3_errprint( "agbnp3_neighbor_lists(): error in agbnp3_getatomneighbors().\n");
    return AGBNP_ERR;
  }else  if(error==2){
    /*
#pragma omp critical
    */
    agbnp3_errprint("agbnp3_neighbor_lists(): unable to (re)allocate near_nl neighbor list (natoms=%d, size=%d)\n",natoms, nlsize);
    return AGBNP_ERR;
  }else if(error==3){
    /*
#pragma omp critical
    */
    agbnp3_errprint("agbnp3_neighbor_lists(): unable to (re)allocate far_nl neighbor list (natoms=%d, size=%d)\n",natoms, nlsize);
    return AGBNP_ERR;
  }else if(error==4){
    agbnp3_errprint( "agbnp3_neighbor_lists(): fatal error: unable to create vpji hash table.\n");
    return AGBNP_ERR;
  }else if(error==5){
    agbnp3_errprint( "agbnp3_neighbor_lists(): fatal error: unable to create vpji buffer.\n");
    return AGBNP_ERR;
  }else if(error==6){
    agbnp3_errprint( "agbnp3_neighbor_lists(): fatal error: unable to create q4ji hash table.\n");
    return AGBNP_ERR;
  }else if(error==7){
    agbnp3_errprint( "agbnp3_neighbor_lists(): fatal error: unable to create q4ji buffer.\n");
    return AGBNP_ERR;
  }

  /* (re)allocation of i4() memory cache */
  if(!agbw->q4cache || 4*(nnl+nnlrc) > agbw->nq4cache){
    agbw->nq4cache = 4*(nnl+nnlrc);
    agbw->q4cache = (float  *)realloc(agbw->q4cache, agbw->nq4cache*sizeof(float ));
    if(!agbw->q4cache){
      agbnp3_errprint( "agbnp3_neighbor_lists(): fatal error: can't allocate memory for q4cache (%d floats)!\n", agbw->nq4cache);
      return AGBNP_ERR;
    }
  }

  return AGBNP_OK;
 }

