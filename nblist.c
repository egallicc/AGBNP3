/* -------------------------------------------------------------------------- *
 * copyright (c) 2014-2015 Emilio Gallicchio                                  *
 * Authors: Emilio Gallicchio <egallicchio@brooklyn.cuny.edu>                 *
 * Contributors:                                                              *
 *                                                                            *
 *  This program is free software: you can redistribute it and/or modify      *
 *  it under the terms of the GNU Lesser General Public License version 3     *
 *  as published by the Free Software Foundation.                             *
 *                                                                            *
 *  The program is distributed in the hope that it will be useful,            *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of            *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
 *  GNU General Public License for more details.                              *
 *                                                                            *
 *  You should have received a copy of the GNU General Public License         *
 *  along this program.  If not, see <http://www.gnu.org/licenses/>.          *
 *                                                                            *
 * -------------------------------------------------------------------------- */


#include <stdlib.h>
#include <stdio.h>
#include "nblist.h"

/*                                                                   *
 * a simple interface to create/reallocate a Verlet neighbor list    *
 * (a NeighList structure as defined in nblist.h)                    */

/* reset neighbor list */
void nblist_reset_neighbor_list(NeighList *neigh_list){
  neigh_list->natoms = 0;
  neigh_list->neighl_size = 0;
  neigh_list->nne = NULL;
  neigh_list->neighl1 = NULL;
  neigh_list->neighl = NULL;
  neigh_list->idx_remap = 0;
  neigh_list->int2ext = NULL;
  neigh_list->ext2int = NULL;
  neigh_list->pbc = 0;
  neigh_list->pbc_trans1 = NULL;
  neigh_list->pbc_trans = NULL;
  neigh_list->data = 0;
  neigh_list->data_index = NULL;
  neigh_list->data_index1 = NULL;
}

/* delete neighbor list */
void nblist_delete_neighbor_list(NeighList *neigh_list){
  if(neigh_list){
    if(neigh_list->nne) free(neigh_list->nne);
    if(neigh_list->neighl) free(neigh_list->neighl);
    if(neigh_list->neighl1) free(neigh_list->neighl1);
    if(neigh_list->int2ext) free(neigh_list->int2ext);
    if(neigh_list->ext2int) free(neigh_list->ext2int);
    if(neigh_list->pbc_trans) free(neigh_list->pbc_trans);
    if(neigh_list->pbc_trans1) free(neigh_list->pbc_trans1);
    if(neigh_list->data_index) free(neigh_list->data_index);
    if(neigh_list->data_index1) free(neigh_list->data_index1);
  }
}


/* reallocates neighbor list */
int nblist_reallocate_neighbor_list(NeighList *neigh_list, 
					      int natoms, int size){
  int iat, index;

  if(natoms > neigh_list->natoms){
    /* reallocates nne and neighl arrays */
    neigh_list->nne = (int *)realloc(neigh_list->nne, natoms*sizeof(int));
    if(!neigh_list->nne){
      fprintf(stderr,"nblist_reallocate_neighbor_list(): error: unable to (re)allocate nne array (%d integers).\n", natoms);
      return NBLIST_ERR;
    }

    neigh_list->neighl = (int **)realloc(neigh_list->neighl, 
					 natoms*sizeof(int *));
    if(!neigh_list->neighl){
      fprintf(stderr,"nblist_reallocate_neighbor_list(): unable to (re)allocate neighl array (%d pointers to integers).\n", natoms);
      return NBLIST_ERR;
    }

    if(neigh_list->idx_remap){
      /* reallocate translation map */
      neigh_list->int2ext = 
	(int *)realloc(neigh_list->int2ext, natoms*sizeof(int));
      neigh_list->ext2int = 
	(int *)realloc(neigh_list->ext2int, natoms*sizeof(int));
      if(!neigh_list->int2ext || !neigh_list->ext2int){
	fprintf(stderr,
		     "mmnblist_reallocate_neighbor_list(): unable to (re)allocate atom indeces remapping arrays (%d integers).\n", 2*natoms);
	return NBLIST_ERR;
      }
    }

    if(neigh_list->pbc){
      /* reallocates PBC translation indexes */
      neigh_list->pbc_trans = 
	(NeighVector **)realloc(neigh_list->pbc_trans, natoms*sizeof(NeighVector *));
      if(!neigh_list->pbc_trans){
	fprintf(stderr,
		     "mmnblist_reallocate_neighbor_list(): unable to (re)allocate pbc_trans array (%d pointers to Vectors).\n", natoms);
	return NBLIST_ERR;
      }
    }

    if(neigh_list->data){
      /* reallocates data indexes */
      neigh_list->data_index = 
	(void ***)realloc(neigh_list->data_index, natoms*sizeof(void **));
      if(!neigh_list->data_index){
	
	fprintf(stderr, "nblist_reallocate_neighbor_list(): unable to (re)allocate data_index array (%d pointers to pointers to voids).\n", natoms);
	return NBLIST_ERR;
      }
    }
  }
  
  if(size > neigh_list->neighl_size){
    /* reallocates neighl1 array */
    
    neigh_list->neighl1 = (int *)realloc(neigh_list->neighl1, 
					 size*sizeof(int));
    if(!neigh_list->neighl1){
      fprintf(stderr,"nblist_reallocate_neighbor_list(): unable to (re)allocate neighl1 array (%d integers).\n", size);
      return NBLIST_ERR;
    }
    /* rebuilds array of pointers */
    index = 0;
    for(iat=0;iat<neigh_list->natoms;iat++){
      neigh_list->neighl[iat] = &(neigh_list->neighl1[index]);
      index += neigh_list->nne[iat];
    }

    if(neigh_list->pbc){
      /* reallocates pbc_trans1 array */    
      neigh_list->pbc_trans1 = (NeighVector *)realloc(neigh_list->pbc_trans1, 
						 size*sizeof(NeighVector));
      if(!neigh_list->pbc_trans1){
	fprintf(stderr,
		     "nblist_reallocate_neighbor_list(): unable to (re)allocate pbc_trans1 array (%d Vectors).\n", size);
	return NBLIST_ERR;
      }
      /* rebuilds array of pointers */
      index = 0;
      for(iat=0;iat<neigh_list->natoms;iat++){
	neigh_list->pbc_trans[iat] = &(neigh_list->pbc_trans1[index]);
	index += neigh_list->nne[iat];
      }
    }

    if(neigh_list->data){
      /* reallocates data_index1 array */    
      neigh_list->data_index1 = (void **)realloc(neigh_list->data_index1, 
						 size*sizeof(void *));
      if(!neigh_list->data_index1){
	fprintf(stderr,
		     "nblist_reallocate_neighbor_list(): unable to (re)allocate data_index1 array (%d pointers to voids).\n", size);
	return NBLIST_ERR;
      }
      /* rebuilds array of pointers */
      index = 0;
      for(iat=0;iat<neigh_list->natoms;iat++){
	neigh_list->data_index[iat] = &(neigh_list->data_index1[index]);
	index += neigh_list->nne[iat];
      }
    }

    /* update neighbor list size */
    neigh_list->neighl_size = size;
    
  }

  if(natoms > neigh_list->natoms){
    /* update number of atoms */
    neigh_list->natoms = natoms;    
  }
  
  return NBLIST_OK;
}
