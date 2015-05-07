#ifndef LIBNBLIST_H
#define LIBNBLIST_H
#ifdef __cplusplus
extern "C" {
#endif

/* return codes */
#define NBLIST_OK 0
#define NBLIST_ERR 1

/*                                                                   *
 * a simple interface to create/reallocate a Verlet neighbor list    *
 *                                                                   */

typedef struct vector_ { /* A 3D vector */
  double x, y, z;
} NeighVector;

/* neighbor list structure */
typedef struct neighlist_ {
  /* number of atoms */
  int natoms;
  /* neighbor list size */
  int neighl_size;
  /* number of neighbors for each atom */
  int *nne;
  /* where the actual atom indexes are stored */
  int *neighl1;
  /* pointer into neighl1 for each atom, neighbors of atom iat are retrived
     as jat=neighl[iat][j] ; 0 <= i < nne[iat] */ 
  int **neighl;

  int idx_remap; /* whether (yes > 0, no = 0) to do atom indexes remapping */
  int *int2ext; /* an optional map to translate from internal to external
		   atom indeces */
  int *ext2int; /* an optional map to translate from external to internal
		   atom indeces */

  /* flag, if not zero neighbor list contains PBC translation vectors */
  int pbc;
  /* same as neighl and neighl1 but stores PBC translation vectors */
  NeighVector  *pbc_trans1;
  NeighVector  **pbc_trans;

  /* additional data  */
  int data; /* flag, if not zero additional data exists */
  /* same as neighl and neighl1 but for additional data */
  void  **data_index1; /* pointers to anything */
  void  ***data_index;

} NeighList;

/* function prototypes */

/* reset neighbor list */
void nblist_reset_neighbor_list(NeighList *neigh_list);
/* delete neighbor list */
void nblist_delete_neighbor_list(NeighList *neigh_list);
/* reallocates neighbor list */
int nblist_reallocate_neighbor_list(NeighList *neigh_list, 
				     int natoms, int size);

#ifdef __cplusplus
}
#endif
#endif
