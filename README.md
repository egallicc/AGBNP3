# AGBNP3: Analytic Generalized-Born + Non-Polar model version 3

Emilio Gallicchio <egallicchio@brooklyn.cuny.edu>

Last Modified: May 2015

 AGBNP is an analytical implicit solvent model based on the pairwise
 descreening (PD) Generalized Born (GB) model, a non-polar
 solvation free energy (NP) estimator, which takes into account
 independently the work of cavity formation and the solute-solvent van
 der Waals interaction energy, and a short-range solute-solvent hydrogen binding model. The model and its derivation are  described in detail in the following paper:
 
Gallicchio E, Paris K, and Levy RM. The AGBNP2 Implicit Solvent Model. J Chem Theory Comput, 5, 2544-2564 (2009). 
 
## License

This software is released under the LGPL license. See LICENSE.

## Credits

This software is written and maintained by Emilio Gallicchio <egallicchio@brooklyn.cuny.edu> with support from a grant from the National Science Foundation (ACI 1440665).

The AGBNP model has been developed by the author, Ronald M Levy <ronlevy@temple.edu>, and collaborators as described in the following publications:

Gallicchio E, Paris K, and Levy RM. The AGBNP2 Implicit Solvent Model. J Chem Theory Comput, 5, 2544-2564 (2009).

Gallicchio E., and R.M. Levy. AGBNP, an analytic implicit solvent model suitable for molecular dynamics simulations and high-resolution modeling, J. Comp. Chem. 25, 479-499 (2004).

Portions of this software, limited to the driver file `libagbnp3.c` and the `libnblist.c` utility library, have been adapted from the AGBNP2 library written by the author while at Rutgers University in the group of Ronald M. Levy. 

The sse_mathfun library has been written by Julien Pommier and released under the zlib license. See license declaration in `sse_mathfun.h`.

## Installation
 
On a Linux system with gcc type `make install` in source directory. Link your code to the `libagbnp3.a` and `libnblist.a` libraries generated. Add the source directory to include path.

For example if the source directory is `/src/AGBNP3` so
```
gcc -I/src/AGBNP3 mycode.c -o mycode -L/src/AGBNP3/ -lagbnp3 -lnblist
```
For other architecture and/or compilers modify the `mach.macros` file as needed. 

## AGBNP C API

 The header file `agbnp3.h` must be included to access the AGBNP3 API functions.

```
 #include "agbnp3.h"
```

```
 > int agbnp_initialize( void );
```` 

 Initializes libagbnp3 library. No library functions can be used until
 this call is made.
 
 Return values:
 AGBNP_OK - library initialized OK.
 AGBNP_ERR - error in initialization.
 
```
 > void agbnp_terminate( void );
```

 Terminates libagbnp3 library. Frees all associated storage.  Once this
 call is made then the library can no longer be used until it is
 initialized again by calling agbnp_initialize().
 
```
typedef double float_i; //can be set to float, see agbnp3.h 
> int agbnp3_new(int *tag, int natoms, 
	      float_i *x, float_i *y, float_i *z, float_i *r, 
	      float_i *charge, float_i dielectric_in, float_i dielectric_out,
	      float_i *igamma, float_i *sgamma,
	      float_i *ialpha, float_i *salpha,
	      int *hbtype, float_i *hbcorr,
	      int nhydrogen, int *ihydrogen, 
	      NeighList *conntbl,
	      int verbose);
 ```

 Creates a new instance of AGBNP3. If successful
 this function returns in the 'tag' argument a non-negative integer id
 that identifies the instance. All subsequent operations on this
 instance should reference this tag. Any number of AGBNP3 instances can
 be generated by calling the agbnp3_new() function. Each instance is
 independent from the others, that is each may correspond to different
 systems, parameters, etc.
 
 Input parameters (units in []):
 
 natoms:   number of atoms of the system. Atoms are numbered
           from 0 to natoms-1.
 
 x, y, z:  Cartesian coordinates of each atom [Angstroms].

 r:  van der Waals radius of each atom [Angstroms]
 
 charge:   partial charge of each atom [e].
 
 dielectric_in:  value of the interior (solute) relative dielectric
                 constant.
 
 dielectric_out: value of the exterior (solvent) relative dielectric
                 constant.
 
 igamma:   value of the gamma non-polar parameter for each atom 
           [(kcal/mol)/Ang^2]
 
 sgamma:   value of the gamma non-polar correction parameter for each
           atom [(kcal/mol)/Ang^2]
  
 ialpha:   value of the "alpha" van der Waals non-polar parameter for each
           atom [(kcal/mol) Ang^3]
 
 salpha    value of the "alpha" van der Waals non-polar correction
           parameter for each atom [(kcal/mol) Ang^3]
 
 Note:  The values of the non-polar parameters used internally are the
 sum of the ideal and correction values. However the non-polar energy
 derived from each is reported separately as a pure non-polar energy
 and a correction energy term. If you are
 confused about this just set the correction parameters to zero.
 
 hbtype: hydrogen bonding type for each type, one of the AGBNP_HB_* atom geometry identifiers listed in agbnp3.h

 hbcorr: hydrogen bonding strength parameter [kcal/mol]
 
 nhydrogen: number of hydrogen atoms in the system.
 
 ihydrogen: atom index of each hydrogen atom. Note that atom indexes start
            from zero.
  
 conntbl: atom connection table. It lists the atoms directly bonded to 
          each atom. It is specified in terms a neigh_list structure 
          defined with the libnblist library - included in the AGBNP3 
          distribution. For documentation pertaining to libnblist see below. 
  
 Return values:
 AGBNP_OK - AGBNP instance created, id tag is returned in 'tag'.
 AGBNP_ERR - error creating AGBNP instance. Consult error message
             on stderr.
 
 ```
 > int agbnp_delete(int tag);
 ```

 Deletes the AGBNP instance referenced by 'tag'. Associated memory is freed.

 Return values:
 AGBNP_OK - AGBNP instance deleted.
 AGBNP_ERR - error deleting AGBNP instance. Consult error message
             on stderr.
 
``` 
int agbnp3_ener(int tag, int init,
		float_i *x, float_i *y, float_i *z,
		float_i *sp, float_i *br, 
		float_i *mol_volume, float_i *surf_area, 
		float_i *egb, float_i (*dgbdr)[3],
		float_i *evdw, float_i *ecorr_vdw, float_i (*dvwdr)[3], 
		float_i *ecav, float_i *ecorr_cav, float_i (*decav)[3],
		float_i *ehb,  float_i (*dehb)[3]);
 ````

 Calculates energies and derivatives of instance referenced by tag.
 
 Input parameters:

 init: unused

 x, y, z: current atomic positions [Angstroms]
 
 Output:

 sp: atomic volume scaling factor

 br: Born radius [Angstroms]

 mol_volume: molecular volume

 surf_area: solvent-exposed surface area of each atom [Ang^2]

 egb: generalized Born energy [kcal/mol]

 evdw, ecorr_vdw: solute solvent van der Waals energy [kcal/mol]

 ecav, ecorr_cav: cavity energy [kcal/mol]

 ehb: solute-solvent hydrogen bonding energy [kcal/mol]

 dgbdr, dvwdr, decav, dehb: gradients of egb, evdw, ecav and ehb.

 Return values:
 AGBNP_OK - AGBNP energy calculated.
 AGBNP_ERR - Error calculating energy. Consult error message
             on stderr.
 
 
### Verlet Neighbor List Utility Functions (libnblist)
 
 The libnblist library provides a data structure to hold a Verlet
 neighbor list and a set of utility functions to manage the memory
 associated with the Verlet neighbor list.
 
 To use this facility the libnblist header file must be included

``` 
 #include "libnblist.h"
 ```
 this file is automatically included by `agbnp.h`
  
 libnblist defines the NeighList data type, a C structure composed of
 the following members:
 
 - Core members:
 
    int natoms; number of atoms.
 
    int neighl_size; Holds the allocated size of the Verlet neighbor
                     list, this is the maximum number of neighbors that
                     can be stored in the list.

    int *nne; an array of size natoms that holds the number of
              neighbors for each atom.
 
    int *neighl1; the pointer to the memory space where the neighbor
                  list is actually stored.
 
    int **neighl; an array of pointers into neighl1 of size natoms
                  pointing to the start of the list of neighbors for
                  each atom. So for example, neighl[iat][j], 0 <= j <
                  nne[iat], are the atom indexes of the neighbors of
                  atom iat.
 
 - Atom indexes remapping members:

    int idx_remap; whether (yes > 0, no = 0) to do atom indexes remapping.
 
    int *int2ext; an optional array of size natoms to translate from
                  internal to external atom indexes.
 
    int *ext2int; an optional array of size max(external index)+1 to
                  translate from external to internal atom indexes.
 
    Note: libnblist can be used to store Verlet neighbor lists for
    calling programs using different atom indexing schemes - starting
    from zero, starting from one, random, etc . To do so libnblist
    makes a distinction between internal and external atom
    indexes. This is best illustrated with an example. Let us say that
    in the caller program 5 atoms are defined with atomic indexes 7,
    9, 4, 6, and 2. The neighbor lists for these atoms are stored in
    the Verlet neighbor list in this order so that neighl[0][j]
    identifies the neighbors of atom 7, neighl[1][j] identifies the
    neighbors of atom 9, etc. 0 is the internal index for atom 7, 1 is
    the internal atom index for atom 9, etc. In order to reference and
    dereference the neighbor list mapping arrays are constructed. The
    content of the int2ext array in this case would be:
 
    int2ext = { 7, 9, 4, 6, 2 };
 
    and the content of the ext2int array would be:
 
    ext2int = { -, -, 4, -, 2, -, 3, 0, -, 1 };
 
    So that for example int2ext[4] = 2 and ext2int[9] = 1. The
    neighbors of atom 9 are neighl[ext2int[9]][j], 0 <= j <
    nne[ext2int[9]]. The Verlet neighbor list stores the external atom
    indexes of the neighbors so that for example in iat =
    neighl[ext2int[9]][1], iat may be 6.  AGBNP uses the libnblist
    indexes remapping arrays to convert from internal AGBNP atom
    indexes (which start from 0 and end at natoms-1) to external atom
    indexes (that could be anything) and viceversa. It is the
    responsibility of the caller to properly set up the remapping
    arrays.
 
 - Other data:
 
    The libnblist data structure has place holders for other types of
    data (Periodic Boundary Conditions translation vectors and general
    additional data). These data structures are not described here as
    they are not relevant to AGBNP usage.
 
 ### Libnblist C API
 
 libnblist provides simple memory management functions for Verlet
 neighbor lists.
 
 The caller usually creates a libnblist structure by allocating memory
 for it:

``` 
 NeighList *nl;
 nl = (NeighList *)malloc(sizeof(NeighList));
```

 `nl` is then used as an argument for the utility functions that follow.
 
```
 > void nblist_reset_neighbor_list(NeighList *neigh_list);
```

 resets structure to pristine state. All pointers NULL, variables set to zero.

``` 
 > int nblist_reallocate_neighbor_list(NeighList *nl, int natoms, int nlsize);
``` 

 Reallocates memory for the Verlet neighbor list pointed by
 'nl'. 'natoms' is the new number of atoms and 'nlsize' is the new list
 size. If natoms > nl->natoms the list is resized in terms of the
 number of atoms. If nlsize > nl->neighl_size the size of the Verlet
 neighbor list is reallocated as well. In either case the previous
 contents of the neighbor list structure are preserved, including index
 remapping arrays and additional data.
 Return values:
      NBLIST_OK - Neighbor list memory reallocated.
      NBLIST_ERR - Error reallocating memory for neighbor
                   list. Neighbor list may be unusable at this point. 
                   Consult error messages on stderr.
 
```
 > int nblist_delete_neighbor_list(NeighList *nl);
```

 Frees memory associated with the Verlet neighbor list referenced by
 'nl'. The memory associated with the neighbor list structure itself
 is not freed. The calling program is responsible for calling free(nl)
 if needed.
 Return values:
      NBLIST_OK - Neighbor list memory freed.
      NBLIST_ERR - Error freeing memory of neighbor list.
                   Consult error messages on stderr.




