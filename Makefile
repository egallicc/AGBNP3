OBJS =  agbnp3.$(O) agbnp3_cpu_simd_kernels.$(O) agbnp3_utils.$(O)
HOBJS =  agbnp3.h agbnp3_private.h
SRC = agbnp3.c
AGBNPLIB = libagbnp3.$(LIBEXT)

include global.macros
include mach.macros

CFLAGS += $(OPENMP_CFLAG) $(AGBNP3_ARCH_FLAGS)
LOCAL_LIBS += $(OPENMP_LIB)

globals:

all: install

install: libagbnp3.$(LIBEXT) libnblist.$(LIBEXT)

clean: 
	rm -rf *.$(O) *.$(STATIC_LIBEXT) *.$(SHARED_LIBEXT)

libagbnp3.$(STATIC_LIBEXT):  $(OBJS)
	$(AR) $(ARFLAGS) $(STATIC_OUT_FLAG)libagbnp3.$(STATIC_LIBEXT) $(OBJS)

libagbnp3.$(SHARED_LIBEXT):  $(OBJS)
	$(SHARED_LINKER) -o libagbnp3.$(SHARED_LIBEXT) $(OBJS)

agbnp3_cpu_simd_kernels.$(O): agbnp3_cpu_simd_kernels.c agbnp3.h agbnp3_private.h
	$(CC) $(CFLAGS) $(OBJ_OUT_FLAG)agbnp3_cpu_simd_kernels.$(O) agbnp3_cpu_simd_kernels.c

agbnp3_utils.$(O): agbnp3_utils.c agbnp3.h agbnp3_private.h
	$(CC) $(CFLAGS) $(OBJ_OUT_FLAG)agbnp3_utils.$(O) agbnp3_utils.c

agbnp3.$(O): agbnp3.c agbnp3.h agbnp3_private.h
	$(CC) $(CFLAGS) $(OBJ_OUT_FLAG)agbnp3.$(O) agbnp3.c

libnblist.$(STATIC_LIBEXT):  nblist.$(O)
	$(AR) $(ARFLAGS) $(STATIC_OUT_FLAG)libnblist.$(STATIC_LIBEXT) nblist.$(O)

libnblist.$(SHARED_LIBEXT):  nblist.$(O)
	$(SHARED_LINKER) -o libnblist.$(SHARED_LIBEXT) nblist.$(O)	

nblist.$(O): nblist.c nblist.h
	$(CC) $(CFLAGS) $(OBJ_OUT_FLAG)nblist.$(O) nblist.c



