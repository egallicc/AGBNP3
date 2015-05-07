OBJS =  libagbnp3.$(O) agbnp3_cpu_simd_kernels.$(O) agbnp3_utils.$(O)
HOBJS =  agbnp3.h agbnp3_private.h
SRC = libagbnp3.c
AGBNPLIB = libagbnp3.$(LIBEXT)

include global.macros
include mach.macros

CFLAGS += $(OPENMP_CFLAG) $(AGBNP3_ARCH_FLAGS)
LOCAL_LIBS += $(OPENMP_LIB)

globals:

all: install

install: libagbnp3.$(LIBEXT) libnblist.$(LIBEXT)

clean: 
	rm -rf *.$(O) *.$(LIBEXT)

libagbnp3.$(STATIC_LIBEXT):  $(OBJS)
	$(AR) $(ARFLAGS) $(STATIC_OUT_FLAG)libagbnp3.$(STATIC_LIBEXT) $(OBJS)

agbnp3_cpu_simd_kernels.$(O): agbnp3_cpu_simd_kernels.c agbnp3.h agbnp3_private.h
	$(CC) $(CFLAGS) $(OBJ_OUT_FLAG)agbnp3_cpu_simd_kernels.$(O) agbnp3_cpu_simd_kernels.c

agbnp3_utils.$(O): agbnp3_utils.c agbnp3.h agbnp3_private.h
	$(CC) $(CFLAGS) $(OBJ_OUT_FLAG)agbnp3_utils.$(O) agbnp3_utils.c

libagbnp3.$(O): libagbnp3.c agbnp3.h agbnp3_private.h
	$(CC) $(CFLAGS) $(OBJ_OUT_FLAG)libagbnp3.$(O) libagbnp3.c

libnblist.$(STATIC_LIBEXT):  libnblist.$(O)
	$(AR) $(ARFLAGS) $(STATIC_OUT_FLAG)libnblist.$(STATIC_LIBEXT) libnblist.$(O)

libnblist.$(O): libnblist.c libnblist.h
	$(CC) $(CFLAGS) $(OBJ_OUT_FLAG)libnblist.$(O) libnblist.c



