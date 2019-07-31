# for Mac OS
# make all "CFLAGS=-Wall -DMACOSX"

CC= gcc

CFLAGS= -Wall

LIBS= -lm

PROGS = testfun maxpower

all: testfun maxpower

nmsimplex: nmsimplex.c
	${CC} ${CFLAGS} -c nmsimplex.o ${LIBS}

testfun: testfun.o nmsimplex.o
	${CC} ${CFLAGS} -o $@ testfun.c nmsimplex.o ${LIBS}

maxpower: maxpower.o nmsimplex.o
	${CC} ${CFLAGS} -o $@ maxpower.c nmsimplex.o ${LIBS}


clean:
		rm -f ./*.o ${PROGS}