#include "auto_f2c.h"
#include "auto_c.h"
#include "auto_types.h"

/* The memory for these are taken care of in autobv_ and autoae_ */
extern struct {
  integer irtn;
  integer *nrtn;
} global_rotations;

#ifdef DEBUG
#include <stdio.h>
#include <stdarg.h>
void log_message(char *fmt, ...)
{
  va_list argp;
  FILE *log_file;
  log_file=fopen("/tmp/redrod_log","a");
  va_start(argp, fmt);
  vfprintf(log_file, fmt, argp);
  va_end(argp);
  fflush(log_file);
  fclose(log_file);
}
#endif

#ifdef MPI
double ***local_storage_aa=NULL;
double ***local_storage_bb=NULL;
double ***local_storage_cc=NULL;
double ***local_storage_dd=NULL;
double ***local_storage_fa=NULL;
double ***local_storage_fc=NULL;

int mpi_worker() {
  MPI_Status stat;
  int my_rank;
  int message_type;
  integer funi_icni_params[5]={0,0,0,0,0};
  integer setup_common_blocks=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  
  while(1) {
    MPI_Recv(&message_type,1,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&stat);
    switch(message_type){
    case AUTO_MPI_KILL_MESSAGE:  /*The kill message*/
      MPI_Finalize();
      exit(0);
      break;
    case AUTO_MPI_INIT_MESSAGE:
      MPI_Bcast(funi_icni_params,5,MPI_LONG,0,MPI_COMM_WORLD);
      /* After we INIT we want setubv to initialize the common blocks
     for the functions is autlib3.c and autlib5.c */
      setup_common_blocks = 1;
      break;
    case AUTO_MPI_SETUBV_MESSAGE:  /*The setubv message is not done yet*/
      mpi_setubv_worker(funi_icni_params,setup_common_blocks);
      /* We only want to initialize the blocks once for each
     outer loop of main() */
      setup_common_blocks = 0;
      break;
    case AUTO_MPI_CONPAR_MESSAGE: /*The setubv message*/
      mpi_conpar_worker();
      break;
    default:
      fprintf(stderr,"Unknown message recieved: %d\n",message_type);
      break;
    }
  }
  return 0;
}
#endif

#ifdef MPI
int mpi_conpar_worker() {
  conpar_parallel_arglist data;
  MPI_Status stat;
  int my_rank;
  integer params[6];
  
  /* find out which worker I am */
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  /* input scalars */
  MPI_Recv(&(data.loop_start) ,1,MPI_LONG,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&stat);
  MPI_Recv(&(data.loop_end)   ,1,MPI_LONG,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&stat);

  MPI_Bcast(params        ,6,MPI_LONG,0,MPI_COMM_WORLD);
  data.nov = &params[0];
  data.nra = &params[1];
  data.nca = &params[2];
  data.ncb = &params[3];
  data.nbc = &params[4];
  data.nrc = &params[5];

  /* input/output arrays */

  data.irf = (integer *)malloc(sizeof(integer)*((data.loop_end-data.loop_start))*(*(data.nra)));
  data.icf = (integer *)malloc(sizeof(integer)*((data.loop_end-data.loop_start))*(*(data.nca)));

  MPI_Scatterv(NULL,NULL,NULL,MPI_LONG,
           data.irf,(int)((data.loop_end-data.loop_start))*(*(data.nra)),MPI_LONG,
           0,MPI_COMM_WORLD);
  MPI_Scatterv(NULL,NULL,NULL,MPI_LONG,
           data.icf,(int)((data.loop_end-data.loop_start))*(*(data.nca)),MPI_LONG,
           0,MPI_COMM_WORLD);

  data.a=local_storage_aa;
  data.b=local_storage_bb;
  data.c=local_storage_cc;

  /* This gets initialized to 0.0 in conpar_process */
  data.d = dmatrix(*data.nrc, *data.ncb);

  conpar_process(&data);
  
  {
    /*I create a temporary send buffer for the MPI_Reduce
      command.  It seems strange that I have to do this, since
      this guy doesn't recieve anything, but there
      you go*/
    double **dtemp;
    dtemp = dmatrix(*(data.nrc), *(data.ncb));
    
    MPI_Reduce(data.d[0],NULL,(*(data.ncb))*(*(data.nrc)),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    free(dtemp);
  }
  MPI_Gatherv(data.a[0][0],(*(data.nca))*(*(data.nra))*((data.loop_end-data.loop_start)),MPI_DOUBLE,
          NULL,NULL,NULL,MPI_DOUBLE,
          0,MPI_COMM_WORLD);
  MPI_Gatherv(data.b[0][0],(*(data.ncb))*(*(data.nra))*((data.loop_end-data.loop_start)),MPI_DOUBLE,
          NULL,NULL,NULL,MPI_DOUBLE,
          0,MPI_COMM_WORLD);
  MPI_Gatherv(data.c[0][0],(*(data.nca))*(*(data.nrc))*((data.loop_end-data.loop_start)),MPI_DOUBLE,
          NULL,NULL,NULL,MPI_DOUBLE,
          0,MPI_COMM_WORLD);
  MPI_Gatherv(data.irf,((data.loop_end-data.loop_start))*(*(data.nra)),MPI_LONG,
          NULL,NULL,NULL,MPI_LONG,
          0,MPI_COMM_WORLD);
  MPI_Gatherv(data.icf,((data.loop_end-data.loop_start))*(*(data.nca)),MPI_LONG,
          NULL,NULL,NULL,MPI_LONG,
          0,MPI_COMM_WORLD);
  
  /*free arrays*/
  free(data.irf);
  free(data.icf); 
  free(data.d);
  return 1;
}
#endif

#ifdef MPI
int mpi_setubv_worker(integer *funi_icni_params,integer setup_common_blocks) {
  setubv_parallel_arglist data;
  MPI_Status stat;
  int my_rank,comm_size;
  int i;
  integer params[11];

  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
  /* input scalars */
  MPI_Recv(&(data.loop_start) ,1,MPI_LONG,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&stat);
  MPI_Recv(&(data.loop_end)   ,1,MPI_LONG,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&stat);
  MPI_Recv(&(data.loop_offset),1,MPI_LONG,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&stat);

  MPI_Bcast(params          ,11,MPI_LONG,0,MPI_COMM_WORLD);
  data.na=params[0];
  data.ndim=params[1];
  data.ips=params[2];
  data.ncol=params[3];
  data.nbc=params[4];
  data.nint=params[5];
  data.ncb=params[6];
  data.nrc=params[7];
  data.nra=params[8];
  data.nca=params[9];
  data.ndxloc=params[10];

  data.iap=(iap_type *)malloc(sizeof(iap_type));
  data.rap=(rap_type *)malloc(sizeof(rap_type));
  data.par=(double *)malloc(sizeof(double)*NPARX2);
  data.icp=(integer *)malloc(sizeof(integer)*NPARX2);
  data.ups=dmatrix(data.ndxloc, data.ndim*data.ncol);
  data.uoldps=dmatrix(data.ndxloc, data.ndim*data.ncol);
  data.wp=dmatrix(data.ncol + 1, data.ncol);
  data.wt=dmatrix(data.ncol + 1, data.ncol);
  data.wi=(double *)malloc(sizeof(double)*(data.ncol + 1));
  data.udotps=dmatrix(data.ndxloc, data.ndim*data.ncol);
  data.upoldp=dmatrix(data.ndxloc, data.ndim*data.ncol);
  data.thu=(double *)malloc(sizeof(double)*(data.ndim)*8);
  data.thl=(double *)malloc(sizeof(double)*NPARX);
  data.rldot=(double *)malloc(sizeof(double)*NPARX);

  {
    int position=0;
    void *buffer;
    int bufsize;
    int size_int,size_double;
    int niap,nrap;
    /* Here we compute the number of elements in the iap and rap structures.
       Since each of the structures is homogeneous we just divide the total
       size by the size of the individual elements.*/
    niap = sizeof(iap_type)/sizeof(integer);
    nrap = sizeof(rap_type)/sizeof(doublereal);

    MPI_Pack_size(niap+NPARX2,MPI_LONG,MPI_COMM_WORLD,&size_int);
    MPI_Pack_size(nrap+NPARX2+
          (data.ndxloc)*(data.ndim)*(data.ncol)+
          (data.ndxloc)*(data.ndim)*(data.ncol)+
          (data.ncol + 1)*(data.ncol)+
          (data.ncol + 1)*(data.ncol)+
          (data.ncol + 1)+
          (data.ndxloc)*(data.ndim)*(data.ncol)+
          (data.ndxloc)*(data.ndim)*(data.ncol)+
          (data.ndim)*8+
          NPARX+
          NPARX,
          MPI_DOUBLE,MPI_COMM_WORLD,&size_double);

    bufsize = size_int + size_double;
    buffer=malloc((unsigned)bufsize);

    MPI_Bcast(buffer  ,bufsize,MPI_PACKED,0,MPI_COMM_WORLD);

    MPI_Unpack(buffer,bufsize,&position,data.iap     ,niap,MPI_LONG,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.rap     ,nrap,MPI_DOUBLE,MPI_COMM_WORLD);
    /***********************************/
    MPI_Unpack(buffer,bufsize,&position,data.par    ,NPARX2,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.icp    ,NPARX2,MPI_LONG,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.ups[0]   ,(data.ndxloc)*(data.ndim)*(data.ncol),MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.uoldps[0],(data.ndxloc)*(data.ndim)*(data.ncol),MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.wp[0]  ,(data.ncol + 1)*(data.ncol),MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.wt[0]  ,(data.ncol + 1)*(data.ncol),MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.wi     ,(data.ncol + 1),MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.udotps[0],(data.ndxloc)*(data.ndim)*(data.ncol),MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.upoldp[0],(data.ndxloc)*(data.ndim)*(data.ncol),MPI_DOUBLE,MPI_COMM_WORLD);

    MPI_Unpack(buffer,bufsize,&position,data.thu    ,(data.ndim)*8,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.thl    ,NPARX,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Unpack(buffer,bufsize,&position,data.rldot  ,NPARX,MPI_DOUBLE,MPI_COMM_WORLD);

  }

  if(setup_common_blocks) {
    /* At this point the iap structure is set up, so we can allocate some
       arrays which are used in autlib3.c and autlib5.c */
    allocate_global_memory(*(data.iap));
    
    /* This function sets up the blrtn global.  Some of the routines
       pointed to by data.icni and data.bcni use this structure */
    setrtn(data.iap,&(data.iap->ndim),&(data.ndxloc),data.ups,data.par);
  } 

  data.dtm=(double *)malloc(sizeof(double)*(data.loop_end-data.loop_start));
  MPI_Scatterv(NULL,NULL,NULL,MPI_DOUBLE,
          data.dtm,data.loop_end-data.loop_start,MPI_DOUBLE,
          0,MPI_COMM_WORLD);

  /* output arrays */
  if(local_storage_aa==NULL)
    local_storage_aa=dmatrix_3d(data.loop_end-data.loop_start, data.nra, data.nca);
  data.aa = local_storage_aa;

  if(local_storage_bb==NULL)
    local_storage_bb=dmatrix_3d(data.loop_end-data.loop_start, data.nra, data.ncb);
  data.bb = local_storage_bb;

  if(local_storage_cc==NULL)
    local_storage_cc=dmatrix_3d(data.loop_end-data.loop_start, data.nrc, data.nca);
  data.cc = local_storage_cc;


  /* dd is filled in by the master */
  data.dd = NULL;

  /*zero arrays*/
  for(i=0;i<(data.nca)*(data.nra)*((data.loop_end-data.loop_start));i++)
    data.aa[0][0][i]=0.0;
  for(i=0;i<(data.ncb)*(data.nra)*((data.loop_end-data.loop_start));i++)
    data.bb[0][0][i]=0.0;
  for(i=0;i<(data.nca)*(data.nrc)*((data.loop_end-data.loop_start));i++)
    data.cc[0][0][i]=0.0;


  /* figure out what funi and icni are from
     the iap array.  This is originally done 
     in autlib1.c.  We do it here, since I
     don't know how to pass function pointers
     through MPI in a possibly heterogeneous 
     environment :-) */
  {
    data.iap->ips  = funi_icni_params[0];
    data.iap->irs  = funi_icni_params[1];
    data.iap->isw  = funi_icni_params[2];
    data.iap->itp  = funi_icni_params[3];
    data.iap->nfpr = funi_icni_params[4];
    if(set_funi_and_icni(data.iap,&data)) {
      MPI_Finalize();
      exit(0);
    }
  }

  setubv_make_aa_bb_cc(&data);

  /*free input arrays*/
  free(data.iap);
  free(data.rap);
  free(data.par);
  free(data.icp);
  free_dmatrix(data.ups);
  free_dmatrix(data.uoldps);
  free(data.dtm);
  free(data.wp);
  free(data.wt);
  free_dmatrix(data.udotps);
  free_dmatrix(data.upoldp);
  free(data.wi);
  free(data.thu);
  free(data.thl);
  free(data.rldot);
  return 1;
}

int set_funi_and_icni(iap_type *iap,setubv_parallel_arglist *data) {
  function_list list;
  set_function_pointers(*iap,&list);

  if (list.type != AUTOBV) {
    printf("Illegal problem type in set_funi_and_icni\n");
    exit(1);
  }
  data->funi = list.bvlist.funi;
  data->icni = list.bvlist.icni;
  data->bcni = list.bvlist.bcni;

  return 0;
}


#endif



