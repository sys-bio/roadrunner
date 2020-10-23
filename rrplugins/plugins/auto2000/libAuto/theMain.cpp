#include "libf2c/auto_f2c.h"
#include "auto_c.h"
#include <time.h>

int AUTO_main(int argc,char *argv[]);


#if defined(_WIN32) || defined(__CODEGEARC__)/* @@edc: added for windows getopt */
extern char *optarg;
int getopt(int nargc, char * const *nargv, const char *ostr);
#endif

#ifdef FLOATING_POINT_TRAP
#include <fpu_control.h>
/* This is a x86 specific function only used for debugging.
It turns on various floating point traps.  */
static int trapfpe()
{
    fpu_control_t traps;
    traps = _FPU_DEFAULT & (~(_FPU_MASK_IM | _FPU_MASK_ZM | _FPU_MASK_OM));
    _FPU_SETCW(traps);
}
#endif

namespace autolib {

FILE *fp2  = NULL;
FILE *fp3  = NULL;
FILE *fp6  = NULL;
FILE *fp7  = NULL;
FILE *fp8  = NULL;
FILE *fp9  = NULL;
FILE *fp12 = NULL;

void CloseAllFiles()
{
    if (fp2 != NULL) {fflush(fp2); fclose(fp2); fp2= NULL;}
    if (fp3 != NULL) {fflush(fp3);fclose(fp3); fp3= NULL;}
    if (fp6 != NULL) {fflush(fp6);fclose(fp6); fp6= NULL;}
    if (fp7 != NULL) {fflush(fp7);fclose(fp7); fp7= NULL;}
    if (fp8 != NULL) {fflush(fp8);fclose(fp8); fp8= NULL;}
    if (fp9 != NULL) {fflush(fp9);fclose(fp9); fp9= NULL;}
    if (fp12 != NULL) {fflush(fp12);fclose(fp12); fp12= NULL;}
}


// initialize these... they may be reset before a call to AUTO_main
int num_model_pars = 10;
int num_total_pars = 2*num_model_pars;
int sysoff = num_model_pars;

// add changable fort names for ease
char fort_name[13][512];
int fort_names_are_valid = 0;

void SetFortNames(const char *key)
{
    sprintf(fort_name[2], "%s.2", key);
    sprintf(fort_name[3], "%s.3", key);
    sprintf(fort_name[6], "%s.6", key);
    sprintf(fort_name[7], "%s.7", key);
    sprintf(fort_name[8], "%s.8", key);
    sprintf(fort_name[9], "%s.9", key);
    sprintf(fort_name[12], "%s.12", key);
}

int global_conpar_type=CONPAR_DEFAULT;
int global_setubv_type=SETUBV_DEFAULT;
int global_reduce_type=REDUCE_DEFAULT;
int global_num_procs=1;
int global_verbose_flag=0;

static void options(){
    fprintf(fp6, "-v:    Give verbose output.\n");
    fprintf(fp6, "-m:    Use the Message Passing Interface library for parallelization.\n");
    fprintf(fp6, "-t:    Use the Pthreads library for parallelization.\n");
    fprintf(fp6, "       This option takes one of three arguements.\n"); 
    fprintf(fp6, "       'conpar' parallelizes the condensation of parameters routine.\n");
    fprintf(fp6, "       'setubv' parallelizes the jacobian setup routine.\n");
    fprintf(fp6, "       'reduce' parallelizes the nested dissection routine.\n");
    fprintf(fp6, "       'all'    parallelizes all routines.\n");
    fprintf(fp6, "       In general the recommeneded option is 'all'.\n");
    fprintf(fp6, "-#:    The number of processing units to use (currently only used with the -t option)\n");
}

static void scheme_not_supported_error(char *scheme) {
    fprintf(fp6, "'%s' not available in this binary\n",scheme);
    fprintf(fp6, "Support for '%s' needs to be included at compile time\n",scheme);
    throw "not supported";
    //return 0;
    //exit(0);
}

int AUTO_main(int argc, char *argv[])
//int main(int argc,char *argv[])
{
#if defined(_WIN32) || defined(__WIN32__)
    clock_t time0, time1;
#else
//     struct timeval  *time0,*time1;
    clock_t time0;
    clock_t time1;
#endif
    integer *icp = new integer[num_total_pars];
    doublereal *par = new doublereal[num_total_pars], *thl = new doublereal[num_total_pars];
    doublereal *thu = NULL;
    integer *iuz= NULL;
    doublereal *vuz= NULL;
    iap_type iap;
    rap_type rap;
    function_list list;

#ifdef USAGE
    struct rusage *init_usage,*total_usage;
    usage_start(&init_usage);
    usage_start(&total_usage);
#endif

#ifdef FLOATING_POINT_TRAP
    trapfpe();
#endif

#ifdef MPI
    MPI_Init(&argc, &argv);
#endif

    // check if we should set fort_names or not
    if(!fort_names_are_valid)
    {
        SetFortNames("fort");
        //strcpy(fort_name[6], "stdout");
    }

OPEN_FP3:
    fp3 = fopen(fort_name[3],"r");
    if(fp3 == NULL) {
        //fprintf(stderr,"warning:  Could not open %s\n", fort_name[3]);

        // don't die, just "touch" the file
        fp3 = fopen(fort_name[3], "w");
        if(fp3 == NULL)
        {
            fprintf(stderr,"Error: couldn't not create %s\n", fort_name[3]);
            throw "Error:  Could not open fort.3";
            //return -1;
            //exit(1);
        }
        fclose(fp3);
        goto OPEN_FP3;
    }

    fp6 = fopen(fort_name[6],"w");
    if(fp6 == NULL) 
    {
        fprintf(stderr,"Error:  Could not open %s\n", fort_name[6]);
        throw "Error:  Could not open fort.6";
            //return -1;
            //exit(1);
    }


    //#ifdef LIBRARY_ONLY
    //if(strcmp(fort_name[6], "stdout"))
    //{
    //  fp6 = fopen(fort_name[6],"w");
    //  if(fp6 == NULL) {
    //      fprintf(stderr,"Error:  Could not open %s\n", fort_name[6]);
    //      throw "Error:  Could not open fort.2";
    //      //return -1;
    //      //exit(1);
    //  }
    //}
    //else
    //  fp6 = stdout;
    //#endif
    fp7 = fopen(fort_name[7],"w");
    if(fp7 == NULL) 
    {
        fprintf(stderr,"Error:  Could not open %s\n", fort_name[7]);
        throw "Error:  Could not open fort.7";
            //return -1;
            //exit(1);
    }
    fp9 = fopen(fort_name[9],"w");
    if(fp9 == NULL) 
    {
        fprintf(stderr,"Error:  Could not open %s\n", fort_name[9]);
        throw "Error:  Could not open fort.9";
            //return -1;
            //exit(1);
    }

    {
        int c;
        while (1)
        {
            c = getopt(argc, argv, "mt:?#:v");
            if (c == -1)
                break;
            switch (c){
    case 'v':
        global_verbose_flag=1;
        break;
    case 'm':
#ifdef MPI
        global_conpar_type = CONPAR_MPI;
        global_setubv_type = SETUBV_MPI;
        break;
#endif
        scheme_not_supported_error((char*)"mpi");
        break;
    case 't':
#ifdef PTHREADS
        if(strcmp(optarg,"setubv")==0) {
            global_setubv_type = SETUBV_PTHREADS;
        }
        else if(strcmp(optarg,"conpar")==0) {
            global_conpar_type = CONPAR_PTHREADS;
        }
        else if(strcmp(optarg,"reduce")==0) {
            global_conpar_type = REDUCE_PTHREADS;
        }
        else if(strcmp(optarg,"all")==0) {
            global_conpar_type = CONPAR_PTHREADS;
            global_setubv_type = SETUBV_PTHREADS;
            global_reduce_type = REDUCE_PTHREADS;
        }
        else {
            fprintf(stderr,"Unknown type for threads '%s'.  Using 'all'\n",optarg);
            global_conpar_type = CONPAR_PTHREADS;
            global_setubv_type = SETUBV_PTHREADS;
        }
        break;
#endif
        scheme_not_supported_error((char*)"threads");
        break;
    case '#':
        global_num_procs=atoi(optarg);
        break;
    case '?':
        options();
        return 0;
        //exit(0);
        //break;
    default:
        printf ("?? getopt returned character code 0%o ??\n", c);
        options();
                    return 0;
            //exit(1);

        //exit(0);
            }
        } // while
    } // scope

#ifdef MPI
    {
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int myid,namelen;
        MPI_Comm_rank(MPI_COMM_WORLD,&myid);
        MPI_Get_processor_name(processor_name,&namelen);
        if(global_verbose_flag) {
            fprintf(stderr,"Process %d on %s with pid %ld\n",
                myid, processor_name, (long int)getpid());
        }
        if(myid!=0) {
            global_conpar_type = CONPAR_MPI;
            global_setubv_type = SETUBV_MPI;
            mpi_worker();
        }
    }    
#endif    
    /* Initialization : */

    iap.mynode = mynode();
    iap.numnodes = numnodes();
    if (iap.numnodes > 1) {
        iap.parallel_flag = 1;
    } else {
        iap.parallel_flag = 0;
    }


    while(1){
        time_start(&time0);
        time_start(&time1);
        /* NOTE:  thu is allocated inside this function, and the
        pointer is passed back.  I know this is ugly, but
        this function does a bit of work to get thu setup correctly,
        as well as figuring out the size the array should be.
        What really should happen is to have one function which
        reads fort.2 and another fuction which initializes the array.
        That way the allocation could happen between the two calles.
        */
        {
            logical eof;
            init(&iap, &rap, par, icp, thl, &thu, &iuz, &vuz, &eof);

            if (eof) {
                break;
            }
        }

        /* Find restart label and determine type of restart point. */
        if (iap.irs > 0) {
            logical found = FALSE_;

            findlb(&iap, &rap, iap.irs, &(iap.nfpr), &found);
            if (! found) {
                if (iap.mynode == 0) {
                    fprintf(stderr,"\nRestart label %4ld not found\n",iap.irs);
                }
                throw "Restart label not found";
                //return 0;
                //exit(0);
            }
        }
#ifdef MPI

        if(global_setubv_type==SETUBV_MPI) {
            /* A few words about what is going on here.  ips, irs, isw, itp, and
            nfpr are used to choose which functions are used for funi, icni, bcni, etc.
            unfortunately, their values are changed in init1 and chdim.  In the
            old version of AUTO the functions were already choosen by the point
            these values were modified, so there was no problem.  Now, in the
            message passing parallel version, the workers need both versions, since
            they both need to select the appropriate functions (using the old values)
            and actually compute (using the new values).  */
            int comm_size,i;
            integer funi_icni_params[5];
            MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
            funi_icni_params[0]=iap.ips;
            funi_icni_params[1]=iap.irs;
            funi_icni_params[2]=iap.isw;
            funi_icni_params[3]=iap.itp;
            funi_icni_params[4]=iap.nfpr;
            for(i=1;i<comm_size;i++){
                /*Send message to get worker into init mode*/
                {
                    int message=AUTO_MPI_INIT_MESSAGE;
                    MPI_Send(&message,1,MPI_INT,i,0,MPI_COMM_WORLD);
                }
            }
            MPI_Bcast(funi_icni_params,5,MPI_LONG,0,MPI_COMM_WORLD);
        }
#endif
        set_function_pointers(iap,&list);
        init1(&iap, &rap, icp, par);
        chdim(&iap);

        /* Create the allocations for the global structures used in 
        autlib3.c and autlib5.c.  These are purely an efficiency thing.
        The allocation and deallocation of these scratch areas takes
        up a nontrivial amount of time if done directly in the
        wrapper functions in autlib3.c*/
        allocate_global_memory(iap);

        /* ---------------------------------------------------------- */
        /* ---------------------------------------------------------- */
        /*  One-parameter continuations */
        /* ---------------------------------------------------------- */
        /* ---------------------------------------------------------- */

#ifdef AUTO_CONSTRUCT_DESTRUCT
        user_construct(argc,argv);
#endif
#ifdef USAGE
        usage_end(init_usage,"main initialization");
#endif

        if(list.type==AUTOAE)
            autoae(&iap, &rap, par, icp, list.aelist.funi, list.aelist.stpnt, list.aelist.pvli, thl, thu, iuz, vuz);
        if(list.type==AUTOBV)
            autobv(&iap, &rap, par, icp, list.bvlist.funi, list.bvlist.bcni,
            list.bvlist.icni, list.bvlist.stpnt, list.bvlist.pvli, thl, thu, iuz, vuz);

#ifdef USAGE
        usage_end(total_usage,"total");

#endif
        time_end(time0,(char*)"Total Time ",fp9);
        fprintf(fp9,"----------------------------------------------");
        fprintf(fp9,"----------------------------------------------\n");
        time_end(time1,(char*)"",fp6);
#ifdef AUTO_CONSTRUCT_DESTRUCT
        user_destruct();
#endif
    }
#ifdef MPI
    {
        int message = AUTO_MPI_KILL_MESSAGE;
        int size,i;
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        for(i=1;i<size;i++)
            MPI_Send(&message,1,MPI_INT,i,0,MPI_COMM_WORLD);
    }

    MPI_Finalize();
#endif

    delete [] icp;
    delete [] par;
    delete [] thl;

    if (thu) free(thu);
    if (iuz) free(iuz);
    if (vuz) free(vuz);

    // close files
    CloseAllFiles();
    return 0;
}

}
