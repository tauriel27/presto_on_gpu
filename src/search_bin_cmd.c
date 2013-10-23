/*****
  command line parser -- generated by clig
  (http://wsd.iitb.fhg.de/~kir/clighome/)

  The command line parser `clig':
  (C) 1995-2004 Harald Kirsch (clig@geggus.net)
*****/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <float.h>
#include <math.h>

#include "search_bin_cmd.h"

char *Program;

/*@-null*/

static Cmdline cmd = {
  /***** -ncand: Number of candidates to try to return */
  /* ncandP = */ 1,
  /* ncand = */ 100,
  /* ncandC = */ 1,
  /***** -minfft: Power-of-2 length of the shortest miniFFT */
  /* minfftP = */ 1,
  /* minfft = */ 32,
  /* minfftC = */ 1,
  /***** -maxfft: Power-of-2 length of the longest miniFFT */
  /* maxfftP = */ 1,
  /* maxfft = */ 65536,
  /* maxfftC = */ 1,
  /***** -flo: The low frequency (Hz) to check */
  /* floP = */ 1,
  /* flo = */ 1.0,
  /* floC = */ 1,
  /***** -fhi: The high frequency (Hz) to check */
  /* fhiP = */ 1,
  /* fhi = */ 2000.0,
  /* fhiC = */ 1,
  /***** -rlo: The low Fourier frequency to check */
  /* rloP = */ 0,
  /* rlo = */ (int)0,
  /* rloC = */ 0,
  /***** -rhi: The high Fourier frequency to check */
  /* rhiP = */ 0,
  /* rhi = */ (int)0,
  /* rhiC = */ 0,
  /***** -lobin: The first Fourier frequency in the data file */
  /* lobinP = */ 1,
  /* lobin = */ 0,
  /* lobinC = */ 1,
  /***** -overlap: Fraction of a short FFT length to shift before performing another */
  /* overlapP = */ 1,
  /* overlap = */ 0.25,
  /* overlapC = */ 1,
  /***** -harmsum: Number of harmonics to sum in the miniFFTs */
  /* harmsumP = */ 1,
  /* harmsum = */ 3,
  /* harmsumC = */ 1,
  /***** -numbetween: Number of points to interpolate per Fourier bin (2 gives the usual bin value and an interbin) */
  /* numbetweenP = */ 1,
  /* numbetween = */ 2,
  /* numbetweenC = */ 1,
  /***** -stack: Number of stacked power spectra making up the data.  (The default means the data are complex amplitudes) */
  /* stackP = */ 1,
  /* stack = */ 0,
  /* stackC = */ 1,
  /***** -interbin: Use interbinning instead of full-blown Fourier interpolation.  (Faster but less accurate and sensitive) */
  /* interbinP = */ 0,
  /***** -noalias: Do not add aliased powers to the harmonic sum.  (Faster but less accurate and sensitive) */
  /* noaliasP = */ 0,
  /***** uninterpreted rest of command line */
  /* argc = */ 0,
  /* argv = */ (char**)0,
  /***** the original command line concatenated */
  /* full_cmd_line = */ NULL
};

/*@=null*/

/***** let LCLint run more smoothly */
/*@-predboolothers*/
/*@-boolops*/


/******************************************************************/
/*****
 This is a bit tricky. We want to make a difference between overflow
 and underflow and we want to allow v==Inf or v==-Inf but not
 v>FLT_MAX. 

 We don't use fabs to avoid linkage with -lm.
*****/
static void
checkFloatConversion(double v, char *option, char *arg)
{
  char *err = NULL;

  if( (errno==ERANGE && v!=0.0) /* even double overflowed */
      || (v<HUGE_VAL && v>-HUGE_VAL && (v<0.0?-v:v)>(double)FLT_MAX) ) {
    err = "large";
  } else if( (errno==ERANGE && v==0.0) 
	     || (v!=0.0 && (v<0.0?-v:v)<(double)FLT_MIN) ) {
    err = "small";
  }
  if( err ) {
    fprintf(stderr, 
	    "%s: parameter `%s' of option `%s' to %s to represent\n",
	    Program, arg, option, err);
    exit(EXIT_FAILURE);
  }
}

int
getIntOpt(int argc, char **argv, int i, int *value, int force)
{
  char *end;
  long v;

  if( ++i>=argc ) goto nothingFound;

  errno = 0;
  v = strtol(argv[i], &end, 0);

  /***** check for conversion error */
  if( end==argv[i] ) goto nothingFound;

  /***** check for surplus non-whitespace */
  while( isspace((int) *end) ) end+=1;
  if( *end ) goto nothingFound;

  /***** check if it fits into an int */
  if( errno==ERANGE || v>(long)INT_MAX || v<(long)INT_MIN ) {
    fprintf(stderr, 
	    "%s: parameter `%s' of option `%s' to large to represent\n",
	    Program, argv[i], argv[i-1]);
    exit(EXIT_FAILURE);
  }
  *value = (int)v;

  return i;

nothingFound:
  if( !force ) return i-1;

  fprintf(stderr, 
	  "%s: missing or malformed integer value after option `%s'\n",
	  Program, argv[i-1]);
    exit(EXIT_FAILURE);
}
/**********************************************************************/

int
getIntOpts(int argc, char **argv, int i, 
	   int **values,
	   int cmin, int cmax)
/*****
  We want to find at least cmin values and at most cmax values.
  cmax==-1 then means infinitely many are allowed.
*****/
{
  int alloced, used;
  char *end;
  long v;
  if( i+cmin >= argc ) {
    fprintf(stderr, 
	    "%s: option `%s' wants at least %d parameters\n",
	    Program, argv[i], cmin);
    exit(EXIT_FAILURE);
  }

  /***** 
    alloc a bit more than cmin values. It does not hurt to have room
    for a bit more values than cmax.
  *****/
  alloced = cmin + 4;
  *values = (int*)calloc((size_t)alloced, sizeof(int));
  if( ! *values ) {
outMem:
    fprintf(stderr, 
	    "%s: out of memory while parsing option `%s'\n",
	    Program, argv[i]);
    exit(EXIT_FAILURE);
  }

  for(used=0; (cmax==-1 || used<cmax) && used+i+1<argc; used++) {
    if( used==alloced ) {
      alloced += 8;
      *values = (int *) realloc(*values, alloced*sizeof(int));
      if( !*values ) goto outMem;
    }

    errno = 0;
    v = strtol(argv[used+i+1], &end, 0);

    /***** check for conversion error */
    if( end==argv[used+i+1] ) break;

    /***** check for surplus non-whitespace */
    while( isspace((int) *end) ) end+=1;
    if( *end ) break;

    /***** check for overflow */
    if( errno==ERANGE || v>(long)INT_MAX || v<(long)INT_MIN ) {
      fprintf(stderr, 
	      "%s: parameter `%s' of option `%s' to large to represent\n",
	      Program, argv[i+used+1], argv[i]);
      exit(EXIT_FAILURE);
    }

    (*values)[used] = (int)v;

  }
    
  if( used<cmin ) {
    fprintf(stderr, 
	    "%s: parameter `%s' of `%s' should be an "
	    "integer value\n",
	    Program, argv[i+used+1], argv[i]);
    exit(EXIT_FAILURE);
  }

  return i+used;
}
/**********************************************************************/

int
getLongOpt(int argc, char **argv, int i, long *value, int force)
{
  char *end;

  if( ++i>=argc ) goto nothingFound;

  errno = 0;
  *value = strtol(argv[i], &end, 0);

  /***** check for conversion error */
  if( end==argv[i] ) goto nothingFound;

  /***** check for surplus non-whitespace */
  while( isspace((int) *end) ) end+=1;
  if( *end ) goto nothingFound;

  /***** check for overflow */
  if( errno==ERANGE ) {
    fprintf(stderr, 
	    "%s: parameter `%s' of option `%s' to large to represent\n",
	    Program, argv[i], argv[i-1]);
    exit(EXIT_FAILURE);
  }
  return i;

nothingFound:
  /***** !force means: this parameter may be missing.*/
  if( !force ) return i-1;

  fprintf(stderr, 
	  "%s: missing or malformed value after option `%s'\n",
	  Program, argv[i-1]);
    exit(EXIT_FAILURE);
}
/**********************************************************************/

int
getLongOpts(int argc, char **argv, int i, 
	    long **values,
	    int cmin, int cmax)
/*****
  We want to find at least cmin values and at most cmax values.
  cmax==-1 then means infinitely many are allowed.
*****/
{
  int alloced, used;
  char *end;

  if( i+cmin >= argc ) {
    fprintf(stderr, 
	    "%s: option `%s' wants at least %d parameters\n",
	    Program, argv[i], cmin);
    exit(EXIT_FAILURE);
  }

  /***** 
    alloc a bit more than cmin values. It does not hurt to have room
    for a bit more values than cmax.
  *****/
  alloced = cmin + 4;
  *values = (long int *)calloc((size_t)alloced, sizeof(long));
  if( ! *values ) {
outMem:
    fprintf(stderr, 
	    "%s: out of memory while parsing option `%s'\n",
	    Program, argv[i]);
    exit(EXIT_FAILURE);
  }

  for(used=0; (cmax==-1 || used<cmax) && used+i+1<argc; used++) {
    if( used==alloced ) {
      alloced += 8;
      *values = (long int*) realloc(*values, alloced*sizeof(long));
      if( !*values ) goto outMem;
    }

    errno = 0;
    (*values)[used] = strtol(argv[used+i+1], &end, 0);

    /***** check for conversion error */
    if( end==argv[used+i+1] ) break;

    /***** check for surplus non-whitespace */
    while( isspace((int) *end) ) end+=1; 
    if( *end ) break;

    /***** check for overflow */
    if( errno==ERANGE ) {
      fprintf(stderr, 
	      "%s: parameter `%s' of option `%s' to large to represent\n",
	      Program, argv[i+used+1], argv[i]);
      exit(EXIT_FAILURE);
    }

  }
    
  if( used<cmin ) {
    fprintf(stderr, 
	    "%s: parameter `%s' of `%s' should be an "
	    "integer value\n",
	    Program, argv[i+used+1], argv[i]);
    exit(EXIT_FAILURE);
  }

  return i+used;
}
/**********************************************************************/

int
getFloatOpt(int argc, char **argv, int i, float *value, int force)
{
  char *end;
  double v;

  if( ++i>=argc ) goto nothingFound;

  errno = 0;
  v = strtod(argv[i], &end);

  /***** check for conversion error */
  if( end==argv[i] ) goto nothingFound;

  /***** check for surplus non-whitespace */
  while( isspace((int) *end) ) end+=1;
  if( *end ) goto nothingFound;

  /***** check for overflow */
  checkFloatConversion(v, argv[i-1], argv[i]);

  *value = (float)v;

  return i;

nothingFound:
  if( !force ) return i-1;

  fprintf(stderr,
	  "%s: missing or malformed float value after option `%s'\n",
	  Program, argv[i-1]);
  exit(EXIT_FAILURE);
 
}
/**********************************************************************/

int
getFloatOpts(int argc, char **argv, int i, 
	   float **values,
	   int cmin, int cmax)
/*****
  We want to find at least cmin values and at most cmax values.
  cmax==-1 then means infinitely many are allowed.
*****/
{
  int alloced, used;
  char *end;
  double v;

  if( i+cmin >= argc ) {
    fprintf(stderr, 
	    "%s: option `%s' wants at least %d parameters\n",
	    Program, argv[i], cmin);
    exit(EXIT_FAILURE);
  }

  /***** 
    alloc a bit more than cmin values.
  *****/
  alloced = cmin + 4;
  *values = (float*)calloc((size_t)alloced, sizeof(float));
  if( ! *values ) {
outMem:
    fprintf(stderr, 
	    "%s: out of memory while parsing option `%s'\n",
	    Program, argv[i]);
    exit(EXIT_FAILURE);
  }

  for(used=0; (cmax==-1 || used<cmax) && used+i+1<argc; used++) {
    if( used==alloced ) {
      alloced += 8;
      *values = (float *) realloc(*values, alloced*sizeof(float));
      if( !*values ) goto outMem;
    }

    errno = 0;
    v = strtod(argv[used+i+1], &end);

    /***** check for conversion error */
    if( end==argv[used+i+1] ) break;

    /***** check for surplus non-whitespace */
    while( isspace((int) *end) ) end+=1;
    if( *end ) break;

    /***** check for overflow */
    checkFloatConversion(v, argv[i], argv[i+used+1]);
    
    (*values)[used] = (float)v;
  }
    
  if( used<cmin ) {
    fprintf(stderr, 
	    "%s: parameter `%s' of `%s' should be a "
	    "floating-point value\n",
	    Program, argv[i+used+1], argv[i]);
    exit(EXIT_FAILURE);
  }

  return i+used;
}
/**********************************************************************/

int
getDoubleOpt(int argc, char **argv, int i, double *value, int force)
{
  char *end;

  if( ++i>=argc ) goto nothingFound;

  errno = 0;
  *value = strtod(argv[i], &end);

  /***** check for conversion error */
  if( end==argv[i] ) goto nothingFound;

  /***** check for surplus non-whitespace */
  while( isspace((int) *end) ) end+=1;
  if( *end ) goto nothingFound;

  /***** check for overflow */
  if( errno==ERANGE ) {
    fprintf(stderr, 
	    "%s: parameter `%s' of option `%s' to %s to represent\n",
	    Program, argv[i], argv[i-1],
	    (*value==0.0 ? "small" : "large"));
    exit(EXIT_FAILURE);
  }

  return i;

nothingFound:
  if( !force ) return i-1;

  fprintf(stderr,
	  "%s: missing or malformed value after option `%s'\n",
	  Program, argv[i-1]);
  exit(EXIT_FAILURE);
 
}
/**********************************************************************/

int
getDoubleOpts(int argc, char **argv, int i, 
	   double **values,
	   int cmin, int cmax)
/*****
  We want to find at least cmin values and at most cmax values.
  cmax==-1 then means infinitely many are allowed.
*****/
{
  int alloced, used;
  char *end;

  if( i+cmin >= argc ) {
    fprintf(stderr, 
	    "%s: option `%s' wants at least %d parameters\n",
	    Program, argv[i], cmin);
    exit(EXIT_FAILURE);
  }

  /***** 
    alloc a bit more than cmin values.
  *****/
  alloced = cmin + 4;
  *values = (double*)calloc((size_t)alloced, sizeof(double));
  if( ! *values ) {
outMem:
    fprintf(stderr, 
	    "%s: out of memory while parsing option `%s'\n",
	    Program, argv[i]);
    exit(EXIT_FAILURE);
  }

  for(used=0; (cmax==-1 || used<cmax) && used+i+1<argc; used++) {
    if( used==alloced ) {
      alloced += 8;
      *values = (double *) realloc(*values, alloced*sizeof(double));
      if( !*values ) goto outMem;
    }

    errno = 0;
    (*values)[used] = strtod(argv[used+i+1], &end);

    /***** check for conversion error */
    if( end==argv[used+i+1] ) break;

    /***** check for surplus non-whitespace */
    while( isspace((int) *end) ) end+=1;
    if( *end ) break;

    /***** check for overflow */
    if( errno==ERANGE ) {
      fprintf(stderr, 
	      "%s: parameter `%s' of option `%s' to %s to represent\n",
	      Program, argv[i+used+1], argv[i],
	      ((*values)[used]==0.0 ? "small" : "large"));
      exit(EXIT_FAILURE);
    }

  }
    
  if( used<cmin ) {
    fprintf(stderr, 
	    "%s: parameter `%s' of `%s' should be a "
	    "double value\n",
	    Program, argv[i+used+1], argv[i]);
    exit(EXIT_FAILURE);
  }

  return i+used;
}
/**********************************************************************/

/**
  force will be set if we need at least one argument for the option.
*****/
int
getStringOpt(int argc, char **argv, int i, char **value, int force)
{
  i += 1;
  if( i>=argc ) {
    if( force ) {
      fprintf(stderr, "%s: missing string after option `%s'\n",
	      Program, argv[i-1]);
      exit(EXIT_FAILURE);
    } 
    return i-1;
  }
  
  if( !force && argv[i][0] == '-' ) return i-1;
  *value = argv[i];
  return i;
}
/**********************************************************************/

int
getStringOpts(int argc, char **argv, int i, 
	   char*  **values,
	   int cmin, int cmax)
/*****
  We want to find at least cmin values and at most cmax values.
  cmax==-1 then means infinitely many are allowed.
*****/
{
  int alloced, used;

  if( i+cmin >= argc ) {
    fprintf(stderr, 
	    "%s: option `%s' wants at least %d parameters\n",
	    Program, argv[i], cmin);
    exit(EXIT_FAILURE);
  }

  alloced = cmin + 4;
    
  *values = (char**)calloc((size_t)alloced, sizeof(char*));
  if( ! *values ) {
outMem:
    fprintf(stderr, 
	    "%s: out of memory during parsing of option `%s'\n",
	    Program, argv[i]);
    exit(EXIT_FAILURE);
  }

  for(used=0; (cmax==-1 || used<cmax) && used+i+1<argc; used++) {
    if( used==alloced ) {
      alloced += 8;
      *values = (char **)realloc(*values, alloced*sizeof(char*));
      if( !*values ) goto outMem;
    }

    if( used>=cmin && argv[used+i+1][0]=='-' ) break;
    (*values)[used] = argv[used+i+1];
  }
    
  if( used<cmin ) {
    fprintf(stderr, 
    "%s: less than %d parameters for option `%s', only %d found\n",
	    Program, cmin, argv[i], used);
    exit(EXIT_FAILURE);
  }

  return i+used;
}
/**********************************************************************/

void
checkIntLower(char *opt, int *values, int count, int max)
{
  int i;

  for(i=0; i<count; i++) {
    if( values[i]<=max ) continue;
    fprintf(stderr, 
	    "%s: parameter %d of option `%s' greater than max=%d\n",
	    Program, i+1, opt, max);
    exit(EXIT_FAILURE);
  }
}
/**********************************************************************/

void
checkIntHigher(char *opt, int *values, int count, int min)
{
  int i;

  for(i=0; i<count; i++) {
    if( values[i]>=min ) continue;
    fprintf(stderr, 
	    "%s: parameter %d of option `%s' smaller than min=%d\n",
	    Program, i+1, opt, min);
    exit(EXIT_FAILURE);
  }
}
/**********************************************************************/

void
checkLongLower(char *opt, long *values, int count, long max)
{
  int i;

  for(i=0; i<count; i++) {
    if( values[i]<=max ) continue;
    fprintf(stderr, 
	    "%s: parameter %d of option `%s' greater than max=%ld\n",
	    Program, i+1, opt, max);
    exit(EXIT_FAILURE);
  }
}
/**********************************************************************/

void
checkLongHigher(char *opt, long *values, int count, long min)
{
  int i;

  for(i=0; i<count; i++) {
    if( values[i]>=min ) continue;
    fprintf(stderr, 
	    "%s: parameter %d of option `%s' smaller than min=%ld\n",
	    Program, i+1, opt, min);
    exit(EXIT_FAILURE);
  }
}
/**********************************************************************/

void
checkFloatLower(char *opt, float *values, int count, float max)
{
  int i;

  for(i=0; i<count; i++) {
    if( values[i]<=max ) continue;
    fprintf(stderr, 
	    "%s: parameter %d of option `%s' greater than max=%f\n",
	    Program, i+1, opt, max);
    exit(EXIT_FAILURE);
  }
}
/**********************************************************************/

void
checkFloatHigher(char *opt, float *values, int count, float min)
{
  int i;

  for(i=0; i<count; i++) {
    if( values[i]>=min ) continue;
    fprintf(stderr, 
	    "%s: parameter %d of option `%s' smaller than min=%f\n",
	    Program, i+1, opt, min);
    exit(EXIT_FAILURE);
  }
}
/**********************************************************************/

void
checkDoubleLower(char *opt, double *values, int count, double max)
{
  int i;

  for(i=0; i<count; i++) {
    if( values[i]<=max ) continue;
    fprintf(stderr, 
	    "%s: parameter %d of option `%s' greater than max=%f\n",
	    Program, i+1, opt, max);
    exit(EXIT_FAILURE);
  }
}
/**********************************************************************/

void
checkDoubleHigher(char *opt, double *values, int count, double min)
{
  int i;

  for(i=0; i<count; i++) {
    if( values[i]>=min ) continue;
    fprintf(stderr, 
	    "%s: parameter %d of option `%s' smaller than min=%f\n",
	    Program, i+1, opt, min);
    exit(EXIT_FAILURE);
  }
}
/**********************************************************************/

static char *
catArgv(int argc, char **argv)
{
  int i;
  size_t l;
  char *s, *t;

  for(i=0, l=0; i<argc; i++) l += (1+strlen(argv[i]));
  s = (char *)malloc(l);
  if( !s ) {
    fprintf(stderr, "%s: out of memory\n", Program);
    exit(EXIT_FAILURE);
  }
  strcpy(s, argv[0]);
  t = s;
  for(i=1; i<argc; i++) {
    t = t+strlen(t);
    *t++ = ' ';
    strcpy(t, argv[i]);
  }
  return s;
}
/**********************************************************************/

void
showOptionValues(void)
{
  int i;

  printf("Full command line is:\n`%s'\n", cmd.full_cmd_line);

  /***** -ncand: Number of candidates to try to return */
  if( !cmd.ncandP ) {
    printf("-ncand not found.\n");
  } else {
    printf("-ncand found:\n");
    if( !cmd.ncandC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%d'\n", cmd.ncand);
    }
  }

  /***** -minfft: Power-of-2 length of the shortest miniFFT */
  if( !cmd.minfftP ) {
    printf("-minfft not found.\n");
  } else {
    printf("-minfft found:\n");
    if( !cmd.minfftC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%d'\n", cmd.minfft);
    }
  }

  /***** -maxfft: Power-of-2 length of the longest miniFFT */
  if( !cmd.maxfftP ) {
    printf("-maxfft not found.\n");
  } else {
    printf("-maxfft found:\n");
    if( !cmd.maxfftC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%d'\n", cmd.maxfft);
    }
  }

  /***** -flo: The low frequency (Hz) to check */
  if( !cmd.floP ) {
    printf("-flo not found.\n");
  } else {
    printf("-flo found:\n");
    if( !cmd.floC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%.40g'\n", cmd.flo);
    }
  }

  /***** -fhi: The high frequency (Hz) to check */
  if( !cmd.fhiP ) {
    printf("-fhi not found.\n");
  } else {
    printf("-fhi found:\n");
    if( !cmd.fhiC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%.40g'\n", cmd.fhi);
    }
  }

  /***** -rlo: The low Fourier frequency to check */
  if( !cmd.rloP ) {
    printf("-rlo not found.\n");
  } else {
    printf("-rlo found:\n");
    if( !cmd.rloC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%d'\n", cmd.rlo);
    }
  }

  /***** -rhi: The high Fourier frequency to check */
  if( !cmd.rhiP ) {
    printf("-rhi not found.\n");
  } else {
    printf("-rhi found:\n");
    if( !cmd.rhiC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%d'\n", cmd.rhi);
    }
  }

  /***** -lobin: The first Fourier frequency in the data file */
  if( !cmd.lobinP ) {
    printf("-lobin not found.\n");
  } else {
    printf("-lobin found:\n");
    if( !cmd.lobinC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%d'\n", cmd.lobin);
    }
  }

  /***** -overlap: Fraction of a short FFT length to shift before performing another */
  if( !cmd.overlapP ) {
    printf("-overlap not found.\n");
  } else {
    printf("-overlap found:\n");
    if( !cmd.overlapC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%.40g'\n", cmd.overlap);
    }
  }

  /***** -harmsum: Number of harmonics to sum in the miniFFTs */
  if( !cmd.harmsumP ) {
    printf("-harmsum not found.\n");
  } else {
    printf("-harmsum found:\n");
    if( !cmd.harmsumC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%d'\n", cmd.harmsum);
    }
  }

  /***** -numbetween: Number of points to interpolate per Fourier bin (2 gives the usual bin value and an interbin) */
  if( !cmd.numbetweenP ) {
    printf("-numbetween not found.\n");
  } else {
    printf("-numbetween found:\n");
    if( !cmd.numbetweenC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%d'\n", cmd.numbetween);
    }
  }

  /***** -stack: Number of stacked power spectra making up the data.  (The default means the data are complex amplitudes) */
  if( !cmd.stackP ) {
    printf("-stack not found.\n");
  } else {
    printf("-stack found:\n");
    if( !cmd.stackC ) {
      printf("  no values\n");
    } else {
      printf("  value = `%d'\n", cmd.stack);
    }
  }

  /***** -interbin: Use interbinning instead of full-blown Fourier interpolation.  (Faster but less accurate and sensitive) */
  if( !cmd.interbinP ) {
    printf("-interbin not found.\n");
  } else {
    printf("-interbin found:\n");
  }

  /***** -noalias: Do not add aliased powers to the harmonic sum.  (Faster but less accurate and sensitive) */
  if( !cmd.noaliasP ) {
    printf("-noalias not found.\n");
  } else {
    printf("-noalias found:\n");
  }
  if( !cmd.argc ) {
    printf("no remaining parameters in argv\n");
  } else {
    printf("argv =");
    for(i=0; i<cmd.argc; i++) {
      printf(" `%s'", cmd.argv[i]);
    }
    printf("\n");
  }
}
/**********************************************************************/

void
usage(void)
{
  fprintf(stderr,"%s","   [-ncand ncand] [-minfft minfft] [-maxfft maxfft] [-flo flo] [-fhi fhi] [-rlo rlo] [-rhi rhi] [-lobin lobin] [-overlap overlap] [-harmsum harmsum] [-numbetween numbetween] [-stack stack] [-interbin] [-noalias] [--] infile\n");
  fprintf(stderr,"%s","      Searches a long FFT for binary pulsar candidates using a phase modulation search.\n");
  fprintf(stderr,"%s","         -ncand: Number of candidates to try to return\n");
  fprintf(stderr,"%s","                 1 int value between 1 and 10000\n");
  fprintf(stderr,"%s","                 default: `100'\n");
  fprintf(stderr,"%s","        -minfft: Power-of-2 length of the shortest miniFFT\n");
  fprintf(stderr,"%s","                 1 int value between 8 and 1048576\n");
  fprintf(stderr,"%s","                 default: `32'\n");
  fprintf(stderr,"%s","        -maxfft: Power-of-2 length of the longest miniFFT\n");
  fprintf(stderr,"%s","                 1 int value between 8 and 1048576\n");
  fprintf(stderr,"%s","                 default: `65536'\n");
  fprintf(stderr,"%s","           -flo: The low frequency (Hz) to check\n");
  fprintf(stderr,"%s","                 1 float value between 0 and oo\n");
  fprintf(stderr,"%s","                 default: `1.0'\n");
  fprintf(stderr,"%s","           -fhi: The high frequency (Hz) to check\n");
  fprintf(stderr,"%s","                 1 float value between 0 and oo\n");
  fprintf(stderr,"%s","                 default: `2000.0'\n");
  fprintf(stderr,"%s","           -rlo: The low Fourier frequency to check\n");
  fprintf(stderr,"%s","                 1 int value between 0 and oo\n");
  fprintf(stderr,"%s","           -rhi: The high Fourier frequency to check\n");
  fprintf(stderr,"%s","                 1 int value between 0 and oo\n");
  fprintf(stderr,"%s","         -lobin: The first Fourier frequency in the data file\n");
  fprintf(stderr,"%s","                 1 int value between 0 and oo\n");
  fprintf(stderr,"%s","                 default: `0'\n");
  fprintf(stderr,"%s","       -overlap: Fraction of a short FFT length to shift before performing another\n");
  fprintf(stderr,"%s","                 1 double value between 0.05 and 1.0\n");
  fprintf(stderr,"%s","                 default: `0.25'\n");
  fprintf(stderr,"%s","       -harmsum: Number of harmonics to sum in the miniFFTs\n");
  fprintf(stderr,"%s","                 1 int value between 1 and 20\n");
  fprintf(stderr,"%s","                 default: `3'\n");
  fprintf(stderr,"%s","    -numbetween: Number of points to interpolate per Fourier bin (2 gives the usual bin value and an interbin)\n");
  fprintf(stderr,"%s","                 1 int value between 1 and 16\n");
  fprintf(stderr,"%s","                 default: `2'\n");
  fprintf(stderr,"%s","         -stack: Number of stacked power spectra making up the data.  (The default means the data are complex amplitudes)\n");
  fprintf(stderr,"%s","                 1 int value between 0 and oo\n");
  fprintf(stderr,"%s","                 default: `0'\n");
  fprintf(stderr,"%s","      -interbin: Use interbinning instead of full-blown Fourier interpolation.  (Faster but less accurate and sensitive)\n");
  fprintf(stderr,"%s","       -noalias: Do not add aliased powers to the harmonic sum.  (Faster but less accurate and sensitive)\n");
  fprintf(stderr,"%s","         infile: Input file name (no suffix) of floating point fft data.  A '.inf' file of the same name must also exist\n");
  fprintf(stderr,"%s","                 1 value\n");
  fprintf(stderr,"%s","  version: 05Mar13\n");
  fprintf(stderr,"%s","  ");
  exit(EXIT_FAILURE);
}
/**********************************************************************/
Cmdline *
parseCmdline(int argc, char **argv)
{
  int i;

  Program = argv[0];
  cmd.full_cmd_line = catArgv(argc, argv);
  for(i=1, cmd.argc=1; i<argc; i++) {
    if( 0==strcmp("--", argv[i]) ) {
      while( ++i<argc ) argv[cmd.argc++] = argv[i];
      continue;
    }

    if( 0==strcmp("-ncand", argv[i]) ) {
      int keep = i;
      cmd.ncandP = 1;
      i = getIntOpt(argc, argv, i, &cmd.ncand, 1);
      cmd.ncandC = i-keep;
      checkIntLower("-ncand", &cmd.ncand, cmd.ncandC, 10000);
      checkIntHigher("-ncand", &cmd.ncand, cmd.ncandC, 1);
      continue;
    }

    if( 0==strcmp("-minfft", argv[i]) ) {
      int keep = i;
      cmd.minfftP = 1;
      i = getIntOpt(argc, argv, i, &cmd.minfft, 1);
      cmd.minfftC = i-keep;
      checkIntLower("-minfft", &cmd.minfft, cmd.minfftC, 1048576);
      checkIntHigher("-minfft", &cmd.minfft, cmd.minfftC, 8);
      continue;
    }

    if( 0==strcmp("-maxfft", argv[i]) ) {
      int keep = i;
      cmd.maxfftP = 1;
      i = getIntOpt(argc, argv, i, &cmd.maxfft, 1);
      cmd.maxfftC = i-keep;
      checkIntLower("-maxfft", &cmd.maxfft, cmd.maxfftC, 1048576);
      checkIntHigher("-maxfft", &cmd.maxfft, cmd.maxfftC, 8);
      continue;
    }

    if( 0==strcmp("-flo", argv[i]) ) {
      int keep = i;
      cmd.floP = 1;
      i = getFloatOpt(argc, argv, i, &cmd.flo, 1);
      cmd.floC = i-keep;
      checkFloatHigher("-flo", &cmd.flo, cmd.floC, 0);
      continue;
    }

    if( 0==strcmp("-fhi", argv[i]) ) {
      int keep = i;
      cmd.fhiP = 1;
      i = getFloatOpt(argc, argv, i, &cmd.fhi, 1);
      cmd.fhiC = i-keep;
      checkFloatHigher("-fhi", &cmd.fhi, cmd.fhiC, 0);
      continue;
    }

    if( 0==strcmp("-rlo", argv[i]) ) {
      int keep = i;
      cmd.rloP = 1;
      i = getIntOpt(argc, argv, i, &cmd.rlo, 1);
      cmd.rloC = i-keep;
      checkIntHigher("-rlo", &cmd.rlo, cmd.rloC, 0);
      continue;
    }

    if( 0==strcmp("-rhi", argv[i]) ) {
      int keep = i;
      cmd.rhiP = 1;
      i = getIntOpt(argc, argv, i, &cmd.rhi, 1);
      cmd.rhiC = i-keep;
      checkIntHigher("-rhi", &cmd.rhi, cmd.rhiC, 0);
      continue;
    }

    if( 0==strcmp("-lobin", argv[i]) ) {
      int keep = i;
      cmd.lobinP = 1;
      i = getIntOpt(argc, argv, i, &cmd.lobin, 1);
      cmd.lobinC = i-keep;
      checkIntHigher("-lobin", &cmd.lobin, cmd.lobinC, 0);
      continue;
    }

    if( 0==strcmp("-overlap", argv[i]) ) {
      int keep = i;
      cmd.overlapP = 1;
      i = getDoubleOpt(argc, argv, i, &cmd.overlap, 1);
      cmd.overlapC = i-keep;
      checkDoubleLower("-overlap", &cmd.overlap, cmd.overlapC, 1.0);
      checkDoubleHigher("-overlap", &cmd.overlap, cmd.overlapC, 0.05);
      continue;
    }

    if( 0==strcmp("-harmsum", argv[i]) ) {
      int keep = i;
      cmd.harmsumP = 1;
      i = getIntOpt(argc, argv, i, &cmd.harmsum, 1);
      cmd.harmsumC = i-keep;
      checkIntLower("-harmsum", &cmd.harmsum, cmd.harmsumC, 20);
      checkIntHigher("-harmsum", &cmd.harmsum, cmd.harmsumC, 1);
      continue;
    }

    if( 0==strcmp("-numbetween", argv[i]) ) {
      int keep = i;
      cmd.numbetweenP = 1;
      i = getIntOpt(argc, argv, i, &cmd.numbetween, 1);
      cmd.numbetweenC = i-keep;
      checkIntLower("-numbetween", &cmd.numbetween, cmd.numbetweenC, 16);
      checkIntHigher("-numbetween", &cmd.numbetween, cmd.numbetweenC, 1);
      continue;
    }

    if( 0==strcmp("-stack", argv[i]) ) {
      int keep = i;
      cmd.stackP = 1;
      i = getIntOpt(argc, argv, i, &cmd.stack, 1);
      cmd.stackC = i-keep;
      checkIntHigher("-stack", &cmd.stack, cmd.stackC, 0);
      continue;
    }

    if( 0==strcmp("-interbin", argv[i]) ) {
      cmd.interbinP = 1;
      continue;
    }

    if( 0==strcmp("-noalias", argv[i]) ) {
      cmd.noaliasP = 1;
      continue;
    }

    if( argv[i][0]=='-' ) {
      fprintf(stderr, "\n%s: unknown option `%s'\n\n",
              Program, argv[i]);
      usage();
    }
    argv[cmd.argc++] = argv[i];
  }/* for i */


  /*@-mustfree*/
  cmd.argv = argv+1;
  /*@=mustfree*/
  cmd.argc -= 1;

  if( 1>cmd.argc ) {
    fprintf(stderr, "%s: there should be at least 1 non-option argument(s)\n",
            Program);
    exit(EXIT_FAILURE);
  }
  if( 1<cmd.argc ) {
    fprintf(stderr, "%s: there should be at most 1 non-option argument(s)\n",
            Program);
    exit(EXIT_FAILURE);
  }
  /*@-compmempass*/  return &cmd;
}

