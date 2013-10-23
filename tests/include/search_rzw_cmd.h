#ifndef __search_rzw_cmd__
#define __search_rzw_cmd__
/*****
  command line parser interface -- generated by clig 
  (http://wsd.iitb.fhg.de/~geg/clighome/)

  The command line parser `clig':
  (C) 1995-2004 Harald Kirsch (clig@geggus.net)
*****/

typedef struct s_Cmdline {
  /***** -ncand: Number of candidates to try to return */
  char ncandP;
  int ncand;
  int ncandC;
  /***** -zlo: The low Fourier frequency derivative to search */
  char zloP;
  int zlo;
  int zloC;
  /***** -zhi: The high Fourier frequency derivative to search */
  char zhiP;
  int zhi;
  int zhiC;
  /***** -rlo: The lowest Fourier frequency to search */
  char rloP;
  int rlo;
  int rloC;
  /***** -rhi: The highest Fourier frequency to search */
  char rhiP;
  int rhi;
  int rhiC;
  /***** -flo: The lowest frequency (Hz) to search */
  char floP;
  int flo;
  int floC;
  /***** -fhi: The highest frequency (Hz) to search */
  char fhiP;
  int fhi;
  int fhiC;
  /***** -lobin: The first Fourier frequency in the data file */
  char lobinP;
  int lobin;
  int lobinC;
  /***** -zapfile: A file containing a list of freqs to ignore (i.e. RFI) */
  char zapfileP;
  char* zapfile;
  int zapfileC;
  /***** -baryv: The earth's radial velocity component (v/c) towards the observation (used to convert topocentric birdie freqs to barycentric) */
  char baryvP;
  double baryv;
  int baryvC;
  /***** -photon: Data is poissonian so use freq 0 as power normalization */
  char photonP;
  /***** uninterpreted command line parameters */
  int argc;
  /*@null*/char **argv;
  /***** the whole command line concatenated */
  char *full_cmd_line;
} Cmdline;


extern char *Program;
extern void usage(void);
extern /*@shared*/Cmdline *parseCmdline(int argc, char **argv);

extern void showOptionValues(void);

#endif

