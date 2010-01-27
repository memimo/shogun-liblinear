/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 *  Implementation of SVM-Ocas solver.
 *
 *  Linear unbiased binary SVM solver.
 *
 * Written (W) 1999-2009 Vojtech Franc
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 *
 * Modifications:
 * 23-oct-2007, VF
 * 10-oct-2007, VF, created.
 * 14-nov-2007, VF, updates
 * ----------------------------------------------------------------------*/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>

#include "classifier/svm/libocas.h"
#include "classifier/svm/libocas_common.h"
#include "classifier/svm/qpssvmlib.h"

namespace shogun
{

static const uint32_t QPSolverMaxIter = 10000000;

static float64_t *H;
static uint32_t BufSize;

/*----------------------------------------------------------------------
 Returns pointer at i-th column of Hessian matrix.
  ----------------------------------------------------------------------*/
static const void *get_col( uint32_t i)
{
  return( &H[ BufSize*i ] );
} 

/*----------------------------------------------------------------------
  Returns time of the day in seconds. 
  ----------------------------------------------------------------------*/
static float64_t get_time()
{
	struct timeval tv;
	if (gettimeofday(&tv, NULL)==0)
		return (float64_t) (tv.tv_sec+((double)(tv.tv_usec))/1e6);
	else
		return 0.0;
}

/*----------------------------------------------------------------------
  SVM-Ocas solver.
  ----------------------------------------------------------------------*/
ocas_return_value_T svm_ocas_solver(
            float64_t C,
            uint32_t nData, 
            float64_t TolRel,
            float64_t TolAbs,
            float64_t QPBound,
            uint32_t _BufSize,
            uint8_t Method,
            void (*compute_W)(float64_t*, float64_t*, float64_t*, uint32_t, void*),
            float64_t (*update_W)(float64_t, void*),
            void (*add_new_cut)(float64_t*, uint32_t*, uint32_t, uint32_t, void*),
            void (*compute_output)(float64_t*, void* ),
            void (*sort)(float64_t*, uint32_t*, uint32_t),
			void* user_data) 
{
  ocas_return_value_T ocas;
  float64_t *b, *alpha, *diag_H;
  float64_t *output, *old_output;
  float64_t xi, sq_norm_W, QPSolverTolRel, dot_prod_WoldW, dummy, sq_norm_oldW;
  float64_t A0, B0, GradVal, t, t1=0, t2=0, *Ci, *Bi, *hpf;
  float64_t start_time;
  uint32_t *hpi;
  uint32_t cut_length;
  uint32_t i, *new_cut;
  uint16_t *I;
  int8_t qp_exitflag;
  float64_t gap;

  ocas.ocas_time = get_time();
  ocas.solver_time = 0;
  ocas.output_time = 0;
  ocas.sort_time = 0;
  ocas.add_time = 0;
  ocas.w_time = 0;

  BufSize = _BufSize;

  QPSolverTolRel = TolRel*0.5;

  H=NULL;
  b=NULL;
  alpha=NULL;
  new_cut=NULL;
  I=NULL;
  diag_H=NULL;
  output=NULL;
  old_output=NULL;
  hpf=NULL;
  hpi=NULL;
  Ci=NULL;
  Bi=NULL;

  /* Hessian matrix contains dot product of normal vectors of selected cutting planes */
  H = (float64_t*)OCAS_CALLOC(BufSize*BufSize,sizeof(float64_t));
  if(H == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }
  
  /* bias of cutting planes */
  b = (float64_t*)OCAS_CALLOC(BufSize,sizeof(float64_t));
  if(b == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  alpha = (float64_t*)OCAS_CALLOC(BufSize,sizeof(float64_t));
  if(alpha == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* indices of examples which define a new cut */
  new_cut = (uint32_t*)OCAS_CALLOC(nData,sizeof(uint32_t));
  if(new_cut == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  I = (uint16_t*)OCAS_CALLOC(BufSize,sizeof(uint16_t));
  if(I == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  for(i=0; i< BufSize; i++) I[i] = 1;

  diag_H = (float64_t*)OCAS_CALLOC(BufSize,sizeof(float64_t));
  if(diag_H == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  output = (float64_t*)OCAS_CALLOC(nData,sizeof(float64_t));
  if(output == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  old_output = (float64_t*)OCAS_CALLOC(nData,sizeof(float64_t));
  if(old_output == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* array of hinge points used in line-serach  */
  hpf = (float64_t*) OCAS_CALLOC(nData, sizeof(hpf[0]));
  if(hpf == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  hpi = (uint32_t*) OCAS_CALLOC(nData, sizeof(hpi[0]));
  if(hpi == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  /* vectors Ci, Bi are used in the line search procedure */
  Ci = (float64_t*)OCAS_CALLOC(nData,sizeof(float64_t));
  if(Ci == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }

  Bi = (float64_t*)OCAS_CALLOC(nData,sizeof(float64_t));
  if(Bi == NULL)
  {
	  ocas.exitflag=-2;
	  goto cleanup;
  }
  
  ocas.nCutPlanes = 0;
  ocas.exitflag = 0;
  ocas.nIter = 0;

  /* Compute initial value of Q_P assuming that W is zero vector.*/
  sq_norm_W = 0;
  xi = nData;
  ocas.Q_P = 0.5*sq_norm_W + C*xi;
  ocas.Q_D = 0;

  /* Compute the initial cutting plane */
  cut_length = nData;
  for(i=0; i < nData; i++)
    new_cut[i] = i;
    
  gap=(ocas.Q_P-ocas.Q_D)/CMath::abs(ocas.Q_P);
  SG_SABS_PROGRESS(gap, -CMath::log10(gap), -CMath::log10(1), -CMath::log10(TolRel), 6);

  /* main loop */
  while( ocas.exitflag == 0 )
  {
    ocas.nIter++;

    /* append a new cut to the buffer and update H */
    b[ocas.nCutPlanes] = -(float64_t)cut_length;

    start_time = get_time();

    add_new_cut( &H[INDEX2(0,ocas.nCutPlanes,BufSize)], new_cut, cut_length, ocas.nCutPlanes, user_data );

    ocas.add_time += get_time() - start_time;

    /* copy new added row:  H(ocas.nCutPlanes,ocas.nCutPlanes,1:ocas.nCutPlanes-1) = H(1:ocas.nCutPlanes-1:ocas.nCutPlanes)' */
    diag_H[ocas.nCutPlanes] = H[INDEX2(ocas.nCutPlanes,ocas.nCutPlanes,BufSize)];
    for(i=0; i < ocas.nCutPlanes; i++) {
      H[INDEX2(ocas.nCutPlanes,i,BufSize)] = H[INDEX2(i,ocas.nCutPlanes,BufSize)];
    }

    ocas.nCutPlanes++;    
    
    /* call inner QP solver */
    start_time = get_time();

    qp_exitflag = qpssvm_solver( &get_col, diag_H, b, C, I, alpha, 
                ocas.nCutPlanes, QPSolverMaxIter, 0.0, QPSolverTolRel, &ocas.Q_D, &dummy, 0 ); 

    ocas.solver_time += get_time() - start_time;

    ocas.Q_D = -ocas.Q_D;

    ocas.nNZAlpha = 0;
    for(i=0; i < ocas.nCutPlanes; i++) {
      if( alpha[i] != 0) ocas.nNZAlpha++;
    }

    sq_norm_oldW = sq_norm_W;
    start_time = get_time();
    compute_W( &sq_norm_W, &dot_prod_WoldW, alpha, ocas.nCutPlanes, user_data );
    ocas.w_time += get_time() - start_time;
    
    /* select a new cut */
    switch( Method )
    {
      /* cutting plane algorithm implemented in SVMperf and BMRM */
      case 0: 

        start_time = get_time();
        compute_output( output, user_data );
        ocas.output_time += get_time()-start_time;

        xi = 0;
        cut_length = 0;
        ocas.trn_err = 0;
        for(i=0; i < nData; i++)
        { 
          if(output[i] <= 0) ocas.trn_err++;
          
          if(output[i] <= 1) {
            xi += 1 - output[i];
            new_cut[cut_length] = i; 
            cut_length++;
          }
        }
        ocas.Q_P = 0.5*sq_norm_W + C*xi;

        gap=(ocas.Q_P-ocas.Q_D)/CMath::abs(ocas.Q_P);
        SG_SABS_PROGRESS(gap, -CMath::log10(gap), -CMath::log10(1), -CMath::log10(TolRel), 6);

        break;


      /* Ocas strategy */
      case 1:

        /* Linesearch */
        A0 = sq_norm_W -2*dot_prod_WoldW + sq_norm_oldW;
        B0 = dot_prod_WoldW - sq_norm_oldW;

        memcpy( old_output, output, sizeof(float64_t)*nData );

        start_time = get_time();
        compute_output( output, user_data );
        ocas.output_time += get_time()-start_time;

        uint32_t num_hp = 0;
        GradVal = B0;
        for(i=0; i< nData; i++) {

          Ci[i] = C*(1-old_output[i]);
          Bi[i] = C*(old_output[i] - output[i]);

          float64_t val;
          if(Bi[i] != 0)
            val = -Ci[i]/Bi[i];
          else
            val = -OCAS_PLUS_INF;
          
          if (val>0)
          {
            hpi[num_hp] = i;
            hpf[num_hp] = val;
            num_hp++;
          }

          if( (Bi[i] < 0 && val > 0) || (Bi[i] > 0 && val <= 0)) 
            GradVal += Bi[i];
          
        }

        t = 0;
        if( GradVal < 0 )
        {
        start_time = get_time();
        sort(hpf, hpi, num_hp);
        ocas.sort_time += get_time() - start_time;

          float64_t t_new, GradVal_new;
          i = 0;
          while( GradVal < 0 && i < num_hp )
          {
            t_new = hpf[i];
            GradVal_new = GradVal + CMath::abs(Bi[hpi[i]]) + A0*(t_new-t);

            if( GradVal_new >= 0 )
            {
              t = t + GradVal*(t-t_new)/(GradVal_new - GradVal);
            }
            else
            {
              t = t_new;
              i++;
            }

            GradVal = GradVal_new;
          }
        }

        /*
        t = hpf[0] - 1;
        i = 0;
        GradVal = t*A0 + Bsum;
        while( GradVal < 0 && i < num_hp && hpf[i] < OCAS_PLUS_INF ) {
          t = hpf[i];
          Bsum = Bsum + CMath::abs(Bi[hpi[i]]);
          GradVal = t*A0 + Bsum;
          i++;
        }
        */
        t = CMath::max(t,0.0);          /* just sanity check; t < 0 should not ocure */

        t1 = t;                /* new (best so far) W */
        t2 = t+(1.0-t)/10.0;   /* new cutting plane */

        /* update W to be the best so far solution */
        sq_norm_W = update_W( t1, user_data );

        /* select a new cut */
        xi = 0;
        cut_length = 0;
        ocas.trn_err = 0;
        for(i=0; i < nData; i++ ) {

          if( (old_output[i]*(1-t2) + t2*output[i]) <= 1 ) 
          {
            new_cut[cut_length] = i; 
            cut_length++;
          }

          output[i] = old_output[i]*(1-t1) + t1*output[i];

          if( output[i] <= 1) xi += 1-output[i];
          if( output[i] <= 0) ocas.trn_err++;

        }

        ocas.Q_P = 0.5*sq_norm_W + C*xi;

        gap=(ocas.Q_P-ocas.Q_D)/CMath::abs(ocas.Q_P);
        SG_SABS_PROGRESS(gap, -CMath::log10(gap), -CMath::log10(1), -CMath::log10(TolRel), 6);

        break;
    }

    /* Stopping conditions */
    if( ocas.Q_P - ocas.Q_D <= TolRel*CMath::abs(ocas.Q_P)) ocas.exitflag = 1; 
    if( ocas.Q_P - ocas.Q_D <= TolAbs) ocas.exitflag = 2; 
    if( ocas.Q_P <= QPBound) ocas.exitflag = 3; 
    if(ocas.nCutPlanes >= BufSize) ocas.exitflag = -1;
         
  } /* end of the main loop */

cleanup:

  OCAS_FREE(H);
  OCAS_FREE(b);
  OCAS_FREE(alpha);
  OCAS_FREE(new_cut);
  OCAS_FREE(I);
  OCAS_FREE(diag_H);
  OCAS_FREE(output);
  OCAS_FREE(old_output);
  OCAS_FREE(hpf);
  OCAS_FREE(hpi);
  OCAS_FREE(Ci);
  OCAS_FREE(Bi);

  ocas.ocas_time = get_time() - ocas.ocas_time;

  return(ocas);
}
}
