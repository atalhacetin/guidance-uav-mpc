/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#define S_FUNCTION_NAME acados_solver_sfunction_quad
#define S_FUNCTION_LEVEL 2

#define MDL_START

// acados
// #include "acados/utils/print.h"
#include "acados_c/sim_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "quad_model/quad_model.h"
#include "acados_solver_quad.h"

#include "simstruc.h"

#define SAMPLINGTIME 0.1

static void mdlInitializeSizes (SimStruct *S)
{
    // specify the number of continuous and discrete states
    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 0);// specify the number of input ports
    if ( !ssSetNumInputPorts(S, 10) )
        return;

    // specify the number of output ports
    if ( !ssSetNumOutputPorts(S, 6) )
        return;

    // specify dimension information for the input ports
    // lbx_0
    ssSetInputPortVectorDimension(S, 0, 11);
    // ubx_0
    ssSetInputPortVectorDimension(S, 1, 11);
    // parameters
    ssSetInputPortVectorDimension(S, 2, (10+1) * 3);
    // y_ref_0
    ssSetInputPortVectorDimension(S, 3, 14);
    // y_ref
    ssSetInputPortVectorDimension(S, 4, 126);
    // y_ref_e
    ssSetInputPortVectorDimension(S, 5, 11);
    // lbx_e
    ssSetInputPortVectorDimension(S, 6, 6);
    // ubx_e
    ssSetInputPortVectorDimension(S, 7, 6);
    // lbu
    ssSetInputPortVectorDimension(S, 8, 30);
    // ubu
    ssSetInputPortVectorDimension(S, 9, 30);/* specify dimension information for the OUTPUT ports */
    ssSetOutputPortVectorDimension(S, 0, 3 );
    ssSetOutputPortVectorDimension(S, 1, 1 );
    ssSetOutputPortVectorDimension(S, 2, 1 );
    ssSetOutputPortVectorDimension(S, 3, 11 ); // state at shooting node 1
    ssSetOutputPortVectorDimension(S, 4, 1);
    ssSetOutputPortVectorDimension(S, 5, 1 );

    // specify the direct feedthrough status
    // should be set to 1 for all inputs used in mdlOutputs
    ssSetInputPortDirectFeedThrough(S, 0, 1);
    ssSetInputPortDirectFeedThrough(S, 1, 1);
    ssSetInputPortDirectFeedThrough(S, 2, 1);
    ssSetInputPortDirectFeedThrough(S, 3, 1);
    ssSetInputPortDirectFeedThrough(S, 4, 1);
    ssSetInputPortDirectFeedThrough(S, 5, 1);
    ssSetInputPortDirectFeedThrough(S, 6, 1);
    ssSetInputPortDirectFeedThrough(S, 7, 1);
    ssSetInputPortDirectFeedThrough(S, 8, 1);
    ssSetInputPortDirectFeedThrough(S, 9, 1);

    // one sample time
    ssSetNumSampleTimes(S, 1);
}


#if defined(MATLAB_MEX_FILE)

#define MDL_SET_INPUT_PORT_DIMENSION_INFO
#define MDL_SET_OUTPUT_PORT_DIMENSION_INFO

static void mdlSetInputPortDimensionInfo(SimStruct *S, int_T port, const DimsInfo_T *dimsInfo)
{
    if ( !ssSetInputPortDimensionInfo(S, port, dimsInfo) )
         return;
}

static void mdlSetOutputPortDimensionInfo(SimStruct *S, int_T port, const DimsInfo_T *dimsInfo)
{
    if ( !ssSetOutputPortDimensionInfo(S, port, dimsInfo) )
         return;
}

#endif /* MATLAB_MEX_FILE */


static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, SAMPLINGTIME);
    ssSetOffsetTime(S, 0, 0.0);
}


static void mdlStart(SimStruct *S)
{
    quad_solver_capsule *capsule = quad_acados_create_capsule();
    quad_acados_create(capsule);

    ssSetUserData(S, (void*)capsule);
}


static void mdlOutputs(SimStruct *S, int_T tid)
{
    quad_solver_capsule *capsule = ssGetUserData(S);
    ocp_nlp_config *nlp_config = quad_acados_get_nlp_config(capsule);
    ocp_nlp_dims *nlp_dims = quad_acados_get_nlp_dims(capsule);
    ocp_nlp_in *nlp_in = quad_acados_get_nlp_in(capsule);
    ocp_nlp_out *nlp_out = quad_acados_get_nlp_out(capsule);

    InputRealPtrsType in_sign;      

    // local buffer
    real_t buffer[14];

    /* go through inputs */
    // lbx_0
    in_sign = ssGetInputPortRealSignalPtrs(S, 0);
    for (int i = 0; i < 11; i++)
        buffer[i] = (double)(*in_sign[i]);

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", buffer);
    // ubx_0
    in_sign = ssGetInputPortRealSignalPtrs(S, 1);
    for (int i = 0; i < 11; i++)
        buffer[i] = (double)(*in_sign[i]);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", buffer);
    // parameters - stage-variant !!!
    in_sign = ssGetInputPortRealSignalPtrs(S, 2);

    // update value of parameters
    for (int ii = 0; ii <= 10; ii++)
    {
        for (int jj = 0; jj < 3; jj++)
            buffer[jj] = (double)(*in_sign[ii*3+jj]);
        quad_acados_update_params(capsule, ii, buffer, 3);
    }

  
    // y_ref_0
    in_sign = ssGetInputPortRealSignalPtrs(S, 3);

    for (int i = 0; i < 14; i++)
        buffer[i] = (double)(*in_sign[i]);

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", (void *) buffer);

  
    // y_ref - for stages 1 to N-1
    in_sign = ssGetInputPortRealSignalPtrs(S, 4);

    for (int ii = 1; ii < 10; ii++)
    {
        for (int jj = 0; jj < 14; jj++)
            buffer[jj] = (double)(*in_sign[(ii-1)*14+jj]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, ii, "yref", (void *) buffer);
    }

  
    // y_ref_e
    in_sign = ssGetInputPortRealSignalPtrs(S, 5);

    for (int i = 0; i < 11; i++)
        buffer[i] = (double)(*in_sign[i]);

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 10, "yref", (void *) buffer);
    // lbx_e
    in_sign = ssGetInputPortRealSignalPtrs(S, 6);

    for (int i = 0; i < 6; i++)
        buffer[i] = (double)(*in_sign[i]);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 10, "lbx", buffer);
    // ubx_e
    in_sign = ssGetInputPortRealSignalPtrs(S, 7);

    for (int i = 0; i < 6; i++)
        buffer[i] = (double)(*in_sign[i]);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 10, "ubx", buffer);
    // lbu
    in_sign = ssGetInputPortRealSignalPtrs(S, 8);
    for (int ii = 0; ii < 10; ii++)
    {
        for (int jj = 0; jj < 3; jj++)
            buffer[jj] = (double)(*in_sign[ii*3+jj]);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, ii, "lbu", (void *) buffer);
    }
    // ubu
    in_sign = ssGetInputPortRealSignalPtrs(S, 9);
    for (int ii = 0; ii < 10; ii++)
    {
        for (int jj = 0; jj < 3; jj++)
            buffer[jj] = (double)(*in_sign[ii*3+jj]);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, ii, "ubu", (void *) buffer);
    }

    /* call solver */
    int rti_phase = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "rti_phase", &rti_phase);
    int acados_status = quad_acados_solve(capsule);


    /* set outputs */
    // assign pointers to output signals
    real_t *out_u0, *out_utraj, *out_xtraj, *out_status, *out_sqp_iter, *out_KKT_res, *out_x1, *out_cpu_time, *out_cpu_time_sim, *out_cpu_time_qp, *out_cpu_time_lin;
    int tmp_int;
    out_u0 = ssGetOutputPortRealSignal(S, 0);
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "u", (void *) out_u0);

  
    out_status = ssGetOutputPortRealSignal(S, 1);
    *out_status = (real_t) acados_status;
    out_KKT_res = ssGetOutputPortRealSignal(S, 2);
    *out_KKT_res = (real_t) nlp_out->inf_norm_res;
    out_x1 = ssGetOutputPortRealSignal(S, 3);
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 1, "x", (void *) out_x1);
    out_cpu_time = ssGetOutputPortRealSignal(S, 4);
    // get solution time
    ocp_nlp_get(nlp_config, capsule->nlp_solver, "time_tot", (void *) out_cpu_time);
    out_sqp_iter = ssGetOutputPortRealSignal(S, 5);
    // get sqp iter
    ocp_nlp_get(nlp_config, capsule->nlp_solver, "sqp_iter", (void *) &tmp_int);
    *out_sqp_iter = (real_t) tmp_int;

}

static void mdlTerminate(SimStruct *S)
{
    quad_solver_capsule *capsule = ssGetUserData(S);

    quad_acados_free(capsule);
    quad_acados_free_capsule(capsule);
}


#ifdef  MATLAB_MEX_FILE
#include "simulink.c"
#else
#include "cg_sfun.h"
#endif
