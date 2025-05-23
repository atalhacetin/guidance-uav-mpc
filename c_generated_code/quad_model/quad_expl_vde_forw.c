/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) quad_expl_vde_forw_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_project CASADI_PREFIX(project)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s12 CASADI_PREFIX(s12)
#define casadi_s13 CASADI_PREFIX(s13)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
#define casadi_sparsify CASADI_PREFIX(sparsify)
#define casadi_trans CASADI_PREFIX(trans)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

#define CASADI_CAST(x,y) ((x) y)

void casadi_sparsify(const casadi_real* x, casadi_real* y, const casadi_int* sp_y, casadi_int tr) {
  casadi_int nrow_y, ncol_y, i, el;
  const casadi_int *colind_y, *row_y;
  nrow_y = sp_y[0];
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y+ncol_y+3;
  if (tr) {
    for (i=0; i<ncol_y; ++i) {
      for (el=colind_y[i]; el!=colind_y[i+1]; ++el) {
        *y++ = CASADI_CAST(casadi_real, x[i + row_y[el]*ncol_y]);
      }
    }
  } else {
    for (i=0; i<ncol_y; ++i) {
      for (el=colind_y[i]; el!=colind_y[i+1]; ++el) {
        *y++ = CASADI_CAST(casadi_real, x[row_y[el]]);
      }
      x += nrow_y;
    }
  }
}

void casadi_trans(const casadi_real* x, const casadi_int* sp_x, casadi_real* y,
    const casadi_int* sp_y, casadi_int* tmp) {
  casadi_int ncol_x, nnz_x, ncol_y, k;
  const casadi_int* row_x, *colind_y;
  ncol_x = sp_x[1];
  nnz_x = sp_x[2 + ncol_x];
  row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}

void casadi_project(const casadi_real* x, const casadi_int* sp_x, casadi_real* y, const casadi_int* sp_y, casadi_real* w) {
  casadi_int ncol_x, ncol_y, i, el;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  for (i=0; i<ncol_x; ++i) {
    for (el=colind_y[i]; el<colind_y[i+1]; ++el) w[row_y[el]] = 0;
    for (el=colind_x[i]; el<colind_x[i+1]; ++el) w[row_x[el]] = x[el];
    for (el=colind_y[i]; el<colind_y[i+1]; ++el) y[el] = w[row_y[el]];
  }
}

static const casadi_int casadi_s0[5] = {3, 1, 0, 1, 0};
static const casadi_int casadi_s1[4] = {0, 5, 7, 9};
static const casadi_int casadi_s2[5] = {3, 1, 0, 1, 1};
static const casadi_int casadi_s3[5] = {3, 1, 0, 1, 2};
static const casadi_int casadi_s4[4] = {2, 4, 6, 11};
static const casadi_int casadi_s5[18] = {11, 3, 0, 4, 8, 12, 3, 7, 8, 9, 4, 6, 8, 9, 5, 6, 7, 9};
static const casadi_int casadi_s6[26] = {3, 11, 0, 0, 0, 0, 1, 2, 3, 5, 7, 9, 12, 12, 0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 2};
static const casadi_int casadi_s7[30] = {11, 3, 0, 8, 16, 24, 0, 1, 2, 3, 7, 8, 9, 10, 0, 1, 2, 4, 6, 8, 9, 10, 0, 1, 2, 5, 6, 7, 9, 10};
static const casadi_int casadi_s8[18] = {11, 3, 0, 4, 8, 12, 0, 1, 2, 10, 0, 1, 2, 10, 0, 1, 2, 10};
static const casadi_int casadi_s9[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s10[135] = {11, 11, 0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s11[39] = {11, 3, 0, 11, 22, 33, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s12[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s13[58] = {11, 11, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 0, 1, 2, 10, 0, 1, 2, 10, 0, 1, 2, 10, 0, 1, 2, 10, 0, 1, 2, 10, 0, 1, 2, 10, 0, 1, 2, 10, 0, 1, 2, 10, 0, 1, 2, 10, 0, 1, 2, 10, 0, 1, 2, 10};

/* quad_expl_vde_forw:(i0[11],i1[11x11],i2[11x3],i3[3],i4[3])->(o0[11],o1[11x11,44nz],o2[11x3,24nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real *rr, *ss, *tt;
  const casadi_int *cii;
  const casadi_real *cs;
  casadi_real *w0=w+11, w1, w2, w3, w4, w5, w6, w7, w8, *w9=w+22, *w10=w+143, *w11=w+154, *w12=w+165, *w13=w+176, *w14=w+187, *w15=w+198, *w16=w+209, *w17=w+220, *w18=w+231, *w19=w+242, *w20=w+253, *w21=w+264, *w22=w+267, w27, w28, *w30=w+281, *w31=w+285, *w32=w+289, *w33=w+301, *w34=w+325, *w37=w+358, *w38=w+362;
  /* #0: @0 = input[0][1] */
  casadi_copy(arg[0] ? arg[0]+3 : 0, 3, w0);
  /* #1: output[0][0] = @0 */
  casadi_copy(w0, 3, res[0]);
  /* #2: @1 = input[3][0] */
  w1 = arg[3] ? arg[3][0] : 0;
  /* #3: output[0][1] = @1 */
  if (res[0]) res[0][3] = w1;
  /* #4: @2 = input[3][1] */
  w2 = arg[3] ? arg[3][1] : 0;
  /* #5: output[0][2] = @2 */
  if (res[0]) res[0][4] = w2;
  /* #6: @3 = input[3][2] */
  w3 = arg[3] ? arg[3][2] : 0;
  /* #7: output[0][3] = @3 */
  if (res[0]) res[0][5] = w3;
  /* #8: @0 = input[4][0] */
  casadi_copy(arg[4], 3, w0);
  /* #9: @4 = @0[2] */
  for (rr=(&w4), ss=w0+2; ss!=w0+3; ss+=1) *rr++ = *ss;
  /* #10: @5 = (@2*@4) */
  w5  = (w2*w4);
  /* #11: @6 = @0[1] */
  for (rr=(&w6), ss=w0+1; ss!=w0+2; ss+=1) *rr++ = *ss;
  /* #12: @7 = (@3*@6) */
  w7  = (w3*w6);
  /* #13: @5 = (@5-@7) */
  w5 -= w7;
  /* #14: output[0][4] = @5 */
  if (res[0]) res[0][6] = w5;
  /* #15: @5 = @0[0] */
  for (rr=(&w5), ss=w0+0; ss!=w0+1; ss+=1) *rr++ = *ss;
  /* #16: @7 = (@3*@5) */
  w7  = (w3*w5);
  /* #17: @8 = (@1*@4) */
  w8  = (w1*w4);
  /* #18: @7 = (@7-@8) */
  w7 -= w8;
  /* #19: output[0][5] = @7 */
  if (res[0]) res[0][7] = w7;
  /* #20: @7 = (@1*@6) */
  w7  = (w1*w6);
  /* #21: @8 = (@2*@5) */
  w8  = (w2*w5);
  /* #22: @7 = (@7-@8) */
  w7 -= w8;
  /* #23: output[0][6] = @7 */
  if (res[0]) res[0][8] = w7;
  /* #24: @0 = vertcat(@1, @2, @3) */
  rr=w0;
  *rr++ = w1;
  *rr++ = w2;
  *rr++ = w3;
  /* #25: @1 = dot(@0, @0) */
  w1 = casadi_dot(3, w0, w0);
  /* #26: output[0][7] = @1 */
  if (res[0]) res[0][9] = w1;
  /* #27: @1 = input[0][3] */
  w1 = arg[0] ? arg[0][9] : 0;
  /* #28: output[0][8] = @1 */
  if (res[0]) res[0][10] = w1;
  /* #29: @9 = input[1][0] */
  casadi_copy(arg[1], 121, w9);
  /* #30: {@10, @11, @12, @13, @14, @15, @16, @17, @18, @19, @20} = horzsplit(@9) */
  casadi_copy(w9, 11, w10);
  casadi_copy(w9+11, 11, w11);
  casadi_copy(w9+22, 11, w12);
  casadi_copy(w9+33, 11, w13);
  casadi_copy(w9+44, 11, w14);
  casadi_copy(w9+55, 11, w15);
  casadi_copy(w9+66, 11, w16);
  casadi_copy(w9+77, 11, w17);
  casadi_copy(w9+88, 11, w18);
  casadi_copy(w9+99, 11, w19);
  casadi_copy(w9+110, 11, w20);
  /* #31: {NULL, @21, NULL, @1, NULL} = vertsplit(@10) */
  casadi_copy(w10+3, 3, w21);
  w1 = w10[9];
  /* #32: output[1][0] = @21 */
  casadi_copy(w21, 3, res[1]);
  /* #33: output[1][1] = @1 */
  if (res[1]) res[1][3] = w1;
  /* #34: {NULL, @21, NULL, @1, NULL} = vertsplit(@11) */
  casadi_copy(w11+3, 3, w21);
  w1 = w11[9];
  /* #35: output[1][2] = @21 */
  if (res[1]) casadi_copy(w21, 3, res[1]+4);
  /* #36: output[1][3] = @1 */
  if (res[1]) res[1][7] = w1;
  /* #37: {NULL, @21, NULL, @1, NULL} = vertsplit(@12) */
  casadi_copy(w12+3, 3, w21);
  w1 = w12[9];
  /* #38: output[1][4] = @21 */
  if (res[1]) casadi_copy(w21, 3, res[1]+8);
  /* #39: output[1][5] = @1 */
  if (res[1]) res[1][11] = w1;
  /* #40: {NULL, @21, NULL, @1, NULL} = vertsplit(@13) */
  casadi_copy(w13+3, 3, w21);
  w1 = w13[9];
  /* #41: output[1][6] = @21 */
  if (res[1]) casadi_copy(w21, 3, res[1]+12);
  /* #42: output[1][7] = @1 */
  if (res[1]) res[1][15] = w1;
  /* #43: {NULL, @21, NULL, @1, NULL} = vertsplit(@14) */
  casadi_copy(w14+3, 3, w21);
  w1 = w14[9];
  /* #44: output[1][8] = @21 */
  if (res[1]) casadi_copy(w21, 3, res[1]+16);
  /* #45: output[1][9] = @1 */
  if (res[1]) res[1][19] = w1;
  /* #46: {NULL, @21, NULL, @1, NULL} = vertsplit(@15) */
  casadi_copy(w15+3, 3, w21);
  w1 = w15[9];
  /* #47: output[1][10] = @21 */
  if (res[1]) casadi_copy(w21, 3, res[1]+20);
  /* #48: output[1][11] = @1 */
  if (res[1]) res[1][23] = w1;
  /* #49: {NULL, @21, NULL, @1, NULL} = vertsplit(@16) */
  casadi_copy(w16+3, 3, w21);
  w1 = w16[9];
  /* #50: output[1][12] = @21 */
  if (res[1]) casadi_copy(w21, 3, res[1]+24);
  /* #51: output[1][13] = @1 */
  if (res[1]) res[1][27] = w1;
  /* #52: {NULL, @21, NULL, @1, NULL} = vertsplit(@17) */
  casadi_copy(w17+3, 3, w21);
  w1 = w17[9];
  /* #53: output[1][14] = @21 */
  if (res[1]) casadi_copy(w21, 3, res[1]+28);
  /* #54: output[1][15] = @1 */
  if (res[1]) res[1][31] = w1;
  /* #55: {NULL, @21, NULL, @1, NULL} = vertsplit(@18) */
  casadi_copy(w18+3, 3, w21);
  w1 = w18[9];
  /* #56: output[1][16] = @21 */
  if (res[1]) casadi_copy(w21, 3, res[1]+32);
  /* #57: output[1][17] = @1 */
  if (res[1]) res[1][35] = w1;
  /* #58: {NULL, @21, NULL, @1, NULL} = vertsplit(@19) */
  casadi_copy(w19+3, 3, w21);
  w1 = w19[9];
  /* #59: output[1][18] = @21 */
  if (res[1]) casadi_copy(w21, 3, res[1]+36);
  /* #60: output[1][19] = @1 */
  if (res[1]) res[1][39] = w1;
  /* #61: {NULL, @21, NULL, @1, NULL} = vertsplit(@20) */
  casadi_copy(w20+3, 3, w21);
  w1 = w20[9];
  /* #62: output[1][20] = @21 */
  if (res[1]) casadi_copy(w21, 3, res[1]+40);
  /* #63: output[1][21] = @1 */
  if (res[1]) res[1][43] = w1;
  /* #64: @22 = zeros(3x11,12nz) */
  casadi_clear(w22, 12);
  /* #65: @23 = zeros(3x1,0nz) */
  /* #66: @1 = ones(3x1,1nz) */
  w1 = 1.;
  /* #67: {@2, NULL, NULL} = vertsplit(@1) */
  w2 = w1;
  /* #68: @24 = 00 */
  /* #69: @25 = 00 */
  /* #70: @26 = 00 */
  /* #71: @1 = (@4*@2) */
  w1  = (w4*w2);
  /* #72: @1 = (-@1) */
  w1 = (- w1 );
  /* #73: @3 = (@6*@2) */
  w3  = (w6*w2);
  /* #74: @7 = project(@0) */
  casadi_sparsify(w0, (&w7), casadi_s0, 0);
  /* #75: @8 = vertcat(@2, @24, @25) */
  rr=(&w8);
  *rr++ = w2;
  /* #76: @27 = dot(@7, @8) */
  w27 = casadi_dot(1, (&w7), (&w8));
  /* #77: @7 = project(@0) */
  casadi_sparsify(w0, (&w7), casadi_s0, 0);
  /* #78: @28 = dot(@8, @7) */
  w28 = casadi_dot(1, (&w8), (&w7));
  /* #79: @27 = (@27+@28) */
  w27 += w28;
  /* #80: @29 = 00 */
  /* #81: @30 = vertcat(@23, @2, @24, @25, @26, @1, @3, @27, @29) */
  rr=w30;
  *rr++ = w2;
  *rr++ = w1;
  *rr++ = w3;
  *rr++ = w27;
  /* #82: @31 = @30[:4] */
  for (rr=w31, ss=w30+0; ss!=w30+4; ss+=1) *rr++ = *ss;
  /* #83: (@22[0, 5, 7, 9] = @31) */
  for (cii=casadi_s1, rr=w22, ss=w31; cii!=casadi_s1+4; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #84: @23 = zeros(3x1,0nz) */
  /* #85: @24 = 00 */
  /* #86: @2 = ones(3x1,1nz) */
  w2 = 1.;
  /* #87: {NULL, @1, NULL} = vertsplit(@2) */
  w1 = w2;
  /* #88: @25 = 00 */
  /* #89: @4 = (@4*@1) */
  w4 *= w1;
  /* #90: @26 = 00 */
  /* #91: @2 = (@5*@1) */
  w2  = (w5*w1);
  /* #92: @2 = (-@2) */
  w2 = (- w2 );
  /* #93: @3 = project(@0) */
  casadi_sparsify(w0, (&w3), casadi_s2, 0);
  /* #94: @27 = vertcat(@24, @1, @25) */
  rr=(&w27);
  *rr++ = w1;
  /* #95: @28 = dot(@3, @27) */
  w28 = casadi_dot(1, (&w3), (&w27));
  /* #96: @3 = project(@0) */
  casadi_sparsify(w0, (&w3), casadi_s2, 0);
  /* #97: @8 = dot(@27, @3) */
  w8 = casadi_dot(1, (&w27), (&w3));
  /* #98: @28 = (@28+@8) */
  w28 += w8;
  /* #99: @29 = 00 */
  /* #100: @31 = vertcat(@23, @24, @1, @25, @4, @26, @2, @28, @29) */
  rr=w31;
  *rr++ = w1;
  *rr++ = w4;
  *rr++ = w2;
  *rr++ = w28;
  /* #101: @30 = @31[:4] */
  for (rr=w30, ss=w31+0; ss!=w31+4; ss+=1) *rr++ = *ss;
  /* #102: (@22[:14:7;1:5:2] = @30) */
  for (rr=w22+0, ss=w30; rr!=w22+14; rr+=7) for (tt=rr+1; tt!=rr+5; tt+=2) *tt = *ss++;
  /* #103: @23 = zeros(3x1,0nz) */
  /* #104: @24 = 00 */
  /* #105: @25 = 00 */
  /* #106: @1 = ones(3x1,1nz) */
  w1 = 1.;
  /* #107: {NULL, NULL, @4} = vertsplit(@1) */
  w4 = w1;
  /* #108: @6 = (@6*@4) */
  w6 *= w4;
  /* #109: @6 = (-@6) */
  w6 = (- w6 );
  /* #110: @5 = (@5*@4) */
  w5 *= w4;
  /* #111: @26 = 00 */
  /* #112: @1 = project(@0) */
  casadi_sparsify(w0, (&w1), casadi_s3, 0);
  /* #113: @2 = vertcat(@24, @25, @4) */
  rr=(&w2);
  *rr++ = w4;
  /* #114: @28 = dot(@1, @2) */
  w28 = casadi_dot(1, (&w1), (&w2));
  /* #115: @1 = project(@0) */
  casadi_sparsify(w0, (&w1), casadi_s3, 0);
  /* #116: @8 = dot(@2, @1) */
  w8 = casadi_dot(1, (&w2), (&w1));
  /* #117: @28 = (@28+@8) */
  w28 += w8;
  /* #118: @29 = 00 */
  /* #119: @30 = vertcat(@23, @24, @25, @4, @6, @5, @26, @28, @29) */
  rr=w30;
  *rr++ = w4;
  *rr++ = w6;
  *rr++ = w5;
  *rr++ = w28;
  /* #120: @31 = @30[:4] */
  for (rr=w31, ss=w30+0; ss!=w30+4; ss+=1) *rr++ = *ss;
  /* #121: (@22[2, 4, 6, 11] = @31) */
  for (cii=casadi_s4, rr=w22, ss=w31; cii!=casadi_s4+4; ++cii, ++ss) if (*cii>=0) rr[*cii] = *ss;
  /* #122: @32 = @22' */
  casadi_trans(w22,casadi_s6, w32, casadi_s5, iw);
  /* #123: @33 = project(@32) */
  casadi_project(w32, casadi_s5, w33, casadi_s7, w);
  /* #124: @34 = input[2][0] */
  casadi_copy(arg[2], 33, w34);
  /* #125: {@20, @19, @18} = horzsplit(@34) */
  casadi_copy(w34, 11, w20);
  casadi_copy(w34+11, 11, w19);
  casadi_copy(w34+22, 11, w18);
  /* #126: {NULL, @0, NULL, @4, NULL} = vertsplit(@20) */
  casadi_copy(w20+3, 3, w0);
  w4 = w20[9];
  /* #127: @23 = 00 */
  /* #128: @24 = 00 */
  /* #129: @25 = 00 */
  /* #130: @26 = 00 */
  /* #131: @29 = 00 */
  /* #132: @35 = 00 */
  /* #133: @36 = 00 */
  /* #134: @31 = vertcat(@0, @23, @24, @25, @26, @29, @35, @36, @4) */
  rr=w31;
  for (i=0, cs=w0; i<3; ++i) *rr++ = *cs++;
  *rr++ = w4;
  /* #135: {NULL, @0, NULL, @4, NULL} = vertsplit(@19) */
  casadi_copy(w19+3, 3, w0);
  w4 = w19[9];
  /* #136: @23 = 00 */
  /* #137: @24 = 00 */
  /* #138: @25 = 00 */
  /* #139: @26 = 00 */
  /* #140: @29 = 00 */
  /* #141: @35 = 00 */
  /* #142: @36 = 00 */
  /* #143: @30 = vertcat(@0, @23, @24, @25, @26, @29, @35, @36, @4) */
  rr=w30;
  for (i=0, cs=w0; i<3; ++i) *rr++ = *cs++;
  *rr++ = w4;
  /* #144: {NULL, @0, NULL, @4, NULL} = vertsplit(@18) */
  casadi_copy(w18+3, 3, w0);
  w4 = w18[9];
  /* #145: @23 = 00 */
  /* #146: @24 = 00 */
  /* #147: @25 = 00 */
  /* #148: @26 = 00 */
  /* #149: @29 = 00 */
  /* #150: @35 = 00 */
  /* #151: @36 = 00 */
  /* #152: @37 = vertcat(@0, @23, @24, @25, @26, @29, @35, @36, @4) */
  rr=w37;
  for (i=0, cs=w0; i<3; ++i) *rr++ = *cs++;
  *rr++ = w4;
  /* #153: @32 = horzcat(@31, @30, @37) */
  rr=w32;
  for (i=0, cs=w31; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w30; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w37; i<4; ++i) *rr++ = *cs++;
  /* #154: @38 = project(@32) */
  casadi_project(w32, casadi_s8, w38, casadi_s7, w);
  /* #155: @33 = (@33+@38) */
  for (i=0, rr=w33, cs=w38; i<24; ++i) (*rr++) += (*cs++);
  /* #156: output[2][0] = @33 */
  casadi_copy(w33, 24, res[2]);
  return 0;
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_forw(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_forw_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_forw_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quad_expl_vde_forw_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_forw_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quad_expl_vde_forw_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quad_expl_vde_forw_incref(void) {
}

CASADI_SYMBOL_EXPORT void quad_expl_vde_forw_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quad_expl_vde_forw_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int quad_expl_vde_forw_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real quad_expl_vde_forw_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quad_expl_vde_forw_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quad_expl_vde_forw_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quad_expl_vde_forw_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s9;
    case 1: return casadi_s10;
    case 2: return casadi_s11;
    case 3: return casadi_s12;
    case 4: return casadi_s12;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quad_expl_vde_forw_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s9;
    case 1: return casadi_s13;
    case 2: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_forw_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 14;
  if (sz_res) *sz_res = 14;
  if (sz_iw) *sz_iw = 4;
  if (sz_w) *sz_w = 386;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
