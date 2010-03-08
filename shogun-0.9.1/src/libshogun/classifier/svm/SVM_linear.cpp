#include "lib/config.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#ifdef HAVE_LAPACK
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "classifier/svm/SVM_linear.h"
#include "classifier/svm/Tron.h"
#include "lib/io.h"


typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int32_t  n)
{   
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

using namespace shogun;

l2_lr_fun::l2_lr_fun(const problem *p, float64_t Cp, float64_t Cn)
: function()
{
	int32_t i;
	int32_t l=p->l;
	int32_t *y=p->y;

	this->prob = p;

	z = new float64_t[l];
	D = new float64_t[l];
	C = new float64_t[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

l2_lr_fun::~l2_lr_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
}


float64_t l2_lr_fun::fun(float64_t *w)
{
	int32_t i;
	float64_t f=0;
	int32_t *y=prob->y;
	int32_t l=prob->l;
	int32_t n=prob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
	        float64_t yz = y[i]*z[i];
		if (yz >= 0)
		        f += C[i]*log(1 + exp(-yz));
		else
		        f += C[i]*(-yz+log(1 + exp(yz)));
	}
	f = 2*f;
	for(i=0;i<n;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void l2_lr_fun::grad(float64_t *w, float64_t *g)
{
	int32_t i;
	int32_t *y=prob->y;
	int32_t l=prob->l;
	int32_t n=prob->n;

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<n;i++)
		g[i] = w[i] + g[i];
}

int32_t l2_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2_lr_fun::Hv(float64_t *s, float64_t *Hs)
{
	int32_t i;
	int32_t l=prob->l;
	int32_t n=prob->n;
	float64_t *wa = new float64_t[l];

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2_lr_fun::Xv(float64_t *v, float64_t *res_Xv)
{
	int32_t l=prob->l;
	int32_t n=prob->n;

	if (prob->use_bias)
		n--;

	for (int32_t i=0;i<l;i++)
	{
		res_Xv[i]=prob->x->dense_dot(i, v, n);

		if (prob->use_bias)
			res_Xv[i]+=v[n];
	}
}

void l2_lr_fun::XTv(float64_t *v, float64_t *res_XTv)
{
	int32_t l=prob->l;
	int32_t n=prob->n;

	if (prob->use_bias)
		n--;

	memset(res_XTv, 0, sizeof(float64_t)*prob->n);

	for (int32_t i=0;i<l;i++)
	{
		prob->x->add_to_dense_vec(v[i], i, res_XTv, n);

		if (prob->use_bias)
			res_XTv[n]+=v[i];
	}
}

l2loss_svm_fun::l2loss_svm_fun(const problem *p, float64_t Cp, float64_t Cn)
: function()
{
	int32_t i;
	int32_t l=p->l;
	int32_t *y=p->y;

	this->prob = p;

	z = new float64_t[l];
	D = new float64_t[l];
	C = new float64_t[l];
	I = new int32_t[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

l2loss_svm_fun::~l2loss_svm_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
	delete[] I;
}

float64_t l2loss_svm_fun::fun(float64_t *w)
{
	int32_t i;
	float64_t f=0;
	int32_t *y=prob->y;
	int32_t l=prob->l;
	int32_t n=prob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
	        z[i] = y[i]*z[i];
		float64_t d = z[i]-1;
		if (d < 0)
			f += C[i]*d*d;
	}
	f = 2*f;
	for(i=0;i<n;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void l2loss_svm_fun::grad(float64_t *w, float64_t *g)
{
	int32_t i;
	int32_t *y=prob->y;
	int32_t l=prob->l;
	int32_t n=prob->n;

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<n;i++)
		g[i] = w[i] + 2*g[i];
}

int32_t l2loss_svm_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2loss_svm_fun::Hv(float64_t *s, float64_t *Hs)
{
	int32_t i;
	int32_t l=prob->l;
	int32_t n=prob->n;
	float64_t *wa = new float64_t[l];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2loss_svm_fun::Xv(float64_t *v, float64_t *res_Xv)
{
	int32_t l=prob->l;
	int32_t n=prob->n;

	if (prob->use_bias)
		n--;

	for (int32_t i=0;i<l;i++)
	{
		res_Xv[i]=prob->x->dense_dot(i, v, n);

		if (prob->use_bias)
			res_Xv[i]+=v[n];
	}
}

void l2loss_svm_fun::subXv(float64_t *v, float64_t *res_Xv)
{
	int32_t n=prob->n;

	if (prob->use_bias)
		n--;

	for (int32_t i=0;i<sizeI;i++)
	{
		res_Xv[i]=prob->x->dense_dot(I[i], v, n);

		if (prob->use_bias)
			res_Xv[i]+=v[n];
	}
}

void l2loss_svm_fun::subXTv(float64_t *v, float64_t *XTv)
{
	int32_t n=prob->n;

	if (prob->use_bias)
		n--;

	memset(XTv, 0, sizeof(float64_t)*prob->n);
	for (int32_t i=0;i<sizeI;i++)
	{
		prob->x->add_to_dense_vec(v[i], I[i], XTv, n);
		
		if (prob->use_bias)
			XTv[n]+=v[i];
	}
}

// A coordinate descent algorithm for 
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

l1r_l2_svc::l1r_l2_svc(problem_l1 *p, float64_t *w, float64_t eps, float64_t Cp, float64_t Cn)
{	
	this->prob_col = p;
	solve_l1r_l2_svc(w, eps, Cp, Cn);
}

void l1r_l2_svc::solve_l1r_l2_svc(float64_t *w, float64_t eps, float64_t Cp, float64_t Cn)
{
	int32_t l = prob_col->l;
	int32_t w_size = prob_col->n;
	int32_t j, s, iter = 0;
	int32_t max_iter = 1000;
	int32_t active_size = w_size;
	int32_t max_num_linesearch = 20;

	float64_t sigma = 0.01;
	float64_t d, G_loss, G, H;
	float64_t Gmax_old = INF;
	float64_t Gmax_new;
	float64_t Gmax_init;
	float64_t d_old, d_diff;
	float64_t loss_old, loss_new;
	float64_t appxcond, cond;

	int32_t *index = new int32_t[w_size];
	schar *y = new schar[l];
	float64_t *b = new float64_t[l]; // b = 1-ywTx
	float64_t *xj_sq = new float64_t[w_size];
	float64_t *C_ex = new float64_t[w_size];
	//TSparseEntry<ST>* x;

	float64_t C[3] = {Cn,0,Cp};
	float64_t *C_vec = new float64_t[l];

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
		{
			y[j] = 1;
			C_ex[j] = C[GETI(j)];
		}
		else
		{		
			y[j] = -1;
			C_ex[j] = C[GETI(j)];
		}
	}

	

	xj_sq=((CSparseFeatures<float64_t>*) prob_col->x)->compute_squared(xj_sq);

	//printf( "a: %f  %f\n", sq_lhs[0], sq_lhs[1]);


	
}



#endif //HAVE_LAPACK
#endif // DOXYGEN_SHOULD_SKIP_THIS
