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

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
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

l1r_l2_svc::l1r_l2_svc(problem *prob_col, float64_t *w, float64_t eps, float64_t Cp, float64_t Cn)
{
	solve_l1r_l2_svc(prob_col, w, eps, Cp, Cn);
}

void l1r_l2_svc::solve_l1r_l2_svc(
	problem *prob_col, float64_t *w, float64_t eps, 
	float64_t Cp, float64_t Cn)
{
	int32_t l = prob_col->l;
	int32_t w_size = prob_col->n;
	int32_t j, k, s, iter = 0;
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

	int32_t *index = new int[w_size];
	schar *y = new schar[l];
	float64_t *b = new double[l]; // b = 1-ywTx
	float64_t *xj_sq = new double[w_size];
	//feature_node *x; //#deleted line
	float64_t **x; //# new line
	int32_t *num_feat;
	int32_t *num_vec;

	// To support weights for instances,
	// replace C[y[i]] with C[i].
	float64_t C[2] = {Cn,Cp};

	//# Load feature matrix data

	prob_col->x->get_feature_matrix(x, num_feat, num_vec);
	
	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = 0;
	}
	for(j=0; j<w_size; j++)
	{
		w[j] = 0;
		index[j] = j;
		xj_sq[j] = 0;
		for(k=0; k<l; k++)
		{
			int32_t ind = k;
			float64_t val = x[k][j];
			x[k][j] *= prob_col->y[ind]; // x->value stores yi*xij
			xj_sq[j] += C[y[ind]]*val*val;
		}
	}

	while(iter < max_iter)
	{
		Gmax_new  = 0;

		for(j=0; j<active_size; j++)
		{
			int32_t i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

			for(k=0; k<l; k++)
			{
				int32_t ind = k;
				if(b[ind] > 0)
				{
					float64_t val = x[k][j];
					float64_t tmp = C[y[ind]]*val;
					G_loss -= tmp*b[ind];
					H += tmp*val;
				}
			}
			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = max(H, 1e-12);

			float64_t Gp = G+1;
			float64_t Gn = G-1;
			float64_t violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);

			// obtain Newton direction d
			if(Gp <= H*w[j])
				d = -Gp/H;
			else if(Gn >= H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			float64_t delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int32_t num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{
					for(k=0; k<l; k++)
					{
						b[k] += d_diff*x[k][j];
					}
					break;
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
					for(k=0; k<l; k++)
					{
						int32_t ind =k;
						if(b[ind] > 0)
							loss_old += C[y[ind]]*b[ind]*b[ind];
						float64_t b_new = b[ind] + d_diff*x[k][j];
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[y[ind]]*b_new*b_new;
					}
				}
				else
				{
					loss_new = 0;
					for(k=0; k<l; k++)
					{
						int32_t ind = k;
						float64_t b_new = b[ind] + d_diff*x[k][j];
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[y[ind]]*b_new*b_new;
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				SG_INFO("#");
				for(int32_t i=0; i<l; i++)
					b[i] = 1;

				for(int32_t i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					for(k=0; k<l; k++)
					{
						b[k] -= w[i]*x[k][j];
					}
				}
			}
		}

		if(iter == 0)
			Gmax_init = Gmax_new;
		iter++;
		if(iter % 10 == 0)
			SG_INFO(".");

		if(Gmax_new <= eps*Gmax_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				SG_INFO("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	SG_INFO("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		SG_INFO("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	float64_t v = 0;
	int32_t nnz = 0;
	for(j=0; j<w_size; j++)
	{
		for(k=0; k<l; k++)
		{
			x[k][j] *= prob_col->y[k]; // restore x->value
		}
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(j=0; j<l; j++)
		if(b[j] > 0)
			v += C[y[j]]*b[j]*b[j];

	SG_INFO("Objective value = %lf\n", v);
	SG_INFO("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}

#endif //HAVE_LAPACK
#endif // DOXYGEN_SHOULD_SKIP_THIS
