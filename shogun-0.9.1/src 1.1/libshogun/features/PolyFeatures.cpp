#include "features/PolyFeatures.h"

using namespace shogun;

CPolyFeatures::CPolyFeatures(CSimpleFeatures<float64_t>* feat, int32_t degree, bool normalize)
	: CDotFeatures(), m_multi_index(NULL), m_multinomial_coefficients(NULL), 
		m_normalization_values(NULL)
{
	ASSERT(feat);

	m_feat = feat;
	SG_REF(m_feat);
	m_degree=degree;
	m_normalize=normalize;
	m_input_dimensions=feat->get_num_features();
	m_output_dimensions=calc_feature_space_dimensions(m_input_dimensions, m_degree);

	store_multi_index();
	store_multinomial_coefficients();
	if (m_normalize)
		store_normalization_values();
}

CPolyFeatures::~CPolyFeatures()
{
	delete[] m_multi_index;
	delete[] m_multinomial_coefficients;
	delete[] m_normalization_values;
	SG_UNREF(m_feat);
}

float64_t CPolyFeatures::dot(int32_t vec_idx1, int32_t vec_idx2)
{
	int32_t len1;
	bool do_free1;
	float64_t* vec1 = m_feat->get_feature_vector(vec_idx1, len1, do_free1);

	int32_t len2;
	bool do_free2;
	float64_t* vec2 = m_feat->get_feature_vector(vec_idx2, len2, do_free2);

	float64_t sum=0;
	int cnt=0;
	for (int j=0; j<m_output_dimensions; j++)
	{
		float64_t out1=m_multinomial_coefficients[j];
		float64_t out2=m_multinomial_coefficients[j];
		for (int k=0; k<m_degree; k++)
		{
			out1*=vec1[m_multi_index[cnt]];
			out2*=vec2[m_multi_index[cnt]];
			cnt++;
		}
		sum+=out1*out2;
	}
	m_feat->free_feature_vector(vec1, len1, do_free1);
	m_feat->free_feature_vector(vec2, len2, do_free2);

	return sum;
}

float64_t CPolyFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	if (vec2_len != m_output_dimensions)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, m_output_dimensions=%d\n", vec2_len, m_output_dimensions);

	int32_t len;
	bool do_free;
	float64_t* vec = m_feat->get_feature_vector(vec_idx1, len, do_free);

	
	int cnt=0;
	float64_t sum=0;
	for (int j=0; j<vec2_len; j++)
	{
		float64_t output=m_multinomial_coefficients[j];
		for (int k=0; k<m_degree; k++)
		{
			output*=vec[m_multi_index[cnt]];
			cnt++;
		}
		sum+=output*vec2[j];
	}
	if (m_normalize)
		sum = sum/m_normalization_values[vec_idx1];

	m_feat->free_feature_vector(vec, len, do_free);
	return sum;
}
void CPolyFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	if (vec2_len != m_output_dimensions)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, m_output_dimensions=%d\n", vec2_len, m_output_dimensions);

	int32_t len;
	bool do_free;
	float64_t* vec = m_feat->get_feature_vector(vec_idx1, len, do_free);

	
	int cnt=0;
	float32_t norm_val=1;
	if (m_normalize)
		norm_val = m_normalization_values[vec_idx1];
	alpha/=norm_val;
	for (int j=0; j<vec2_len; j++)
	{
		float64_t output=m_multinomial_coefficients[j];
		for (int k=0; k<m_degree; k++)
		{
			output*=vec[m_multi_index[cnt]];
			cnt++;
		}
		if (abs_val)
			output=CMath::abs(output); 

		vec2[j]+=alpha*output;
	}
	m_feat->free_feature_vector(vec, len, do_free);
}
void CPolyFeatures::store_normalization_values()
{
	delete[] m_normalization_values;

	int32_t num_vec = this->get_num_vectors();

	m_normalization_values=new float32_t[num_vec];
	for (int i=0; i<num_vec; i++)
	{
		float64_t tmp = CMath::sqrt(dot(i,i)); 
		if (tmp==0)
			// trap division by zero
			m_normalization_values[i]=1;
		else 
			m_normalization_values[i]=tmp;
	}
		
}

void CPolyFeatures::store_multi_index()
{
	delete[] m_multi_index;

        m_multi_index=new uint16_t[m_output_dimensions*m_degree];

        uint16_t* exponents = new uint16_t[m_input_dimensions];
        if (!exponents)
		SG_ERROR( "Error allocating mem \n");	
	/*copy adress: otherwise it will be overwritten in recursion*/
        uint16_t* index = m_multi_index;
        enumerate_multi_index(0, &index, exponents, m_degree);

	delete[] exponents;
}

void CPolyFeatures::enumerate_multi_index(const int32_t feat_idx, uint16_t** index, uint16_t* exponents, const int32_t degree)
{
	if (feat_idx==m_input_dimensions-1 || degree==0)
	{
		if (feat_idx==m_input_dimensions-1)
			exponents[feat_idx] = degree;
		if (degree==0)
			exponents[feat_idx] = 0;
		int32_t i, j;
		for (j=0; j<feat_idx+1; j++)
			for (i=0; i<exponents[j]; i++)
			{
				**index = j;
				(*index)++;
			}
		exponents[feat_idx] = 0;
		return;
	}
	int32_t k;
	for (k=0; k<=degree; k++)
	{
		exponents[feat_idx] =  k;
		enumerate_multi_index(feat_idx+1, index,  exponents, degree-k);	
	}
	return; 

}

void CPolyFeatures::store_multinomial_coefficients()
{
	delete[] m_multinomial_coefficients;

	m_multinomial_coefficients = new float64_t[m_output_dimensions];
	int32_t* exponents = new int32_t[m_input_dimensions];
	if (!exponents)
		SG_ERROR( "Error allocating mem \n");
	int32_t j=0;
	for (j=0; j<m_input_dimensions; j++)
		exponents[j] = 0;
	int32_t k, cnt=0;
	for (j=0; j<m_output_dimensions; j++)
	{
		for (k=0; k<m_degree; k++)
		{
			exponents[m_multi_index[cnt]] ++;
			cnt++;
		}
		m_multinomial_coefficients[j] =  sqrt((double) multinomialcoef(exponents, m_input_dimensions));
		for (k=0; k<m_input_dimensions; k++)
		{
			exponents[k]=0;
		}
	}
	delete[] exponents;
}

int32_t CPolyFeatures::bico2(int32_t n, int32_t k)
{
		
	/* for this problem k is usually small (<=degree), 
	 * thus it is efficient to 
	 * to use recursion and prune end recursions*/	
	if (n<k) 
		return 0;
	if (k>n/2)
		k = n-k;
	if (k<0)
		return 0;
	if (k==0)
		return 1;
	if (k==1)
		return n;
	if (k<4)
		return bico2(n-1, k-1)+bico2(n-1, k);

	/* call function as implemented in numerical recipies:
	 * much more efficient for large binomial coefficients*/
	return bico(n, k);
	
}

int32_t CPolyFeatures::calc_feature_space_dimensions(int32_t N, int32_t D)
{
	if (N==1)
		return 1;
	if (D==0)
		return 1;
	int32_t d;
	int32_t ret = 0;
	for (d=0; d<=D; d++)
		ret += calc_feature_space_dimensions(N-1, d);

	return ret;
}

int32_t CPolyFeatures::multinomialcoef(int32_t* exps, int32_t len)
{
	int32_t ret = 1, i;
	int32_t n = 0;
	for (i=0; i<len; i++)
	{
		n += exps[i];
		ret *= bico2(n, exps[i]);
	}
	return ret; 
}

/* gammln as implemented in the
 * second edition of Numerical Recipes in C */
float64_t CPolyFeatures::gammln(float64_t xx)
{
    float64_t x,y,tmp,ser;
    static float64_t cof[6]={76.18009172947146,    -86.50532032941677,
                          24.01409824083091,    -1.231739572450155,
                          0.1208650973866179e-2,-0.5395239384953e-5};
    int32_t j;

    y=x=xx;
    tmp=x+5.5;
    tmp -= (x+0.5)*log(tmp);
    ser=1.000000000190015;
    for (j=0;j<=5;j++) ser += cof[j]/++y;
    return -tmp+log(2.5066282746310005*ser/x);
}

float64_t CPolyFeatures::factln(int32_t n)
{
	static float64_t a[101];

	if (n < 0) SG_ERROR("Negative factorial in routine factln\n");
	if (n <= 1) return 0.0;
	if (n <= 100) return a[n] ? a[n] : (a[n]=gammln(n+1.0));
	else return gammln(n+1.0);
}

int32_t CPolyFeatures::bico(int32_t n, int32_t k)
{
	/* use floor to clean roundoff errors*/
	return (int32_t) floor(0.5+exp(factln(n)-factln(k)-factln(n-k)));
}
CFeatures* CPolyFeatures::duplicate() const
{
	return new CPolyFeatures(*this);
}
