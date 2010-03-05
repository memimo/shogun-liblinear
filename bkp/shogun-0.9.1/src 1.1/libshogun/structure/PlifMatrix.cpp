#include "structure/PlifMatrix.h"
#include "structure/Plif.h"
#include "structure/PlifArray.h"
#include "structure/PlifBase.h"

using namespace shogun;

CPlifMatrix::CPlifMatrix() : m_PEN(NULL), m_num_plifs(0), m_num_limits(0),
	m_plif_matrix(NULL), m_state_signals(NULL)
{
}

CPlifMatrix::~CPlifMatrix()
{
}

void CPlifMatrix::create_plifs(int32_t num_plifs, int32_t num_limits)
{
	for (int32_t i=0; i<m_num_plifs; i++)	
		delete m_PEN[i];
	delete[] m_PEN;
	m_PEN=NULL;

	m_num_plifs=num_plifs;
	m_num_limits=num_limits;
	m_PEN = new CPlif*[num_plifs] ;
	for (int32_t i=0; i<num_plifs; i++)	
		m_PEN[i]=new CPlif(num_limits) ;
}

void CPlifMatrix::set_plif_ids(int32_t* plif_ids, int32_t num_ids)
{
	if (num_ids!=m_num_plifs)
		SG_ERROR("plif_ids size mismatch (num_ids=%d vs.num_plifs=%d)\n", num_ids, m_num_plifs);

	m_ids.resize_array(m_num_plifs);
	m_ids.set_array(plif_ids, num_ids, true, true);

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_id(id);
	}
}

void CPlifMatrix::set_plif_min_values(float64_t* min_values, int32_t num_values)
{
	if (num_values!=m_num_plifs)
		SG_ERROR("plif_values size mismatch (num_values=%d vs.num_plifs=%d)\n", num_values, m_num_plifs);

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_min_value(min_values[i]);
	}
}

void CPlifMatrix::set_plif_max_values(float64_t* max_values, int32_t num_values)
{
	if (num_values!=m_num_plifs)
		SG_ERROR("plif_values size mismatch (num_values=%d vs.num_plifs=%d)\n", num_values, m_num_plifs);

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_max_value(max_values[i]);
	}
}

void CPlifMatrix::set_plif_use_cache(bool* use_cache, int32_t num_values)
{
	if (num_values!=m_num_plifs)
		SG_ERROR("plif_values size mismatch (num_values=%d vs.num_plifs=%d)\n", num_values, m_num_plifs);

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_use_cache(use_cache[i]);
	}
}

void CPlifMatrix::set_plif_use_svm(int32_t* use_svm, int32_t num_values)
{
	if (num_values!=m_num_plifs)
		SG_ERROR("plif_values size mismatch (num_values=%d vs.num_plifs=%d)\n", num_values, m_num_plifs);

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_use_svm(use_svm[i]);
	}
}

void CPlifMatrix::set_plif_limits(float64_t* limits, int32_t num_plifs, int32_t num_limits)
{
	if (num_plifs!=m_num_plifs ||  num_limits!=m_num_limits)
	{
		SG_ERROR("limits size mismatch expected (%d,%d) got (%d,%d)\n",
				m_num_plifs, m_num_limits, num_plifs, num_limits);
	}

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		float64_t* lim = new float64_t[m_num_limits];

		for (int32_t k=0; k<m_num_limits; k++)
			lim[k] = limits[i*m_num_limits+k];

		int32_t id=get_plif_id(i);
		m_PEN[id]->set_plif_limits(lim, m_num_limits);
	}
}

void CPlifMatrix::set_plif_penalties(float64_t* penalties, int32_t num_plifs, int32_t num_limits)
{
	if (num_plifs!=m_num_plifs ||  num_limits!=m_num_limits)
	{
		SG_ERROR("penalties size mismatch expected (%d,%d) got (%d,%d)\n",
				m_num_plifs, m_num_limits, num_plifs, num_limits);
	}

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		float64_t* pen = new float64_t[m_num_limits];

		for (int32_t k=0; k<m_num_limits; k++)
			pen[k] = penalties[i*m_num_limits+k];

		int32_t id=get_plif_id(i);
		m_PEN[id]->set_plif_penalty(pen, m_num_limits);
	}
}

void CPlifMatrix::set_plif_names(T_STRING<char>* names, int32_t num_values, int32_t maxlen)
{
	if (num_values!=m_num_plifs)
		SG_ERROR("names size mismatch (num_values=%d vs.num_plifs=%d)\n", num_values, m_num_plifs);

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		m_PEN[id]->set_plif_name(CStringFeatures<char>::get_zero_terminated_string_copy(names[i]));
	}
}

void CPlifMatrix::set_plif_transform_type(T_STRING<char>* transform_type, int32_t num_values, int32_t maxlen)
{
	if (num_values!=m_num_plifs)
		SG_ERROR("transform_type size mismatch (num_values=%d vs.num_plifs=%d)\n", num_values, m_num_plifs);

	for (int32_t i=0; i<m_num_plifs; i++)
	{
		int32_t id=get_plif_id(i);
		char* transform_str=CStringFeatures<char>::get_zero_terminated_string_copy(transform_type[i]);

		if (!m_PEN[id]->set_transform_type(transform_str))
		{
			delete[] m_PEN;
			m_PEN=NULL;
			m_num_plifs=0;
			m_num_limits=0;
			SG_ERROR( "transform type not recognized ('%s')\n", transform_str) ;
		}
	}
}


bool CPlifMatrix::compute_plif_matrix(
	float64_t* penalties_array, int32_t* Dim, int32_t numDims)
{
	CPlif** PEN = get_PEN();
	int32_t num_states = Dim[0];
	int32_t num_plifs = get_num_plifs();

	delete[] m_plif_matrix ;
	m_plif_matrix = new CPlifBase*[num_states*num_states] ;

	CArray3<float64_t> penalties(penalties_array, num_states, num_states, Dim[2], true, true) ;

	for (int32_t i=0; i<num_states; i++)
	{
		for (int32_t j=0; j<num_states; j++)
		{
			CPlifArray * plif_array = new CPlifArray() ;
			CPlif * plif = NULL ;
			plif_array->clear() ;
			for (int32_t k=0; k<Dim[2]; k++)
			{
				if (penalties.element(i,j,k)==0)
					continue ;
				int32_t id = (int32_t) penalties.element(i,j,k)-1 ;

				if ((id<0 || id>=num_plifs) && (id!=-1))
				{
					SG_ERROR( "id out of range\n") ;
					CPlif::delete_penalty_struct(PEN, num_plifs) ;
					return false ;
				}
				plif = PEN[id] ;

				plif_array->add_plif(plif) ;
			}

			if (plif_array->get_num_plifs()==0)
			{
				SG_UNREF(plif_array);
				m_plif_matrix[i+j*num_states] = NULL ;
			}
			else if (plif_array->get_num_plifs()==1)
			{
				SG_UNREF(plif_array);
				ASSERT(plif!=NULL) ;
				m_plif_matrix[i+j*num_states] = plif ;
			}
			else
				m_plif_matrix[i+j*num_states] = plif_array ;

		}
	}
	return true;
}

bool  CPlifMatrix::compute_signal_plifs(
	int32_t* state_signals, int32_t feat_dim3, int32_t num_states)
{
	int32_t Nplif = get_num_plifs();
	CPlif** PEN = get_PEN();

	CPlifBase **PEN_state_signal = new CPlifBase*[feat_dim3*num_states] ;
	for (int32_t i=0; i<num_states*feat_dim3; i++)
	{
		int32_t id = (int32_t) state_signals[i]-1 ;
		if ((id<0 || id>=Nplif) && (id!=-1))
		{
			SG_ERROR( "id out of range\n") ;
			CPlif::delete_penalty_struct(PEN, Nplif) ;
			return false ;
		}
		if (id==-1)
			PEN_state_signal[i]=NULL ;
		else
			PEN_state_signal[i]=PEN[id] ;
	}
	m_state_signals=PEN_state_signal;
	return true;
}

void CPlifMatrix::set_plif_state_signal_matrix(
	int32_t *plif_id_matrix, int32_t m, int32_t max_num_signals)
{
	if (m!=m_num_plifs)
		SG_ERROR( "plif_state_signal_matrix size does not match previous info %i!=%i\n", m, m_num_plifs) ;

	/*if (m_seq.get_dim3() != max_num_signals)
		SG_ERROR( "size(plif_state_signal_matrix,2) does not match with size(m_seq,3): %i!=%i\nSorry, Soeren... interface changed\n", m_seq.get_dim3(), max_num_signals) ;

	CArray2<int32_t> id_matrix(plif_id_matrix, m_num_plifs, max_num_signals, false, false) ;
	m_PEN_state_signals.resize_array(m_num_plifs, max_num_signals) ;
	for (int32_t i=0; i<m_num_plifs; i++)
	{
		for (int32_t j=0; j<max_num_signals; j++)
		{
			if (id_matrix.element(i,j)>=0)
				m_PEN_state_signals.element(i,j)=m_plif_list[id_matrix.element(i,j)] ;
			else
				m_PEN_state_signals.element(i,j)=NULL ;
		}
	}*/
}
