
// File: index.xml

// File: classshogun_1_1CDynProg.xml
%feature("docstring") shogun::CDynProg "

Dynamic Programming Class.

Structure and Function collection. This Class implements a Dynamic
Programming functions.

C++ includes: DynProg.h ";

/*  model specific variables.  */

/*  these are p,q,a,b,N,M etc

*/

%feature("docstring")  shogun::CDynProg::CDynProg "

constructor

Parameters:
-----------

p_num_svms:  number of SVMs ";

%feature("docstring")  shogun::CDynProg::~CDynProg "";

%feature("docstring")  shogun::CDynProg::best_path_no_b "

best path no b

Parameters:
-----------

max_iter:  max iter

best_iter:  best iter

my_path:  my path

best path no b ";

%feature("docstring")  shogun::CDynProg::best_path_no_b_trans "

best path no b transition

Parameters:
-----------

max_iter:  max iter

max_best_iter:  max best iter

nbest:  nbest

prob_nbest:  prob_nbest

my_paths:  my paths ";

%feature("docstring")  shogun::CDynProg::set_num_states "

set number of states use this to set N first

Parameters:
-----------

N:  new N ";

%feature("docstring")  shogun::CDynProg::get_num_states "

get num states ";

%feature("docstring")  shogun::CDynProg::get_num_svms "

get num svms ";

%feature("docstring")  shogun::CDynProg::init_content_svm_value_array
"

init CArray for precomputed content svm values with size seq_len x
num_svms

Parameters:
-----------

p_num_svms:  ::  number of svm weight vectors for content prediction
";

%feature("docstring")  shogun::CDynProg::init_tiling_data "

init CArray for precomputed tiling intensitie-plif-values with size
seq_len x num_svms

Parameters:
-----------

probe_pos:  local positions of probes

intensities:  intensities of probes

num_probes:  number of probes ";

%feature("docstring")  shogun::CDynProg::precompute_tiling_plifs "

precompute tiling Plifs

Parameters:
-----------

PEN:  Plif PEN

tiling_plif_ids:  tiling plif id's

num_tiling_plifs:  number of tiling plifs ";

%feature("docstring")  shogun::CDynProg::resize_lin_feat "

append rows to linear features array

Parameters:
-----------

num_new_feat:  number of new rows to add ";

%feature("docstring")  shogun::CDynProg::set_p_vector "

set vector p

Parameters:
-----------

p:  new vector p

N:  size of vector p ";

%feature("docstring")  shogun::CDynProg::set_q_vector "

set vector q

Parameters:
-----------

q:  new vector q

N:  size of vector q ";

%feature("docstring")  shogun::CDynProg::set_a "

set matrix a

Parameters:
-----------

a:  new matrix a

M:  dimension M of matrix a

N:  dimension N of matrix a ";

%feature("docstring")  shogun::CDynProg::set_a_id "

set a id

Parameters:
-----------

a:  new a id (identity?)

M:  dimension M of matrix a

N:  dimension N of matrix a ";

%feature("docstring")  shogun::CDynProg::set_a_trans_matrix "

set a transition matrix

Parameters:
-----------

a_trans:  transition matrix a

num_trans:  number of transitions

N:  dimension N of matrix a ";

%feature("docstring")  shogun::CDynProg::init_mod_words_array "

init mod words array

Parameters:
-----------

p_mod_words_array:  new mod words array

num_elem:  number of array elements

num_columns:  number of columns ";

%feature("docstring")  shogun::CDynProg::check_svm_arrays "

check SVM arrays call this function to check consistency

whether arrays are ok ";

%feature("docstring")  shogun::CDynProg::set_observation_matrix "

set best path seq

Parameters:
-----------

seq:  signal features

dims:  dimensions

ndims:  number of dimensions ";

%feature("docstring")  shogun::CDynProg::get_num_positions "

get number of positions; the dynamic program is sparse encoded and
this function gives the number of positions that can actually be part
of a predicted path

number of positions ";

%feature("docstring")  shogun::CDynProg::set_content_type_array "

set an array of length #(candidate positions) which specifies the
content type of each pos and a mask that determines to which extend
the loss should be applied to this position; this is a way to encode
label confidence via weights between zero and one

Parameters:
-----------

seg_path:  seg path

rows:  rows

cols:  cols ";

%feature("docstring")  shogun::CDynProg::set_pos "

set best path pos

Parameters:
-----------

pos:  the position

seq_len:  length of sequence ";

%feature("docstring")  shogun::CDynProg::set_orf_info "

set best path orf info only for compute_nbest_paths

Parameters:
-----------

orf_info:  the orf info

m:  dimension m

n:  dimension n ";

%feature("docstring")  shogun::CDynProg::set_gene_string "

set best path genesstr

Parameters:
-----------

genestr:  gene string

genestr_len:  length of gene string ";

%feature("docstring")  shogun::CDynProg::set_dict_weights "

set best path dict weights

Parameters:
-----------

dictionary_weights:  dictionary weights

dict_len:  length of dictionary weights

n:  dimension n ";

%feature("docstring")  shogun::CDynProg::best_path_set_segment_loss "

set best path segment loss

Parameters:
-----------

segment_loss:  segment loss

num_segment_id1:  number of segment id1

num_segment_id2:  number of segment id2 ";

%feature("docstring")
shogun::CDynProg::best_path_set_segment_ids_mask "

set best path segmend ids mask

Parameters:
-----------

segment_ids:  segment ids

segment_mask:  segment mask

m:  dimension m ";

%feature("docstring")  shogun::CDynProg::set_sparse_features "

set sparse feature matrices ";

%feature("docstring")  shogun::CDynProg::set_plif_matrices "

set plif matrices

Parameters:
-----------

pm:  plif matrix object ";

%feature("docstring")  shogun::CDynProg::get_scores "

best path get scores

Parameters:
-----------

scores:  scores

n:  dimension n ";

%feature("docstring")  shogun::CDynProg::get_states "

best path get states

Parameters:
-----------

states:  states

m:  dimension m

n:  dimension n ";

%feature("docstring")  shogun::CDynProg::get_positions "

best path get positions

Parameters:
-----------

positions:  positions

m:  dimension m

n:  dimension n ";

%feature("docstring")  shogun::CDynProg::compute_nbest_paths "

run the viterbi algorithm to compute the n best viterbi paths

Parameters:
-----------

max_num_signals:  maximal number of signals for a single state

use_orf:  whether orf shall be used

nbest:  number of best paths (n)

with_loss:  use loss

with_multiple_sequences:  !!!not functional set to false!!! ";

%feature("docstring")  shogun::CDynProg::best_path_trans_deriv "

given a path though the state model and the corresponding positions
compute the features. This can be seen as the derivative of the score
(output of dynamic program) with respect to the parameters

Parameters:
-----------

my_state_seq:  state sequence of the path

my_pos_seq:  sequence of positions

my_seq_len:  length of state and position sequences

seq_array:  array of features

max_num_signals:  maximal number of signals ";

%feature("docstring")  shogun::CDynProg::set_my_state_seq "

set best path my state sequence

Parameters:
-----------

my_state_seq:  my state sequence ";

%feature("docstring")  shogun::CDynProg::set_my_pos_seq "

set best path my position sequence

Parameters:
-----------

my_pos_seq:  my position sequence ";

%feature("docstring")  shogun::CDynProg::get_path_scores "

get path scores

best_path_trans_deriv result retrieval functions

Parameters:
-----------

my_scores:  scores

seq_len:  length of sequence ";

%feature("docstring")  shogun::CDynProg::get_path_losses "

get path losses

best_path_trans_deriv result retrieval functions

Parameters:
-----------

my_losses:  my losses

seq_len:  length of sequence ";

%feature("docstring")  shogun::CDynProg::get_N "

access function for number of states N ";

%feature("docstring")  shogun::CDynProg::set_q "

access function for probability of end states

Parameters:
-----------

offset:  index 0...N-1

value:  value to be set ";

%feature("docstring")  shogun::CDynProg::set_p "

access function for probability of first state

Parameters:
-----------

offset:  index 0...N-1

value:  value to be set ";

%feature("docstring")  shogun::CDynProg::set_a "

access function for matrix a

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...N-1

value:  value to be set ";

%feature("docstring")  shogun::CDynProg::get_q "

access function for probability of end states

Parameters:
-----------

offset:  index 0...N-1

value at offset ";

%feature("docstring")  shogun::CDynProg::get_q_deriv "

access function for derivated probability of end states

Parameters:
-----------

offset:  index 0...N-1

value at offset ";

%feature("docstring")  shogun::CDynProg::get_p "

access function for probability of initial states

Parameters:
-----------

offset:  index 0...N-1

value at offset ";

%feature("docstring")  shogun::CDynProg::get_p_deriv "

access function for derivated probability of initial states

Parameters:
-----------

offset:  index 0...N-1

value at offset ";

%feature("docstring")  shogun::CDynProg::precompute_content_values "

create array of precomputed content svm values ";

%feature("docstring")  shogun::CDynProg::get_lin_feat "

return array of precomputed linear features like content predictions
and PLiFed tiling array data Jonas

lin_feat_array ";

%feature("docstring")  shogun::CDynProg::set_lin_feat "

set your own array of precomputed linear features like content
predictions and PLiFed tiling array data Jonas

Parameters:
-----------

p_lin_feat:  array of features

p_num_svms:  number of tracks

p_seq_len:  number of candidate positions ";

%feature("docstring")  shogun::CDynProg::create_word_string "

create word string from char* Jonas ";

%feature("docstring")  shogun::CDynProg::precompute_stop_codons "

precompute stop codons ";

%feature("docstring")  shogun::CDynProg::get_a "

access function for matrix a

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...N-1

value at position line colum ";

%feature("docstring")  shogun::CDynProg::get_a_deriv "

access function for matrix a derivated

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...N-1

value at position line colum ";

%feature("docstring")  shogun::CDynProg::set_intron_list "

set intron list

Parameters:
-----------

intron_list:

num_plifs:  number of intron plifs ";

%feature("docstring")  shogun::CDynProg::get_segment_loss_object "

get the segment loss object ";

%feature("docstring")  shogun::CDynProg::long_transition_settings "

settings for long transition handling

Parameters:
-----------

use_long_transitions:  use the long transition approximation

threshold:  use long transition for segments larger than

max_len:  allow transitions up to ";


// File: classshogun_1_1CIntronList.xml
%feature("docstring") shogun::CIntronList "

class IntronList

C++ includes: IntronList.h ";

%feature("docstring")  shogun::CIntronList::CIntronList "

constructor ";

%feature("docstring")  shogun::CIntronList::~CIntronList "";

%feature("docstring")  shogun::CIntronList::init_list "

initialize all arrays with the number of candidate positions

Parameters:
-----------

all_pos:  list of candidate positions

len:  number of candidate positions ";

%feature("docstring")  shogun::CIntronList::read_introns "

read introns

Parameters:
-----------

start_pos:  array of start positions

end_pos:  array of end positions

quality:  quality scores for introns in list

len:  number of items in all three previous arguments ";

%feature("docstring")  shogun::CIntronList::get_intron_support "

get coverage and quality score

Parameters:
-----------

values:  values[0]: coverage of that intron; values[1]: associated
quality score

from_pos:  start position of intron

to_pos:  end position of intron ";

%feature("docstring")  shogun::CIntronList::get_name "

object name ";


// File: classshogun_1_1CPlif.xml
%feature("docstring") shogun::CPlif "

class Plif

C++ includes: Plif.h ";

%feature("docstring")  shogun::CPlif::CPlif "

constructor

Parameters:
-----------

len:  len ";

%feature("docstring")  shogun::CPlif::~CPlif "";

%feature("docstring")  shogun::CPlif::init_penalty_struct_cache "

init penalty struct cache ";

%feature("docstring")  shogun::CPlif::lookup_penalty_svm "

lookup penalty SVM

Parameters:
-----------

p_value:  value

d_values:  d values

the penalty ";

%feature("docstring")  shogun::CPlif::lookup_penalty "

lookup penalty float64_t

Parameters:
-----------

p_value:  value

svm_values:  SVM values

the penalty ";

%feature("docstring")  shogun::CPlif::lookup_penalty "

lookup penalty int32_t

Parameters:
-----------

p_value:  value

svm_values:  SVM values

the penalty ";

%feature("docstring")  shogun::CPlif::lookup "

lookup

Parameters:
-----------

p_value:  value

a penalty ";

%feature("docstring")  shogun::CPlif::penalty_clear_derivative "

penalty clear derivative ";

%feature("docstring")  shogun::CPlif::penalty_add_derivative_svm "

penalty add derivative SVM

Parameters:
-----------

p_value:  value

svm_values:  SVM values

factor:  factor weighting the added value ";

%feature("docstring")  shogun::CPlif::penalty_add_derivative "

penalty add derivative

Parameters:
-----------

p_value:  value

svm_values:  SVM values

factor:  factor weighting the added value ";

%feature("docstring")  shogun::CPlif::get_cum_derivative "

get cum derivative

Parameters:
-----------

p_len:  len

cum derivative ";

%feature("docstring")  shogun::CPlif::set_transform_type "

set transform type

Parameters:
-----------

type_str:  type (string)

if setting was successful ";

%feature("docstring")  shogun::CPlif::get_transform_type "

get transform type

type_str type (string) ";

%feature("docstring")  shogun::CPlif::set_id "

set ID

Parameters:
-----------

p_id:  the id to set ";

%feature("docstring")  shogun::CPlif::get_id "

get ID

the ID ";

%feature("docstring")  shogun::CPlif::get_max_id "

get maximum ID

maximum ID ";

%feature("docstring")  shogun::CPlif::set_use_svm "

set use SVM

Parameters:
-----------

p_use_svm:  if SVM shall be used ";

%feature("docstring")  shogun::CPlif::get_use_svm "

get use SVM

if SVM is used ";

%feature("docstring")  shogun::CPlif::uses_svm_values "

check if plif uses SVM values

if plif uses SVM values ";

%feature("docstring")  shogun::CPlif::set_use_cache "

set use cache

Parameters:
-----------

p_use_cache:  if cache shall be used ";

%feature("docstring")  shogun::CPlif::invalidate_cache "

invalidate the cache ";

%feature("docstring")  shogun::CPlif::get_use_cache "

get use cache

if cache is used ";

%feature("docstring")  shogun::CPlif::set_plif "

set plif

Parameters:
-----------

p_len:  len

p_limits:  limit

p_penalties:  penalties ";

%feature("docstring")  shogun::CPlif::set_plif_limits "

set plif_limits

Parameters:
-----------

p_limits:  limit

p_len:  len ";

%feature("docstring")  shogun::CPlif::set_plif_penalty "

set plif penalty

Parameters:
-----------

p_penalties:  penalties

p_len:  len ";

%feature("docstring")  shogun::CPlif::set_plif_length "

set plif length

Parameters:
-----------

p_len:  len ";

%feature("docstring")  shogun::CPlif::get_plif_limits "

get Plif limits

limits ";

%feature("docstring")  shogun::CPlif::get_plif_penalties "

get plif penalty

plif penalty ";

%feature("docstring")  shogun::CPlif::set_max_value "

set maximum value

Parameters:
-----------

p_max_value:  maximum value ";

%feature("docstring")  shogun::CPlif::get_max_value "

get maximum value

maximum value ";

%feature("docstring")  shogun::CPlif::set_min_value "

set minimum value

Parameters:
-----------

p_min_value:  minimum value ";

%feature("docstring")  shogun::CPlif::get_min_value "

get minimum value

minimum value ";

%feature("docstring")  shogun::CPlif::set_plif_name "

set name

Parameters:
-----------

p_name:  name ";

%feature("docstring")  shogun::CPlif::get_plif_name "

get name

name ";

%feature("docstring")  shogun::CPlif::get_do_calc "

get do calc

if calc shall be done ";

%feature("docstring")  shogun::CPlif::set_do_calc "

set do calc

Parameters:
-----------

b:  if calc shall be done ";

%feature("docstring")  shogun::CPlif::get_used_svms "

get SVM_ids and number of SVMs used ";

%feature("docstring")  shogun::CPlif::get_plif_len "

get plif len

plif len ";

%feature("docstring")  shogun::CPlif::list_plif "

print PLIF

lists some properties of the PLIF ";

%feature("docstring")  shogun::CPlif::get_name "

object name ";


// File: classshogun_1_1CPlifArray.xml
%feature("docstring") shogun::CPlifArray "

class PlifArray

C++ includes: PlifArray.h ";

%feature("docstring")  shogun::CPlifArray::CPlifArray "

default constructor ";

%feature("docstring")  shogun::CPlifArray::~CPlifArray "";

%feature("docstring")  shogun::CPlifArray::add_plif "

add plif

Parameters:
-----------

new_plif:  the new plif to be added ";

%feature("docstring")  shogun::CPlifArray::clear "

clear ";

%feature("docstring")  shogun::CPlifArray::get_num_plifs "

get number of plifs

number of plifs ";

%feature("docstring")  shogun::CPlifArray::lookup_penalty "

lookup penalty float64_t

Parameters:
-----------

p_value:  value

svm_values:  SVM values ";

%feature("docstring")  shogun::CPlifArray::lookup_penalty "

lookup penalty int32_t

Parameters:
-----------

p_value:  value

svm_values:  SVM values ";

%feature("docstring")  shogun::CPlifArray::penalty_clear_derivative "

penalty clear derivative ";

%feature("docstring")  shogun::CPlifArray::penalty_add_derivative "

penalty add derivative

Parameters:
-----------

p_value:  value

svm_values:  SVM values

factor:  weighting the added value ";

%feature("docstring")  shogun::CPlifArray::get_max_value "

get maximum value

maximum value ";

%feature("docstring")  shogun::CPlifArray::get_min_value "

get minimum value

minumum value ";

%feature("docstring")  shogun::CPlifArray::uses_svm_values "

check if plif uses SVM values

if plif uses SVM values ";

%feature("docstring")  shogun::CPlifArray::get_max_id "

get maximum ID

maximum ID ";

%feature("docstring")  shogun::CPlifArray::get_used_svms "

get SVM_ids and number of SVMs used

abstract base method ";

%feature("docstring")  shogun::CPlifArray::list_plif "

print PLIF

lists all PLIFs in array ";

%feature("docstring")  shogun::CPlifArray::get_name "

object name ";


// File: classshogun_1_1CPlifBase.xml
%feature("docstring") shogun::CPlifBase "

class PlifBase

C++ includes: PlifBase.h ";

%feature("docstring")  shogun::CPlifBase::CPlifBase "

default constructor ";

%feature("docstring")  shogun::CPlifBase::~CPlifBase "";

%feature("docstring")  shogun::CPlifBase::lookup_penalty "

lookup penalty float64_t

abstract base method

Parameters:
-----------

p_value:  value

svm_values:  SVM values

penalty ";

%feature("docstring")  shogun::CPlifBase::lookup_penalty "

lookup penalty int32_t

abstract base method

Parameters:
-----------

p_value:  value

svm_values:  SVM values

penalty ";

%feature("docstring")  shogun::CPlifBase::penalty_clear_derivative "

penalty clear derivative

abstrace base method ";

%feature("docstring")  shogun::CPlifBase::penalty_add_derivative "

penalty add derivative

abstract base method

Parameters:
-----------

p_value:  value

svm_values:  SVM values

factor:  factor weighting the added value ";

%feature("docstring")  shogun::CPlifBase::get_max_value "

get maximum value

abstract base method

maximum value ";

%feature("docstring")  shogun::CPlifBase::get_min_value "

get minimum value

abstract base method

minimum value ";

%feature("docstring")  shogun::CPlifBase::get_used_svms "

get SVM_ids and number of SVMs used

abstract base method ";

%feature("docstring")  shogun::CPlifBase::uses_svm_values "

if plif uses SVM values

abstract base method

if plif uses SVM values ";

%feature("docstring")  shogun::CPlifBase::get_max_id "

get maximum ID

abstract base method

maximum ID ";

%feature("docstring")  shogun::CPlifBase::list_plif "

print PLIF

abstract base method ";


// File: classshogun_1_1CPlifMatrix.xml
%feature("docstring") shogun::CPlifMatrix "

store plif arrays for all transitions in the model

C++ includes: PlifMatrix.h ";

%feature("docstring")  shogun::CPlifMatrix::CPlifMatrix "

constructor ";

%feature("docstring")  shogun::CPlifMatrix::~CPlifMatrix "

destructor ";

%feature("docstring")  shogun::CPlifMatrix::get_PEN "

get array of all plifs

plif array ";

%feature("docstring")  shogun::CPlifMatrix::get_plif_matrix "

get plif matrix

matrix of plifs ";

%feature("docstring")  shogun::CPlifMatrix::get_state_signals "

get plifs defining the mapping of signals to states

plifs ";

%feature("docstring")  shogun::CPlifMatrix::get_num_plifs "

get number of plifs

number of plifs ";

%feature("docstring")  shogun::CPlifMatrix::get_num_limits "

get number of support points for picewise linear transformations
(PLiFs)

number of support points ";

%feature("docstring")  shogun::CPlifMatrix::create_plifs "

create an empty plif matrix of size num_plifs * num_limits

Parameters:
-----------

num_plifs:  number of plifs

num_limits:  number of plif limits ";

%feature("docstring")  shogun::CPlifMatrix::set_plif_ids "

set plif ids

Parameters:
-----------

ids:  plif ids

num_ids:  number of ids ";

%feature("docstring")  shogun::CPlifMatrix::set_plif_min_values "

set array of min values for all plifs

Parameters:
-----------

min_values:  array of min values

num_values:  length of array ";

%feature("docstring")  shogun::CPlifMatrix::set_plif_max_values "

set array of max values for all plifs

Parameters:
-----------

max_values:  array of max values

num_values:  length of array ";

%feature("docstring")  shogun::CPlifMatrix::set_plif_use_cache "

set plif use cache

Parameters:
-----------

use_cache:  set array of bool values

num_values:  length of array ";

%feature("docstring")  shogun::CPlifMatrix::set_plif_use_svm "

set plif use svm

Parameters:
-----------

use_svm:  use svm

num_values:  length of array ";

%feature("docstring")  shogun::CPlifMatrix::set_plif_limits "

set all abscissa values of the support points for the for the pice
wise linear transformations (PLiFs)

Parameters:
-----------

limits:  array of length num_plifs*num_limits

num_plifs:  number of plifs

num_limits:  number of support vectors ";

%feature("docstring")  shogun::CPlifMatrix::set_plif_penalties "

set all ordinate values of the support points for the for the pice
wise linear transformations (PLiFs)

Parameters:
-----------

penalties:  plif values: array of length num_plifs*num_limits

num_plifs:  number of plifs

num_limits:  number of support vectors ";

%feature("docstring")  shogun::CPlifMatrix::set_plif_names "

set names for the PLiFs

Parameters:
-----------

names:  names

num_values:  number of names

maxlen:  maximal string len of the names ";

%feature("docstring")  shogun::CPlifMatrix::set_plif_transform_type "

set plif transform type; for some features the plifs live in log space
therefore the input values have to be transformed to log space before
the transformation can be applied; the transform type is string coded

Parameters:
-----------

transform_type:  transform type (e.g. LOG(x), LOG(x+1), ...)

num_values:  number of transform strings

maxlen:  of transform strings ";

%feature("docstring")  shogun::CPlifMatrix::get_plif_id "

return plif id for idx

Parameters:
-----------

idx:  idx of plif

id of plif ";

%feature("docstring")  shogun::CPlifMatrix::compute_plif_matrix "

parse an 3D array of plif ids and compute the corresponding 2D plif
matrix by subsuming the third dim into one PlifArray; Note: the class
PlifArray is derived from PlifBase. It computes all individual plifs
and sums them up.

Parameters:
-----------

penalties_array:  3D array of plif ids (nofstates x nofstates x
nof(features for each transition))

Dim:  array of dimensions

numDims:  number of dimensions

success ";

%feature("docstring")  shogun::CPlifMatrix::compute_signal_plifs "

parse an 3D array of plif ids and compute the corresponding 3D plif
array;

Parameters:
-----------

state_signals:  mapping of features to states

feat_dim3:  maximal number of features to be considered in one state

num_states:  number of states

success ";

%feature("docstring")
shogun::CPlifMatrix::set_plif_state_signal_matrix "

set best path plif state signal matrix

Parameters:
-----------

plif_id_matrix:  plif id matrix

m:  dimension m of matrix

n:  dimension n of matrix ";

%feature("docstring")  shogun::CPlifMatrix::get_name "

object name ";


// File: classshogun_1_1CSegmentLoss.xml
%feature("docstring") shogun::CSegmentLoss "

class IntronList

C++ includes: SegmentLoss.h ";

%feature("docstring")  shogun::CSegmentLoss::CSegmentLoss "

constructor ";

%feature("docstring")  shogun::CSegmentLoss::~CSegmentLoss "";

%feature("docstring")  shogun::CSegmentLoss::get_segment_loss "

get segment loss for a given range

Parameters:
-----------

from_pos:  start position

to_pos:  end position

segment_id:  type of the segment ";

%feature("docstring")  shogun::CSegmentLoss::get_segment_loss_extend "

get segment loss for a given range

Parameters:
-----------

from_pos:  start position

to_pos:  end position

segment_id:  type of the segment ";

%feature("docstring")  shogun::CSegmentLoss::set_segment_loss "

set best path segment loss

Parameters:
-----------

segment_loss:  segment loss

m:  number of segment id1

n:  number of segment id2 ";

%feature("docstring")  shogun::CSegmentLoss::set_segment_ids "

set best path segmend ids

Parameters:
-----------

segment_ids:  segment ids ";

%feature("docstring")  shogun::CSegmentLoss::set_segment_mask "

mask parts of the sequence such that there is no loss incured there;
this is used if there is uncertainty in the label

Parameters:
-----------

segment_mask:  mask ";

%feature("docstring")  shogun::CSegmentLoss::set_num_segment_types "

set num segment types

Parameters:
-----------

num_segment_types:  num segment types ";

%feature("docstring")  shogun::CSegmentLoss::compute_loss "

compute loss

Parameters:
-----------

all_pos:  all candidate positions

len:  number of positions ";

%feature("docstring")  shogun::CSegmentLoss::get_name "

object name ";


// File: structshogun_1_1CDynProg_1_1svm__values__struct.xml


// File: namespaceshogun.xml


// File: DynProg_8h.xml


// File: IntronList_8h.xml


// File: Plif_8h.xml


// File: PlifArray_8h.xml


// File: PlifBase_8h.xml


// File: PlifMatrix_8h.xml


// File: SegmentLoss_8h.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml


// File: dir_ec750daebb04793913fb9b894ae5dca0.xml

