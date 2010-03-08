
// File: index.xml

// File: classshogun_1_1CAlphabet.xml
%feature("docstring") shogun::CAlphabet "

The class Alphabet implements an alphabet and alphabet utility
functions.

These utility functions can be used to remap characters to more
(bit-)efficient representations, check if a string is valid, compute
histograms etc.

Currently supported alphabets are DNA, RAWDNA, RNA, PROTEIN, BINARY,
ALPHANUM, CUBE, RAW, IUPAC_NUCLEIC_ACID and IUPAC_AMINO_ACID.

C++ includes: Alphabet.h ";

%feature("docstring")  shogun::CAlphabet::CAlphabet "

default constructor constructor

Parameters:
-----------

alpha:  alphabet to use

len:  len ";

%feature("docstring")  shogun::CAlphabet::CAlphabet "

constructor

Parameters:
-----------

alpha:  alphabet (type) to use ";

%feature("docstring")  shogun::CAlphabet::CAlphabet "

constructor

Parameters:
-----------

alpha:  alphabet to use ";

%feature("docstring")  shogun::CAlphabet::~CAlphabet "";

%feature("docstring")  shogun::CAlphabet::set_alphabet "

set alphabet and initialize mapping table (for remap)

Parameters:
-----------

alpha:  new alphabet ";

%feature("docstring")  shogun::CAlphabet::get_alphabet "

get alphabet

alphabet ";

%feature("docstring")  shogun::CAlphabet::get_num_symbols "

get number of symbols in alphabet

number of symbols ";

%feature("docstring")  shogun::CAlphabet::get_num_bits "

get number of bits necessary to store all symbols in alphabet

number of necessary storage bits ";

%feature("docstring")  shogun::CAlphabet::remap_to_bin "

remap element e.g translate ACGT to 0123

Parameters:
-----------

c:  element to remap

remapped element ";

%feature("docstring")  shogun::CAlphabet::remap_to_char "

remap element e.g translate 0123 to ACGT

Parameters:
-----------

c:  element to remap

remapped element ";

%feature("docstring")  shogun::CAlphabet::clear_histogram "

clear histogram ";

%feature("docstring")  shogun::CAlphabet::add_string_to_histogram "

make histogram for whole string

Parameters:
-----------

p:  string

len:  length of string ";

%feature("docstring")  shogun::CAlphabet::add_byte_to_histogram "

add element to histogram

Parameters:
-----------

p:  element ";

%feature("docstring")  shogun::CAlphabet::print_histogram "

print histogram ";

%feature("docstring")  shogun::CAlphabet::get_hist "

get histogram

Parameters:
-----------

h:  where the histogram will be stored

len:  length of histogram ";

%feature("docstring")  shogun::CAlphabet::get_histogram "

get pointer to histogram ";

%feature("docstring")  shogun::CAlphabet::check_alphabet "

check whether symbols in histogram are valid in alphabet e.g. for DNA
if only letters ACGT appear

Parameters:
-----------

print_error:  if errors shall be printed

if symbols in histogram are valid in alphabet ";

%feature("docstring")  shogun::CAlphabet::is_valid "

check whether symbols are valid in alphabet e.g. for DNA if symbol is
one of the A,C,G or T

Parameters:
-----------

c:  symbol

if symbol is a valid character in alphabet ";

%feature("docstring")  shogun::CAlphabet::check_alphabet_size "

check whether symbols in histogram ALL fit in alphabet

Parameters:
-----------

print_error:  if errors shall be printed

if symbols in histogram ALL fit in alphabet ";

%feature("docstring")  shogun::CAlphabet::get_num_symbols_in_histogram
"

return number of symbols in histogram

number of symbols in histogram ";

%feature("docstring")  shogun::CAlphabet::get_max_value_in_histogram "

return maximum value in histogram

maximum value in histogram ";

%feature("docstring")  shogun::CAlphabet::get_num_bits_in_histogram "

return number of bits required to store all symbols in histogram

number of bits required to store all symbols in histogram ";

%feature("docstring")  shogun::CAlphabet::get_name "

object name ";

%feature("docstring")  shogun::CAlphabet::translate_from_single_order
"";

%feature("docstring")  shogun::CAlphabet::translate_from_single_order
"";

%feature("docstring")  shogun::CAlphabet::translate_from_single_order
"";

%feature("docstring")
shogun::CAlphabet::translate_from_single_order_reversed "";

%feature("docstring")
shogun::CAlphabet::translate_from_single_order_reversed "";

%feature("docstring")
shogun::CAlphabet::translate_from_single_order_reversed "";


// File: classshogun_1_1CAttributeFeatures.xml
%feature("docstring") shogun::CAttributeFeatures "

Implements attributed features, that is in the simplest case a number
of (attribute, value) pairs.

For example

x[0...].attr1 = <value(s)> x[0...].attr2 = <value(s)>.

A more complex example would be nested structures
x[0...].attr1[0...].subattr1 = ..

This might be used to represent (attr, value) pairs, simple
structures, trees ...

C++ includes: AttributeFeatures.h ";

%feature("docstring")  shogun::CAttributeFeatures::CAttributeFeatures
"

default constructor ";

%feature("docstring")  shogun::CAttributeFeatures::~CAttributeFeatures
"

destructor ";

%feature("docstring")  shogun::CAttributeFeatures::get_attribute "

return the feature object matching attribute name

Parameters:
-----------

attr_name:  attribute name

feature object ";

%feature("docstring")
shogun::CAttributeFeatures::get_attribute_by_index "

return the feature object at index

Parameters:
-----------

idx:  index of attribute

attr_name:  attribute name (returned by reference)

attr_obj:  attribute object (returned by reference) ";

%feature("docstring")  shogun::CAttributeFeatures::set_attribute "

set the feature object for attribute name

Parameters:
-----------

attr_name:  attribute name

attr_obj:  feature object to set

true on success ";

%feature("docstring")  shogun::CAttributeFeatures::del_attribute "

delete the attribute matching attribute name

Parameters:
-----------

attr_name:  attribute name

true on success ";

%feature("docstring")  shogun::CAttributeFeatures::get_num_attributes
"

get number of attributes

number of attributes ";

%feature("docstring")  shogun::CAttributeFeatures::get_name "

object name ";


// File: classshogun_1_1CCombinedDotFeatures.xml
%feature("docstring") shogun::CCombinedDotFeatures "

Features that allow stacking of a number of DotFeatures.

They transparently provide all the operations of DotFeatures, i.e.

a way to obtain the dimensionality of the feature space, i.e.
$\\\\mbox{dim}({\\\\cal X})$

dot product between feature vectors:

\\\\[r = {\\\\bf x} \\\\cdot {\\\\bf x'}\\\\]

dot product between feature vector and a dense vector ${\\\\bf z}$:

\\\\[r = {\\\\bf x} \\\\cdot {\\\\bf z}\\\\]

multiplication with a scalar $\\\\alpha$ and addition on to a dense
vector ${\\\\bf z}$:

\\\\[{\\\\bf z'} = \\\\alpha {\\\\bf x} + {\\\\bf z}\\\\]

C++ includes: CombinedDotFeatures.h ";

%feature("docstring")
shogun::CCombinedDotFeatures::CCombinedDotFeatures "

constructor ";

%feature("docstring")
shogun::CCombinedDotFeatures::CCombinedDotFeatures "

copy constructor ";

%feature("docstring")
shogun::CCombinedDotFeatures::~CCombinedDotFeatures "

destructor ";

%feature("docstring")  shogun::CCombinedDotFeatures::get_num_vectors "

get number of examples/vectors

abstract base method

number of examples/vectors ";

%feature("docstring")
shogun::CCombinedDotFeatures::get_dim_feature_space "

obtain the dimensionality of the feature space

dimensionality ";

%feature("docstring")  shogun::CCombinedDotFeatures::dot "

compute dot product between vector1 and vector2, appointed by their
indices

Parameters:
-----------

vec_idx1:  index of first vector

vec_idx2:  index of second vector ";

%feature("docstring")  shogun::CCombinedDotFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CCombinedDotFeatures::add_to_dense_vec
"

add vector 1 multiplied with alpha to dense vector2

Parameters:
-----------

alpha:  scalar alpha

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector

abs_val:  if true add the absolute value ";

%feature("docstring")
shogun::CCombinedDotFeatures::get_nnz_features_for_vector "

get number of non-zero features in vector

Parameters:
-----------

num:  which vector

number of non-zero features in vector ";

%feature("docstring")  shogun::CCombinedDotFeatures::get_feature_type
"

get feature type

templated feature type ";

%feature("docstring")  shogun::CCombinedDotFeatures::get_feature_class
"

get feature class

feature class ";

%feature("docstring")  shogun::CCombinedDotFeatures::get_size "

get memory footprint of one feature

abstract base method

memory footprint of one feature ";

%feature("docstring")  shogun::CCombinedDotFeatures::duplicate "

duplicate feature object

feature object ";

%feature("docstring")  shogun::CCombinedDotFeatures::list_feature_objs
"

list feature objects ";

%feature("docstring")
shogun::CCombinedDotFeatures::get_first_feature_obj "

get first feature object

first feature object ";

%feature("docstring")
shogun::CCombinedDotFeatures::get_first_feature_obj "

get first feature object

Parameters:
-----------

current:  list of features

first feature object ";

%feature("docstring")
shogun::CCombinedDotFeatures::get_next_feature_obj "

get next feature object

next feature object ";

%feature("docstring")
shogun::CCombinedDotFeatures::get_next_feature_obj "

get next feature object

Parameters:
-----------

current:  list of features

next feature object ";

%feature("docstring")
shogun::CCombinedDotFeatures::get_last_feature_obj "

get last feature object

last feature object ";

%feature("docstring")
shogun::CCombinedDotFeatures::insert_feature_obj "

insert feature object

Parameters:
-----------

obj:  feature object to insert

if inserting was successful ";

%feature("docstring")
shogun::CCombinedDotFeatures::append_feature_obj "

append feature object

Parameters:
-----------

obj:  feature object to append

if appending was successful ";

%feature("docstring")
shogun::CCombinedDotFeatures::delete_feature_obj "

delete feature object

if deleting was successful ";

%feature("docstring")
shogun::CCombinedDotFeatures::get_num_feature_obj "

get number of feature objects

number of feature objects ";

%feature("docstring")
shogun::CCombinedDotFeatures::get_subfeature_weights "

get subfeature weights

Parameters:
-----------

weights:  subfeature weights

num_weights:  where number of weights is stored ";

%feature("docstring")
shogun::CCombinedDotFeatures::set_subfeature_weights "

set subfeature weights

Parameters:
-----------

weights:  new subfeature weights

num_weights:  number of subfeature weights ";

%feature("docstring")  shogun::CCombinedDotFeatures::get_name "

object name ";


// File: classshogun_1_1CCombinedFeatures.xml
%feature("docstring") shogun::CCombinedFeatures "

The class CombinedFeatures is used to combine a number of of feature
objects into a single CombinedFeatures object.

It keeps pointers to the added sub-features and is especially useful
to combine kernels working on different domains (c.f. CCombinedKernel)
and to combine kernels looking at independent features.

C++ includes: CombinedFeatures.h ";

%feature("docstring")  shogun::CCombinedFeatures::CCombinedFeatures "

default constructor ";

%feature("docstring")  shogun::CCombinedFeatures::CCombinedFeatures "

copy constructor ";

%feature("docstring")  shogun::CCombinedFeatures::duplicate "

duplicate feature object

feature object ";

%feature("docstring")  shogun::CCombinedFeatures::~CCombinedFeatures "

destructor ";

%feature("docstring")  shogun::CCombinedFeatures::get_feature_type "

get feature type

feature type UNKNOWN ";

%feature("docstring")  shogun::CCombinedFeatures::get_feature_class "

get feature class

feature class SIMPLE ";

%feature("docstring")  shogun::CCombinedFeatures::get_num_vectors "

get number of feature vectors

number of feature vectors ";

%feature("docstring")  shogun::CCombinedFeatures::get_size "

get memory footprint of one feature

memory footprint of one feature ";

%feature("docstring")  shogun::CCombinedFeatures::list_feature_objs "

list feature objects ";

%feature("docstring")
shogun::CCombinedFeatures::check_feature_obj_compatibility "

check feature object compatibility

Parameters:
-----------

comb_feat:  feature to check for compatibility

if feature is compatible ";

%feature("docstring")
shogun::CCombinedFeatures::get_first_feature_obj "

get first feature object

first feature object ";

%feature("docstring")
shogun::CCombinedFeatures::get_first_feature_obj "

get first feature object

Parameters:
-----------

current:  list of features

first feature object ";

%feature("docstring")  shogun::CCombinedFeatures::get_next_feature_obj
"

get next feature object

next feature object ";

%feature("docstring")  shogun::CCombinedFeatures::get_next_feature_obj
"

get next feature object

Parameters:
-----------

current:  list of features

next feature object ";

%feature("docstring")  shogun::CCombinedFeatures::get_last_feature_obj
"

get last feature object

last feature object ";

%feature("docstring")  shogun::CCombinedFeatures::insert_feature_obj "

insert feature object

Parameters:
-----------

obj:  feature object to insert

if inserting was successful ";

%feature("docstring")  shogun::CCombinedFeatures::append_feature_obj "

append feature object

Parameters:
-----------

obj:  feature object to append

if appending was successful ";

%feature("docstring")  shogun::CCombinedFeatures::delete_feature_obj "

delete feature object

if deleting was successful ";

%feature("docstring")  shogun::CCombinedFeatures::get_num_feature_obj
"

get number of feature objects

number of feature objects ";

%feature("docstring")  shogun::CCombinedFeatures::get_name "

object name ";


// File: classshogun_1_1CDotFeatures.xml
%feature("docstring") shogun::CDotFeatures "

Features that support dot products among other operations.

DotFeatures support the following operations:

a way to obtain the dimensionality of the feature space, i.e.
$\\\\mbox{dim}({\\\\cal X})$

dot product between feature vectors:

\\\\[r = {\\\\bf x} \\\\cdot {\\\\bf x'}\\\\]

dot product between feature vector and a dense vector ${\\\\bf z}$:

\\\\[r = {\\\\bf x} \\\\cdot {\\\\bf z}\\\\]

multiplication with a scalar $\\\\alpha$ and addition on to a dense
vector ${\\\\bf z}$:

\\\\[{\\\\bf z'} = \\\\alpha {\\\\bf x} + {\\\\bf z}\\\\]

C++ includes: DotFeatures.h ";

%feature("docstring")  shogun::CDotFeatures::CDotFeatures "

constructor

Parameters:
-----------

size:  cache size ";

%feature("docstring")  shogun::CDotFeatures::CDotFeatures "

copy constructor ";

%feature("docstring")  shogun::CDotFeatures::CDotFeatures "

constructor

Parameters:
-----------

fname:  filename to load features from ";

%feature("docstring")  shogun::CDotFeatures::~CDotFeatures "";

%feature("docstring")  shogun::CDotFeatures::get_dim_feature_space "

obtain the dimensionality of the feature space

(not mix this up with the dimensionality of the input space, usually
obtained via get_num_features())

dimensionality ";

%feature("docstring")  shogun::CDotFeatures::dot "

compute dot product between vector1 and vector2, appointed by their
indices

Parameters:
-----------

vec_idx1:  index of first vector

vec_idx2:  index of second vector ";

%feature("docstring")  shogun::CDotFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CDotFeatures::add_to_dense_vec "

add vector 1 multiplied with alpha to dense vector2

Parameters:
-----------

alpha:  scalar alpha

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector

abs_val:  if true add the absolute value ";

%feature("docstring")  shogun::CDotFeatures::dense_dot_range "

Compute the dot product for a range of vectors. This function makes
use of dense_dot alphas[i] * sparse[i]^T * w + b

Parameters:
-----------

output:  result for the given vector range

start:  start vector range from this idx

stop:  stop vector range at this idx

alphas:  scalars to multiply with, may be NULL

vec:  dense vector to compute dot product with

dim:  length of the dense vector

b:  bias ";

%feature("docstring")
shogun::CDotFeatures::get_nnz_features_for_vector "

get number of non-zero features in vector

(in case accurate estimates are too expensive overestimating is OK)

Parameters:
-----------

num:  which vector

number of sparse features in vector ";

%feature("docstring")
shogun::CDotFeatures::get_combined_feature_weight "

get combined feature weight

combined feature weight ";

%feature("docstring")
shogun::CDotFeatures::set_combined_feature_weight "

set combined kernel weight

Parameters:
-----------

nw:  new combined feature weight ";

%feature("docstring")  shogun::CDotFeatures::get_feature_matrix "

get a copy of the feature matrix (in feature space)
num_feat,num_vectors are returned by reference

Parameters:
-----------

dst:  destination to store matrix in

num_feat:  number of features (rows of matrix)

num_vec:  number of vectors (columns of matrix) ";


// File: classshogun_1_1CDummyFeatures.xml
%feature("docstring") shogun::CDummyFeatures "

The class DummyFeatures implements features that only know the number
of feature objects (but don't actually contain any).

This is used in the CCustomKernel.

C++ includes: DummyFeatures.h ";

%feature("docstring")  shogun::CDummyFeatures::CDummyFeatures "

constructor

Parameters:
-----------

num:  number of feature vectors ";

%feature("docstring")  shogun::CDummyFeatures::CDummyFeatures "

copy constructor ";

%feature("docstring")  shogun::CDummyFeatures::~CDummyFeatures "

destructor ";

%feature("docstring")  shogun::CDummyFeatures::get_num_vectors "

get number of feature vectors ";

%feature("docstring")  shogun::CDummyFeatures::get_size "

get size of features (always 1) ";

%feature("docstring")  shogun::CDummyFeatures::duplicate "

duplicate features ";

%feature("docstring")  shogun::CDummyFeatures::get_feature_type "

get feature type (ANY) ";

%feature("docstring")  shogun::CDummyFeatures::get_feature_class "

get feature class (ANY) ";

%feature("docstring")  shogun::CDummyFeatures::get_name "

object name ";


// File: classshogun_1_1CExplicitSpecFeatures.xml
%feature("docstring") shogun::CExplicitSpecFeatures "

Features that compute the Spectrum Kernel feature space explicitly.

See:  CCommWordStringKernel

C++ includes: ExplicitSpecFeatures.h ";

%feature("docstring")
shogun::CExplicitSpecFeatures::CExplicitSpecFeatures "

constructor

Parameters:
-----------

str:  stringfeatures (of words)

normalize:  whether to use sqrtdiag normalization ";

%feature("docstring")
shogun::CExplicitSpecFeatures::CExplicitSpecFeatures "

copy constructor ";

%feature("docstring")
shogun::CExplicitSpecFeatures::~CExplicitSpecFeatures "

destructor ";

%feature("docstring")  shogun::CExplicitSpecFeatures::duplicate "

duplicate feature object

feature object ";

%feature("docstring")
shogun::CExplicitSpecFeatures::get_dim_feature_space "

obtain the dimensionality of the feature space

(not mix this up with the dimensionality of the input space, usually
obtained via get_num_features())

dimensionality ";

%feature("docstring")  shogun::CExplicitSpecFeatures::dot "

compute dot product between vector1 and vector2, appointed by their
indices

Parameters:
-----------

vec_idx1:  index of first vector

vec_idx2:  index of second vector ";

%feature("docstring")  shogun::CExplicitSpecFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CExplicitSpecFeatures::add_to_dense_vec
"

add vector 1 multiplied with alpha to dense vector2

Parameters:
-----------

alpha:  scalar alpha

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector

abs_val:  if true add the absolute value ";

%feature("docstring")
shogun::CExplicitSpecFeatures::get_nnz_features_for_vector "

get number of non-zero features in vector

Parameters:
-----------

num:  which vector

number of non-zero features in vector ";

%feature("docstring")  shogun::CExplicitSpecFeatures::get_feature_type
"

get feature type

templated feature type ";

%feature("docstring")
shogun::CExplicitSpecFeatures::get_feature_class "

get feature class

feature class ";

%feature("docstring")  shogun::CExplicitSpecFeatures::get_num_vectors
"

get number of strings

number of strings ";

%feature("docstring")  shogun::CExplicitSpecFeatures::get_size "

get size of one element

size of one element ";

%feature("docstring")  shogun::CExplicitSpecFeatures::get_name "

object name ";


// File: classshogun_1_1CFeatures.xml
%feature("docstring") shogun::CFeatures "

The class Features is the base class of all feature objects.

It can be understood as a dense real valued feature matrix (with e.g.
columns as single feature vectors), a set of strings, graphs or any
other arbitrary collection of objects. As a result this class is kept
very general and implements only very weak interfaces to

duplicate the Feature object

obtain the feature type (like F_DREAL, F_SHORT ...)

obtain the feature class (like Simple dense matrices, sparse or
strings)

obtain the number of feature \"vectors\"

In addition it provides helpers to check e.g. for compability of
feature objects.

Currently there are 3 general feature classes, which are
CSimpleFeatures (dense matrices), CSparseFeatures (sparse matrices),
CStringFeatures (a set of strings) from which all the specific
features like CSimpleFeatures<float64_t> (dense real valued feature
matrices) are derived.

C++ includes: Features.h ";

%feature("docstring")  shogun::CFeatures::CFeatures "

constructor

Parameters:
-----------

size:  cache size ";

%feature("docstring")  shogun::CFeatures::CFeatures "

copy constructor ";

%feature("docstring")  shogun::CFeatures::CFeatures "

constructor

Parameters:
-----------

fname:  filename to load features from ";

%feature("docstring")  shogun::CFeatures::duplicate "

duplicate feature object

abstract base method

feature object ";

%feature("docstring")  shogun::CFeatures::~CFeatures "";

%feature("docstring")  shogun::CFeatures::get_feature_type "

get feature type

abstract base method

templated feature type ";

%feature("docstring")  shogun::CFeatures::get_feature_class "

get feature class

abstract base method

feature class like STRING, SIMPLE, SPARSE... ";

%feature("docstring")  shogun::CFeatures::add_preproc "

add preprocessor

Parameters:
-----------

p:  preprocessor to set

something inty ";

%feature("docstring")  shogun::CFeatures::del_preproc "

delete preprocessor from list caller has to clean up returned preproc

Parameters:
-----------

num:  index of preprocessor in list ";

%feature("docstring")  shogun::CFeatures::get_preproc "

get specified preprocessor

Parameters:
-----------

num:  index of preprocessor in list ";

%feature("docstring")  shogun::CFeatures::set_preprocessed "

set applied flag for preprocessor

Parameters:
-----------

num:  index of preprocessor in list ";

%feature("docstring")  shogun::CFeatures::is_preprocessed "

get whether specified preprocessor was already applied

Parameters:
-----------

num:  index of preprocessor in list ";

%feature("docstring")  shogun::CFeatures::get_num_preprocessed "

get the number of applied preprocs

number of applied preprocessors ";

%feature("docstring")  shogun::CFeatures::get_num_preproc "

get number of preprocessors

number of preprocessors ";

%feature("docstring")  shogun::CFeatures::clean_preprocs "

clears all preprocs ";

%feature("docstring")  shogun::CFeatures::get_cache_size "

get cache size

cache size ";

%feature("docstring")  shogun::CFeatures::get_num_vectors "

get number of examples/vectors

abstract base method

number of examples/vectors ";

%feature("docstring")  shogun::CFeatures::reshape "

in case there is a feature matrix allow for reshaping

NOT IMPLEMENTED!

Parameters:
-----------

num_features:  new number of features

num_vectors:  new number of vectors

if reshaping was successful ";

%feature("docstring")  shogun::CFeatures::get_size "

get memory footprint of one feature

abstract base method

memory footprint of one feature ";

%feature("docstring")  shogun::CFeatures::list_feature_obj "

list feature object ";

%feature("docstring")  shogun::CFeatures::load "

load features from file

Parameters:
-----------

fname:  filename to load from

if loading was successful ";

%feature("docstring")  shogun::CFeatures::save "

save features to file

Parameters:
-----------

fname:  filename to save to

if saving was successful ";

%feature("docstring")  shogun::CFeatures::check_feature_compatibility
"

check feature compatibility

Parameters:
-----------

f:  features to check for compatibility

if features are compatible ";

%feature("docstring")  shogun::CFeatures::has_property "

check if features have given property

Parameters:
-----------

p:  feature property

if features have given property ";

%feature("docstring")  shogun::CFeatures::set_property "

set property

Parameters:
-----------

p:  kernel property to set ";

%feature("docstring")  shogun::CFeatures::unset_property "

unset property

Parameters:
-----------

p:  kernel property to unset ";


// File: classshogun_1_1CFKFeatures.xml
%feature("docstring") shogun::CFKFeatures "

The class FKFeatures implements Fischer kernel features obtained from
two Hidden Markov models.

It was used in

K. Tsuda, M. Kawanabe, G. Raetsch, S. Sonnenburg, and K.R. Mueller. A
new discriminative kernel from probabilistic models. Neural
Computation, 14:2397-2414, 2002.

which also has the details.

Note that FK-features are computed on the fly, so to be effective
feature caching should be enabled.

It inherits its functionality from CSimpleFeatures, which should be
consulted for further reference.

C++ includes: FKFeatures.h ";

%feature("docstring")  shogun::CFKFeatures::CFKFeatures "

constructor

Parameters:
-----------

size:  cache size

p:  positive HMM

n:  negative HMM ";

%feature("docstring")  shogun::CFKFeatures::CFKFeatures "

copy constructor ";

%feature("docstring")  shogun::CFKFeatures::~CFKFeatures "";

%feature("docstring")  shogun::CFKFeatures::set_models "

set HMMs

Parameters:
-----------

p:  positive HMM

n:  negative HMM ";

%feature("docstring")  shogun::CFKFeatures::set_a "

set weight a

Parameters:
-----------

a:  weight a ";

%feature("docstring")  shogun::CFKFeatures::get_a "

get weight a

weight a ";

%feature("docstring")  shogun::CFKFeatures::set_feature_matrix "

set feature matrix

something floaty ";

%feature("docstring")  shogun::CFKFeatures::set_opt_a "

set opt a

Parameters:
-----------

a:  a

something floaty ";

%feature("docstring")  shogun::CFKFeatures::get_weight_a "

get weight_a

weight_a ";

%feature("docstring")  shogun::CFKFeatures::get_name "

object name ";


// File: classshogun_1_1CImplicitWeightedSpecFeatures.xml
%feature("docstring") shogun::CImplicitWeightedSpecFeatures "

Features that compute the Weighted Spectrum Kernel feature space
explicitly.

See:  CWeightedCommWordStringKernel

C++ includes: ImplicitWeightedSpecFeatures.h ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::CImplicitWeightedSpecFeatures "

constructor

Parameters:
-----------

str:  stringfeatures (of words)

normalize:  whether to use sqrtdiag normalization ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::CImplicitWeightedSpecFeatures "

copy constructor ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::~CImplicitWeightedSpecFeatures
"";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::duplicate "

duplicate feature object

feature object ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::get_dim_feature_space "

obtain the dimensionality of the feature space

(not mix this up with the dimensionality of the input space, usually
obtained via get_num_features())

dimensionality ";

%feature("docstring")  shogun::CImplicitWeightedSpecFeatures::dot "

compute dot product between vector1 and vector2, appointed by their
indices

Parameters:
-----------

vec_idx1:  index of first vector

vec_idx2:  index of second vector ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::add_to_dense_vec "

add vector 1 multiplied with alpha to dense vector2

Parameters:
-----------

alpha:  scalar alpha

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector

abs_val:  if true add the absolute value ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::get_nnz_features_for_vector "

get number of non-zero features in vector

Parameters:
-----------

num:  which vector

number of non-zero features in vector ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::get_feature_type "

get feature type

templated feature type ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::get_feature_class "

get feature class

feature class ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::get_num_vectors "

get number of strings

number of strings ";

%feature("docstring")  shogun::CImplicitWeightedSpecFeatures::get_size
"

get size of one element

size of one element ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::set_wd_weights "

set weighted degree weights

if setting was successful ";

%feature("docstring")
shogun::CImplicitWeightedSpecFeatures::set_weights "

set custom weights (swig compatible)

Parameters:
-----------

w:  weights

d:  degree (must match number of weights)

if setting was successful ";

%feature("docstring")  shogun::CImplicitWeightedSpecFeatures::get_name
"

object name ";


// File: classshogun_1_1CLabels.xml
%feature("docstring") shogun::CLabels "

The class Labels models labels, i.e. class assignments of objects.

Labels here are always real-valued and thus applicable to
classification (cf. CClassifier) and regression (cf. CRegression)
problems.

C++ includes: Labels.h ";

%feature("docstring")  shogun::CLabels::CLabels "

default constructor ";

%feature("docstring")  shogun::CLabels::CLabels "

constructor

Parameters:
-----------

num_labels:  number of labels ";

%feature("docstring")  shogun::CLabels::CLabels "

constructor

Parameters:
-----------

src:  labels to set

len:  number of labels ";

%feature("docstring")  shogun::CLabels::CLabels "

constructor

Parameters:
-----------

fname:  filename to load labels from ";

%feature("docstring")  shogun::CLabels::~CLabels "";

%feature("docstring")  shogun::CLabels::load "

load labels from file

Parameters:
-----------

fname:  filename to load from

if loading was successful ";

%feature("docstring")  shogun::CLabels::save "

save labels to file

Parameters:
-----------

fname:  filename to save to

if saving was successful ";

%feature("docstring")  shogun::CLabels::set_label "

set label

Parameters:
-----------

idx:  index of label to set

label:  value of label

if setting was successful ";

%feature("docstring")  shogun::CLabels::set_int_label "

set INT label

Parameters:
-----------

idx:  index of label to set

label:  INT value of label

if setting was successful ";

%feature("docstring")  shogun::CLabels::get_label "

get label

Parameters:
-----------

idx:  index of label to get

value of label ";

%feature("docstring")  shogun::CLabels::get_int_label "

get INT label

Parameters:
-----------

idx:  index of label to get

INT value of label ";

%feature("docstring")  shogun::CLabels::is_two_class_labeling "

is two-class labeling

if this is two-class labeling ";

%feature("docstring")  shogun::CLabels::get_num_classes "

return number of classes (for multiclass) labels have to be zero based
0,1,...C missing labels are illegal

number of classes ";

%feature("docstring")  shogun::CLabels::get_labels "

get labels caller has to clean up

Parameters:
-----------

len:  number of labels

the labels ";

%feature("docstring")  shogun::CLabels::get_labels "

get labels (swig compatible)

Parameters:
-----------

dst:  where labels will be stored in

len:  where number of labels will be stored in ";

%feature("docstring")  shogun::CLabels::set_labels "

set labels

Parameters:
-----------

src:  labels to set

len:  number of labels ";

%feature("docstring")  shogun::CLabels::get_int_labels "

get INT label vector caller has to clean up

Parameters:
-----------

len:  number of labels to get

INT labels ";

%feature("docstring")  shogun::CLabels::set_int_labels "

set INT labels caller has to clean up

Parameters:
-----------

labels:  INT labels

len:  number of INT labels ";

%feature("docstring")  shogun::CLabels::get_num_labels "

get number of labels

number of labels ";

%feature("docstring")  shogun::CLabels::get_name "

object name ";


// File: classshogun_1_1CPolyFeatures.xml
%feature("docstring") shogun::CPolyFeatures "

implement DotFeatures for the polynomial kernel

see DotFeatures for further discription

C++ includes: PolyFeatures.h ";

%feature("docstring")  shogun::CPolyFeatures::CPolyFeatures "

constructor

Parameters:
-----------

feat:  real features

degree:  degree of the polynomial kernel

normalize:  normalize kernel ";

%feature("docstring")  shogun::CPolyFeatures::~CPolyFeatures "";

%feature("docstring")  shogun::CPolyFeatures::CPolyFeatures "

copy constructor

not implemented!

Parameters:
-----------

orig:  original PolyFeature ";

%feature("docstring")  shogun::CPolyFeatures::get_dim_feature_space "

get dimensions of feature space

dimensions of feature space ";

%feature("docstring")
shogun::CPolyFeatures::get_nnz_features_for_vector "

get number of non-zero features in vector

Parameters:
-----------

num:  index of vector

number of non-zero features in vector ";

%feature("docstring")  shogun::CPolyFeatures::get_feature_type "

get feature type

feature type ";

%feature("docstring")  shogun::CPolyFeatures::get_feature_class "

get feature class

feature class ";

%feature("docstring")  shogun::CPolyFeatures::get_num_vectors "

get number of vectors

number of vectors ";

%feature("docstring")  shogun::CPolyFeatures::dot "

compute dot product between vector1 and vector2, appointed by their
indices

Parameters:
-----------

vec_idx1:  index of first vector

vec_idx2:  index of second vector ";

%feature("docstring")  shogun::CPolyFeatures::get_size "

size ";

%feature("docstring")  shogun::CPolyFeatures::duplicate "

duplicate feature object

feature object ";

%feature("docstring")  shogun::CPolyFeatures::get_name "

name of class ";

%feature("docstring")  shogun::CPolyFeatures::dense_dot "

compute dot product of vector with index arg1 with an given second
vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  second vector

vec2_len:  length of second vector ";

%feature("docstring")  shogun::CPolyFeatures::add_to_dense_vec "

compute alpha*x+vec2

Parameters:
-----------

alpha:  alpha

vec_idx1:  index of first vector x

vec2:  vec2

vec2_len:  length of vec2

abs_val:  if true add the absolute value ";


// File: classshogun_1_1CRealFileFeatures.xml
%feature("docstring") shogun::CRealFileFeatures "

The class RealFileFeatures implements a dense double-precision
floating point matrix from a file.

It inherits its functionality from CSimpleFeatures, which should be
consulted for further reference.

C++ includes: RealFileFeatures.h ";

%feature("docstring")  shogun::CRealFileFeatures::CRealFileFeatures "

constructor

Parameters:
-----------

size:  cache size

file:  file to load features from ";

%feature("docstring")  shogun::CRealFileFeatures::CRealFileFeatures "

constructor

Parameters:
-----------

size:  cache size

filename:  filename to load features from ";

%feature("docstring")  shogun::CRealFileFeatures::CRealFileFeatures "

copy constructor ";

%feature("docstring")  shogun::CRealFileFeatures::~CRealFileFeatures "";

%feature("docstring")  shogun::CRealFileFeatures::load_feature_matrix
"

load feature matrix

loaded feature matrix ";

%feature("docstring")  shogun::CRealFileFeatures::get_label "

get label at given index

Parameters:
-----------

idx:  index to look at

label at given index ";

%feature("docstring")  shogun::CRealFileFeatures::get_name "

object name ";


// File: classshogun_1_1CSimpleFeatures.xml
%feature("docstring") shogun::CSimpleFeatures "

The class SimpleFeatures implements dense feature matrices.

The feature matrices are stored en-block in memory in fortran order,
i.e. column-by-column, where a column denotes a feature vector.

There are get_num_vectors() many feature vectors, of dimension
get_num_features(). To access a feature vector call
get_feature_vector() and when you are done treating it call
free_feature_vector(). While free_feature_vector() is a NOP in most
cases feature vectors might have been generated on the fly (due to a
number preprocessors being attached to them).

From this template class a number the following dense feature matrix
types are used and supported:

bool matrix - CSimpleFeatures<bool>

8bit char matrix - CSimpleFeatures<char>

8bit Byte matrix - CSimpleFeatures<uint8_t>

16bit Integer matrix - CSimpleFeatures<int16_t>

16bit Word matrix - CSimpleFeatures<uint16_t>

32bit Integer matrix - CSimpleFeatures<int32_t>

32bit Unsigned Integer matrix - CSimpleFeatures<uint32_t>

32bit Float matrix - CSimpleFeatures<float32_t>

64bit Float matrix - CSimpleFeatures<float64_t>

64bit Float matrix in a file - CRealFileFeatures

64bit Tangent of posterior log-odds (TOP) features from HMM -
CTOPFeatures

64bit Fisher Kernel (FK) features from HMM - CTOPFeatures

96bit Float matrix - CSimpleFeatures<floatmax_t>

C++ includes: SimpleFeatures.h ";

%feature("docstring")  shogun::CSimpleFeatures::CSimpleFeatures "

constructor

Parameters:
-----------

size:  cache size ";

%feature("docstring")  shogun::CSimpleFeatures::CSimpleFeatures "

copy constructor ";

%feature("docstring")  shogun::CSimpleFeatures::CSimpleFeatures "

constructor

Parameters:
-----------

src:  feature matrix

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix ";

%feature("docstring")  shogun::CSimpleFeatures::CSimpleFeatures "

constructor

NOT IMPLEMENTED!

Parameters:
-----------

fname:  filename to load features from ";

%feature("docstring")  shogun::CSimpleFeatures::duplicate "

duplicate feature object

feature object ";

%feature("docstring")  shogun::CSimpleFeatures::~CSimpleFeatures "";

%feature("docstring")  shogun::CSimpleFeatures::free_feature_matrix "

free feature matrix ";

%feature("docstring")  shogun::CSimpleFeatures::free_features "

free feature matrix and cache ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_vector "

get feature vector for sample num from the matrix as it is if matrix
is initialized, else return preprocessed compute_feature_vector

Parameters:
-----------

num:  index of feature vector

len:  length is returned by reference

dofree:  whether returned vector must be freed by caller via
free_feature_vector

feature vector ";

%feature("docstring")  shogun::CSimpleFeatures::set_feature_vector "

set feature vector num

( only available in-memory feature matrices )

Parameters:
-----------

src:  vector

len:  length of vector

num:  index where to put vector to ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_vector "

get feature vector num

Parameters:
-----------

dst:  destination to store vector in

len:  length of vector

num:  index of vector ";

%feature("docstring")  shogun::CSimpleFeatures::free_feature_vector "

free feature vector

Parameters:
-----------

feat_vec:  feature vector to free

num:  index in feature cache

dofree:  if vector should be really deleted ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_matrix "

get a copy of the feature matrix num_feat,num_vectors are returned by
reference

Parameters:
-----------

dst:  destination to store matrix in

num_feat:  number of features (rows of matrix)

num_vec:  number of vectors (columns of matrix) ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_matrix "

get the pointer to the feature matrix num_feat,num_vectors are
returned by reference

Parameters:
-----------

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix

feature matrix ";

%feature("docstring")  shogun::CSimpleFeatures::set_feature_matrix "

set feature matrix necessary to set feature_matrix, num_features,
num_vectors, where num_features is the column offset, and columns are
linear in memory see below for definition of feature_matrix

Parameters:
-----------

fm:  feature matrix to se

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix ";

%feature("docstring")  shogun::CSimpleFeatures::copy_feature_matrix "

copy feature matrix store copy of feature_matrix, where num_features
is the column offset, and columns are linear in memory see below for
definition of feature_matrix

Parameters:
-----------

src:  feature matrix to copy

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix ";

%feature("docstring")  shogun::CSimpleFeatures::apply_preproc "

apply preprocessor

Parameters:
-----------

force_preprocessing:  if preprocssing shall be forced

if applying was successful ";

%feature("docstring")  shogun::CSimpleFeatures::get_size "

get memory footprint of one feature

memory footprint of one feature ";

%feature("docstring")  shogun::CSimpleFeatures::get_num_vectors "

get number of feature vectors

number of feature vectors ";

%feature("docstring")  shogun::CSimpleFeatures::get_num_features "

get number of features

number of features ";

%feature("docstring")  shogun::CSimpleFeatures::set_num_features "

set number of features

Parameters:
-----------

num:  number to set ";

%feature("docstring")  shogun::CSimpleFeatures::set_num_vectors "

set number of vectors

Parameters:
-----------

num:  number to set ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_class "

get feature class

feature class SIMPLE ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type

templated feature type ";

%feature("docstring")  shogun::CSimpleFeatures::reshape "

reshape

Parameters:
-----------

p_num_features:  new number of features

p_num_vectors:  new number of vectors

if reshaping was successful ";

%feature("docstring")  shogun::CSimpleFeatures::get_dim_feature_space
"

obtain the dimensionality of the feature space

(not mix this up with the dimensionality of the input space, usually
obtained via get_num_features())

dimensionality ";

%feature("docstring")  shogun::CSimpleFeatures::dot "

compute dot product between vector1 and vector2, appointed by their
indices

Parameters:
-----------

vec_idx1:  index of first vector

vec_idx2:  index of second vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::add_to_dense_vec "

add vector 1 multiplied with alpha to dense vector2

Parameters:
-----------

alpha:  scalar alpha

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector

abs_val:  if true add the absolute value ";

%feature("docstring")
shogun::CSimpleFeatures::get_nnz_features_for_vector "

get number of non-zero features in vector

Parameters:
-----------

num:  which vector

number of non-zero features in vector ";

%feature("docstring")  shogun::CSimpleFeatures::Align_char_features "

align char features

Parameters:
-----------

cf:  char features

Ref:  other char features

gapCost:  gap cost

if aligning was successful ";

%feature("docstring")  shogun::CSimpleFeatures::load "

load features from file

Parameters:
-----------

fname:  filename to load from

if loading was successful ";

%feature("docstring")  shogun::CSimpleFeatures::save "

save features to file

Parameters:
-----------

fname:  filename to save to

if saving was successful ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the BOOL feature can deal with

feature type BOOL ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the CHAR feature can deal with

feature type CHAR ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the BYTE feature can deal with

feature type BYTE ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the SHORT feature can deal with

feature type SHORT ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the WORD feature can deal with

feature type WORD ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the INT feature can deal with

feature type INT ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the UINT feature can deal with

feature type UINT ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the LONG feature can deal with

feature type LONG ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the ULONG feature can deal with

feature type ULONG ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the SHORTREAL feature can deal with

feature type SHORTREAL ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the DREAL feature can deal with

feature type DREAL ";

%feature("docstring")  shogun::CSimpleFeatures::get_feature_type "

get feature type the LONGREAL feature can deal with

feature type LONGREAL ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSimpleFeatures::Align_char_features "

align strings and compute emperical kernel map based on alignment
scores

non functional code - needs updating

Parameters:
-----------

cf:  strings to be aligned to reference

Ref:  reference strings to be aligned to

gapCost:  costs for a gap ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSimpleFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";


// File: classshogun_1_1CSparseFeatures.xml
%feature("docstring") shogun::CSparseFeatures "

Template class SparseFeatures implements sparse matrices.

Features are an array of TSparse, sorted w.r.t. vec_index (increasing)
and withing same vec_index w.r.t. feat_index (increasing);

Sparse feature vectors can be accessed via get_sparse_feature_vector()
and should be freed (this operation is a NOP in most cases) via
free_sparse_feature_vector().

As this is a template class it can directly be used for different data
types like sparse matrices of real valued, integer, byte etc type.

C++ includes: SparseFeatures.h ";

%feature("docstring")  shogun::CSparseFeatures::CSparseFeatures "

constructor

Parameters:
-----------

size:  cache size ";

%feature("docstring")  shogun::CSparseFeatures::CSparseFeatures "

convenience constructor that creates sparse features from the ones
passed as argument

Parameters:
-----------

src:  dense feature matrix

num_feat:  number of features

num_vec:  number of vectors

copy:  true to copy feature matrix ";

%feature("docstring")  shogun::CSparseFeatures::CSparseFeatures "

convenience constructor that creates sparse features from dense
features

Parameters:
-----------

src:  dense feature matrix

num_feat:  number of features

num_vec:  number of vectors ";

%feature("docstring")  shogun::CSparseFeatures::CSparseFeatures "

copy constructor ";

%feature("docstring")  shogun::CSparseFeatures::CSparseFeatures "

constructor

Parameters:
-----------

fname:  filename to load features from ";

%feature("docstring")  shogun::CSparseFeatures::~CSparseFeatures "";

%feature("docstring")
shogun::CSparseFeatures::free_sparse_feature_matrix "

free sparse feature matrix ";

%feature("docstring")  shogun::CSparseFeatures::free_sparse_features "

free sparse feature matrix and cache ";

%feature("docstring")  shogun::CSparseFeatures::duplicate "

duplicate feature object

feature object ";

%feature("docstring")  shogun::CSparseFeatures::get_feature "

get a single feature

Parameters:
-----------

num:  number of feature vector to retrieve

index:  index of feature in this vector

sum of features that match dimension index and 0 if none is found ";

%feature("docstring")
shogun::CSparseFeatures::get_full_feature_vector "

converts a sparse feature vector into a dense one preprocessed
compute_feature_vector caller cleans up

Parameters:
-----------

num:  index of feature vector

len:  length is returned by reference

dense feature vector ";

%feature("docstring")
shogun::CSparseFeatures::get_full_feature_vector "

get the fully expanded dense feature vector num

Parameters:
-----------

dst:  feature vector

len:  length is returned by reference

num:  index of feature vector ";

%feature("docstring")
shogun::CSparseFeatures::get_nnz_features_for_vector "

get number of non-zero features in vector

Parameters:
-----------

num:  which vector

number of non-zero features in vector ";

%feature("docstring")
shogun::CSparseFeatures::get_sparse_feature_vector "

get sparse feature vector for sample num from the matrix as it is if
matrix is initialized, else return preprocessed compute_feature_vector

Parameters:
-----------

num:  index of feature vector

len:  number of sparse entries is returned by reference

vfree:  whether returned vector must be freed by caller via
free_sparse_feature_vector

sparse feature vector ";

%feature("docstring")  shogun::CSparseFeatures::sparse_dot "

compute the dot product between two sparse feature vectors alpha *
vec^T * vec

Parameters:
-----------

alpha:  scalar to multiply with

avec:  first sparse feature vector

alen:  avec's length

bvec:  second sparse feature vector

blen:  bvec's length

dot product between the two sparse feature vectors ";

%feature("docstring")  shogun::CSparseFeatures::dense_dot "

compute the dot product between dense weights and a sparse feature
vector alpha * sparse^T * w + b

Parameters:
-----------

alpha:  scalar to multiply with

num:  index of feature vector

vec:  dense vector to compute dot product with

dim:  length of the dense vector

b:  bias

dot product between dense weights and a sparse feature vector ";

%feature("docstring")  shogun::CSparseFeatures::add_to_dense_vec "

add a sparse feature vector onto a dense one dense+=alpha*sparse

Parameters:
-----------

alpha:  scalar to multiply with

num:  index of feature vector

vec:  dense vector

dim:  length of the dense vector

abs_val:  if true, do dense+=alpha*abs(sparse) ";

%feature("docstring")
shogun::CSparseFeatures::free_sparse_feature_vector "

free sparse feature vector

Parameters:
-----------

feat_vec:  feature vector to free

num:  index of this vector in the cache

free:  if vector should be really deleted ";

%feature("docstring")
shogun::CSparseFeatures::get_sparse_feature_matrix "

get the pointer to the sparse feature matrix num_feat,num_vectors are
returned by reference

Parameters:
-----------

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix

feature matrix ";

%feature("docstring")
shogun::CSparseFeatures::get_sparse_feature_matrix "

get the pointer to the sparse feature matrix (swig compatible)
num_feat,num_vectors are returned by reference

Parameters:
-----------

dst:  feature matrix

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix

nnz:  number of nonzero elements ";

%feature("docstring")  shogun::CSparseFeatures::clean_tsparse "

clean TSparse

Parameters:
-----------

sfm:  sparse feature matrix

num_vec:  number of vectors in matrix ";

%feature("docstring")  shogun::CSparseFeatures::get_transposed "

compute and return the transpose of the sparse feature matrix which
will be prepocessed. num_feat, num_vectors are returned by reference
caller has to clean up

Parameters:
-----------

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix

transposed sparse feature matrix ";

%feature("docstring")
shogun::CSparseFeatures::set_sparse_feature_matrix "

set feature matrix necessary to set feature_matrix, num_features,
num_vectors, where num_features is the column offset, and columns are
linear in memory see below for definition of feature_matrix

Parameters:
-----------

src:  new sparse feature matrix

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix ";

%feature("docstring")
shogun::CSparseFeatures::get_full_feature_matrix "

gets a copy of a full feature matrix num_feat,num_vectors are returned
by reference

Parameters:
-----------

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix

full feature matrix ";

%feature("docstring")
shogun::CSparseFeatures::get_full_feature_matrix "

gets a copy of a full feature matrix (swig compatible)
num_feat,num_vectors are returned by reference

Parameters:
-----------

dst:  full feature matrix

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix ";

%feature("docstring")
shogun::CSparseFeatures::set_full_feature_matrix "

creates a sparse feature matrix from a full dense feature matrix
necessary to set feature_matrix, num_features and num_vectors where
num_features is the column offset, and columns are linear in memory
see above for definition of sparse_feature_matrix

Parameters:
-----------

src:  full feature matrix

num_feat:  number of features in matrix

num_vec:  number of vectors in matrix ";

%feature("docstring")  shogun::CSparseFeatures::apply_preproc "

apply preprocessor

Parameters:
-----------

force_preprocessing:  if preprocssing shall be forced

if applying was successful ";

%feature("docstring")  shogun::CSparseFeatures::get_size "

get memory footprint of one feature

memory footprint of one feature ";

%feature("docstring")  shogun::CSparseFeatures::obtain_from_simple "

obtain sparse features from simple features

Parameters:
-----------

sf:  simple features

if obtaining was successful ";

%feature("docstring")  shogun::CSparseFeatures::get_num_vectors "

get number of feature vectors

number of feature vectors ";

%feature("docstring")  shogun::CSparseFeatures::get_num_features "

get number of features

number of features ";

%feature("docstring")  shogun::CSparseFeatures::set_num_features "

set number of features

Sometimes when loading sparse features not all possible dimensions are
used. This may pose a problem to classifiers when being applied to
higher dimensional test-data. This function allows to artificially
explode the feature space

Parameters:
-----------

num:  the number of features, must be larger than the current number
of features

previous number of features ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_class "

get feature class

feature class SPARSE ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type

templated feature type ";

%feature("docstring")  shogun::CSparseFeatures::free_feature_vector "

free feature vector

Parameters:
-----------

feat_vec:  feature vector to free

num:  index of vector in cache

free:  if vector really should be deleted ";

%feature("docstring")
shogun::CSparseFeatures::get_num_nonzero_entries "

get number of non-zero entries in sparse feature matrix

number of non-zero entries in sparse feature matrix ";

%feature("docstring")  shogun::CSparseFeatures::compute_squared "

compute a^2 on all feature vectors

Parameters:
-----------

sq:  the square for each vector is stored in here

the square for each vector ";

%feature("docstring")  shogun::CSparseFeatures::compute_squared_norm "

compute (a-b)^2 (== a^2+b^2+2ab) usually called by kernels'/distances'
compute functions works on two feature vectors, although it is a
member of a single feature: can either be called by lhs or rhs.

Parameters:
-----------

lhs:  left-hand side features

sq_lhs:  squared values of left-hand side

idx_a:  index of left-hand side's vector to compute

rhs:  right-hand side features

sq_rhs:  squared values of right-hand side

idx_b:  index of right-hand side's vector to compute ";

%feature("docstring")  shogun::CSparseFeatures::load_svmlight_file "

load features from file

Parameters:
-----------

fname:  filename to load from

do_sort_features:  if true features will be sorted to ensure they are
in ascending order

label object with corresponding labels ";

%feature("docstring")  shogun::CSparseFeatures::sort_features "

ensure that features occur in ascending order, only call when no
preprocessors are attached ";

%feature("docstring")  shogun::CSparseFeatures::write_svmlight_file "

write features to file using svm light format

Parameters:
-----------

fname:  filename to write to

label:  Label object (number of labels must correspond to number of
features)

true if successful ";

%feature("docstring")  shogun::CSparseFeatures::get_dim_feature_space
"

obtain the dimensionality of the feature space

(not mix this up with the dimensionality of the input space, usually
obtained via get_num_features())

dimensionality ";

%feature("docstring")  shogun::CSparseFeatures::dot "

compute dot product between vector1 and vector2, appointed by their
indices

Parameters:
-----------

vec_idx1:  index of first vector

vec_idx2:  index of second vector ";

%feature("docstring")  shogun::CSparseFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CSparseFeatures::get_name "

object name ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the BOOL feature can deal with

feature type BOOL ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the CHAR feature can deal with

feature type CHAR ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the BYTE feature can deal with

feature type BYTE ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the SHORT feature can deal with

feature type SHORT ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the WORD feature can deal with

feature type WORD ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the INT feature can deal with

feature type INT ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the UINT feature can deal with

feature type UINT ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the LONG feature can deal with

feature type LONG ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the ULONG feature can deal with

feature type ULONG ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the SHORTREAL feature can deal with

feature type SHORTREAL ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the DREAL feature can deal with

feature type DREAL ";

%feature("docstring")  shogun::CSparseFeatures::get_feature_type "

get feature type the LONGREAL feature can deal with

feature type LONGREAL ";


// File: classshogun_1_1CStringFeatures.xml
%feature("docstring") shogun::CStringFeatures "

Template class StringFeatures implements a list of strings.

As this class is a template the underlying storage type is quite
arbitrary and not limited to character strings, but could also be
sequences of floating point numbers etc. Strings differ from matrices
(cf. CSimpleFeatures) in a way that the dimensionality of the feature
vectors (i.e. the strings) is not fixed; it may vary between strings.

Most string kernels require StringFeatures but a number of them
actually requires strings to have same length.

When preprocessors are attached to string features they may shorten
the string, but are not allowed to return strings longer than
max_string_length, as some algorithms depend on this.

Also note that string features cannot currently be computed on-the-
fly.

C++ includes: StringFeatures.h ";

%feature("docstring")  shogun::CStringFeatures::CStringFeatures "

default constructor ";

%feature("docstring")  shogun::CStringFeatures::CStringFeatures "

constructor

Parameters:
-----------

alpha:  alphabet (type) to use for string features ";

%feature("docstring")  shogun::CStringFeatures::CStringFeatures "

constructor

Parameters:
-----------

p_features:  new features

p_num_vectors:  number of vectors

p_max_string_length:  maximum string length

alpha:  alphabet (type) to use for string features ";

%feature("docstring")  shogun::CStringFeatures::CStringFeatures "

constructor

Parameters:
-----------

alpha:  alphabet to use for string features ";

%feature("docstring")  shogun::CStringFeatures::CStringFeatures "

copy constructor ";

%feature("docstring")  shogun::CStringFeatures::CStringFeatures "

constructor

Parameters:
-----------

fname:  filename to load features from

alpha:  alphabet (type) to use for string features ";

%feature("docstring")  shogun::CStringFeatures::~CStringFeatures "";

%feature("docstring")  shogun::CStringFeatures::cleanup "

cleanup string features ";

%feature("docstring")  shogun::CStringFeatures::cleanup_feature_vector
"

cleanup a single feature vector ";

%feature("docstring")  shogun::CStringFeatures::get_feature_class "

get feature class

feature class STRING ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type

templated feature type ";

%feature("docstring")  shogun::CStringFeatures::get_alphabet "

get alphabet used in string features

alphabet ";

%feature("docstring")  shogun::CStringFeatures::duplicate "

duplicate feature object

feature object ";

%feature("docstring")  shogun::CStringFeatures::get_feature_vector "

get string for selected example num

Parameters:
-----------

dst:  destination where vector will be stored

len:  number of features in vector

num:  index of the string ";

%feature("docstring")  shogun::CStringFeatures::set_feature_vector "

set string for selected example num

Parameters:
-----------

src:  destination where vector will be stored

len:  number of features in vector

num:  index of the string ";

%feature("docstring")
shogun::CStringFeatures::enable_on_the_fly_preprocessing "

call this to preprocess string features upon get_feature_vector ";

%feature("docstring")
shogun::CStringFeatures::disable_on_the_fly_preprocessing "

call this to disable on the fly feature preprocessing on
get_feature_vector. Useful when you manually apply preprocessors. ";

%feature("docstring")  shogun::CStringFeatures::get_feature_vector "

get feature vector for sample num

Parameters:
-----------

num:  index of feature vector

len:  length is returned by reference

dofree:  whether returned vector must be freed by caller via
free_feature_vector

feature vector for sample num ";

%feature("docstring")  shogun::CStringFeatures::free_feature_vector "

free feature vector

Parameters:
-----------

feat_vec:  feature vector to free

num:  index in feature cache

dofree:  if vector should be really deleted ";

%feature("docstring")  shogun::CStringFeatures::get_feature "

get feature

Parameters:
-----------

vec_num:  which vector

feat_num:  which feature

feature ";

%feature("docstring")  shogun::CStringFeatures::get_vector_length "

get vector length

Parameters:
-----------

vec_num:  which vector

length of vector ";

%feature("docstring")  shogun::CStringFeatures::get_max_vector_length
"

get maximum vector length

maximum vector/string length ";

%feature("docstring")  shogun::CStringFeatures::get_num_vectors "

get number of vectors

number of vectors ";

%feature("docstring")  shogun::CStringFeatures::get_num_symbols "

get number of symbols

Note: floatmax_t sounds weird, but LONG is not long enough

number of symbols ";

%feature("docstring")  shogun::CStringFeatures::get_max_num_symbols "

get maximum number of symbols

Note: floatmax_t sounds weird, but int64_t is not long enough (and
there is no int128_t type)

maximum number of symbols ";

%feature("docstring")
shogun::CStringFeatures::get_original_num_symbols "

number of symbols before higher order mapping

original number of symbols ";

%feature("docstring")  shogun::CStringFeatures::get_order "

order used for higher order mapping

order ";

%feature("docstring")  shogun::CStringFeatures::get_masked_symbols "

a higher order mapped symbol will be shaped such that the symbols
specified by bits in the mask will be returned.

Parameters:
-----------

symbol:  symbol to mask

mask:  mask to apply

masked symbol ";

%feature("docstring")  shogun::CStringFeatures::shift_offset "

shift offset to the left by amount

Parameters:
-----------

offset:  offset to shift

amount:  amount to shift the offset

shifted offset ";

%feature("docstring")  shogun::CStringFeatures::shift_symbol "

shift symbol to the right by amount (taking care of custom symbol
sizes)

Parameters:
-----------

symbol:  symbol to shift

amount:  amount to shift the symbol

shifted symbol ";

%feature("docstring")  shogun::CStringFeatures::load "

load features from file

Parameters:
-----------

fname:  filename to load from

if loading was successful ";

%feature("docstring")  shogun::CStringFeatures::load_dna_file "

load DNA features from file

Parameters:
-----------

fname:  filename to load from

remap_to_bin:  if remap_to_bin

if loading was successful ";

%feature("docstring")  shogun::CStringFeatures::load_fasta_file "

load fasta file as string features

Parameters:
-----------

fname:  filename to load from

ignore_invalid:  if set to true, characters other than A,C,G,T are
converted to A

if loading was successful ";

%feature("docstring")  shogun::CStringFeatures::load_fastq_file "

load fastq file as string features

Parameters:
-----------

fname:  filename to load from

ignore_invalid:  if set to true, characters other than A,C,G,T are
converted to A

bitremap_in_single_string:  if set to true, do binary embedding of
symbols

if loading was successful ";

%feature("docstring")  shogun::CStringFeatures::load_from_directory "

load features from directory

Parameters:
-----------

dirname:  directory name to load from

if loading was successful ";

%feature("docstring")  shogun::CStringFeatures::set_features "

set features

Parameters:
-----------

p_features:  new features

p_num_vectors:  number of vectors

p_max_string_length:  maximum string length

if setting was successful ";

%feature("docstring")  shogun::CStringFeatures::get_features "

get_features

Parameters:
-----------

num_str:  number of strings (returned)

max_str_len:  maximal string length (returned)

string features ";

%feature("docstring")  shogun::CStringFeatures::copy_features "

copy_features

Parameters:
-----------

num_str:  number of strings (returned)

max_str_len:  maximal string length (returned)

string features ";

%feature("docstring")  shogun::CStringFeatures::get_features "

get_features (swig compatible)

Parameters:
-----------

dst:  string features (returned)

num_str:  number of strings (returned) ";

%feature("docstring")  shogun::CStringFeatures::save "

save features to file

Parameters:
-----------

dest:  filename to save to

if saving was successful ";

%feature("docstring")  shogun::CStringFeatures::load_compressed "

load compressed features from file

Parameters:
-----------

src:  filename to load from

decompress:  whether to decompress on loading

if loading was successful ";

%feature("docstring")  shogun::CStringFeatures::save_compressed "

save compressed features to file

Parameters:
-----------

dest:  filename to save to

compression:  compressor to use

level:  compression level to use (1-9)

if saving was successful ";

%feature("docstring")  shogun::CStringFeatures::get_size "

get memory footprint of one feature

memory footprint of one feature ";

%feature("docstring")  shogun::CStringFeatures::apply_preproc "

apply preprocessor

Parameters:
-----------

force_preprocessing:  if preprocssing shall be forced

if applying was successful ";

%feature("docstring")
shogun::CStringFeatures::obtain_by_sliding_window "

slides a window of size window_size over the current single string
step_size is the amount by which the window is shifted. creates
(string_len-window_size)/step_size many feature obj if skip is
nonzero, skip the first 'skip' characters of each string

Parameters:
-----------

window_size:  window size

step_size:  step size

skip:  skip

something inty ";

%feature("docstring")
shogun::CStringFeatures::obtain_by_position_list "

extracts windows of size window_size from first string using the
positions in list

Parameters:
-----------

window_size:  window size

positions:  positions

skip:  skip

something inty ";

%feature("docstring")  shogun::CStringFeatures::obtain_from_char "

obtain string features from char features

wrapper for template method

Parameters:
-----------

sf:  string features

start:  start

p_order:  order

gap:  gap

rev:  reverse

if obtaining was successful ";

%feature("docstring")
shogun::CStringFeatures::obtain_from_char_features "

template obtain from char features

Parameters:
-----------

sf:  string features

start:  start

p_order:  order

gap:  gap

rev:  reverse

if obtaining was successful ";

%feature("docstring")  shogun::CStringFeatures::have_same_length "

check if length of each vector in this feature object equals the given
length.

Parameters:
-----------

len:  vector length to check against

if length of each vector in this feature object equals the given
length. ";

%feature("docstring")  shogun::CStringFeatures::embed_features "

embed string features in bit representation in-place ";

%feature("docstring")
shogun::CStringFeatures::compute_symbol_mask_table "

compute symbol mask table

required to access bit-based symbols ";

%feature("docstring")  shogun::CStringFeatures::unembed_word "

remap bit-based word to character sequence

Parameters:
-----------

word:  word to remap

seq:  sequence of size len that remapped characters are written to

len:  length of sequence and word ";

%feature("docstring")  shogun::CStringFeatures::embed_word "

embed a single word

Parameters:
-----------

seq:  sequence of size len in a bitfield

len:  ";

%feature("docstring")
shogun::CStringFeatures::determine_maximum_string_length "

determine new maximum string length ";

%feature("docstring")  shogun::CStringFeatures::set_feature_vector "

set feature vector for sample num

Parameters:
-----------

num:  index of feature vector

string:  string with the feature vector's content

len:  length of the string ";

%feature("docstring")  shogun::CStringFeatures::get_name "

object name ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the char feature can deal with

feature type char ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the char feature can deal with

feature type char ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the BYTE feature can deal with

feature type BYTE ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the SHORT feature can deal with

feature type SHORT ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the WORD feature can deal with

feature type WORD ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the INT feature can deal with

feature type INT ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the INT feature can deal with

feature type INT ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the LONG feature can deal with

feature type LONG ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the ULONG feature can deal with

feature type ULONG ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the SHORTREAL feature can deal with

feature type SHORTREAL ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the DREAL feature can deal with

feature type DREAL ";

%feature("docstring")  shogun::CStringFeatures::get_feature_type "

get feature type the LONGREAL feature can deal with

feature type LONGREAL ";

%feature("docstring")  shogun::CStringFeatures::get_masked_symbols "";

%feature("docstring")  shogun::CStringFeatures::get_masked_symbols "";

%feature("docstring")  shogun::CStringFeatures::get_masked_symbols "";

%feature("docstring")  shogun::CStringFeatures::get_masked_symbols "";

%feature("docstring")  shogun::CStringFeatures::shift_offset "";

%feature("docstring")  shogun::CStringFeatures::shift_offset "";

%feature("docstring")  shogun::CStringFeatures::shift_offset "";

%feature("docstring")  shogun::CStringFeatures::shift_offset "";

%feature("docstring")  shogun::CStringFeatures::shift_symbol "";

%feature("docstring")  shogun::CStringFeatures::shift_symbol "";

%feature("docstring")  shogun::CStringFeatures::shift_symbol "";

%feature("docstring")  shogun::CStringFeatures::shift_symbol "";

%feature("docstring")
shogun::CStringFeatures::obtain_from_char_features "";

%feature("docstring")
shogun::CStringFeatures::obtain_from_char_features "";

%feature("docstring")
shogun::CStringFeatures::obtain_from_char_features "";

%feature("docstring")  shogun::CStringFeatures::embed_features "";

%feature("docstring")  shogun::CStringFeatures::embed_features "";

%feature("docstring")  shogun::CStringFeatures::embed_features "";

%feature("docstring")
shogun::CStringFeatures::compute_symbol_mask_table "";

%feature("docstring")
shogun::CStringFeatures::compute_symbol_mask_table "";

%feature("docstring")
shogun::CStringFeatures::compute_symbol_mask_table "";

%feature("docstring")  shogun::CStringFeatures::embed_word "";

%feature("docstring")  shogun::CStringFeatures::embed_word "";

%feature("docstring")  shogun::CStringFeatures::embed_word "";

%feature("docstring")  shogun::CStringFeatures::unembed_word "";

%feature("docstring")  shogun::CStringFeatures::unembed_word "";

%feature("docstring")  shogun::CStringFeatures::unembed_word "";


// File: classshogun_1_1CStringFileFeatures.xml
%feature("docstring") shogun::CStringFileFeatures "

File based string features.

StringFeatures that are file based. Underneath memory mapped files are
used. Derived from CStringFeatures thus transparently enabling all of
the StringFeature functionality.

Supported file format contains one string per line, lines of variable
length are supported and must be separated by ' '.

C++ includes: StringFileFeatures.h ";

%feature("docstring")
shogun::CStringFileFeatures::CStringFileFeatures "

default constructor ";

%feature("docstring")
shogun::CStringFileFeatures::CStringFileFeatures "

constructor

Parameters:
-----------

fname:  filename of the file containing line based features

alpha:  alphabet (type) to use for string features ";

%feature("docstring")
shogun::CStringFileFeatures::~CStringFileFeatures "

default destructor ";


// File: classshogun_1_1CTOPFeatures.xml
%feature("docstring") shogun::CTOPFeatures "

The class TOPFeatures implements TOP kernel features obtained from two
Hidden Markov models.

It was used in

K. Tsuda, M. Kawanabe, G. Raetsch, S. Sonnenburg, and K.R. Mueller. A
new discriminative kernel from probabilistic models. Neural
Computation, 14:2397-2414, 2002.

which also has the details.

Note that TOP-features are computed on the fly, so to be effective
feature caching should be enabled.

It inherits its functionality from CSimpleFeatures, which should be
consulted for further reference.

C++ includes: TOPFeatures.h ";

%feature("docstring")  shogun::CTOPFeatures::CTOPFeatures "

constructor

Parameters:
-----------

size:  cache size

p:  positive HMM

n:  negative HMM

neglin:  if negative HMM is of linear shape

poslin:  if positive HMM is of linear shape ";

%feature("docstring")  shogun::CTOPFeatures::CTOPFeatures "

copy constructor ";

%feature("docstring")  shogun::CTOPFeatures::~CTOPFeatures "";

%feature("docstring")  shogun::CTOPFeatures::set_models "

set HMMs

Parameters:
-----------

p:  positive HMM

n:  negative HMM ";

%feature("docstring")  shogun::CTOPFeatures::set_feature_matrix "

set feature matrix

something floaty ";

%feature("docstring")  shogun::CTOPFeatures::compute_num_features "

compute number of features

number of features ";

%feature("docstring")  shogun::CTOPFeatures::compute_relevant_indizes
"

compute relevant indices

Parameters:
-----------

hmm:  HMM to compute for

hmm_idx:  HMM index

if computing was successful ";

%feature("docstring")  shogun::CTOPFeatures::get_name "

object name ";


// File: classshogun_1_1CWDFeatures.xml
%feature("docstring") shogun::CWDFeatures "

Features that compute the Weighted Degreee Kernel feature space
explicitly.

See:  CWeightedDegreeStringKernel

C++ includes: WDFeatures.h ";

%feature("docstring")  shogun::CWDFeatures::CWDFeatures "

constructor

Parameters:
-----------

str:  stringfeatures (of bytes)

order:  of wd kernel

from_order:  use first order weights from higher order weighting ";

%feature("docstring")  shogun::CWDFeatures::CWDFeatures "

copy constructor ";

%feature("docstring")  shogun::CWDFeatures::~CWDFeatures "

destructor ";

%feature("docstring")  shogun::CWDFeatures::get_dim_feature_space "

obtain the dimensionality of the feature space

(not mix this up with the dimensionality of the input space, usually
obtained via get_num_features())

dimensionality ";

%feature("docstring")  shogun::CWDFeatures::dot "

compute dot product between vector1 and vector2, appointed by their
indices

Parameters:
-----------

vec_idx1:  index of first vector

vec_idx2:  index of second vector ";

%feature("docstring")  shogun::CWDFeatures::dense_dot "

compute dot product between vector1 and a dense vector

Parameters:
-----------

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector ";

%feature("docstring")  shogun::CWDFeatures::add_to_dense_vec "

add vector 1 multiplied with alpha to dense vector2

Parameters:
-----------

alpha:  scalar alpha

vec_idx1:  index of first vector

vec2:  pointer to real valued vector

vec2_len:  length of real valued vector

abs_val:  if true add the absolute value ";

%feature("docstring")
shogun::CWDFeatures::get_nnz_features_for_vector "

get number of non-zero features in vector

Parameters:
-----------

num:  which vector

number of non-zero features in vector ";

%feature("docstring")  shogun::CWDFeatures::duplicate "

duplicate feature object

feature object ";

%feature("docstring")  shogun::CWDFeatures::get_feature_type "

get feature type

templated feature type ";

%feature("docstring")  shogun::CWDFeatures::get_feature_class "

get feature class

feature class ";

%feature("docstring")  shogun::CWDFeatures::get_num_vectors "

get number of examples/vectors

abstract base method

number of examples/vectors ";

%feature("docstring")  shogun::CWDFeatures::get_size "

get memory footprint of one feature

abstract base method

memory footprint of one feature ";

%feature("docstring")  shogun::CWDFeatures::get_name "

object name ";


// File: structshogun_1_1T__ATTRIBUTE.xml
%feature("docstring") shogun::T_ATTRIBUTE "

Attribute Struct

C++ includes: AttributeFeatures.h ";


// File: structshogun_1_1T__HMM__INDIZES.xml
%feature("docstring") shogun::T_HMM_INDIZES "

HMM indices

C++ includes: TOPFeatures.h ";


// File: classshogun_1_1T__STRING.xml
%feature("docstring") shogun::T_STRING "

template class T_STRING

C++ includes: StringFeatures.h ";


// File: structshogun_1_1TSparse.xml
%feature("docstring") shogun::TSparse "

template class TSparse

C++ includes: SparseFeatures.h ";


// File: structshogun_1_1TSparseEntry.xml
%feature("docstring") shogun::TSparseEntry "

template class TSparseEntry

C++ includes: SparseFeatures.h ";


// File: namespaceshogun.xml


// File: Alphabet_8h.xml


// File: AttributeFeatures_8h.xml


// File: CombinedDotFeatures_8h.xml


// File: CombinedFeatures_8h.xml


// File: DotFeatures_8h.xml


// File: DummyFeatures_8h.xml


// File: ExplicitSpecFeatures_8h.xml


// File: Features_8h.xml


// File: FeatureTypes_8h.xml


// File: FKFeatures_8h.xml


// File: ImplicitWeightedSpecFeatures_8h.xml


// File: Labels_8h.xml


// File: PolyFeatures_8h.xml


// File: RealFileFeatures_8h.xml


// File: SimpleFeatures_8h.xml


// File: SparseFeatures_8h.xml


// File: StringFeatures_8h.xml


// File: StringFileFeatures_8h.xml


// File: TOPFeatures_8h.xml


// File: WDFeatures_8h.xml


// File: dir_532d87e6696c390dc65ad3a79bd6599e.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml

