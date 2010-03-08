
// File: index.xml

// File: classshogun_1_1CAUCKernel.xml
%feature("docstring") shogun::CAUCKernel "

The AUC kernel can be used to maximize the area under the receiver
operator characteristic curve (AUC) instead of margin in SVM training.

It takes as argument a sub-kernel and Labels based on which number of
positive labels times number of negative labels many ``virtual''
examples are created that ensure that all positive examples get a
higher score than all negative examples in training.

C++ includes: AUCKernel.h ";

%feature("docstring")  shogun::CAUCKernel::CAUCKernel "

constructor

Parameters:
-----------

size:  cache size

subkernel:  the subkernel ";

%feature("docstring")  shogun::CAUCKernel::~CAUCKernel "

destructor ";

%feature("docstring")  shogun::CAUCKernel::setup_auc_maximization "

initialize kernel based on current labeling and subkernel

Parameters:
-----------

labels:  - current labeling

new label object to be used together with this kernel in SVM training
for AUC maximization ";

%feature("docstring")  shogun::CAUCKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CAUCKernel::get_kernel_type "

return what type of kernel we are

kernel type AUC ";

%feature("docstring")  shogun::CAUCKernel::get_name "

return the kernel's name

name AUC ";


// File: classshogun_1_1CAvgDiagKernelNormalizer.xml
%feature("docstring") shogun::CAvgDiagKernelNormalizer "

Normalize the kernel by either a constant or the average value of the
diagonal elements (depending on argument c of the constructor).

In case c <= 0 compute scale as \\\\[ \\\\mbox{scale} =
\\\\frac{1}{N}\\\\sum_{i=1}^N k(x_i,x_i) \\\\]

otherwise use scale=c and normalize the kernel via

\\\\[ k'(x,x')= \\\\frac{k(x,x')}{scale} \\\\]

C++ includes: AvgDiagKernelNormalizer.h ";

%feature("docstring")
shogun::CAvgDiagKernelNormalizer::CAvgDiagKernelNormalizer "

constructor

Parameters:
-----------

c:  scale parameter, if <= 0 scaling will be computed from the avg of
the kernel diagonal elements ";

%feature("docstring")
shogun::CAvgDiagKernelNormalizer::~CAvgDiagKernelNormalizer "

default destructor ";

%feature("docstring")  shogun::CAvgDiagKernelNormalizer::init "

initialization of the normalizer (if needed)

Parameters:
-----------

k:  kernel ";

%feature("docstring")  shogun::CAvgDiagKernelNormalizer::normalize "

normalize the kernel value

Parameters:
-----------

value:  kernel value

idx_lhs:  index of left hand side vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")  shogun::CAvgDiagKernelNormalizer::normalize_lhs
"

normalize only the left hand side vector

Parameters:
-----------

value:  value of a component of the left hand side feature vector

idx_lhs:  index of left hand side vector ";

%feature("docstring")  shogun::CAvgDiagKernelNormalizer::normalize_rhs
"

normalize only the right hand side vector

Parameters:
-----------

value:  value of a component of the right hand side feature vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")  shogun::CAvgDiagKernelNormalizer::get_name "

object name ";


// File: classshogun_1_1CChi2Kernel.xml
%feature("docstring") shogun::CChi2Kernel "

The Chi2 kernel operating on realvalued vectors computes the chi-
squared distance between sets of histograms.

It is a very useful distance in image recognition (used to detect
objects).

It is defined as \\\\[ k({\\\\bf x},({\\\\bf x'})=
e^{-\\\\frac{1}{width}
\\\\sum_{i=0}^{l}\\\\frac{(x_i-x'_i)^2}{(x_i+x'_i)}} \\\\]

C++ includes: Chi2Kernel.h ";

%feature("docstring")  shogun::CChi2Kernel::CChi2Kernel "

constructor

Parameters:
-----------

size:  cache size

width:  width ";

%feature("docstring")  shogun::CChi2Kernel::CChi2Kernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

width:  width

size:  cache size ";

%feature("docstring")  shogun::CChi2Kernel::~CChi2Kernel "";

%feature("docstring")  shogun::CChi2Kernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CChi2Kernel::get_kernel_type "

return what type of kernel we are

kernel type CHI2 ";

%feature("docstring")  shogun::CChi2Kernel::get_name "

return the kernel's name

name Chi2 ";


// File: classshogun_1_1CCombinedKernel.xml
%feature("docstring") shogun::CCombinedKernel "

The Combined kernel is used to combine a number of kernels into a
single CombinedKernel object by linear combination.

It keeps pointers to the added sub-kernels $k_m({\\\\bf x}, {\\\\bf
x'})$ and for each sub-kernel - a kernel specific weight $\\\\beta_m$.

It is especially useful to combine kernels working on different
domains and to combine kernels looking at independent features and
requires CCombinedFeatures to be used.

It is defined as:

\\\\[ k_{combined}({\\\\bf x}, {\\\\bf x'}) = \\\\sum_{m=1}^M
\\\\beta_m k_m({\\\\bf x}, {\\\\bf x'}) \\\\]

C++ includes: CombinedKernel.h ";

%feature("docstring")  shogun::CCombinedKernel::CCombinedKernel "

constructor

Parameters:
-----------

size:  cache size

append_subkernel_weights:  if subkernel weights shall be appended ";

%feature("docstring")  shogun::CCombinedKernel::~CCombinedKernel "";

%feature("docstring")  shogun::CCombinedKernel::init "

initialize kernel

Parameters:
-----------

lhs:  features of left-hand side

rhs:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CCombinedKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CCombinedKernel::get_kernel_type "

return what type of kernel we are

kernel type COMBINED ";

%feature("docstring")  shogun::CCombinedKernel::get_feature_type "

return feature type the kernel can deal with

feature type UNKNOWN ";

%feature("docstring")  shogun::CCombinedKernel::get_feature_class "

return feature class the kernel can deal with

feature class COMBINED ";

%feature("docstring")  shogun::CCombinedKernel::get_name "

return the kernel's name

name Combined ";

%feature("docstring")  shogun::CCombinedKernel::list_kernels "

list kernels ";

%feature("docstring")  shogun::CCombinedKernel::get_first_kernel "

get first kernel

first kernel ";

%feature("docstring")  shogun::CCombinedKernel::get_first_kernel "

get first kernel

Parameters:
-----------

current:

first kernel ";

%feature("docstring")  shogun::CCombinedKernel::get_kernel "

get kernel

Parameters:
-----------

idx:  index of kernel

kernel at index idx ";

%feature("docstring")  shogun::CCombinedKernel::get_last_kernel "

get last kernel

last kernel ";

%feature("docstring")  shogun::CCombinedKernel::get_next_kernel "

get next kernel

next kernel ";

%feature("docstring")  shogun::CCombinedKernel::get_next_kernel "

get next kernel multi-thread safe

Parameters:
-----------

current:

next kernel ";

%feature("docstring")  shogun::CCombinedKernel::insert_kernel "

insert kernel

Parameters:
-----------

k:  kernel

if inserting was successful ";

%feature("docstring")  shogun::CCombinedKernel::append_kernel "

append kernel

Parameters:
-----------

k:  kernel

if appending was successful ";

%feature("docstring")  shogun::CCombinedKernel::delete_kernel "

delete kernel

if deleting was successful ";

%feature("docstring")
shogun::CCombinedKernel::get_append_subkernel_weights "

check if subkernel weights are appended

if subkernel weigths are appended ";

%feature("docstring")  shogun::CCombinedKernel::get_num_subkernels "

get number of subkernels

number of subkernels ";

%feature("docstring")  shogun::CCombinedKernel::has_features "

test whether features have been assigned to lhs and rhs

true if features are assigned ";

%feature("docstring")  shogun::CCombinedKernel::remove_lhs "

remove lhs from kernel ";

%feature("docstring")  shogun::CCombinedKernel::remove_rhs "

remove rhs from kernel ";

%feature("docstring")  shogun::CCombinedKernel::remove_lhs_and_rhs "

remove lhs and rhs from kernel ";

%feature("docstring")  shogun::CCombinedKernel::init_optimization "

initialize optimization

Parameters:
-----------

count:  count

IDX:  index

weights:  weights

if initializing was successful ";

%feature("docstring")  shogun::CCombinedKernel::delete_optimization "

delete optimization

if deleting was successful ";

%feature("docstring")  shogun::CCombinedKernel::compute_optimized "

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")  shogun::CCombinedKernel::compute_batch "

computes output for a batch of examples in an optimized fashion
(favorable if kernel supports it, i.e. has KP_BATCHEVALUATION. to the
outputvector target (of length num_vec elements) the output for the
examples enumerated in vec_idx are added. therefore make sure that it
is initialized with ZERO. the following num_suppvec, IDX, alphas
arguments are the number of support vectors, their indices and weights
";

%feature("docstring")  shogun::CCombinedKernel::emulate_compute_batch
"

emulates batch computation, via linadd optimization w^t x or even down
to sum_i alpha_i K(x_i,x)

Parameters:
-----------

k:  kernel

num_vec:  number of vectors

vec_idx:  vector index

target:  target

num_suppvec:  number of support vectors

IDX:  IDX

weights:  weights ";

%feature("docstring")  shogun::CCombinedKernel::add_to_normal "

add to normal vector

Parameters:
-----------

idx:  where to add

weight:  what to add ";

%feature("docstring")  shogun::CCombinedKernel::clear_normal "

clear normal vector ";

%feature("docstring")  shogun::CCombinedKernel::compute_by_subkernel "

compute by subkernel

Parameters:
-----------

idx:  index

subkernel_contrib:  subkernel contribution ";

%feature("docstring")  shogun::CCombinedKernel::get_subkernel_weights
"

get subkernel weights

Parameters:
-----------

num_weights:  where number of weights is stored

subkernel weights ";

%feature("docstring")  shogun::CCombinedKernel::get_subkernel_weights
"

get subkernel weights (swig compatible)

Parameters:
-----------

weights:  subkernel weights

num_weights:  number of weights ";

%feature("docstring")  shogun::CCombinedKernel::set_subkernel_weights
"

set subkernel weights

Parameters:
-----------

weights:  new subkernel weights

num_weights:  number of subkernel weights ";

%feature("docstring")  shogun::CCombinedKernel::set_optimization_type
"

set optimization type

Parameters:
-----------

t:  optimization type ";

%feature("docstring")  shogun::CCombinedKernel::precompute_subkernels
"

precompute all sub-kernels ";


// File: classshogun_1_1CCommUlongStringKernel.xml
%feature("docstring") shogun::CCommUlongStringKernel "

The CommUlongString kernel may be used to compute the spectrum kernel
from strings that have been mapped into unsigned 64bit integers.

These 64bit integers correspond to k-mers. To be applicable in this
kernel they need to be sorted (e.g. via the SortUlongString pre-
processor).

It basically uses the algorithm in the unix \"comm\" command (hence
the name) to compute:

\\\\[ k({\\\\bf x},({\\\\bf x'})= \\\\Phi_k({\\\\bf x})\\\\cdot
\\\\Phi_k({\\\\bf x'}) \\\\]

where $\\\\Phi_k$ maps a sequence ${\\\\bf x}$ that consists of
letters in $\\\\Sigma$ to a feature vector of size $|\\\\Sigma|^k$. In
this feature vector each entry denotes how often the k-mer appears in
that ${\\\\bf x}$.

Note that this representation enables spectrum kernels of order 8 for
8bit alphabets (like binaries) and order 32 for 2-bit alphabets like
DNA.

For this kernel the linadd speedups are implemented (though there is
room for improvement here when a whole set of sequences is ADDed)
using sorted lists.

C++ includes: CommUlongStringKernel.h ";

%feature("docstring")
shogun::CCommUlongStringKernel::CCommUlongStringKernel "

constructor

Parameters:
-----------

size:  cache size

use_sign:  if sign shall be used ";

%feature("docstring")
shogun::CCommUlongStringKernel::CCommUlongStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

use_sign:  if sign shall be used

size:  cache size ";

%feature("docstring")
shogun::CCommUlongStringKernel::~CCommUlongStringKernel "";

%feature("docstring")  shogun::CCommUlongStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CCommUlongStringKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CCommUlongStringKernel::get_kernel_type
"

return what type of kernel we are

kernel type COMMULONGSTRING ";

%feature("docstring")  shogun::CCommUlongStringKernel::get_name "

return the kernel's name

name CommUlongString ";

%feature("docstring")
shogun::CCommUlongStringKernel::init_optimization "

initialize optimization

Parameters:
-----------

count:  count

IDX:  index

weights:  weights

if initializing was successful ";

%feature("docstring")
shogun::CCommUlongStringKernel::delete_optimization "

delete optimization

if deleting was successful ";

%feature("docstring")
shogun::CCommUlongStringKernel::compute_optimized "

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")
shogun::CCommUlongStringKernel::merge_dictionaries "

merge dictionaries

Parameters:
-----------

t:  t

j:  j

k:  k

vec:  vector

dic:  dictionary

dic_weights:  dictionary weights

weight:  weight

vec_idx:  vector index ";

%feature("docstring")  shogun::CCommUlongStringKernel::add_to_normal "

add to normal

Parameters:
-----------

idx:  where to add

weight:  what to add ";

%feature("docstring")  shogun::CCommUlongStringKernel::clear_normal "

clear normal ";

%feature("docstring")  shogun::CCommUlongStringKernel::remove_lhs "

remove lhs from kernel ";

%feature("docstring")  shogun::CCommUlongStringKernel::remove_rhs "

remove rhs from kernel ";

%feature("docstring")
shogun::CCommUlongStringKernel::get_feature_type "

return feature type the kernel can deal with

feature type ULONG ";

%feature("docstring")  shogun::CCommUlongStringKernel::get_dictionary
"

get dictionary

Parameters:
-----------

dsize:  dictionary size will be stored in here

dict:  dictionary will be stored in here

dweights:  dictionary weights will be stored in here ";


// File: classshogun_1_1CCommWordStringKernel.xml
%feature("docstring") shogun::CCommWordStringKernel "

The CommWordString kernel may be used to compute the spectrum kernel
from strings that have been mapped into unsigned 16bit integers.

These 16bit integers correspond to k-mers. To applicable in this
kernel they need to be sorted (e.g. via the SortWordString pre-
processor).

It basically uses the algorithm in the unix \"comm\" command (hence
the name) to compute:

\\\\[ k({\\\\bf x},({\\\\bf x'})= \\\\Phi_k({\\\\bf x})\\\\cdot
\\\\Phi_k({\\\\bf x'}) \\\\]

where $\\\\Phi_k$ maps a sequence ${\\\\bf x}$ that consists of
letters in $\\\\Sigma$ to a feature vector of size $|\\\\Sigma|^k$. In
this feature vector each entry denotes how often the k-mer appears in
that ${\\\\bf x}$.

Note that this representation is especially tuned to small alphabets
(like the 2-bit alphabet DNA), for which it enables spectrum kernels
of order up to 8.

For this kernel the linadd speedups are quite efficiently implemented
using direct maps.

C++ includes: CommWordStringKernel.h ";

%feature("docstring")
shogun::CCommWordStringKernel::CCommWordStringKernel "

constructor

Parameters:
-----------

size:  cache size

use_sign:  if sign shall be used ";

%feature("docstring")
shogun::CCommWordStringKernel::CCommWordStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

use_sign:  if sign shall be used

size:  cache size ";

%feature("docstring")
shogun::CCommWordStringKernel::~CCommWordStringKernel "";

%feature("docstring")  shogun::CCommWordStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CCommWordStringKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CCommWordStringKernel::get_kernel_type
"

return what type of kernel we are

kernel type COMMWORDSTRING ";

%feature("docstring")  shogun::CCommWordStringKernel::get_name "

return the kernel's name

name CommWordString ";

%feature("docstring")  shogun::CCommWordStringKernel::init_dictionary
"

initialize dictionary

Parameters:
-----------

size:  size ";

%feature("docstring")
shogun::CCommWordStringKernel::init_optimization "

initialize optimization

Parameters:
-----------

count:  count

IDX:  index

weights:  weights

if initializing was successful ";

%feature("docstring")
shogun::CCommWordStringKernel::delete_optimization "

delete optimization

if deleting was successful ";

%feature("docstring")
shogun::CCommWordStringKernel::compute_optimized "

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")  shogun::CCommWordStringKernel::add_to_normal "

add to normal

Parameters:
-----------

idx:  where to add

weight:  what to add ";

%feature("docstring")  shogun::CCommWordStringKernel::clear_normal "

clear normal ";

%feature("docstring")  shogun::CCommWordStringKernel::get_feature_type
"

return feature type the kernel can deal with

feature type WORD ";

%feature("docstring")  shogun::CCommWordStringKernel::get_dictionary "

get dictionary

Parameters:
-----------

dsize:  dictionary size will be stored in here

dweights:  dictionary weights will be stored in here ";

%feature("docstring")  shogun::CCommWordStringKernel::compute_scoring
"

compute scoring

Parameters:
-----------

max_degree:  maximum degree

num_feat:  number of features

num_sym:  number of symbols

target:  target

num_suppvec:  number of support vectors

IDX:  IDX

alphas:  alphas

do_init:  if initialization shall be performed

computed scores ";

%feature("docstring")
shogun::CCommWordStringKernel::compute_consensus "

compute consensus

Parameters:
-----------

num_feat:  number of features

num_suppvec:  number of support vectors

IDX:  IDX

alphas:  alphas

computed consensus ";

%feature("docstring")
shogun::CCommWordStringKernel::set_use_dict_diagonal_optimization "

set_use_dict_diagonal_optimization

Parameters:
-----------

flag:  enable diagonal optimization ";

%feature("docstring")
shogun::CCommWordStringKernel::get_use_dict_diagonal_optimization "

get.use.dict.diagonal.optimization

true if diagonal optimization is on ";


// File: classshogun_1_1CConstKernel.xml
%feature("docstring") shogun::CConstKernel "

The Constant Kernel returns a constant for all elements.

A ``kernel'' that simply returns a single constant, i.e. $k({\\\\bf
x}, {\\\\bf x'})= c$

C++ includes: ConstKernel.h ";

%feature("docstring")  shogun::CConstKernel::CConstKernel "

constructor

Parameters:
-----------

c:  constant c ";

%feature("docstring")  shogun::CConstKernel::CConstKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

c:  constant c ";

%feature("docstring")  shogun::CConstKernel::~CConstKernel "";

%feature("docstring")  shogun::CConstKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CConstKernel::get_kernel_type "

return what type of kernel we are

kernel type CONST ";

%feature("docstring")  shogun::CConstKernel::get_feature_type "

return feature type the kernel can deal with

feature type ANY ";

%feature("docstring")  shogun::CConstKernel::get_feature_class "

return feature class the kernel can deal with

feature class ANY ";

%feature("docstring")  shogun::CConstKernel::get_name "

return the kernel's name

name Const ";


// File: classshogun_1_1CCustomKernel.xml
%feature("docstring") shogun::CCustomKernel "

The Custom Kernelallows for custom user provided kernel matrices.

For squared training matrices it allows to store only the upper
triangle of the kernel to save memory: Full symmetric kernel matrices
can be stored as is or can be internally converted into (or directly
given in) upper triangle representation. Also note that values are
stored as 32bit floats.

C++ includes: CustomKernel.h ";

%feature("docstring")  shogun::CCustomKernel::CCustomKernel "

default constructor ";

%feature("docstring")  shogun::CCustomKernel::CCustomKernel "

constructor

compute custom kernel from given kernel matrix

Parameters:
-----------

k:  kernel matrix ";

%feature("docstring")  shogun::CCustomKernel::CCustomKernel "

constructor

sets full kernel matrix from full kernel matrix

Parameters:
-----------

km:  kernel matrix

rows:  number of rows in matrix

cols:  number of cols in matrix

if setting was successful ";

%feature("docstring")  shogun::CCustomKernel::~CCustomKernel "";

%feature("docstring")  shogun::CCustomKernel::dummy_init "

initialize kernel with dummy features

Kernels always need feature objects assigned. As the custom kernel
does not really require this it creates some magic dummy features that
only know about the number of vectors

Parameters:
-----------

rows:  features of left-hand side

cols:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CCustomKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CCustomKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CCustomKernel::get_kernel_type "

return what type of kernel we are

kernel type CUSTOM ";

%feature("docstring")  shogun::CCustomKernel::get_feature_type "

return feature type the kernel can deal with

feature type ANY ";

%feature("docstring")  shogun::CCustomKernel::get_feature_class "

return feature class the kernel can deal with

feature class ANY ";

%feature("docstring")  shogun::CCustomKernel::get_name "

return the kernel's name

name Custom ";

%feature("docstring")
shogun::CCustomKernel::set_triangle_kernel_matrix_from_triangle "

set kernel matrix (only elements from upper triangle) from elements of
upper triangle (concat'd), including the main diagonal

Parameters:
-----------

km:  kernel matrix

len:  denotes the size of the array and should match
len=cols*(cols+1)/2

if setting was successful ";

%feature("docstring")
shogun::CCustomKernel::set_triangle_kernel_matrix_from_full "

set kernel matrix (only elements from upper triangle) from squared
matrix

Parameters:
-----------

km:  kernel matrix

rows:  number of rows in matrix

cols:  number of cols in matrix

if setting was successful ";

%feature("docstring")
shogun::CCustomKernel::set_full_kernel_matrix_from_full "

set full kernel matrix from full kernel matrix

Parameters:
-----------

km:  kernel matrix

rows:  number of rows in matrix

cols:  number of cols in matrix

if setting was successful ";

%feature("docstring")  shogun::CCustomKernel::get_num_vec_lhs "

get number of vectors of lhs features

number of vectors of left-hand side ";

%feature("docstring")  shogun::CCustomKernel::get_num_vec_rhs "

get number of vectors of rhs features

number of vectors of right-hand side ";

%feature("docstring")  shogun::CCustomKernel::has_features "

test whether features have been assigned to lhs and rhs

true if features are assigned ";


// File: classshogun_1_1CDiagKernel.xml
%feature("docstring") shogun::CDiagKernel "

The Diagonal Kernel returns a constant for the diagonal and zero
otherwise.

A kernel that returns zero for all non-diagonal elements and a single
constant otherwise, i.e. $k({\\\\bf x_i}, {\\\\bf x_j})=
\\\\delta_{ij} c$

C++ includes: DiagKernel.h ";

%feature("docstring")  shogun::CDiagKernel::CDiagKernel "

constructor

Parameters:
-----------

size:  cache size

diag:  diagonal ";

%feature("docstring")  shogun::CDiagKernel::CDiagKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

diag:  diagonal ";

%feature("docstring")  shogun::CDiagKernel::~CDiagKernel "";

%feature("docstring")  shogun::CDiagKernel::get_feature_type "

return feature type the kernel can deal with

feature type ANY ";

%feature("docstring")  shogun::CDiagKernel::get_feature_class "

return feature class the kernel can deal with

feature class ANY ";

%feature("docstring")  shogun::CDiagKernel::get_kernel_type "

return what type of kernel we are

kernel type CUSTOM ";

%feature("docstring")  shogun::CDiagKernel::get_name "

return the kernel's name

name Custom ";


// File: classshogun_1_1CDiceKernelNormalizer.xml
%feature("docstring") shogun::CDiceKernelNormalizer "

DiceKernelNormalizer performs kernel normalization inspired by the
Dice coefficient (seehttp://en.wikipedia.org/wiki/Dice's_coefficient).

\\\\[ k'({\\\\bf x},{\\\\bf x'}) = \\\\frac{2k({\\\\bf x},{\\\\bf
x'})}{k({\\\\bf x},{\\\\bf x})+k({\\\\bf x'},{\\\\bf x'}} \\\\]

C++ includes: DiceKernelNormalizer.h ";

%feature("docstring")
shogun::CDiceKernelNormalizer::CDiceKernelNormalizer "

default constructor

Parameters:
-----------

use_opt_diag:  - some kernels support faster diagonal compuation via
compute_diag(idx), this flag enables this ";

%feature("docstring")
shogun::CDiceKernelNormalizer::~CDiceKernelNormalizer "

default destructor ";

%feature("docstring")  shogun::CDiceKernelNormalizer::init "

initialization of the normalizer

Parameters:
-----------

k:  kernel ";

%feature("docstring")  shogun::CDiceKernelNormalizer::normalize "

normalize the kernel value

Parameters:
-----------

value:  kernel value

idx_lhs:  index of left hand side vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")  shogun::CDiceKernelNormalizer::normalize_lhs "

normalize only the left hand side vector

Parameters:
-----------

value:  value of a component of the left hand side feature vector

idx_lhs:  index of left hand side vector ";

%feature("docstring")  shogun::CDiceKernelNormalizer::normalize_rhs "

normalize only the right hand side vector

Parameters:
-----------

value:  value of a component of the right hand side feature vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")
shogun::CDiceKernelNormalizer::alloc_and_compute_diag "

alloc and compute the vector containing the square root of the
diagonal elements of this kernel. ";


// File: classshogun_1_1CDistanceKernel.xml
%feature("docstring") shogun::CDistanceKernel "

The Distance kernel takes a distance as input.

It turns a distance into something kernel like by computing

\\\\[ k({\\\\bf x}, {\\\\bf x'}) = e^{-\\\\frac{dist({\\\\bf x},
{\\\\bf x'})}{width}} \\\\]

C++ includes: DistanceKernel.h ";

%feature("docstring")  shogun::CDistanceKernel::CDistanceKernel "

constructor

Parameters:
-----------

cache:  cache size

width:  width

dist:  distance ";

%feature("docstring")  shogun::CDistanceKernel::CDistanceKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

width:  width

dist:  distance ";

%feature("docstring")  shogun::CDistanceKernel::~CDistanceKernel "";

%feature("docstring")  shogun::CDistanceKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CDistanceKernel::get_kernel_type "

return what type of kernel we are

kernel type DISTANCE ";

%feature("docstring")  shogun::CDistanceKernel::get_feature_type "

return feature type the kernel can deal with

feature type of distance used ";

%feature("docstring")  shogun::CDistanceKernel::get_feature_class "

return feature class the kernel can deal with

feature class of distance used ";

%feature("docstring")  shogun::CDistanceKernel::get_name "

return the kernel's name

name Distance ";


// File: classshogun_1_1CFirstElementKernelNormalizer.xml
%feature("docstring") shogun::CFirstElementKernelNormalizer "

Normalize the kernel by a constant obtained from the first element of
the kernel matrix, i.e. $ c=k({\\\\bf x},{\\\\bf x})$.

\\\\[ k'(x,x')= \\\\frac{k(x,x')}{c} \\\\]

useful if the kernel returns constant elements along the diagonal
anyway and all one wants is to scale the kernel down to 1 on the
diagonal.

C++ includes: FirstElementKernelNormalizer.h ";

%feature("docstring")
shogun::CFirstElementKernelNormalizer::CFirstElementKernelNormalizer "

constructor ";

%feature("docstring")
shogun::CFirstElementKernelNormalizer::~CFirstElementKernelNormalizer
"

default destructor ";

%feature("docstring")  shogun::CFirstElementKernelNormalizer::init "

initialization of the normalizer (if needed)

Parameters:
-----------

k:  kernel ";

%feature("docstring")
shogun::CFirstElementKernelNormalizer::normalize "

normalize the kernel value

Parameters:
-----------

value:  kernel value

idx_lhs:  index of left hand side vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")
shogun::CFirstElementKernelNormalizer::normalize_lhs "

normalize only the left hand side vector

Parameters:
-----------

value:  value of a component of the left hand side feature vector

idx_lhs:  index of left hand side vector ";

%feature("docstring")
shogun::CFirstElementKernelNormalizer::normalize_rhs "

normalize only the right hand side vector

Parameters:
-----------

value:  value of a component of the right hand side feature vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")  shogun::CFirstElementKernelNormalizer::get_name
"

object name ";


// File: classshogun_1_1CFixedDegreeStringKernel.xml
%feature("docstring") shogun::CFixedDegreeStringKernel "

The FixedDegree String kernel takes as input two strings of same size
and counts the number of matches of length d.

\\\\[ k({\\\\bf x}, {\\\\bf x'}) = \\\\sum_{i=0}^{l-d} I({\\\\bf
x}_{i,i+1,\\\\dots,i+d-1} = {\\\\bf x'}_{i,i+1,\\\\dots,i+d-1}) \\\\]

Note that additional normalisation is applied, i.e. \\\\[ k'({\\\\bf
x}, {\\\\bf x'})=\\\\frac{k({\\\\bf x}, {\\\\bf
x'})}{\\\\sqrt{k({\\\\bf x}, {\\\\bf x})k({\\\\bf x'}, {\\\\bf x'})}}
\\\\]

C++ includes: FixedDegreeStringKernel.h ";

%feature("docstring")
shogun::CFixedDegreeStringKernel::CFixedDegreeStringKernel "

constructor

Parameters:
-----------

size:  cache size

degree:  the degree ";

%feature("docstring")
shogun::CFixedDegreeStringKernel::CFixedDegreeStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

degree:  the degree ";

%feature("docstring")
shogun::CFixedDegreeStringKernel::~CFixedDegreeStringKernel "";

%feature("docstring")  shogun::CFixedDegreeStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CFixedDegreeStringKernel::cleanup "

clean up kernel ";

%feature("docstring")
shogun::CFixedDegreeStringKernel::get_kernel_type "

return what type of kernel we are

kernel type FIXEDDEGREE ";

%feature("docstring")  shogun::CFixedDegreeStringKernel::get_name "

return the kernel's name

name FixedDegree ";


// File: classshogun_1_1CGaussianKernel.xml
%feature("docstring") shogun::CGaussianKernel "

The well known Gaussian kernel (swiss army knife for SVMs) on dense
real valued features.

It is computed as

\\\\[ k({\\\\bf x},{\\\\bf x'})= exp(-\\\\frac{||{\\\\bf x}-{\\\\bf
x'}||^2}{\\\\tau}) \\\\]

where $\\\\tau$ is the kernel width.

C++ includes: GaussianKernel.h ";

%feature("docstring")  shogun::CGaussianKernel::CGaussianKernel "

default constructor ";

%feature("docstring")  shogun::CGaussianKernel::CGaussianKernel "

constructor

Parameters:
-----------

size:  cache size

width:  width ";

%feature("docstring")  shogun::CGaussianKernel::CGaussianKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

width:  width

size:  cache size ";

%feature("docstring")  shogun::CGaussianKernel::~CGaussianKernel "";

%feature("docstring")  shogun::CGaussianKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CGaussianKernel::get_kernel_type "

return what type of kernel we are

kernel type GAUSSIAN ";

%feature("docstring")  shogun::CGaussianKernel::get_name "

return the kernel's name

name Gaussian ";


// File: classshogun_1_1CGaussianShiftKernel.xml
%feature("docstring") shogun::CGaussianShiftKernel "

An experimental kernel inspired by the
WeightedDegreePositionStringKernel and the Gaussian kernel.

It is computed as

\\\\[ k({\\\\bf x},{\\\\bf x'})= \\\\exp(-\\\\frac{||{\\\\bf
x}-{\\\\bf x'}||^2}{\\\\tau}) +
\\\\sum_{s=1}^{S_{\\\\mathrm{max}}/S_{\\\\mathrm{step}}}
\\\\frac{1}{2s} \\\\exp(-\\\\frac{||{\\\\bf x}_{[1:|{\\\\bf
x}|-sS_{\\\\mathrm{step}}]}-{\\\\bf
x'}_{[sS_{\\\\mathrm{step}}:|{\\\\bf x}|]}||^2}{\\\\tau}) +
\\\\sum_{s=1}^{S_{max}/S_{step}} \\\\frac{1}{2s}
\\\\exp(-\\\\frac{||{\\\\bf x}_{[sS_{\\\\mathrm{step}}:|{\\\\bf
x}|]}-{\\\\bf x'}_{[1:|{\\\\bf
x}|-sS_{\\\\mathrm{step}}]}||^2}{\\\\tau}) + \\\\]

where $\\\\tau$ is the kernel width. The idea is to shift the
dimensions of the input vectors against eachother.
$S_{\\\\mathrm{step}}$ is the step size (parameter shift_step) of the
shifts and $S_{\\\\mathrm{max}}$ (parameter max_shift) is the maximal
shift.

C++ includes: GaussianShiftKernel.h ";

%feature("docstring")
shogun::CGaussianShiftKernel::CGaussianShiftKernel "

constructor

Parameters:
-----------

size:  cache size

width:  width

max_shift:  maximum shift

shift_step:  shift step ";

%feature("docstring")
shogun::CGaussianShiftKernel::CGaussianShiftKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

width:  width

max_shift:  maximum shift

shift_step:  shift step

size:  cache size ";

%feature("docstring")
shogun::CGaussianShiftKernel::~CGaussianShiftKernel "";

%feature("docstring")  shogun::CGaussianShiftKernel::get_kernel_type "

return what type of kernel we are

kernel type GAUSSIANSHIFT ";

%feature("docstring")  shogun::CGaussianShiftKernel::get_name "

return the kernel's name

name GaussianShift ";


// File: classshogun_1_1CGaussianShortRealKernel.xml
%feature("docstring") shogun::CGaussianShortRealKernel "

The well known Gaussian kernel (swiss army knife for SVMs) on dense
short-real valued features.

It is computed as

\\\\[ k({\\\\bf x},{\\\\bf x'})= exp(-\\\\frac{||{\\\\bf x}-{\\\\bf
x'}||^2}{\\\\tau}) \\\\]

where $\\\\tau$ is the kernel width.

C++ includes: GaussianShortRealKernel.h ";

%feature("docstring")
shogun::CGaussianShortRealKernel::CGaussianShortRealKernel "

constructor

Parameters:
-----------

size:  cache size

width:  width ";

%feature("docstring")
shogun::CGaussianShortRealKernel::CGaussianShortRealKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

width:  width

size:  cache size ";

%feature("docstring")
shogun::CGaussianShortRealKernel::~CGaussianShortRealKernel "";

%feature("docstring")  shogun::CGaussianShortRealKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")
shogun::CGaussianShortRealKernel::get_kernel_type "

return what type of kernel we are

kernel type GAUSSIAN ";

%feature("docstring")  shogun::CGaussianShortRealKernel::get_name "

return the kernel's name

name GaussianShortReal ";


// File: classshogun_1_1CHistogramWordStringKernel.xml
%feature("docstring") shogun::CHistogramWordStringKernel "

The HistogramWordString computes the TOP kernel on inhomogeneous
Markov Chains.

C++ includes: HistogramWordStringKernel.h ";

%feature("docstring")
shogun::CHistogramWordStringKernel::CHistogramWordStringKernel "

constructor

Parameters:
-----------

size:  cache size

pie:  plugin estimate ";

%feature("docstring")
shogun::CHistogramWordStringKernel::CHistogramWordStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

pie:  plugin estimate ";

%feature("docstring")
shogun::CHistogramWordStringKernel::~CHistogramWordStringKernel "";

%feature("docstring")  shogun::CHistogramWordStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CHistogramWordStringKernel::cleanup "

clean up kernel ";

%feature("docstring")
shogun::CHistogramWordStringKernel::get_kernel_type "

return what type of kernel we are

kernel type HISTOGRAM ";

%feature("docstring")  shogun::CHistogramWordStringKernel::get_name "

return the kernel's name

name Histogram ";


// File: classshogun_1_1CIdentityKernelNormalizer.xml
%feature("docstring") shogun::CIdentityKernelNormalizer "

Identity Kernel Normalization, i.e. no normalization is applied.

C++ includes: IdentityKernelNormalizer.h ";

%feature("docstring")
shogun::CIdentityKernelNormalizer::CIdentityKernelNormalizer "

default constructor ";

%feature("docstring")
shogun::CIdentityKernelNormalizer::~CIdentityKernelNormalizer "

default destructor ";

%feature("docstring")  shogun::CIdentityKernelNormalizer::init "

initialization of the normalizer (if needed)

Parameters:
-----------

k:  kernel ";

%feature("docstring")  shogun::CIdentityKernelNormalizer::normalize "

normalize the kernel value

Parameters:
-----------

value:  kernel value

idx_lhs:  index of left hand side vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")
shogun::CIdentityKernelNormalizer::normalize_lhs "

normalize only the left hand side vector

Parameters:
-----------

value:  value of a component of the left hand side feature vector

idx_lhs:  index of left hand side vector ";

%feature("docstring")
shogun::CIdentityKernelNormalizer::normalize_rhs "

normalize only the right hand side vector

Parameters:
-----------

value:  value of a component of the right hand side feature vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")  shogun::CIdentityKernelNormalizer::get_name "

object name ";


// File: classshogun_1_1CKernel.xml
%feature("docstring") shogun::CKernel "

The Kernel base class.

Non-mathematically spoken, a kernel is a function that given two input
objects ${\\\\bf x}$ and ${\\\\bf x'}$ returns a score describing the
similarity of the vectors. The score should be larger when the objects
are more similar.

It can be defined as

\\\\[ k({\\\\bf x},{\\\\bf x'})= \\\\Phi_k({\\\\bf x})\\\\cdot
\\\\Phi_k({\\\\bf x'}) \\\\]

where $\\\\Phi$ maps the objects into some potentially high
dimensional feature space.

Apart from the input features, the base kernel takes only one argument
(the size of the kernel cache) that is used to efficiently train
kernel-machines like e.g. SVMs.

In case you would like to define your own kernel, you only have to
define a new compute() function (and the kernel name via get_name()
and the kernel type get_kernel_type()). A good example to look at is
the GaussianKernel.

C++ includes: Kernel.h ";

%feature("docstring")  shogun::CKernel::CKernel "

default constructor ";

%feature("docstring")  shogun::CKernel::CKernel "

constructor

Parameters:
-----------

size:  cache size ";

%feature("docstring")  shogun::CKernel::CKernel "

constructor

Parameters:
-----------

l:  features for left-hand side

r:  features for right-hand side

size:  cache size ";

%feature("docstring")  shogun::CKernel::~CKernel "";

%feature("docstring")  shogun::CKernel::kernel "

get kernel function for lhs feature vector a and rhs feature vector b

Parameters:
-----------

idx_a:  index of feature vector a

idx_b:  index of feature vector b

computed kernel function ";

%feature("docstring")  shogun::CKernel::get_kernel_matrix "

get kernel matrix

Parameters:
-----------

dst:  destination where matrix will be stored

m:  dimension m of matrix

n:  dimension n of matrix ";

%feature("docstring")  shogun::CKernel::get_kernel_matrix "

get kernel matrix real

Parameters:
-----------

m:  dimension m of matrix

n:  dimension n of matrix

target:  the kernel matrix

the kernel matrix ";

%feature("docstring")  shogun::CKernel::init "

initialize kernel e.g. setup lhs/rhs of kernel, precompute
normalization constants etc. make sure to check that your kernel can
deal with the supplied features (!)

Parameters:
-----------

lhs:  features for left-hand side

rhs:  features for right-hand side

if init was successful ";

%feature("docstring")  shogun::CKernel::set_normalizer "

set the current kernel normalizer

if successful ";

%feature("docstring")  shogun::CKernel::get_normalizer "

obtain the current kernel normalizer

the kernel normalizer ";

%feature("docstring")  shogun::CKernel::init_normalizer "

initialize the current kernel normalizer if init was successful ";

%feature("docstring")  shogun::CKernel::cleanup "

clean up your kernel

base method only removes lhs and rhs overload to add further cleanup
but make sure CKernel::cleanup() is called ";

%feature("docstring")  shogun::CKernel::load "

load the kernel matrix

Parameters:
-----------

fname:  filename to load from

if loading was successful ";

%feature("docstring")  shogun::CKernel::save "

save kernel matrix

Parameters:
-----------

fname:  filename to save to

if saving was successful ";

%feature("docstring")  shogun::CKernel::get_lhs "

get left-hand side of features used in kernel

features of left-hand side ";

%feature("docstring")  shogun::CKernel::get_rhs "

get right-hand side of features used in kernel

features of right-hand side ";

%feature("docstring")  shogun::CKernel::get_num_vec_lhs "

get number of vectors of lhs features

number of vectors of left-hand side ";

%feature("docstring")  shogun::CKernel::get_num_vec_rhs "

get number of vectors of rhs features

number of vectors of right-hand side ";

%feature("docstring")  shogun::CKernel::has_features "

test whether features have been assigned to lhs and rhs

true if features are assigned ";

%feature("docstring")  shogun::CKernel::lhs_equals_rhs "

test whether features on lhs and rhs are the same

true if features are the same ";

%feature("docstring")  shogun::CKernel::remove_lhs_and_rhs "

remove lhs and rhs from kernel ";

%feature("docstring")  shogun::CKernel::remove_lhs "

remove lhs from kernel ";

%feature("docstring")  shogun::CKernel::remove_rhs "

remove rhs from kernel ";

%feature("docstring")  shogun::CKernel::get_kernel_type "

return what type of kernel we are, e.g. Linear,Polynomial,
Gaussian,...

abstract base method

kernel type ";

%feature("docstring")  shogun::CKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CKernel::get_feature_class "

return feature class the kernel can deal with

abstract base method

feature class ";

%feature("docstring")  shogun::CKernel::set_cache_size "

set the size of the kernel cache

Parameters:
-----------

size:  of kernel cache ";

%feature("docstring")  shogun::CKernel::get_cache_size "

return the size of the kernel cache

size of kernel cache ";

%feature("docstring")  shogun::CKernel::list_kernel "

list kernel ";

%feature("docstring")  shogun::CKernel::has_property "

check if kernel has given property

Parameters:
-----------

p:  kernel property

if kernel has given property ";

%feature("docstring")  shogun::CKernel::clear_normal "

for optimizable kernels, i.e. kernels where the weight vector can be
computed explicitly (if it fits into memory) ";

%feature("docstring")  shogun::CKernel::add_to_normal "

add vector*factor to 'virtual' normal vector

Parameters:
-----------

vector_idx:  index

weight:  weight ";

%feature("docstring")  shogun::CKernel::get_optimization_type "

get optimization type

optimization type ";

%feature("docstring")  shogun::CKernel::set_optimization_type "

set optimization type

Parameters:
-----------

t:  optimization type to set ";

%feature("docstring")  shogun::CKernel::get_is_initialized "

check if optimization is initialized

if optimization is initialized ";

%feature("docstring")  shogun::CKernel::init_optimization "

initialize optimization

Parameters:
-----------

count:  count

IDX:  index

weights:  weights

if initializing was successful ";

%feature("docstring")  shogun::CKernel::delete_optimization "

delete optimization

if deleting was successful ";

%feature("docstring")  shogun::CKernel::init_optimization_svm "

initialize optimization

Parameters:
-----------

svm:  svm model

if initializing was successful ";

%feature("docstring")  shogun::CKernel::compute_optimized "

compute optimized

Parameters:
-----------

vector_idx:  index to compute

optimized value at given index ";

%feature("docstring")  shogun::CKernel::compute_batch "

computes output for a batch of examples in an optimized fashion
(favorable if kernel supports it, i.e. has KP_BATCHEVALUATION. to the
outputvector target (of length num_vec elements) the output for the
examples enumerated in vec_idx are added. therefore make sure that it
is initialized with ZERO. the following num_suppvec, IDX, alphas
arguments are the number of support vectors, their indices and weights
";

%feature("docstring")  shogun::CKernel::get_combined_kernel_weight "

get combined kernel weight

combined kernel weight ";

%feature("docstring")  shogun::CKernel::set_combined_kernel_weight "

set combined kernel weight

Parameters:
-----------

nw:  new combined kernel weight ";

%feature("docstring")  shogun::CKernel::get_num_subkernels "

get number of subkernels

number of subkernels ";

%feature("docstring")  shogun::CKernel::compute_by_subkernel "

compute by subkernel

Parameters:
-----------

vector_idx:  index

subkernel_contrib:  subkernel contribution ";

%feature("docstring")  shogun::CKernel::get_subkernel_weights "

get subkernel weights

Parameters:
-----------

num_weights:  number of weights will be stored here

subkernel weights ";

%feature("docstring")  shogun::CKernel::set_subkernel_weights "

set subkernel weights

Parameters:
-----------

weights:  subkernel weights

num_weights:  number of weights ";


// File: classshogun_1_1CKernelNormalizer.xml
%feature("docstring") shogun::CKernelNormalizer "

The class Kernel Normalizer defines a function to postprocess kernel
values.

Formally it defines f(.,.,.)

\\\\[ k'({\\\\bf x},{\\\\bf x'}) = f(k({\\\\bf x},{\\\\bf x'}),{\\\\bf
x},{\\\\bf x'}) \\\\]

examples for f(.,.,.) would be scaling with a constant

\\\\[ f(k({\\\\bf x},{\\\\bf x'}), ., .)= \\\\frac{1}{c}\\\\cdot
k({\\\\bf x},{\\\\bf x'}) \\\\]

as can be found in class CAvgDiagKernelNormalizer, the identity (cf.
CIdentityKernelNormalizer), dividing by the Square Root of the product
of the diagonal elements which effectively normalizes the vectors in
feature space to norm 1 (see CSqrtDiagKernelNormalizer)

\\\\[ k'({\\\\bf x},{\\\\bf x'}) = \\\\frac{k({\\\\bf x},{\\\\bf
x'})}{\\\\sqrt{k({\\\\bf x},{\\\\bf x})k({\\\\bf x'},{\\\\bf x'})}}
\\\\]

C++ includes: KernelNormalizer.h ";

%feature("docstring")  shogun::CKernelNormalizer::CKernelNormalizer "

default constructor ";

%feature("docstring")  shogun::CKernelNormalizer::~CKernelNormalizer "

default destructor ";

%feature("docstring")  shogun::CKernelNormalizer::init "

initialization of the normalizer (if needed)

Parameters:
-----------

k:  kernel ";

%feature("docstring")  shogun::CKernelNormalizer::normalize "

normalize the kernel value

Parameters:
-----------

value:  kernel value

idx_lhs:  index of left hand side vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")  shogun::CKernelNormalizer::normalize_lhs "

normalize only the left hand side vector

Parameters:
-----------

value:  value of a component of the left hand side feature vector

idx_lhs:  index of left hand side vector ";

%feature("docstring")  shogun::CKernelNormalizer::normalize_rhs "

normalize only the right hand side vector

Parameters:
-----------

value:  value of a component of the right hand side feature vector

idx_rhs:  index of right hand side vector ";


// File: classshogun_1_1CLinearByteKernel.xml
%feature("docstring") shogun::CLinearByteKernel "

Computes the standard linear kernel on dense byte valued features.

Formally, it computes

\\\\[ k({\\\\bf x},{\\\\bf x'})= {\\\\bf x}\\\\cdot {\\\\bf x'} \\\\]

C++ includes: LinearByteKernel.h ";

%feature("docstring")  shogun::CLinearByteKernel::CLinearByteKernel "

constructor ";

%feature("docstring")  shogun::CLinearByteKernel::CLinearByteKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CLinearByteKernel::~CLinearByteKernel "";

%feature("docstring")  shogun::CLinearByteKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CLinearByteKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CLinearByteKernel::get_kernel_type "

return what type of kernel we are

kernel type LINEAR ";

%feature("docstring")  shogun::CLinearByteKernel::get_name "

return the kernel's name

name FixedDegree ";

%feature("docstring")  shogun::CLinearByteKernel::init_optimization "

optimizable kernel, i.e. precompute normal vector and as phi(x) = x do
scalar product in input space

Parameters:
-----------

num_suppvec:  number of support vectors

sv_idx:  support vector index

alphas:  alphas

if optimization was successful ";

%feature("docstring")  shogun::CLinearByteKernel::delete_optimization
"

delete optimization

if deleting was successful ";

%feature("docstring")  shogun::CLinearByteKernel::compute_optimized "

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")  shogun::CLinearByteKernel::clear_normal "

clear normal vector ";

%feature("docstring")  shogun::CLinearByteKernel::add_to_normal "

add to normal vector

Parameters:
-----------

idx:  where to add

weight:  what to add ";


// File: classshogun_1_1CLinearKernel.xml
%feature("docstring") shogun::CLinearKernel "

Computes the standard linear kernel on dense real valued features.

Formally, it computes

\\\\[ k({\\\\bf x},{\\\\bf x'})= {\\\\bf x}\\\\cdot {\\\\bf x'} \\\\]

C++ includes: LinearKernel.h ";

%feature("docstring")  shogun::CLinearKernel::CLinearKernel "

constructor ";

%feature("docstring")  shogun::CLinearKernel::CLinearKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CLinearKernel::~CLinearKernel "";

%feature("docstring")  shogun::CLinearKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CLinearKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CLinearKernel::get_kernel_type "

return what type of kernel we are

kernel type LINEAR ";

%feature("docstring")  shogun::CLinearKernel::get_name "

return the kernel's name

name Lineaer ";

%feature("docstring")  shogun::CLinearKernel::init_optimization "

optimizable kernel, i.e. precompute normal vector and as phi(x) = x do
scalar product in input space

Parameters:
-----------

num_suppvec:  number of support vectors

sv_idx:  support vector index

alphas:  alphas

if optimization was successful ";

%feature("docstring")  shogun::CLinearKernel::delete_optimization "

delete optimization

if deleting was successful ";

%feature("docstring")  shogun::CLinearKernel::compute_optimized "

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")  shogun::CLinearKernel::clear_normal "

clear normal vector ";

%feature("docstring")  shogun::CLinearKernel::add_to_normal "

add to normal vector

Parameters:
-----------

idx:  where to add

weight:  what to add ";

%feature("docstring")  shogun::CLinearKernel::get_normal "

get normal

Parameters:
-----------

len:  where length of normal vector will be stored

normal vector ";

%feature("docstring")  shogun::CLinearKernel::get_w "

get normal vector (swig compatible)

Parameters:
-----------

dst_w:  store w in this argument

dst_dims:  dimension of w ";

%feature("docstring")  shogun::CLinearKernel::set_w "

set normal vector (swig compatible)

Parameters:
-----------

src_w:  new w

src_w_dim:  dimension of new w - must fit dim of lhs ";


// File: classshogun_1_1CLinearStringKernel.xml
%feature("docstring") shogun::CLinearStringKernel "

Computes the standard linear kernel on dense char valued features.

Formally, it computes

\\\\[ k({\\\\bf x},{\\\\bf x'})= \\\\frac{1}{scale}{\\\\bf x}\\\\cdot
{\\\\bf x'} \\\\]

Note: Basically the same as LinearByteKernel but on signed chars.

C++ includes: LinearStringKernel.h ";

%feature("docstring")
shogun::CLinearStringKernel::CLinearStringKernel "

constructor ";

%feature("docstring")
shogun::CLinearStringKernel::CLinearStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")
shogun::CLinearStringKernel::~CLinearStringKernel "";

%feature("docstring")  shogun::CLinearStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CLinearStringKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CLinearStringKernel::get_kernel_type "

return what type of kernel we are

kernel type LINEAR ";

%feature("docstring")  shogun::CLinearStringKernel::get_name "

return the kernel's name

name Linear ";

%feature("docstring")  shogun::CLinearStringKernel::init_optimization
"

optimizable kernel, i.e. precompute normal vector and as phi(x) = x do
scalar product in input space

Parameters:
-----------

num_suppvec:  number of support vectors

sv_idx:  support vector index

alphas:  alphas

if optimization was successful ";

%feature("docstring")
shogun::CLinearStringKernel::delete_optimization "

delete optimization

if deleting was successful ";

%feature("docstring")  shogun::CLinearStringKernel::compute_optimized
"

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")  shogun::CLinearStringKernel::clear_normal "

clear normal ";

%feature("docstring")  shogun::CLinearStringKernel::add_to_normal "

add to normal

Parameters:
-----------

idx:  where to add

weight:  what to add ";


// File: classshogun_1_1CLinearWordKernel.xml
%feature("docstring") shogun::CLinearWordKernel "

Computes the standard linear kernel on dense word (2-byte) valued
features.

Formally, it computes

\\\\[ k({\\\\bf x},{\\\\bf x'})= \\\\frac{1}{scale}{\\\\bf x}\\\\cdot
{\\\\bf x'} \\\\]

C++ includes: LinearWordKernel.h ";

%feature("docstring")  shogun::CLinearWordKernel::CLinearWordKernel "

constructor ";

%feature("docstring")  shogun::CLinearWordKernel::CLinearWordKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CLinearWordKernel::~CLinearWordKernel "";

%feature("docstring")  shogun::CLinearWordKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CLinearWordKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CLinearWordKernel::get_kernel_type "

return what type of kernel we are

kernel type LINEAR ";

%feature("docstring")  shogun::CLinearWordKernel::get_name "

return the kernel's name

name Linear ";

%feature("docstring")  shogun::CLinearWordKernel::init_optimization "

initialize optimization

Parameters:
-----------

num_suppvec:  number of support vectors

sv_idx:  support vector index

alphas:  alphas

if initializing was successful ";

%feature("docstring")  shogun::CLinearWordKernel::delete_optimization
"

delete optimization

if deleting was successful ";

%feature("docstring")  shogun::CLinearWordKernel::compute_optimized "

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")  shogun::CLinearWordKernel::clear_normal "

clear normal ";

%feature("docstring")  shogun::CLinearWordKernel::add_to_normal "

add to normal

Parameters:
-----------

idx:  where to add

weight:  what to add ";


// File: classshogun_1_1CLocalAlignmentStringKernel.xml
%feature("docstring") shogun::CLocalAlignmentStringKernel "

The LocalAlignmentString kernel compares two sequences through all
possible local alignments between the two sequences.

The implementation is taken fromhttp://www.mloss.org/software/view/40/
and only adjusted to work with shogun.

C++ includes: LocalAlignmentStringKernel.h ";

%feature("docstring")
shogun::CLocalAlignmentStringKernel::CLocalAlignmentStringKernel "

constructor

Parameters:
-----------

size:  cache size ";

%feature("docstring")
shogun::CLocalAlignmentStringKernel::CLocalAlignmentStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")
shogun::CLocalAlignmentStringKernel::~CLocalAlignmentStringKernel "";

%feature("docstring")  shogun::CLocalAlignmentStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CLocalAlignmentStringKernel::cleanup "

clean up kernel ";

%feature("docstring")
shogun::CLocalAlignmentStringKernel::get_kernel_type "

return what type of kernel we are

kernel type LOCALALIGNMENT ";

%feature("docstring")  shogun::CLocalAlignmentStringKernel::get_name "

return the kernel's name

name LocalAlignment ";


// File: classshogun_1_1CLocalityImprovedStringKernel.xml
%feature("docstring") shogun::CLocalityImprovedStringKernel "

The LocalityImprovedString kernel is inspired by the polynomial
kernel. Comparing neighboring characters it puts emphasize on local
features.

It can be defined as \\\\[ K({\\\\bf x},{\\\\bf
x'})=\\\\left(\\\\sum_{i=0}^{T-1}\\\\left(\\\\sum_{j=-l}^{+l}w_jI_{i+j}({\\\\bf
x},{\\\\bf x'})\\\\right)^{d_1}\\\\right)^{d_2}, \\\\] where $
I_i({\\\\bf x},{\\\\bf x'})=1$ if $x_i=x'_i$ and 0 otherwise.

C++ includes: LocalityImprovedStringKernel.h ";

%feature("docstring")
shogun::CLocalityImprovedStringKernel::CLocalityImprovedStringKernel "

constructor

Parameters:
-----------

size:  cache size

length:  length

inner_degree:  inner degree

outer_degree:  outer degree ";

%feature("docstring")
shogun::CLocalityImprovedStringKernel::CLocalityImprovedStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

length:  length

inner_degree:  inner degree

outer_degree:  outer degree ";

%feature("docstring")
shogun::CLocalityImprovedStringKernel::~CLocalityImprovedStringKernel
"";

%feature("docstring")  shogun::CLocalityImprovedStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")
shogun::CLocalityImprovedStringKernel::get_kernel_type "

return what type of kernel we are

kernel type LOCALITYIMPROVED ";

%feature("docstring")  shogun::CLocalityImprovedStringKernel::get_name
"

return the kernel's name

name LocalityImproved ";


// File: classshogun_1_1CMatchWordStringKernel.xml
%feature("docstring") shogun::CMatchWordStringKernel "

The class MatchWordStringKernel computes a variant of the polynomial
kernel on strings of same length converted to a word alphabet.

It is computed as

\\\\[ k({\\\\bf x},{\\\\bf x'})= \\\\sum_{i=0}^L I(x_i=x'_i)+c)^d
\\\\]

where I is the indicator function which evaluates to 1 if its argument
is true and to 0 otherwise.

Note that additional normalisation is applied, i.e. \\\\[ k'({\\\\bf
x}, {\\\\bf x'})=\\\\frac{k({\\\\bf x}, {\\\\bf
x'})}{\\\\sqrt{k({\\\\bf x}, {\\\\bf x})k({\\\\bf x'}, {\\\\bf x'})}}
\\\\]

C++ includes: MatchWordStringKernel.h ";

%feature("docstring")
shogun::CMatchWordStringKernel::CMatchWordStringKernel "

constructor

Parameters:
-----------

size:  cache size

d:  degree ";

%feature("docstring")
shogun::CMatchWordStringKernel::CMatchWordStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

degree:  degree ";

%feature("docstring")
shogun::CMatchWordStringKernel::~CMatchWordStringKernel "";

%feature("docstring")  shogun::CMatchWordStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CMatchWordStringKernel::get_kernel_type
"

return what type of kernel we are

kernel type LINEAR ";

%feature("docstring")  shogun::CMatchWordStringKernel::get_name "

return the kernel's name

name MatchWordString ";


// File: classshogun_1_1CMultitaskKernelNormalizer.xml
%feature("docstring") shogun::CMultitaskKernelNormalizer "

The MultitaskKernel allows Multitask Learning via a modified kernel
function.

This effectively normalizes the vectors in feature space to norm 1
(see CSqrtDiagKernelNormalizer)

\\\\[ k'({\\\\bf x},{\\\\bf x'}) = ... \\\\]

C++ includes: MultitaskKernelNormalizer.h ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::CMultitaskKernelNormalizer "

default constructor ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::CMultitaskKernelNormalizer "

default constructor

Parameters:
-----------

task_lhs:  task vector with containing task_id for each example for
left hand side

task_rhs:  task vector with containing task_id for each example for
right hand side ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::~CMultitaskKernelNormalizer "

default destructor ";

%feature("docstring")  shogun::CMultitaskKernelNormalizer::init "

initialization of the normalizer

Parameters:
-----------

k:  kernel ";

%feature("docstring")  shogun::CMultitaskKernelNormalizer::normalize "

normalize the kernel value

Parameters:
-----------

value:  kernel value

idx_lhs:  index of left hand side vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::normalize_lhs "

normalize only the left hand side vector

Parameters:
-----------

value:  value of a component of the left hand side feature vector

idx_lhs:  index of left hand side vector ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::normalize_rhs "

normalize only the right hand side vector

Parameters:
-----------

value:  value of a component of the right hand side feature vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::get_task_vector_lhs "

vec task vector with containing task_id for each example on left hand
side ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::set_task_vector_lhs "

Parameters:

vec:  task vector with containing task_id for each example ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::get_task_vector_rhs "

vec task vector with containing task_id for each example on right hand
side ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::set_task_vector_rhs "

Parameters:

vec:  task vector with containing task_id for each example ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::set_task_vector "

Parameters:

vec:  task vector with containing task_id for each example ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::get_task_similarity "

Parameters:

task_lhs:  task_id on left hand side

task_rhs:  task_id on right hand side

similarity between tasks ";

%feature("docstring")
shogun::CMultitaskKernelNormalizer::set_task_similarity "

Parameters:

task_lhs:  task_id on left hand side

task_rhs:  task_id on right hand side

similarity:  similarity between tasks ";

%feature("docstring")  shogun::CMultitaskKernelNormalizer::get_name "

object name ";


// File: classshogun_1_1COligoStringKernel.xml
%feature("docstring") shogun::COligoStringKernel "

This class offers access to the Oligo Kernel introduced by Meinicke et
al. in 2004.

The class has functions to preprocess the data such that the kernel
computation can be pursued faster. The kernel function is then
kernelOligoFast or kernelOligo.

Requires significant speedup, should be working but as is might be
applicable only to academic small scale problems:

the kernel should only ever see encoded sequences, which however
requires another OligoFeatures object (using CSimpleFeatures of pairs)

Uses CSqrtDiagKernelNormalizer, as the vanilla kernel seems to be very
diagonally dominant.

C++ includes: OligoStringKernel.h ";

%feature("docstring")  shogun::COligoStringKernel::COligoStringKernel
"

Constructor

Parameters:
-----------

cache_size:  cache size for kernel

k:  k-mer length

width:  - equivalent to 2*sigma^2 ";

%feature("docstring")  shogun::COligoStringKernel::~COligoStringKernel
"

Destructor ";

%feature("docstring")  shogun::COligoStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::COligoStringKernel::get_kernel_type "

return what type of kernel we are

kernel type OLIGO ";

%feature("docstring")  shogun::COligoStringKernel::get_name "

return the kernel's name

name Oligo ";

%feature("docstring")  shogun::COligoStringKernel::compute "

compute kernel function for features a and b idx_{a,b} denote the
index of the feature vectors in the corresponding feature object

abstract base method

Parameters:
-----------

x:  index a

y:  index b

computed kernel function at indices a,b ";

%feature("docstring")  shogun::COligoStringKernel::cleanup "

clean up your kernel ";


// File: classshogun_1_1CPolyKernel.xml
%feature("docstring") shogun::CPolyKernel "

Computes the standard polynomial kernel on dense real valued features.

Formally, it computes

\\\\[ k({\\\\bf x},{\\\\bf x'})= ({\\\\bf x}\\\\cdot {\\\\bf x'}+c)^d
\\\\]

Note that additional normalisation is applied, i.e. \\\\[ k'({\\\\bf
x}, {\\\\bf x'})=\\\\frac{k({\\\\bf x}, {\\\\bf
x'})}{\\\\sqrt{k({\\\\bf x}, {\\\\bf x})k({\\\\bf x'}, {\\\\bf x'})}}
\\\\]

C++ includes: PolyKernel.h ";

%feature("docstring")  shogun::CPolyKernel::CPolyKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

d:  degree

inhom:  is inhomogeneous

size:  cache size ";

%feature("docstring")  shogun::CPolyKernel::CPolyKernel "

constructor

Parameters:
-----------

size:  cache size

degree:  degree

inhomogene:  is inhomogeneous ";

%feature("docstring")  shogun::CPolyKernel::~CPolyKernel "";

%feature("docstring")  shogun::CPolyKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CPolyKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CPolyKernel::get_kernel_type "

return what type of kernel we are

kernel type POLY ";

%feature("docstring")  shogun::CPolyKernel::get_name "

return the kernel's name

name Poly ";


// File: classshogun_1_1CPolyMatchStringKernel.xml
%feature("docstring") shogun::CPolyMatchStringKernel "

The class PolyMatchStringKernel computes a variant of the polynomial
kernel on strings of same length.

It is computed as

\\\\[ k({\\\\bf x},{\\\\bf x'})= (\\\\sum_{i=0}^L I(x_i=x'_i)+c)^d
\\\\]

where I is the indicator function which evaluates to 1 if its argument
is true and to 0 otherwise.

Note that additional normalisation is applied, i.e. \\\\[ k'({\\\\bf
x}, {\\\\bf x'})=\\\\frac{k({\\\\bf x}, {\\\\bf
x'})}{\\\\sqrt{k({\\\\bf x}, {\\\\bf x})k({\\\\bf x'}, {\\\\bf x'})}}
\\\\]

C++ includes: PolyMatchStringKernel.h ";

%feature("docstring")
shogun::CPolyMatchStringKernel::CPolyMatchStringKernel "

constructor

Parameters:
-----------

size:  cache size

degree:  degree

inhomogene:  is inhomogeneous ";

%feature("docstring")
shogun::CPolyMatchStringKernel::CPolyMatchStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

degree:  degree

inhomogene:  is inhomogeneous ";

%feature("docstring")
shogun::CPolyMatchStringKernel::~CPolyMatchStringKernel "";

%feature("docstring")  shogun::CPolyMatchStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CPolyMatchStringKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CPolyMatchStringKernel::get_kernel_type
"

return what type of kernel we are

kernel type POLYMATCH ";

%feature("docstring")  shogun::CPolyMatchStringKernel::get_name "

return the kernel's name

name PolyMatchString ";


// File: classshogun_1_1CPolyMatchWordStringKernel.xml
%feature("docstring") shogun::CPolyMatchWordStringKernel "

The class PolyMatchWordStringKernel computes a variant of the
polynomial kernel on word-features.

It makes sense for strings of same length mapped to word features and
is computed as

\\\\[ k({\\\\bf x},{\\\\bf x'})= (\\\\sum_{i=0}^L I(x_i=x'_i)+c)^d
\\\\]

where I is the indicator function which evaluates to 1 if its argument
is true and to 0 otherwise.

Note that additional normalisation is applied, i.e. \\\\[ k'({\\\\bf
x}, {\\\\bf x'})=\\\\frac{k({\\\\bf x}, {\\\\bf
x'})}{\\\\sqrt{k({\\\\bf x}, {\\\\bf x})k({\\\\bf x'}, {\\\\bf x'})}}
\\\\]

C++ includes: PolyMatchWordStringKernel.h ";

%feature("docstring")
shogun::CPolyMatchWordStringKernel::CPolyMatchWordStringKernel "

constructor

Parameters:
-----------

size:  cache size

degree:  degree

inhomogene:  is inhomogeneous ";

%feature("docstring")
shogun::CPolyMatchWordStringKernel::CPolyMatchWordStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

degree:  degree

inhomogene:  is inhomogeneous ";

%feature("docstring")
shogun::CPolyMatchWordStringKernel::~CPolyMatchWordStringKernel "";

%feature("docstring")  shogun::CPolyMatchWordStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CPolyMatchWordStringKernel::cleanup "

clean up kernel ";

%feature("docstring")
shogun::CPolyMatchWordStringKernel::get_kernel_type "

return what type of kernel we are

kernel type POLYMATCH ";

%feature("docstring")  shogun::CPolyMatchWordStringKernel::get_name "

return the kernel's name

name PolyMatchWord ";


// File: classshogun_1_1CPyramidChi2.xml
%feature("docstring") shogun::CPyramidChi2 "

Pyramid Kernel over Chi2 matched histograms.

The Pyramid Chi2 Kernel often used in image classification with sum
inside the exponential. TODO: add adaptive width computation via
median

C++ includes: PyramidChi2.h ";

%feature("docstring")  shogun::CPyramidChi2::CPyramidChi2 "

constructor

Parameters:
-----------

size:  cache size in MB

num_cells2:  - the number of pyramid cells

weights_foreach_cell2:  the vector of weights for each cell with which
the Chi2 distance gets weighted

width_computation_type2:  - 0 use the following parameter as fixed
width, 1- use mean of inner distances, 2 - use median of inner
distances in cases 1 and 2 the value of parameter width is
important!!!

width2:  - in case of width_computation_type ==0 it is the width, in
case of width_computation_type > 0 its value determines the how many
random features are used for determining the width in case of
width_computation_type > 0 set width2 <=1 to use all LEFT HAND SIDE
features for width estimation ";

%feature("docstring")  shogun::CPyramidChi2::CPyramidChi2 "

constructor

Parameters:
-----------

l:  features lhs convention: concatenated features along all cells,
i.e. [feature for cell1, feature for cell2, ... feature for last cell]
, the dimensionality of the base feature is equal to dividing the
total feature length by the number ofcells

r:  features rhs the same convention as for param l applies here

size:  size

num_cells2:  - the number of pyramid cells

weights_foreach_cell2:  the vector of weights for each cell with which
the Chi2 distance gets weighted

width_computation_type:  - 0 use the following parameter as fixed
width, 1- use mean of inner distances in case 1 the value of parameter
width is important!!!

width2:  - in case of width_computation_type ==0 it is the width, in
case of width_computation_type > 0 its value determines the how many
random features are used for determining the width in case of
width_computation_type > 0 set width2 <=1 to use all LEFT HAND SIDE
features for width estimation ";

%feature("docstring")  shogun::CPyramidChi2::init "

init

Parameters:
-----------

l:  features lhs

r:  reatures rhs ";

%feature("docstring")  shogun::CPyramidChi2::~CPyramidChi2 "";

%feature("docstring")  shogun::CPyramidChi2::cleanup "

cleanup ";

%feature("docstring")  shogun::CPyramidChi2::get_kernel_type "

return what type of kernel we are Linear,Polynomial, Gaussian,... ";

%feature("docstring")  shogun::CPyramidChi2::get_name "

return the name of a kernel ";


// File: classshogun_1_1CRegulatoryModulesStringKernel.xml
%feature("docstring") shogun::CRegulatoryModulesStringKernel "

The Regulaty Modules kernel, based on the WD kernel, as published in
Schultheiss et al., Bioinformatics (2009) on regulatory sequences.

C++ includes: RegulatoryModulesStringKernel.h ";

%feature("docstring")
shogun::CRegulatoryModulesStringKernel::CRegulatoryModulesStringKernel
"

constructor

Parameters:
-----------

size:  cache size

width:  width of gaussian kernel

degree:  degree of wds kernel

shift:  shift of wds kernel

window:  size of window around motifs to compute wds kernels on ";

%feature("docstring")
shogun::CRegulatoryModulesStringKernel::CRegulatoryModulesStringKernel
"

constructor

Parameters:
-----------

lstr:  string features of left-hand side

rstr:  string features of right-hand side

lpos:  motif positions on lhs

rpos:  motif positions on rhs

width:  width of gaussian kernel

degree:  degree of wds kernel

shift:  shift of wds kernel

window:  size of window around motifs to compute wds kernels on

size:  cache size ";

%feature("docstring")
shogun::CRegulatoryModulesStringKernel::~CRegulatoryModulesStringKernel
"

default destructor ";

%feature("docstring")  shogun::CRegulatoryModulesStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")
shogun::CRegulatoryModulesStringKernel::get_kernel_type "

return what type of kernel we are

kernel type ";

%feature("docstring")
shogun::CRegulatoryModulesStringKernel::get_name "

return the kernel's name

name Regulatory Modules ";

%feature("docstring")
shogun::CRegulatoryModulesStringKernel::set_motif_positions "

set motif positions

Parameters:
-----------

positions_lhs:  motif positions on lhs

positions_rhs:  motif positions on rhs ";


// File: classshogun_1_1CRidgeKernelNormalizer.xml
%feature("docstring") shogun::CRidgeKernelNormalizer "

Normalize the kernel by adding a constant term to its diagonal. This
aids kernels to become positive definite (even though they are not -
often caused by numerical problems).

Formally,

\\\\[ k'(x,x')= \\\\frac{k(x,x')}+ R\\\\cdot {\\\\bf E} \\\\]

where E is a matrix with ones on the diagonal and R is the scalar
ridge term. The ridge term R is computed as $R=r\\\\dot c$.

Typically,

r=1e-10 and c=0.0 will add mean(diag(K))*1e-10 to the diagonal

r=0.1 and c=1 will add 0.1 to the diagonal

In case c <= 0, c is compute as the mean of the kernel diagonal \\\\[
\\\\mbox{c} = \\\\frac{1}{N}\\\\sum_{i=1}^N k(x_i,x_i) \\\\]

C++ includes: RidgeKernelNormalizer.h ";

%feature("docstring")
shogun::CRidgeKernelNormalizer::CRidgeKernelNormalizer "

constructor

Parameters:
-----------

r:  ridge parameter

c:  scale parameter, if <= 0 scaling will be computed from the avg of
the kernel diagonal elements

the scalar r*c will be added to the kernel diagonal, typical use
cases: r=1e-10 and c=0.0 will add mean(diag(K))*1e-10 to the diagonal

r=0.1 and c=1 will add 0.1 to the diagonal ";

%feature("docstring")
shogun::CRidgeKernelNormalizer::~CRidgeKernelNormalizer "

default destructor ";

%feature("docstring")  shogun::CRidgeKernelNormalizer::init "

initialization of the normalizer (if needed)

Parameters:
-----------

k:  kernel ";

%feature("docstring")  shogun::CRidgeKernelNormalizer::normalize "

normalize the kernel value

Parameters:
-----------

value:  kernel value

idx_lhs:  index of left hand side vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")  shogun::CRidgeKernelNormalizer::normalize_lhs "

normalize only the left hand side vector

Parameters:
-----------

value:  value of a component of the left hand side feature vector

idx_lhs:  index of left hand side vector ";

%feature("docstring")  shogun::CRidgeKernelNormalizer::normalize_rhs "

normalize only the right hand side vector

Parameters:
-----------

value:  value of a component of the right hand side feature vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")  shogun::CRidgeKernelNormalizer::get_name "

object name ";


// File: classshogun_1_1CSalzbergWordStringKernel.xml
%feature("docstring") shogun::CSalzbergWordStringKernel "

The SalzbergWordString kernel implements the Salzberg kernel.

It is described in

Engineering Support Vector Machine Kernels That Recognize Translation
Initiation Sites A. Zien, G.Raetsch, S. Mika, B. Schoelkopf, T.
Lengauer, K.-R. Mueller

C++ includes: SalzbergWordStringKernel.h ";

%feature("docstring")
shogun::CSalzbergWordStringKernel::CSalzbergWordStringKernel "

constructor

Parameters:
-----------

size:  cache size

pie:  the plugin estimate

labels:  optional labels to set prior from ";

%feature("docstring")
shogun::CSalzbergWordStringKernel::CSalzbergWordStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

pie:  the plugin estimate

labels:  optional labels to set prior from ";

%feature("docstring")
shogun::CSalzbergWordStringKernel::~CSalzbergWordStringKernel "";

%feature("docstring")
shogun::CSalzbergWordStringKernel::set_prior_probs "

set prior probs

Parameters:
-----------

pos_prior_:  positive prior

neg_prior_:  negative prior ";

%feature("docstring")
shogun::CSalzbergWordStringKernel::set_prior_probs_from_labels "

set prior probs from labels

Parameters:
-----------

labels:  labels to set prior probabilites from ";

%feature("docstring")  shogun::CSalzbergWordStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CSalzbergWordStringKernel::cleanup "

clean up kernel ";

%feature("docstring")
shogun::CSalzbergWordStringKernel::get_kernel_type "

return what type of kernel we are

kernel type SALZBERG ";

%feature("docstring")  shogun::CSalzbergWordStringKernel::get_name "

return the kernel's name

name Salzberg ";


// File: classshogun_1_1CSigmoidKernel.xml
%feature("docstring") shogun::CSigmoidKernel "

The standard Sigmoid kernel computed on dense real valued features.

Formally, it is computed as

\\\\[ k({\\\\bf x},{\\\\bf x'})=\\\\mbox{tanh}(\\\\gamma {\\\\bf
x}\\\\cdot{\\\\bf x'}+c) \\\\]

C++ includes: SigmoidKernel.h ";

%feature("docstring")  shogun::CSigmoidKernel::CSigmoidKernel "

constructor

Parameters:
-----------

size:  cache size

gamma:  gamma

coef0:  coefficient 0 ";

%feature("docstring")  shogun::CSigmoidKernel::CSigmoidKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

size:  cache size

gamma:  gamma

coef0:  coefficient 0 ";

%feature("docstring")  shogun::CSigmoidKernel::~CSigmoidKernel "";

%feature("docstring")  shogun::CSigmoidKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CSigmoidKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CSigmoidKernel::get_kernel_type "

return what type of kernel we are

kernel type SIGMOID ";

%feature("docstring")  shogun::CSigmoidKernel::get_name "

return the kernel's name

name Sigmoid ";


// File: classshogun_1_1CSimpleKernel.xml
%feature("docstring") shogun::CSimpleKernel "

Template class SimpleKernel is the base class for kernels working on
Simple features.

CSimpleFeatures are dense Matrix like Features and Kernels operating
on them all derive from this class (cf., e.g., CGaussianKernel)

C++ includes: SimpleKernel.h ";

%feature("docstring")  shogun::CSimpleKernel::CSimpleKernel "

default constructor ";

%feature("docstring")  shogun::CSimpleKernel::CSimpleKernel "

constructor

Parameters:
-----------

cachesize:  cache size ";

%feature("docstring")  shogun::CSimpleKernel::CSimpleKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CSimpleKernel::init "

initialize kernel e.g. setup lhs/rhs of kernel, precompute
normalization constants etc. make sure to check that your kernel can
deal with the supplied features (!)

Parameters:
-----------

l:  features for left-hand side

r:  features for right-hand side

if init was successful ";

%feature("docstring")  shogun::CSimpleKernel::get_feature_class "

return feature class the kernel can deal with

feature class SIMPLE ";

%feature("docstring")  shogun::CSimpleKernel::get_feature_type "

return feature type the kernel can deal with

templated feature type ";

%feature("docstring")  shogun::CSimpleKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSimpleKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSimpleKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSimpleKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSimpleKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSimpleKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSimpleKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSimpleKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";


// File: classshogun_1_1CSimpleLocalityImprovedStringKernel.xml
%feature("docstring") shogun::CSimpleLocalityImprovedStringKernel "

SimpleLocalityImprovedString kernel, is a ``simplified'' and better
performing version of the Locality improved kernel.

It can be defined as

\\\\[ K({\\\\bf x},{\\\\bf
x'})=\\\\left(\\\\sum_{i=0}^{T-1}\\\\left(\\\\sum_{j=-l}^{+l}w_jI_{i+j}({\\\\bf
x},{\\\\bf x'})\\\\right)^{d_1}\\\\right)^{d_2}, \\\\] where $
I_i({\\\\bf x},{\\\\bf x'})=1$ if $x_i=x'_i$ and 0 otherwise.

C++ includes: SimpleLocalityImprovedStringKernel.h ";

%feature("docstring")
shogun::CSimpleLocalityImprovedStringKernel::CSimpleLocalityImprovedStringKernel
"

constructor

Parameters:
-----------

size:  cache size

length:  length

inner_degree:  inner degree

outer_degree:  outer degree ";

%feature("docstring")
shogun::CSimpleLocalityImprovedStringKernel::CSimpleLocalityImprovedStringKernel
"

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

length:  length

inner_degree:  inner degree

outer_degree:  outer degree ";

%feature("docstring")
shogun::CSimpleLocalityImprovedStringKernel::~CSimpleLocalityImprovedStringKernel
"";

%feature("docstring")
shogun::CSimpleLocalityImprovedStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")
shogun::CSimpleLocalityImprovedStringKernel::cleanup "

clean up kernel ";

%feature("docstring")
shogun::CSimpleLocalityImprovedStringKernel::get_kernel_type "

return what type of kernel we are

kernel type SIMPLELOCALITYIMPROVED ";

%feature("docstring")
shogun::CSimpleLocalityImprovedStringKernel::get_name "

return the kernel's name

name SimpleLocalityImproved ";


// File: classshogun_1_1CSparseGaussianKernel.xml
%feature("docstring") shogun::CSparseGaussianKernel "

The well known Gaussian kernel (swiss army knife for SVMs) on sparse
real valued features.

It is computed as

\\\\[ k({\\\\bf x},{\\\\bf x'})= exp(-\\\\frac{||{\\\\bf x}-{\\\\bf
x'}||^2}{\\\\tau}) \\\\]

where $\\\\tau$ is the kernel width.

C++ includes: SparseGaussianKernel.h ";

%feature("docstring")
shogun::CSparseGaussianKernel::CSparseGaussianKernel "

constructor

Parameters:
-----------

size:  cache size

width:  width ";

%feature("docstring")
shogun::CSparseGaussianKernel::CSparseGaussianKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

width:  width ";

%feature("docstring")
shogun::CSparseGaussianKernel::~CSparseGaussianKernel "";

%feature("docstring")  shogun::CSparseGaussianKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CSparseGaussianKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CSparseGaussianKernel::get_kernel_type
"

return what type of kernel we are

kernel type SPARSEGAUSSIAN ";

%feature("docstring")  shogun::CSparseGaussianKernel::get_feature_type
"

return feature type the kernel can deal with

feature type DREAL ";

%feature("docstring")  shogun::CSparseGaussianKernel::get_name "

return the kernel's name

name SparseGaussian ";


// File: classshogun_1_1CSparseKernel.xml
%feature("docstring") shogun::CSparseKernel "

Template class SparseKernel, is the base class of kernels working on
sparse features.

See e.g. the CSparseGaussianKernel for an example.

C++ includes: SparseKernel.h ";

%feature("docstring")  shogun::CSparseKernel::CSparseKernel "

constructor

Parameters:
-----------

cachesize:  cache size ";

%feature("docstring")  shogun::CSparseKernel::CSparseKernel "

constructor

Parameters:
-----------

l:  features for left-hand side

r:  features for right-hand side ";

%feature("docstring")  shogun::CSparseKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CSparseKernel::get_feature_class "

return feature class the kernel can deal with

feature class SPARSE ";

%feature("docstring")  shogun::CSparseKernel::get_feature_type "

return feature type the kernel can deal with

templated feature type ";

%feature("docstring")  shogun::CSparseKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSparseKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSparseKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSparseKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSparseKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSparseKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CSparseKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";


// File: classshogun_1_1CSparseLinearKernel.xml
%feature("docstring") shogun::CSparseLinearKernel "

Computes the standard linear kernel on sparse real valued features.

Formally, it computes

\\\\[ k({\\\\bf x},{\\\\bf x'})= \\\\Phi_k({\\\\bf x})\\\\cdot
\\\\Phi_k({\\\\bf x'}) \\\\]

C++ includes: SparseLinearKernel.h ";

%feature("docstring")
shogun::CSparseLinearKernel::CSparseLinearKernel "

constructor ";

%feature("docstring")
shogun::CSparseLinearKernel::CSparseLinearKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")
shogun::CSparseLinearKernel::~CSparseLinearKernel "";

%feature("docstring")  shogun::CSparseLinearKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CSparseLinearKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CSparseLinearKernel::get_kernel_type "

return what type of kernel we are

kernel type SPARSELINEAR ";

%feature("docstring")  shogun::CSparseLinearKernel::get_name "

return the kernel's name

name FixedDegree ";

%feature("docstring")  shogun::CSparseLinearKernel::init_optimization
"

optimizable kernel, i.e. precompute normal vector and as phi(x) = x do
scalar product in input space

Parameters:
-----------

num_suppvec:  number of support vectors

sv_idx:  support vector index

alphas:  alphas

if optimization was successful ";

%feature("docstring")
shogun::CSparseLinearKernel::delete_optimization "

delete optimization

if deleting was successful ";

%feature("docstring")  shogun::CSparseLinearKernel::compute_optimized
"

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")  shogun::CSparseLinearKernel::clear_normal "

clear normal ";

%feature("docstring")  shogun::CSparseLinearKernel::add_to_normal "

add to normal

Parameters:
-----------

idx:  where to add

weight:  what to add ";

%feature("docstring")  shogun::CSparseLinearKernel::get_normal "

get normal

Parameters:
-----------

len:  length of normal vector will be stored here

the normal vector ";


// File: classshogun_1_1CSparsePolyKernel.xml
%feature("docstring") shogun::CSparsePolyKernel "

Computes the standard polynomial kernel on sparse real valued
features.

Formally, it computes

\\\\[ k({\\\\bf x},{\\\\bf x'})= ({\\\\bf x}\\\\cdot {\\\\bf x'}+c)^d
\\\\]

Note that additional normalisation is applied, i.e. \\\\[ k'({\\\\bf
x}, {\\\\bf x'})=\\\\frac{k({\\\\bf x}, {\\\\bf
x'})}{\\\\sqrt{k({\\\\bf x}, {\\\\bf x})k({\\\\bf x'}, {\\\\bf x'})}}
\\\\]

C++ includes: SparsePolyKernel.h ";

%feature("docstring")  shogun::CSparsePolyKernel::CSparsePolyKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

size:  cache size

d:  degree

inhom:  is inhomogeneous ";

%feature("docstring")  shogun::CSparsePolyKernel::CSparsePolyKernel "

constructor

Parameters:
-----------

size:  cache size

degree:  degree

inhomogene:  is inhomogeneous ";

%feature("docstring")  shogun::CSparsePolyKernel::~CSparsePolyKernel "";

%feature("docstring")  shogun::CSparsePolyKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CSparsePolyKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CSparsePolyKernel::get_feature_type "

return feature type the kernel can deal with

feature type DREAL ";

%feature("docstring")  shogun::CSparsePolyKernel::get_kernel_type "

return what type of kernel we are

kernel type POLY ";

%feature("docstring")  shogun::CSparsePolyKernel::get_name "

return the kernel's name

name SparsePoly ";


// File: classshogun_1_1CSqrtDiagKernelNormalizer.xml
%feature("docstring") shogun::CSqrtDiagKernelNormalizer "

SqrtDiagKernelNormalizer divides by the Square Root of the product of
the diagonal elements.

This effectively normalizes the vectors in feature space to norm 1
(see CSqrtDiagKernelNormalizer)

\\\\[ k'({\\\\bf x},{\\\\bf x'}) = \\\\frac{k({\\\\bf x},{\\\\bf
x'})}{\\\\sqrt{k({\\\\bf x},{\\\\bf x})k({\\\\bf x'},{\\\\bf x'})}}
\\\\]

C++ includes: SqrtDiagKernelNormalizer.h ";

%feature("docstring")
shogun::CSqrtDiagKernelNormalizer::CSqrtDiagKernelNormalizer "

default constructor

Parameters:
-----------

use_opt_diag:  - some kernels support faster diagonal compuation via
compute_diag(idx), this flag enables this ";

%feature("docstring")
shogun::CSqrtDiagKernelNormalizer::~CSqrtDiagKernelNormalizer "

default destructor ";

%feature("docstring")  shogun::CSqrtDiagKernelNormalizer::init "

initialization of the normalizer

Parameters:
-----------

k:  kernel ";

%feature("docstring")  shogun::CSqrtDiagKernelNormalizer::normalize "

normalize the kernel value

Parameters:
-----------

value:  kernel value

idx_lhs:  index of left hand side vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")
shogun::CSqrtDiagKernelNormalizer::normalize_lhs "

normalize only the left hand side vector

Parameters:
-----------

value:  value of a component of the left hand side feature vector

idx_lhs:  index of left hand side vector ";

%feature("docstring")
shogun::CSqrtDiagKernelNormalizer::normalize_rhs "

normalize only the right hand side vector

Parameters:
-----------

value:  value of a component of the right hand side feature vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")
shogun::CSqrtDiagKernelNormalizer::alloc_and_compute_diag "

alloc and compute the vector containing the square root of the
diagonal elements of this kernel. ";

%feature("docstring")  shogun::CSqrtDiagKernelNormalizer::get_name "

object name ";


// File: classshogun_1_1CStringKernel.xml
%feature("docstring") shogun::CStringKernel "

Template class StringKernel, is the base class of all String Kernels.

For a (very complex) example see e.g. CWeightedDegreeStringKernel

C++ includes: StringKernel.h ";

%feature("docstring")  shogun::CStringKernel::CStringKernel "

constructor

Parameters:
-----------

cachesize:  cache size ";

%feature("docstring")  shogun::CStringKernel::CStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CStringKernel::init "

initialize kernel e.g. setup lhs/rhs of kernel, precompute
normalization constants etc. make sure to check that your kernel can
deal with the supplied features (!)

Parameters:
-----------

l:  features for left-hand side

r:  features for right-hand side

if init was successful ";

%feature("docstring")  shogun::CStringKernel::get_feature_class "

return feature class the kernel can deal with

feature class STRING ";

%feature("docstring")  shogun::CStringKernel::get_feature_type "

return feature type the kernel can deal with

templated feature type ";

%feature("docstring")  shogun::CStringKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CStringKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CStringKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CStringKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CStringKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CStringKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";

%feature("docstring")  shogun::CStringKernel::get_feature_type "

return feature type the kernel can deal with

abstract base method

feature type ";


// File: classshogun_1_1CTanimotoKernelNormalizer.xml
%feature("docstring") shogun::CTanimotoKernelNormalizer "

TanimotoKernelNormalizer performs kernel normalization inspired by the
Tanimoto coefficient (seehttp://en.wikipedia.org/wiki/Jaccard_index ).

\\\\[ k'({\\\\bf x},{\\\\bf x'}) = \\\\frac{k({\\\\bf x},{\\\\bf
x'})}{k({\\\\bf x},{\\\\bf x})+k({\\\\bf x'},{\\\\bf x'})-k({\\\\bf
x},{\\\\bf x'})} \\\\]

C++ includes: TanimotoKernelNormalizer.h ";

%feature("docstring")
shogun::CTanimotoKernelNormalizer::CTanimotoKernelNormalizer "

default constructor

Parameters:
-----------

use_opt_diag:  - some kernels support faster diagonal compuation via
compute_diag(idx), this flag enables this ";

%feature("docstring")
shogun::CTanimotoKernelNormalizer::~CTanimotoKernelNormalizer "

default destructor ";

%feature("docstring")  shogun::CTanimotoKernelNormalizer::init "

initialization of the normalizer

Parameters:
-----------

k:  kernel ";

%feature("docstring")  shogun::CTanimotoKernelNormalizer::normalize "

normalize the kernel value

Parameters:
-----------

value:  kernel value

idx_lhs:  index of left hand side vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")
shogun::CTanimotoKernelNormalizer::normalize_lhs "

normalize only the left hand side vector

Parameters:
-----------

value:  value of a component of the left hand side feature vector

idx_lhs:  index of left hand side vector ";

%feature("docstring")
shogun::CTanimotoKernelNormalizer::normalize_rhs "

normalize only the right hand side vector

Parameters:
-----------

value:  value of a component of the right hand side feature vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")
shogun::CTanimotoKernelNormalizer::alloc_and_compute_diag "

alloc and compute the vector containing the square root of the
diagonal elements of this kernel. ";


// File: classshogun_1_1CTensorProductPairKernel.xml
%feature("docstring") shogun::CTensorProductPairKernel "

Computes the Tensor Product Pair Kernel (TPPK).

Formally, it computes

\\\\[ k_{\\\\mbox{tppk}}(({\\\\bf a},{\\\\bf b}), ({\\\\bf c},{\\\\bf
d}))= k({\\\\bf a}, {\\\\bf c})\\\\cdot k({\\\\bf b}, {\\\\bf c}) +
k({\\\\bf a},{\\\\bf d})\\\\cdot k({\\\\bf b}, {\\\\bf c}) \\\\]

It is defined on pairs of inputs and a subkernel $k$. The subkernel
has to be given on initialization. The pairs are specified via indizes
(ab)using 2-dimensional integer features.

Its feature space $\\\\Phi_{\\\\mbox{tppk}}$ is the tensor product of
the feature spaces of the subkernel $k(.,.)$ on its input.

It is often used in bioinformatics, e.g., to predict protein-protein
interactions.

C++ includes: TensorProductPairKernel.h ";

%feature("docstring")
shogun::CTensorProductPairKernel::CTensorProductPairKernel "

constructor

Parameters:
-----------

size:  cache size

subkernel:  the subkernel ";

%feature("docstring")
shogun::CTensorProductPairKernel::CTensorProductPairKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

subkernel:  the subkernel ";

%feature("docstring")
shogun::CTensorProductPairKernel::~CTensorProductPairKernel "";

%feature("docstring")  shogun::CTensorProductPairKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")
shogun::CTensorProductPairKernel::get_kernel_type "

return what type of kernel we are

kernel type TPPK ";

%feature("docstring")  shogun::CTensorProductPairKernel::get_name "

return the kernel's name

name TPPK ";


// File: classshogun_1_1CVarianceKernelNormalizer.xml
%feature("docstring") shogun::CVarianceKernelNormalizer "

VarianceKernelNormalizer divides by the ``variance''.

This effectively normalizes the vectors in feature space to variance 1
(see CVarianceKernelNormalizer)

\\\\[ k'({\\\\bf x},{\\\\bf x'}) = \\\\frac{k({\\\\bf x},{\\\\bf
x'})}{\\\\frac{1}{N}\\\\sum_{i=1}^N k({\\\\bf x}_i, {\\\\bf x}_i) -
\\\\sum_{i,j=1}^N, k({\\\\bf x}_i,{\\\\bf x'}_j)/N^2} \\\\]

C++ includes: VarianceKernelNormalizer.h ";

%feature("docstring")
shogun::CVarianceKernelNormalizer::CVarianceKernelNormalizer "

default constructor ";

%feature("docstring")
shogun::CVarianceKernelNormalizer::~CVarianceKernelNormalizer "

default destructor ";

%feature("docstring")  shogun::CVarianceKernelNormalizer::init "

initialization of the normalizer

Parameters:
-----------

k:  kernel ";

%feature("docstring")  shogun::CVarianceKernelNormalizer::normalize "

normalize the kernel value

Parameters:
-----------

value:  kernel value

idx_lhs:  index of left hand side vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")
shogun::CVarianceKernelNormalizer::normalize_lhs "

normalize only the left hand side vector

Parameters:
-----------

value:  value of a component of the left hand side feature vector

idx_lhs:  index of left hand side vector ";

%feature("docstring")
shogun::CVarianceKernelNormalizer::normalize_rhs "

normalize only the right hand side vector

Parameters:
-----------

value:  value of a component of the right hand side feature vector

idx_rhs:  index of right hand side vector ";

%feature("docstring")  shogun::CVarianceKernelNormalizer::get_name "

object name ";


// File: classshogun_1_1CWeightedCommWordStringKernel.xml
%feature("docstring") shogun::CWeightedCommWordStringKernel "

The WeightedCommWordString kernel may be used to compute the weighted
spectrum kernel (i.e. a spectrum kernel for 1 to K-mers, where each
k-mer length is weighted by some coefficient $\\\\beta_k$) from
strings that have been mapped into unsigned 16bit integers.

These 16bit integers correspond to k-mers. To applicable in this
kernel they need to be sorted (e.g. via the SortWordString pre-
processor).

It basically uses the algorithm in the unix \"comm\" command (hence
the name) to compute:

\\\\[ k({\\\\bf x},({\\\\bf x'})=
\\\\sum_{k=1}^K\\\\beta_k\\\\Phi_k({\\\\bf x})\\\\cdot
\\\\Phi_k({\\\\bf x'}) \\\\]

where $\\\\Phi_k$ maps a sequence ${\\\\bf x}$ that consists of
letters in $\\\\Sigma$ to a feature vector of size $|\\\\Sigma|^k$. In
this feature vector each entry denotes how often the k-mer appears in
that ${\\\\bf x}$.

Note that this representation is especially tuned to small alphabets
(like the 2-bit alphabet DNA), for which it enables spectrum kernels
of order 8.

For this kernel the linadd speedups are quite efficiently implemented
using direct maps.

C++ includes: WeightedCommWordStringKernel.h ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::CWeightedCommWordStringKernel "

constructor

Parameters:
-----------

size:  cache size

use_sign:  if sign shall be used ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::CWeightedCommWordStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

use_sign:  if sign shall be used

size:  cache size ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::~CWeightedCommWordStringKernel
"";

%feature("docstring")  shogun::CWeightedCommWordStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CWeightedCommWordStringKernel::cleanup
"

clean up kernel ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::compute_optimized "

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::add_to_normal "

add to normal

Parameters:
-----------

idx:  where to add

weight:  what to add ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::merge_normal "

merge normal ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::set_wd_weights "

set weighted degree weights

if setting was successful ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::set_weights "

set custom weights (swig compatible)

Parameters:
-----------

w:  weights

d:  degree (must match number of weights)

if setting was successful ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::get_kernel_type "

return what type of kernel we are

kernel type WEIGHTEDCOMMWORDSTRING ";

%feature("docstring")  shogun::CWeightedCommWordStringKernel::get_name
"

return the kernel's name

name WeightedCommWordString ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::get_feature_type "

return feature type the kernel can deal with

feature type WORD ";

%feature("docstring")
shogun::CWeightedCommWordStringKernel::compute_scoring "

compute scoring

Parameters:
-----------

max_degree:  maximum degree

num_feat:  number of features

num_sym:  number of symbols

target:  target

num_suppvec:  number of support vectors

IDX:  IDX

alphas:  alphas

do_init:  if initialization shall be performed

computed score ";


// File: classshogun_1_1CWeightedDegreePositionStringKernel.xml
%feature("docstring") shogun::CWeightedDegreePositionStringKernel "

The Weighted Degree Position String kernel (Weighted Degree kernel
with shifts).

The WD-shift kernel of order d compares two sequences ${\\\\bf x}$ and
${\\\\bf x'}$ of length L by summing all contributions of k-mer
matches of lengths $k\\\\in\\\\{1,\\\\dots,d\\\\}$, weighted by
coefficients $\\\\beta_k$ allowing for a positional tolerance of up to
shift s.

It is formally defined as \\\\begin{eqnarray*}
&&\\\\!\\\\!\\\\!\\\\!\\\\!\\\\!\\\\!k({\\\\bf x}_i,{\\\\bf
x}_j)=\\\\sum_{k=1}^d\\\\beta_k\\\\sum_{l=1}^{\\\\!\\\\!\\\\!\\\\!L-k+1\\\\!\\\\!\\\\!\\\\!}\\\\gamma_l\\\\sum_{\\\\begin{array}{c}s=0\\\\\\\\
\\\\!\\\\!\\\\!\\\\!s+l\\\\leq
L\\\\!\\\\!\\\\!\\\\!\\\\end{array}}^{S(l)}
\\\\delta_s\\\\;\\\\mu_{k,l,s,{{\\\\bf x}_i},{{\\\\bf x}_j}},\\\\\\\\
&&\\\\!\\\\!\\\\!\\\\!\\\\!\\\\!\\\\!\\\\!\\\\!\\\\! {\\\\footnotesize
\\\\mu_{k,l,s,{{\\\\bf x}_i},{{\\\\bf x}_j}}\\\\!\\\\!\\\\!
=\\\\!\\\\! I({\\\\bf u}_{k,l+s}({\\\\bf x}_i)\\\\! =\\\\!{\\\\bf
u}_{k,l}({\\\\bf x}_j))\\\\! +\\\\!I({\\\\bf u}_{k,l}({\\\\bf
x}_i)\\\\! =\\\\!{\\\\bf u}_{k,l+s}({\\\\bf x}_j))},\\\\nonumber
\\\\end{eqnarray*} where $\\\\beta_j$ are the weighting coefficients
of the j-mers, $\\\\gamma_l$ is a weighting over the position in the
sequence, $\\\\delta_s=1/(2(s+1))$ is the weight assigned to shifts
(in either direction) of extent s, and S(l) determines the shift range
at position l.

C++ includes: WeightedDegreePositionStringKernel.h ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::CWeightedDegreePositionStringKernel
"

constructor

Parameters:
-----------

size:  cache size

degree:  degree

max_mismatch:  maximum mismatch

mkl_stepsize:  MKL stepsize ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::CWeightedDegreePositionStringKernel
"

constructor

Parameters:
-----------

size:  cache size

weights:  weights

degree:  degree

max_mismatch:  maximum mismatch

shift:  position shifts

shift_len:  number of shifts

mkl_stepsize:  MKL stepsize ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::CWeightedDegreePositionStringKernel
"

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

degree:  degree ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::~CWeightedDegreePositionStringKernel
"";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::cleanup "

clean up kernel ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::get_kernel_type "

return what type of kernel we are

kernel type WEIGHTEDDEGREEPOS ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::get_name "

return the kernel's name

name WeightedDegreePos ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_optimization "

initialize optimization

Parameters:
-----------

p_count:  count

IDX:  index

alphas:  alphas

if initializing was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_optimization "

initialize optimization do initialization for tree_num up to
upto_tree, use tree_num=-1 to construct all trees

Parameters:
-----------

count:  count

IDX:  IDX

alphas:  alphas

tree_num:  which tree

upto_tree:  up to this tree

if initializing was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::delete_optimization "

delete optimization

if deleting was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::compute_optimized "

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::compute_batch "

compute batch

Parameters:
-----------

num_vec:  number of vectors

vec_idx:  vector index

target:  target

num_suppvec:  number of support vectors

IDX:  IDX

alphas:  alphas

factor:  factor ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::clear_normal "

clear normal subkernel functionality ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::add_to_normal "

add to normal

Parameters:
-----------

idx:  where to add

weight:  what to add ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::get_num_subkernels "

get number of subkernels

number of subkernels ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::compute_by_subkernel "

compute by subkernel

Parameters:
-----------

idx:  index

subkernel_contrib:  subkernel contribution ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::get_subkernel_weights "

get subkernel weights

Parameters:
-----------

num_weights:  number of weights will be stored here

subkernel weights ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::set_subkernel_weights "

set subkernel weights

Parameters:
-----------

weights2:  weights

num_weights2:  number of weights ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::compute_abs_weights "

compute abs weights

Parameters:
-----------

len:  len

computed abs weights ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::is_tree_initialized "

check if tree is initialized

if tree is initialized ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::get_max_mismatch "

get maximum mismatch

maximum mismatch ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::get_degree "

get degree

the degree ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::get_degree_weights "

get degree weights

Parameters:
-----------

d:  degree weights will be stored here

len:  number of degree weights will be stored here ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::get_weights "

get weights

Parameters:
-----------

num_weights:  number of weights will be stored here

weights ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::get_position_weights "

get position weights

Parameters:
-----------

len:  number of position weights will be stored here

position weights ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::set_shifts "

set shifts

Parameters:
-----------

shifts:  new shifts

len:  number of shifts ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::set_weights "

set weights

Parameters:
-----------

weights:  new weights

d:  degree

len:  number of weights ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::set_wd_weights "

set wd weights

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::set_position_weights "

set position weights

Parameters:
-----------

pws:  new position weights

len:  number of position weights

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::set_position_weights_lhs
"

set position weights for left-hand side

Parameters:
-----------

pws:  new position weights

len:  len

num:  num

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::set_position_weights_rhs
"

set position weights for right-hand side

Parameters:
-----------

pws:  new position weights

len:  len

num:  num

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_block_weights "

initialize block weights

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_block_weights_from_wd
"

initialize block weights from weighted degree

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_block_weights_from_wd_external
"

initialize block weights from external weighted degree

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_block_weights_const
"

initialize block weights constant

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_block_weights_linear
"

initialize block weights linear

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_block_weights_sqpoly
"

initialize block weights squared polynomial

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_block_weights_cubicpoly
"

initialize block weights cubic polynomial

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_block_weights_exp "

initialize block weights exponential

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_block_weights_log "

initialize block weights logarithmic

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::init_block_weights_external
"

initialize block weights external

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::delete_position_weights "

delete position weights

if deleting was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::delete_position_weights_lhs
"

delete position weights left-hand side

if deleting was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::delete_position_weights_rhs
"

delete position weights right-hand side

if deleting was successful ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::compute_by_tree "

compute by tree

Parameters:
-----------

idx:  index

computed value ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::compute_by_tree "

compute by tree

Parameters:
-----------

idx:  index

LevelContrib:  level contribution ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::compute_scoring "

compute positional scoring function, which assigns a weight per
position, per symbol in the sequence

Parameters:
-----------

max_degree:  maximum degree

num_feat:  number of features

num_sym:  number of symbols

target:  target

num_suppvec:  number of support vectors

IDX:  IDX

weights:  weights

computed scores ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::compute_consensus "

compute consensus string

Parameters:
-----------

num_feat:  number of features

num_suppvec:  number of support vectors

IDX:  IDX

alphas:  alphas

consensus string ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::extract_w "

extract w

Parameters:
-----------

max_degree:  maximum degree

num_feat:  number of features

num_sym:  number of symbols

w_result:  w

num_suppvec:  number of support vectors

IDX:  IDX

alphas:  alphas

w ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::compute_POIM "

compute POIM

Parameters:
-----------

max_degree:  maximum degree

num_feat:  number of features

num_sym:  number of symbols

poim_result:  poim

num_suppvec:  number of support vectors

IDX:  IDX

alphas:  alphas

distrib:  distribution

computed POIMs ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::prepare_POIM2 "

prepare POIM2

Parameters:
-----------

num_feat:  number of features

num_sym:  number of symbols

distrib:  distribution ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::compute_POIM2 "

compute POIM2

Parameters:
-----------

max_degree:  maximum degree

svm:  SVM ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::get_POIM2 "

get POIM2

Parameters:
-----------

poim:  POIMs (returned)

result_len:  (returned) ";

%feature("docstring")
shogun::CWeightedDegreePositionStringKernel::cleanup_POIM2 "

cleanup POIM2 ";


// File: classshogun_1_1CWeightedDegreeStringKernel.xml
%feature("docstring") shogun::CWeightedDegreeStringKernel "

The Weighted Degree String kernel.

The WD kernel of order d compares two sequences ${\\\\bf x}$ and
${\\\\bf x'}$ of length L by summing all contributions of k-mer
matches of lengths $k\\\\in\\\\{1,\\\\dots,d\\\\}$, weighted by
coefficients $\\\\beta_k$. It is defined as \\\\[ k({\\\\bf x},{\\\\bf
x'})=\\\\sum_{k=1}^d\\\\beta_k\\\\sum_{l=1}^{L-k+1}I({\\\\bf
u}_{k,l}({\\\\bf x})={\\\\bf u}_{k,l}({\\\\bf x'})). \\\\] Here,
${\\\\bf u}_{k,l}({\\\\bf x})$ is the string of length k starting at
position l of the sequence ${\\\\bf x}$ and $I(\\\\cdot)$ is the
indicator function which evaluates to 1 when its argument is true and
to 0 otherwise.

C++ includes: WeightedDegreeStringKernel.h ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::CWeightedDegreeStringKernel "

constructor

Parameters:
-----------

degree:  degree

type:  weighted degree kernel type ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::CWeightedDegreeStringKernel "

constructor

Parameters:
-----------

weights:  kernel's weights

degree:  degree ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::CWeightedDegreeStringKernel "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

degree:  degree ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::~CWeightedDegreeStringKernel "";

%feature("docstring")  shogun::CWeightedDegreeStringKernel::init "

initialize kernel

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if initializing was successful ";

%feature("docstring")  shogun::CWeightedDegreeStringKernel::cleanup "

clean up kernel ";

%feature("docstring")  shogun::CWeightedDegreeStringKernel::get_type "

get WD kernel weighting type

weighting type

See:  EWDKernType ";

%feature("docstring")  shogun::CWeightedDegreeStringKernel::get_degree
"

get degree of WD kernel

degree of the kernel ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_max_mismatch "

get the number of mismatches that are allowed in WD kernel computation

number of mismatches ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_kernel_type "

return what type of kernel we are

kernel type WEIGHTEDDEGREE ";

%feature("docstring")  shogun::CWeightedDegreeStringKernel::get_name "

return the kernel's name

name WeightedDegree ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_optimization "

initialize optimization

Parameters:
-----------

count:  count

IDX:  index

alphas:  alphas

if initializing was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_optimization "

initialize optimization do initialization for tree_num up to
upto_tree, use tree_num=-1 to construct all trees

Parameters:
-----------

count:  count

IDX:  IDX

alphas:  alphas

tree_num:  which tree

if initializing was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::delete_optimization "

delete optimization

if deleting was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::compute_optimized "

compute optimized

Parameters:
-----------

idx:  index to compute

optimized value at given index ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::compute_batch "

compute batch

Parameters:
-----------

num_vec:  number of vectors

vec_idx:  vector index

target:  target

num_suppvec:  number of support vectors

IDX:  IDX

alphas:  alphas

factor:  factor ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::clear_normal "

clear normal subkernel functionality ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::add_to_normal "

add to normal

Parameters:
-----------

idx:  where to add

weight:  what to add ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_num_subkernels "

get number of subkernels

number of subkernels ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::compute_by_subkernel "

compute by subkernel

Parameters:
-----------

idx:  index

subkernel_contrib:  subkernel contribution ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_subkernel_weights "

get subkernel weights

Parameters:
-----------

num_weights:  number of weights will be stored here

subkernel weights ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::set_subkernel_weights "

set subkernel weights

Parameters:
-----------

weights2:  weights

num_weights2:  number of weights ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::compute_abs_weights "

compute abs weights

Parameters:
-----------

len:  len

computed abs weights ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::compute_by_tree "

compute by tree

Parameters:
-----------

idx:  index

LevelContrib:  level contribution

computed value ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::is_tree_initialized "

check if tree is initialized

if tree is initialized ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_degree_weights "

get degree weights

Parameters:
-----------

d:  degree weights will be stored here

len:  number of degree weights will be stored here ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_weights "

get weights

Parameters:
-----------

num_weights:  number of weights will be stored here

weights ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_position_weights "

get position weights

Parameters:
-----------

len:  number of position weights will be stored here

position weights ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::set_wd_weights_by_type "

set wd weights

Parameters:
-----------

type:  weighted degree kernel type

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::set_wd_weights "

set wd weights

Parameters:
-----------

p_weights:  new eights

d:  degree

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::set_weights "

set weights

Parameters:
-----------

weights:  new weights

d:  degree

len:  number of weights ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::set_position_weights "

set position weights

Parameters:
-----------

position_weights:  new position weights

len:  number of position weights

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_block_weights "

initialize block weights

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_block_weights_from_wd "

initialize block weights from weighted degree

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_block_weights_from_wd_external
"

initialize block weights from external weighted degree

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_block_weights_const "

initialize block weights constant

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_block_weights_linear "

initialize block weights linear

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_block_weights_sqpoly "

initialize block weights squared polynomial

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_block_weights_cubicpoly "

initialize block weights cubic polynomial

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_block_weights_exp "

initialize block weights exponential

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_block_weights_log "

initialize block weights logarithmic

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::init_block_weights_external "

initialize block weights external

if initialization was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::delete_position_weights "

delete position weights

if deleting was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::set_max_mismatch "

set maximum mismatch

Parameters:
-----------

max:  new maximum mismatch

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_max_mismatch "

get maximum mismatch

maximum mismatch ";

%feature("docstring")  shogun::CWeightedDegreeStringKernel::set_degree
"

set degree

Parameters:
-----------

deg:  new degree

if setting was successful ";

%feature("docstring")  shogun::CWeightedDegreeStringKernel::get_degree
"

get degree

degree ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::set_use_block_computation "

set if block computation shall be performed

Parameters:
-----------

block:  if block computation shall be performed

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_use_block_computation "

check if block computation is performed

if block computation is performed ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::set_mkl_stepsize "

set MKL steps ize

Parameters:
-----------

step:  new step size

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_mkl_stepsize "

get MKL step size

MKL step size ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::set_which_degree "

set which degree

Parameters:
-----------

which:  which degree

if setting was successful ";

%feature("docstring")
shogun::CWeightedDegreeStringKernel::get_which_degree "

get which degree

which degree ";


// File: structshogun_1_1K__THREAD__PARAM.xml
%feature("docstring") shogun::K_THREAD_PARAM "

kernel thread parameters

C++ includes: Kernel.h ";


// File: namespaceshogun.xml


// File: AUCKernel_8h.xml


// File: AvgDiagKernelNormalizer_8h.xml


// File: Chi2Kernel_8h.xml


// File: CombinedKernel_8h.xml


// File: CommUlongStringKernel_8h.xml


// File: CommWordStringKernel_8h.xml


// File: ConstKernel_8h.xml


// File: CustomKernel_8h.xml


// File: DiagKernel_8h.xml


// File: DiceKernelNormalizer_8h.xml


// File: DistanceKernel_8h.xml


// File: FirstElementKernelNormalizer_8h.xml


// File: FixedDegreeStringKernel_8h.xml


// File: GaussianKernel_8h.xml


// File: GaussianShiftKernel_8h.xml


// File: GaussianShortRealKernel_8h.xml


// File: HistogramWordStringKernel_8h.xml


// File: IdentityKernelNormalizer_8h.xml


// File: Kernel_8h.xml


// File: KernelNormalizer_8h.xml


// File: LinearByteKernel_8h.xml


// File: LinearKernel_8h.xml


// File: LinearStringKernel_8h.xml


// File: LinearWordKernel_8h.xml


// File: LocalAlignmentStringKernel_8h.xml


// File: LocalityImprovedStringKernel_8h.xml


// File: MatchWordStringKernel_8h.xml


// File: MultitaskKernelNormalizer_8h.xml


// File: OligoStringKernel_8h.xml


// File: PolyKernel_8h.xml


// File: PolyMatchStringKernel_8h.xml


// File: PolyMatchWordStringKernel_8h.xml


// File: PyramidChi2_8h.xml


// File: RegulatoryModulesStringKernel_8h.xml


// File: RidgeKernelNormalizer_8h.xml


// File: SalzbergWordStringKernel_8h.xml


// File: SigmoidKernel_8h.xml


// File: SimpleKernel_8h.xml


// File: SimpleLocalityImprovedStringKernel_8h.xml


// File: SparseGaussianKernel_8h.xml


// File: SparseKernel_8h.xml


// File: SparseLinearKernel_8h.xml


// File: SparsePolyKernel_8h.xml


// File: SqrtDiagKernelNormalizer_8h.xml


// File: StringKernel_8h.xml


// File: TanimotoKernelNormalizer_8h.xml


// File: TensorProductPairKernel_8h.xml


// File: VarianceKernelNormalizer_8h.xml


// File: WeightedCommWordStringKernel_8h.xml


// File: WeightedDegreePositionStringKernel_8h.xml


// File: WeightedDegreeStringKernel_8h.xml


// File: dir_430180f02556cc126694f29e97cdd05a.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml

