
// File: index.xml

// File: classshogun_1_1CClassifier.xml
%feature("docstring") shogun::CClassifier "

A generic classifier interface.

A classifier takes as input CLabels. Later subclasses may specialize
the classifier to require labels and a kernel or labels and (real-
valued) features.

A classifier needs to override the train() function for training, the
function classify_example() (optionally classify() to predict on the
whole set of examples) and the load and save routines.

C++ includes: Classifier.h ";

%feature("docstring")  shogun::CClassifier::CClassifier "

constructor ";

%feature("docstring")  shogun::CClassifier::~CClassifier "";

%feature("docstring")  shogun::CClassifier::train "

train classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CClassifier::classify "

classify objects using the currently set features

classified labels ";

%feature("docstring")  shogun::CClassifier::classify "

classify objects

Parameters:
-----------

data:  (test) data to be classified

classified labels ";

%feature("docstring")  shogun::CClassifier::classify_example "

classify one example

abstract base method

Parameters:
-----------

num:  which example to classify

infinite float value ";

%feature("docstring")  shogun::CClassifier::load "

load Classifier from file

abstract base method

Parameters:
-----------

srcfile:  file to load from

failure ";

%feature("docstring")  shogun::CClassifier::save "

save Classifier to file

abstract base method

Parameters:
-----------

dstfile:  file to save to

failure ";

%feature("docstring")  shogun::CClassifier::set_labels "

set labels

Parameters:
-----------

lab:  labels ";

%feature("docstring")  shogun::CClassifier::get_labels "

get labels

labels ";

%feature("docstring")  shogun::CClassifier::get_label "

get one specific label

Parameters:
-----------

i:  index of label to get

value of label at index i ";

%feature("docstring")  shogun::CClassifier::set_max_train_time "

set maximum training time

Parameters:
-----------

t:  maximimum training time ";

%feature("docstring")  shogun::CClassifier::get_max_train_time "

get maximum training time

maximum training time ";

%feature("docstring")  shogun::CClassifier::get_classifier_type "

get classifier type

classifier type NONE ";

%feature("docstring")  shogun::CClassifier::set_solver_type "

set solver type

Parameters:
-----------

st:  solver type ";

%feature("docstring")  shogun::CClassifier::get_solver_type "

get solver type

solver ";


// File: classshogun_1_1CDistanceMachine.xml
%feature("docstring") shogun::CDistanceMachine "

A generic DistanceMachine interface.

A distance machine is based on a a-priori choosen distance.

C++ includes: DistanceMachine.h ";

%feature("docstring")  shogun::CDistanceMachine::CDistanceMachine "

default constructor ";

%feature("docstring")  shogun::CDistanceMachine::~CDistanceMachine "";

%feature("docstring")  shogun::CDistanceMachine::set_distance "

set distance

Parameters:
-----------

d:  distance to set ";

%feature("docstring")  shogun::CDistanceMachine::get_distance "

get distance

distance ";

%feature("docstring")  shogun::CDistanceMachine::distances_lhs "

get distance functions for lhs feature vectors going from a1 to a2 and
rhs feature vector b

Parameters:
-----------

result:  array of distance values

idx_a1:  first feature vector a1 at idx_a1

idx_a2:  last feature vector a2 at idx_a2

idx_b:  feature vector b at idx_b ";

%feature("docstring")  shogun::CDistanceMachine::distances_rhs "

get distance functions for rhs feature vectors going from b1 to b2 and
lhs feature vector a

Parameters:
-----------

result:  array of distance values

idx_b1:  first feature vector a1 at idx_b1

idx_b2:  last feature vector a2 at idx_b2

idx_a:  feature vector a at idx_a ";


// File: classshogun_1_1CDomainAdaptationSVM.xml
%feature("docstring") shogun::CDomainAdaptationSVM "

class DomainAdaptiveSVM

C++ includes: DomainAdaptationSVM.h ";

%feature("docstring")
shogun::CDomainAdaptationSVM::CDomainAdaptationSVM "

default constructor ";

%feature("docstring")
shogun::CDomainAdaptationSVM::CDomainAdaptationSVM "

constructor

Parameters:
-----------

C:  cost constant C

k:  kernel

lab:  labels

presvm:  trained SVM to regularize against

B:  trade-off constant B ";

%feature("docstring")
shogun::CDomainAdaptationSVM::~CDomainAdaptationSVM "

destructor ";

%feature("docstring")  shogun::CDomainAdaptationSVM::init "

init SVM

Parameters:
-----------

presvm:  trained SVM to regularize against

B:  trade-off constant B ";

%feature("docstring")  shogun::CDomainAdaptationSVM::train "

train SVM classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")
shogun::CDomainAdaptationSVM::get_classifier_type "

get classifier type

classifier type LIGHT ";

%feature("docstring")  shogun::CDomainAdaptationSVM::classify "

classify objects

Parameters:
-----------

data:  (test) data to be classified

classified labels ";

%feature("docstring")  shogun::CDomainAdaptationSVM::get_presvm "

returns SVM that is used as prior information

presvm ";

%feature("docstring")  shogun::CDomainAdaptationSVM::get_B "

getter for regularization parameter B

regularization parameter B ";

%feature("docstring")  shogun::CDomainAdaptationSVM::get_name "

object name ";


// File: classshogun_1_1CGMNPLib.xml
%feature("docstring") shogun::CGMNPLib "

class GMNPLib Library of solvers for Generalized Minimal Norm Problem
(GMNP).

Generalized Minimal Norm Problem to solve is

min 0.5*alpha'*H*alpha + c'*alpha

subject to sum(alpha) = 1, alpha(i) >= 0

H [dim x dim] is symmetric positive definite matrix. c [dim x 1] is an
arbitrary vector.

The precision of the found solution is given by the parameters tmax,
tolabs and tolrel which define the stopping conditions:

UB-LB <= tolabs -> exit_flag = 1 Abs. tolerance. UB-LB <= UB*tolrel ->
exit_flag = 2 Relative tolerance. LB > th -> exit_flag = 3 Threshold
on lower bound. t >= tmax -> exit_flag = 0 Number of iterations.

UB ... Upper bound on the optimal solution. LB ... Lower bound on the
optimal solution. t ... Number of iterations. History ... Value of LB
and UB wrt. number of iterations.

The following algorithms are implemented:
..............................................

GMNP solver based on improved MDM algorithm 1 (u fixed v optimized)
exitflag = gmnp_imdm( &get_col, diag_H, vector_c, dim, tmax, tolabs,
tolrel, th, &alpha, &t, &History, verb );

For more info refer to V.Franc: Optimization Algorithms for Kernel
Methods. Research report. CTU-CMP-2005-22. CTU FEL Prague.
2005.ftp://cmp.felk.cvut.cz/pub/cmp/articles/franc/Franc-PhD.pdf .

C++ includes: gmnplib.h ";

%feature("docstring")  shogun::CGMNPLib::CGMNPLib "

constructor

Parameters:
-----------

vector_y:  vector y

kernel:  kernel

num_data:  number of data

num_virtual_data:  number of virtual data

num_classes:  number of classes

reg_const:  reg const ";

%feature("docstring")  shogun::CGMNPLib::~CGMNPLib "";

%feature("docstring")  shogun::CGMNPLib::gmnp_imdm "

-------------------------------------------------------------- GMNP
solver based on improved MDM algorithm 1.

Search strategy: u determined by common rule and v is optimized.

Usage: exitflag = gmnp_imdm( &get_col, diag_H, vector_c, dim, tmax,
tolabs, tolrel, th, &alpha, &t, &History );
-------------------------------------------------------------- ";

%feature("docstring")  shogun::CGMNPLib::get_indices2 "

get indices2

Parameters:
-----------

index:  index

c:  c

i:  i ";


// File: classshogun_1_1CGMNPSVM.xml
%feature("docstring") shogun::CGMNPSVM "

Class GMNPSVM implements a one vs. rest MultiClass SVM.

It uses CGMNPLib for training (in true multiclass-SVM fashion).

C++ includes: GMNPSVM.h ";

%feature("docstring")  shogun::CGMNPSVM::CGMNPSVM "

default constructor ";

%feature("docstring")  shogun::CGMNPSVM::CGMNPSVM "

constructor

Parameters:
-----------

C:  constant C

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CGMNPSVM::~CGMNPSVM "

default destructor ";

%feature("docstring")  shogun::CGMNPSVM::train "

train SVM

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CGMNPSVM::get_classifier_type "

get classifier type

classifier type GMNPSVM ";

%feature("docstring")  shogun::CGMNPSVM::getbasealphas "

required for CMKLMulticlass constraint computation

Parameters:
-----------

basealphas:  basealphas[k][j] is the alpha for class k and sample j
which is untransformed compared to the alphas stored in CSVM* members
";

%feature("docstring")  shogun::CGMNPSVM::get_name "

object name ";


// File: classshogun_1_1CGNPPLib.xml
%feature("docstring") shogun::CGNPPLib "

class GNPPLib, a Library of solvers for Generalized Nearest Point
Problem (GNPP).

C++ includes: gnpplib.h ";

%feature("docstring")  shogun::CGNPPLib::CGNPPLib "

constructor

Parameters:
-----------

vector_y:  vector y

kernel:  kernel

num_data:  number of data

reg_const:  reg const ";

%feature("docstring")  shogun::CGNPPLib::~CGNPPLib "";

%feature("docstring")  shogun::CGNPPLib::gnpp_mdm "

-------------------------------------------------------------- QP
solver based on MDM algorithm.

Usage: exitflag = gnpp_mdm(diag_H, vector_c, vector_y, dim, tmax,
tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
-------------------------------------------------------------- ";

%feature("docstring")  shogun::CGNPPLib::gnpp_imdm "

-------------------------------------------------------------- QP
solver based on improved MDM algorithm (u fixed v optimized)

Usage: exitflag = gnpp_imdm( diag_H, vector_c, vector_y, dim, tmax,
tolabs, tolrel, th, &alpha, &t, &aHa11, &aHa22, &History );
-------------------------------------------------------------- ";

%feature("docstring")  shogun::CGNPPLib::get_name "

object name ";


// File: classshogun_1_1CGNPPSVM.xml
%feature("docstring") shogun::CGNPPSVM "

class GNPPSVM

C++ includes: GNPPSVM.h ";

%feature("docstring")  shogun::CGNPPSVM::CGNPPSVM "

default constructor ";

%feature("docstring")  shogun::CGNPPSVM::CGNPPSVM "

constructor

Parameters:
-----------

C:  constant C

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CGNPPSVM::~CGNPPSVM "";

%feature("docstring")  shogun::CGNPPSVM::train "

train SVM classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CGNPPSVM::get_classifier_type "

get classifier type

classifier type GNPPSVM ";

%feature("docstring")  shogun::CGNPPSVM::get_name "

object name ";


// File: classshogun_1_1CGPBTSVM.xml
%feature("docstring") shogun::CGPBTSVM "

class GPBTSVM

C++ includes: GPBTSVM.h ";

%feature("docstring")  shogun::CGPBTSVM::CGPBTSVM "

default constructor ";

%feature("docstring")  shogun::CGPBTSVM::CGPBTSVM "

constructor

Parameters:
-----------

C:  constant C

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CGPBTSVM::~CGPBTSVM "";

%feature("docstring")  shogun::CGPBTSVM::train "

train SVM classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CGPBTSVM::get_classifier_type "

get classifier type

classifier type GPBT ";

%feature("docstring")  shogun::CGPBTSVM::get_name "

object name ";


// File: classshogun_1_1CKernelMachine.xml
%feature("docstring") shogun::CKernelMachine "

A generic KernelMachine interface.

A kernel machine is defined as \\\\[ f({\\\\bf x})=\\\\sum_{i=0}^{N-1}
\\\\alpha_i k({\\\\bf x}, {\\\\bf x_i})+b \\\\]

where $N$ is the number of training examples $\\\\alpha_i$ are the
weights assigned to each training example $k(x,x')$ is the kernel and
$b$ the bias.

Using an a-priori choosen kernel, the $\\\\alpha_i$ and bias are
determined in a training procedure.

C++ includes: KernelMachine.h ";

%feature("docstring")  shogun::CKernelMachine::CKernelMachine "

default constructor ";

%feature("docstring")  shogun::CKernelMachine::~CKernelMachine "

destructor ";

%feature("docstring")  shogun::CKernelMachine::set_kernel "

set kernel

Parameters:
-----------

k:  kernel ";

%feature("docstring")  shogun::CKernelMachine::get_kernel "

get kernel

kernel ";

%feature("docstring")
shogun::CKernelMachine::set_batch_computation_enabled "

set batch computation enabled

Parameters:
-----------

enable:  if batch computation shall be enabled ";

%feature("docstring")
shogun::CKernelMachine::get_batch_computation_enabled "

check if batch computation is enabled

if batch computation is enabled ";

%feature("docstring")  shogun::CKernelMachine::set_linadd_enabled "

set linadd enabled

Parameters:
-----------

enable:  if linadd shall be enabled ";

%feature("docstring")  shogun::CKernelMachine::get_linadd_enabled "

check if linadd is enabled

if linadd is enabled ";

%feature("docstring")  shogun::CKernelMachine::set_bias_enabled "

set state of bias

Parameters:
-----------

enable_bias:  if bias shall be enabled ";

%feature("docstring")  shogun::CKernelMachine::get_bias_enabled "

get state of bias

state of bias ";

%feature("docstring")  shogun::CKernelMachine::get_bias "

get bias

bias ";

%feature("docstring")  shogun::CKernelMachine::set_bias "

set bias to given value

Parameters:
-----------

bias:  new bias ";

%feature("docstring")  shogun::CKernelMachine::get_support_vector "

get support vector at given index

Parameters:
-----------

idx:  index of support vector

support vector ";

%feature("docstring")  shogun::CKernelMachine::get_alpha "

get alpha at given index

Parameters:
-----------

idx:  index of alpha

alpha ";

%feature("docstring")  shogun::CKernelMachine::set_support_vector "

set support vector at given index to given value

Parameters:
-----------

idx:  index of support vector

val:  new value of support vector

if operation was successful ";

%feature("docstring")  shogun::CKernelMachine::set_alpha "

set alpha at given index to given value

Parameters:
-----------

idx:  index of alpha vector

val:  new value of alpha vector

if operation was successful ";

%feature("docstring")  shogun::CKernelMachine::get_num_support_vectors
"

get number of support vectors

number of support vectors ";

%feature("docstring")  shogun::CKernelMachine::set_alphas "

set alphas to given values

Parameters:
-----------

alphas:  array with all alphas to set

d:  number of alphas (== number of support vectors) ";

%feature("docstring")  shogun::CKernelMachine::set_support_vectors "

set support vectors to given values

Parameters:
-----------

svs:  array with all support vectors to set

d:  number of support vectors ";

%feature("docstring")  shogun::CKernelMachine::get_support_vectors "

get all support vectors (swig compatible)

Parameters:
-----------

svs:  array to contain a copy of the support vectors

num:  number of support vectors in the array ";

%feature("docstring")  shogun::CKernelMachine::get_alphas "

get all alphas (swig compatible)

Parameters:
-----------

alphas:  array to contain a copy of the alphas

d1:  number of alphas in the array ";

%feature("docstring")  shogun::CKernelMachine::create_new_model "

create new model

Parameters:
-----------

num:  number of alphas and support vectors in new model ";

%feature("docstring")
shogun::CKernelMachine::init_kernel_optimization "

initialise kernel optimisation

if operation was successful ";

%feature("docstring")  shogun::CKernelMachine::classify "

classify kernel machine

result labels ";

%feature("docstring")  shogun::CKernelMachine::classify "

classify objects

Parameters:
-----------

data:  (test) data to be classified

classified labels ";

%feature("docstring")  shogun::CKernelMachine::classify_example "

classify one example

Parameters:
-----------

num:  which example to classify

classified value ";


// File: classshogun_1_1CKernelPerceptron.xml
%feature("docstring") shogun::CKernelPerceptron "

Class KernelPerceptron - currently unfinished implementation of a
Kernel Perceptron.

C++ includes: KernelPerceptron.h ";

%feature("docstring")  shogun::CKernelPerceptron::CKernelPerceptron "

constructor ";

%feature("docstring")  shogun::CKernelPerceptron::~CKernelPerceptron "";

%feature("docstring")  shogun::CKernelPerceptron::train "

train kernel perceptron classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CKernelPerceptron::classify_example "

classify one specific example

Parameters:
-----------

num:  which example to classify

classified value ";

%feature("docstring")  shogun::CKernelPerceptron::load "

load KernelPerceptron from file

Parameters:
-----------

srcfile:  file to load from

if load was successful ";

%feature("docstring")  shogun::CKernelPerceptron::save "

save KernelPerceptron to file

Parameters:
-----------

dstfile:  file to save to

if save was successful ";

%feature("docstring")  shogun::CKernelPerceptron::get_classifier_type
"

get classifier type

classifier type KERNELPERCEPTRON ";

%feature("docstring")  shogun::CKernelPerceptron::get_name "

object name ";


// File: classshogun_1_1CKNN.xml
%feature("docstring") shogun::CKNN "

Class KNN, an implementation of the standard k-nearest neigbor
classifier.

An example is classified to belong to the class of which the majority
of the k closest examples belong to.

To avoid ties, k should be an odd number. To define how close examples
are k-NN requires a CDistance object to work with (e.g.,
CEuclideanDistance ).

Note that k-NN has zero training time but classification times
increase dramatically with the number of examples. Also note that k-NN
is capable of multi-class-classification.

C++ includes: KNN.h ";

%feature("docstring")  shogun::CKNN::CKNN "

default constructor ";

%feature("docstring")  shogun::CKNN::CKNN "

constructor

Parameters:
-----------

k:  k

d:  distance

trainlab:  labels for training ";

%feature("docstring")  shogun::CKNN::~CKNN "";

%feature("docstring")  shogun::CKNN::get_classifier_type "

get classifier type

classifier type KNN ";

%feature("docstring")  shogun::CKNN::train "

train k-NN classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CKNN::classify "

classify all examples

resulting labels ";

%feature("docstring")  shogun::CKNN::classify "

classify objects

Parameters:
-----------

data:  (test) data to be classified

classified labels ";

%feature("docstring")  shogun::CKNN::classify_example "

get output for example \"vec_idx\" ";

%feature("docstring")  shogun::CKNN::classify_for_multiple_k "

classify all examples for 1...k

Parameters:
-----------

output:  resulting labels for all k

k_out:  number of columns (k)

num_vec:  number of outputs ";

%feature("docstring")  shogun::CKNN::load "

load from file

Parameters:
-----------

srcfile:  file to load from

if loading was successful ";

%feature("docstring")  shogun::CKNN::save "

save to file

Parameters:
-----------

dstfile:  file to save to

if saving was successful ";

%feature("docstring")  shogun::CKNN::set_k "

set k

Parameters:
-----------

p_k:  new k ";

%feature("docstring")  shogun::CKNN::get_k "

get k

k ";

%feature("docstring")  shogun::CKNN::get_name "

object name ";


// File: classshogun_1_1CLaRank.xml
%feature("docstring") shogun::CLaRank "";

%feature("docstring")  shogun::CLaRank::CLaRank "";

%feature("docstring")  shogun::CLaRank::CLaRank "

constructor

Parameters:
-----------

C:  constant C

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CLaRank::~CLaRank "";

%feature("docstring")  shogun::CLaRank::train "

train classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CLaRank::add "";

%feature("docstring")  shogun::CLaRank::predict "";

%feature("docstring")  shogun::CLaRank::destroy "";

%feature("docstring")  shogun::CLaRank::computeGap "";

%feature("docstring")  shogun::CLaRank::getNumOutputs "";

%feature("docstring")  shogun::CLaRank::getNSV "";

%feature("docstring")  shogun::CLaRank::computeW2 "";

%feature("docstring")  shogun::CLaRank::getDual "";

%feature("docstring")  shogun::CLaRank::get_classifier_type "

get classifier type

classifier type LIBSVM ";

%feature("docstring")  shogun::CLaRank::get_name "

object name ";

%feature("docstring")  shogun::CLaRank::set_batch_mode "";

%feature("docstring")  shogun::CLaRank::get_batch_mode "";

%feature("docstring")  shogun::CLaRank::set_tau "";

%feature("docstring")  shogun::CLaRank::get_tau "";


// File: classshogun_1_1CLibSVM.xml
%feature("docstring") shogun::CLibSVM "

LibSVM.

C++ includes: LibSVM.h ";

%feature("docstring")  shogun::CLibSVM::CLibSVM "

constructor ";

%feature("docstring")  shogun::CLibSVM::CLibSVM "

constructor

Parameters:
-----------

C:  constant C

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CLibSVM::~CLibSVM "";

%feature("docstring")  shogun::CLibSVM::train "

train SVM classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CLibSVM::get_classifier_type "

get classifier type

classifier type LIBSVM ";

%feature("docstring")  shogun::CLibSVM::get_name "

object name ";


// File: classshogun_1_1CLibSVMMultiClass.xml
%feature("docstring") shogun::CLibSVMMultiClass "

class LibSVMMultiClass

C++ includes: LibSVMMultiClass.h ";

%feature("docstring")  shogun::CLibSVMMultiClass::CLibSVMMultiClass "

default constructor ";

%feature("docstring")  shogun::CLibSVMMultiClass::CLibSVMMultiClass "

constructor

Parameters:
-----------

C:  constant C

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CLibSVMMultiClass::~CLibSVMMultiClass "";

%feature("docstring")  shogun::CLibSVMMultiClass::train "

train multiclass SVM classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CLibSVMMultiClass::get_classifier_type
"

get classifier type

classifier type LIBSVMMULTICLASS ";

%feature("docstring")  shogun::CLibSVMMultiClass::get_name "

object name ";


// File: classshogun_1_1CLibSVMOneClass.xml
%feature("docstring") shogun::CLibSVMOneClass "

class LibSVMOneClass

C++ includes: LibSVMOneClass.h ";

%feature("docstring")  shogun::CLibSVMOneClass::CLibSVMOneClass "

default constructor ";

%feature("docstring")  shogun::CLibSVMOneClass::CLibSVMOneClass "

constructor

Parameters:
-----------

C:  constant C

k:  kernel ";

%feature("docstring")  shogun::CLibSVMOneClass::~CLibSVMOneClass "";

%feature("docstring")  shogun::CLibSVMOneClass::train "

train SVM

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CLibSVMOneClass::get_classifier_type "

get classifier type

classifier type LIBSVMONECLASS ";

%feature("docstring")  shogun::CLibSVMOneClass::get_name "

object name ";


// File: classshogun_1_1CLinearClassifier.xml
%feature("docstring") shogun::CLinearClassifier "

Class LinearClassifier is a generic interface for all kinds of linear
classifiers.

A linear classifier computes

\\\\[ f({\\\\bf x})= {\\\\bf w} \\\\cdot {\\\\bf x} + b \\\\]

where ${\\\\bf w}$ are the weights assigned to each feature in
training and $b$ the bias.

To implement a linear classifier all that is required is to define the
train() function that delivers ${\\\\bf w}$ above.

Note that this framework works with linear classifiers of arbitraty
feature type, e.g. dense and sparse and even string based features.
This is implemented by using CDotFeatures that may provide a mapping
function $\\\\Phi({\\\\bf x})\\\\mapsto {\\\\cal R^D}$ encapsulating
all the required operations (like the dot product). The decision
function is thus

\\\\[ f({\\\\bf x})= {\\\\bf w} \\\\cdot \\\\Phi({\\\\bf x}) + b.
\\\\]

The following linear classifiers are implemented Linear Descriminant
Analysis (CLDA)

Linear Programming Machines (CLPM, CLPBoost)

Perceptron ( CPerceptron)

Linear SVMs ( CSVMSGD, CLibLinear, CSVMOcas, CSVMLin, CSubgradientSVM)

See:  CDotFeatures

C++ includes: LinearClassifier.h ";

%feature("docstring")  shogun::CLinearClassifier::CLinearClassifier "

default constructor ";

%feature("docstring")  shogun::CLinearClassifier::~CLinearClassifier "";

%feature("docstring")  shogun::CLinearClassifier::classify_example "

get output for example \"vec_idx\" ";

%feature("docstring")  shogun::CLinearClassifier::get_w "

get w

Parameters:
-----------

dst_w:  store w in this argument

dst_dims:  dimension of w ";

%feature("docstring")  shogun::CLinearClassifier::get_w "

get w (swig compatible)

Parameters:
-----------

dst_w:  store w in this argument

dst_dims:  dimension of w ";

%feature("docstring")  shogun::CLinearClassifier::set_w "

set w

Parameters:
-----------

src_w:  new w

src_w_dim:  dimension of new w ";

%feature("docstring")  shogun::CLinearClassifier::set_bias "

set bias

Parameters:
-----------

b:  new bias ";

%feature("docstring")  shogun::CLinearClassifier::get_bias "

get bias

bias ";

%feature("docstring")  shogun::CLinearClassifier::load "

load from file

Parameters:
-----------

srcfile:  file to load from

if loading was successful ";

%feature("docstring")  shogun::CLinearClassifier::save "

save to file

Parameters:
-----------

dstfile:  file to save to

if saving was successful ";

%feature("docstring")  shogun::CLinearClassifier::set_features "

set features

Parameters:
-----------

feat:  features to set ";

%feature("docstring")  shogun::CLinearClassifier::classify "

classify all examples

resulting labels ";

%feature("docstring")  shogun::CLinearClassifier::classify "

classify objects

Parameters:
-----------

data:  (test) data to be classified

classified labels ";

%feature("docstring")  shogun::CLinearClassifier::get_features "

get features

features ";


// File: classshogun_1_1CMKL.xml
%feature("docstring") shogun::CMKL "

Multiple Kernel Learning.

A support vector machine based method for use with multiple kernels.
In Multiple Kernel Learning (MKL) in addition to the SVM
$\\\\bf\\\\alpha$ and bias term $b$ the kernel weights
$\\\\bf\\\\beta$ are estimated in training. The resulting kernel
method can be stated as

\\\\[ f({\\\\bf x})=\\\\sum_{i=0}^{N-1} \\\\alpha_i \\\\sum_{j=0}^M
\\\\beta_j k_j({\\\\bf x}, {\\\\bf x_i})+b . \\\\]

where $N$ is the number of training examples $\\\\alpha_i$ are the
weights assigned to each training example $\\\\beta_j$ are the weights
assigned to each sub-kernel $k_j(x,x')$ are sub-kernels and $b$ the
bias.

Kernels have to be chosen a-priori. In MKL $\\\\alpha_i,\\\\;\\\\beta$
and bias are determined by solving the following optimization program

\\\\begin{eqnarray*} \\\\mbox{min} &&
\\\\gamma-\\\\sum_{i=1}^N\\\\alpha_i\\\\\\\\ \\\\mbox{w.r.t.} &&
\\\\gamma\\\\in R, \\\\alpha\\\\in R^N \\\\nonumber\\\\\\\\
\\\\mbox{s.t.} && {\\\\bf 0}\\\\leq\\\\alpha\\\\leq{\\\\bf
1}C,\\\\;\\\\;\\\\sum_{i=1}^N \\\\alpha_i y_i=0 \\\\nonumber\\\\\\\\
&& \\\\frac{1}{2}\\\\sum_{i,j=1}^N \\\\alpha_i \\\\alpha_j y_i y_j
k_k({\\\\bf x}_i,{\\\\bf x}_j)\\\\leq \\\\gamma,\\\\;\\\\; \\\\forall
k=1,\\\\ldots,K\\\\nonumber\\\\\\\\ \\\\end{eqnarray*} here C is a
pre-specified regularization parameter.

Within shogun this optimization problem is solved using semi-infinite
programming. For 1-norm MKL using one of the two approaches described
in

Soeren Sonnenburg, Gunnar Raetsch, Christin Schaefer, and Bernhard
Schoelkopf. Large Scale Multiple Kernel Learning. Journal of Machine
Learning Research, 7:1531-1565, July 2006.

The first approach (also called the wrapper algorithm) wrapps around a
single kernel SVMs, alternatingly solving for $\\\\alpha$ and
$\\\\beta$. It is using a traditional SVM to generate new violated
constraings and thus requires a single kernel SVM and any of the SVMs
contained in shogun can be used. In the MKL step either a linear
program is solved via glpk or cplex or analytically or a newton (for
norms>1) step is performed.

The second much faster but also more memory demanding approach
performing interleaved optimization, is integrated into the chunking-
based SVMlight.

In addition sparsity of MKL can be controlled by the choice of the
$L_p$-norm regularizing $\\\\beta$ as described in

Marius Kloft, Ulf Brefeld, Soeren Sonnenburg, and Alexander Zien.
Efficient and accurate lp-norm multiple kernel learning. In Advances
in Neural Information Processing Systems 21. MIT Press, Cambridge, MA,
2009.

C++ includes: MKL.h ";

%feature("docstring")  shogun::CMKL::CMKL "

Constructor

Parameters:
-----------

s:  SVM to use as constraint generator in MKL SIP ";

%feature("docstring")  shogun::CMKL::~CMKL "

Destructor ";

%feature("docstring")  shogun::CMKL::set_constraint_generator "

SVM to use as constraint generator in MKL SIP

Parameters:
-----------

s:  svm ";

%feature("docstring")  shogun::CMKL::set_svm "

SVM to use as constraint generator in MKL SIP

Parameters:
-----------

s:  svm ";

%feature("docstring")  shogun::CMKL::get_svm "

get SVM that is used as constraint generator in MKL SIP

svm ";

%feature("docstring")  shogun::CMKL::train "

train MKL classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CMKL::set_C_mkl "

set C mkl

Parameters:
-----------

C:  new C_mkl ";

%feature("docstring")  shogun::CMKL::set_mkl_norm "

set mkl norm

Parameters:
-----------

norm:  new mkl norm (must be greater equal 1) ";

%feature("docstring")
shogun::CMKL::set_interleaved_optimization_enabled "

set state of optimization (interleaved or wrapper)

Parameters:
-----------

enable:  if true interleaved optimization is used; wrapper otherwise
";

%feature("docstring")
shogun::CMKL::get_interleaved_optimization_enabled "

get state of optimization (interleaved or wrapper)

true if interleaved optimization is used; wrapper otherwise ";

%feature("docstring")  shogun::CMKL::compute_mkl_primal_objective "

compute mkl primal objective

computed mkl primal objective ";

%feature("docstring")  shogun::CMKL::compute_mkl_dual_objective "

compute mkl dual objective

computed dual objective ";

%feature("docstring")  shogun::CMKL::set_mkl_epsilon "

set mkl epsilon (optimization accuracy for kernel weights)

Parameters:
-----------

eps:  new weight_epsilon ";

%feature("docstring")  shogun::CMKL::get_mkl_epsilon "

get mkl epsilon for weights (optimization accuracy for kernel weights)

epsilon for weights ";

%feature("docstring")  shogun::CMKL::get_mkl_iterations "

get number of MKL iterations

mkl_iterations ";

%feature("docstring")  shogun::CMKL::perform_mkl_step "

perform single mkl iteration

given sum of alphas, objectives for current alphas for each kernel and
current kernel weighting compute the corresponding optimal kernel
weighting (all via get/set_subkernel_weights in CCombinedKernel)

Parameters:
-----------

sumw:  vector of 1/2*alpha'*K_j*alpha for each kernel j

suma:  scalar sum_i alpha_i etc. ";

%feature("docstring")  shogun::CMKL::compute_sum_alpha "

compute beta independent term from objective, e.g., in 2-class MKL
sum_i alpha_i etc ";

%feature("docstring")  shogun::CMKL::compute_sum_beta "

compute 1/2*alpha'*K_j*alpha for each kernel j (beta dependent term
from objective)

Parameters:
-----------

sumw:  vector of size num_kernels to hold the result ";


// File: classshogun_1_1CMKLClassification.xml
%feature("docstring") shogun::CMKLClassification "

Multiple Kernel Learning for two-class-classification.

Learns an SVM classifier and its kernel weights. Makes only sense if
multiple kernels are used.

See:   CMKL

C++ includes: MKLClassification.h ";

%feature("docstring")  shogun::CMKLClassification::CMKLClassification
"

Constructor

Parameters:
-----------

s:  SVM to use as constraint generator in MKL SILP ";

%feature("docstring")  shogun::CMKLClassification::~CMKLClassification
"

Destructor ";

%feature("docstring")  shogun::CMKLClassification::compute_sum_alpha "

compute beta independent term from objective, e.g., in 2-class MKL
sum_i alpha_i etc ";


// File: classshogun_1_1CMKLMultiClass.xml
%feature("docstring") shogun::CMKLMultiClass "

MKLMultiClass is a class for L1-norm multiclass MKL.

It is based on the GMNPSVM Multiclass SVM. Its own parameters are the
L2 norm weight change based MKL Its termination criterion set by void
set_mkl_epsilon(float64_t eps ); and the maximal number of MKL
iterations set by void set_max_num_mkliters(int32_t maxnum); It passes
the regularization constants C1 and C2 to GMNPSVM.

C++ includes: MKLMultiClass.h ";

%feature("docstring")  shogun::CMKLMultiClass::CMKLMultiClass "

Class default Constructor ";

%feature("docstring")  shogun::CMKLMultiClass::CMKLMultiClass "

Class Constructor commonly used in Shogun Toolbox

Parameters:
-----------

C:  constant C

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CMKLMultiClass::~CMKLMultiClass "

Class default Destructor ";

%feature("docstring")  shogun::CMKLMultiClass::train "

train Multiclass MKL classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CMKLMultiClass::get_classifier_type "

get classifier type

classifier type GMNPMKL ";

%feature("docstring")  shogun::CMKLMultiClass::getsubkernelweights "

returns MKL weights for the different kernels

Parameters:
-----------

numweights:  is output parameter, is set to zero if no weights have
been computed or to the number of MKL weights which is equal to the
number of kernels

NULL if no weights have been computed or otherwise an array with the
weights, caller has to delete[] the output by itself ";

%feature("docstring")  shogun::CMKLMultiClass::set_mkl_epsilon "

sets MKL termination threshold

Parameters:
-----------

eps:  is the desired threshold value the termination criterion is the
L2 norm between the current MKL weights and their counterpart from the
previous iteration ";

%feature("docstring")  shogun::CMKLMultiClass::set_max_num_mkliters "

sets maximal number of MKL iterations

Parameters:
-----------

maxnum:  is the desired maximal number of MKL iterations; when it is
reached the MKL terminates irrespective of the MKL progress set it to
a nonpositive value in order to turn it off ";


// File: classshogun_1_1CMKLOneClass.xml
%feature("docstring") shogun::CMKLOneClass "

Multiple Kernel Learning for one-class-classification.

Learns a One-Class SVM classifier and its kernel weights. Makes only
sense if multiple kernels are used.

See:   CMKL

C++ includes: MKLOneClass.h ";

%feature("docstring")  shogun::CMKLOneClass::CMKLOneClass "

Constructor

Parameters:
-----------

s:  SVM to use as constraint generator in MKL SILP ";

%feature("docstring")  shogun::CMKLOneClass::~CMKLOneClass "

Destructor ";

%feature("docstring")  shogun::CMKLOneClass::compute_sum_alpha "

compute beta independent term from objective, e.g., in 2-class MKL
sum_i alpha_i etc ";


// File: classshogun_1_1CMPDSVM.xml
%feature("docstring") shogun::CMPDSVM "

class MPDSVM

C++ includes: MPDSVM.h ";

%feature("docstring")  shogun::CMPDSVM::CMPDSVM "

default constructor ";

%feature("docstring")  shogun::CMPDSVM::CMPDSVM "

constructor

Parameters:
-----------

C:  constant C

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CMPDSVM::~CMPDSVM "";

%feature("docstring")  shogun::CMPDSVM::train "

train SVM classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CMPDSVM::get_classifier_type "

get classifier type

classifier type MPD ";

%feature("docstring")  shogun::CMPDSVM::get_name "

object name ";


// File: classshogun_1_1CMultiClassSVM.xml
%feature("docstring") shogun::CMultiClassSVM "

class MultiClassSVM

C++ includes: MultiClassSVM.h ";

%feature("docstring")  shogun::CMultiClassSVM::CMultiClassSVM "

constructor

Parameters:
-----------

type:  type of MultiClassSVM ";

%feature("docstring")  shogun::CMultiClassSVM::CMultiClassSVM "

constructor

Parameters:
-----------

type:  type of MultiClassSVM

C:  constant C

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CMultiClassSVM::~CMultiClassSVM "";

%feature("docstring")  shogun::CMultiClassSVM::create_multiclass_svm "

create multiclass SVM

Parameters:
-----------

num_classes:  number of classes in SVM

if creation was successful ";

%feature("docstring")  shogun::CMultiClassSVM::set_svm "

set SVM

Parameters:
-----------

num:  number to set

svm:  SVM to set

if setting was successful ";

%feature("docstring")  shogun::CMultiClassSVM::get_svm "

get SVM

Parameters:
-----------

num:  which SVM to get

SVM at number num ";

%feature("docstring")  shogun::CMultiClassSVM::get_num_svms "

get number of SVMs

number of SVMs ";

%feature("docstring")  shogun::CMultiClassSVM::cleanup "

cleanup SVM ";

%feature("docstring")  shogun::CMultiClassSVM::classify "

classify all examples

resulting labels ";

%feature("docstring")  shogun::CMultiClassSVM::classify_example "

classify one example

Parameters:
-----------

num:  number of example to classify

resulting classification ";

%feature("docstring")  shogun::CMultiClassSVM::classify_one_vs_rest "

classify one vs rest

resulting labels ";

%feature("docstring")
shogun::CMultiClassSVM::classify_example_one_vs_rest "

classify one example one vs rest

Parameters:
-----------

num:  number of example of classify

resulting classification ";

%feature("docstring")  shogun::CMultiClassSVM::classify_one_vs_one "

classify one vs one

resulting labels ";

%feature("docstring")
shogun::CMultiClassSVM::classify_example_one_vs_one "

classify one example one vs one

Parameters:
-----------

num:  number of example of classify

resulting classification ";

%feature("docstring")  shogun::CMultiClassSVM::load "

load a Multiclass SVM from file

Parameters:
-----------

svm_file:  the file handle ";

%feature("docstring")  shogun::CMultiClassSVM::save "

write a Multiclass SVM to a file

Parameters:
-----------

svm_file:  the file handle ";


// File: classshogun_1_1CPerceptron.xml
%feature("docstring") shogun::CPerceptron "

Class Perceptron implements the standard linear (online) perceptron.

Given a maximum number of iterations (the standard perceptron
algorithm is not guaranteed to converge) and a fixed lerning rate, the
result is a linear classifier.

See:   CLinearClassifier

http://en.wikipedia.org/wiki/Perceptron

C++ includes: Perceptron.h ";

%feature("docstring")  shogun::CPerceptron::CPerceptron "

default constructor ";

%feature("docstring")  shogun::CPerceptron::CPerceptron "

constructor

Parameters:
-----------

traindat:  training features

trainlab:  labels for training features ";

%feature("docstring")  shogun::CPerceptron::~CPerceptron "";

%feature("docstring")  shogun::CPerceptron::get_classifier_type "

get classifier type

classifier type PERCEPTRON ";

%feature("docstring")  shogun::CPerceptron::train "

train classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CPerceptron::set_learn_rate "

set learn rate of gradient descent training algorithm ";

%feature("docstring")  shogun::CPerceptron::set_max_iter "

set maximum number of iterations ";

%feature("docstring")  shogun::CPerceptron::get_name "

object name ";


// File: classshogun_1_1CPluginEstimate.xml
%feature("docstring") shogun::CPluginEstimate "

class PluginEstimate

The class PluginEstimate takes as input two probabilistic models (of
type CLinearHMM, even though general models are possible ) and
classifies examples according to the rule

\\\\[ f({\\\\bf x})= \\\\log(\\\\mbox{Pr}({\\\\bf x}|\\\\theta_+)) -
\\\\log(\\\\mbox{Pr}({\\\\bf x}|\\\\theta_-)) \\\\]

See:  CLinearHMM

CDistribution

C++ includes: PluginEstimate.h ";

%feature("docstring")  shogun::CPluginEstimate::CPluginEstimate "

default constructor

Parameters:
-----------

pos_pseudo:  pseudo for positive model

neg_pseudo:  pseudo for negative model ";

%feature("docstring")  shogun::CPluginEstimate::~CPluginEstimate "";

%feature("docstring")  shogun::CPluginEstimate::train "

train plugin estimate classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CPluginEstimate::classify "

classify objects using the currently set features

classified labels ";

%feature("docstring")  shogun::CPluginEstimate::classify "

classify objects

Parameters:
-----------

data:  (test) data to be classified

classified labels ";

%feature("docstring")  shogun::CPluginEstimate::set_features "

set features

Parameters:
-----------

feat:  features to set ";

%feature("docstring")  shogun::CPluginEstimate::get_features "

get features

features ";

%feature("docstring")  shogun::CPluginEstimate::classify_example "

classify the test feature vector indexed by vec_idx ";

%feature("docstring")
shogun::CPluginEstimate::posterior_log_odds_obsolete "

obsolete posterior log odds

Parameters:
-----------

vector:  vector

len:  len

something floaty ";

%feature("docstring")
shogun::CPluginEstimate::get_parameterwise_log_odds "

get log odds parameter-wise

Parameters:
-----------

obs:  observation

position:  position

log odd at position ";

%feature("docstring")
shogun::CPluginEstimate::log_derivative_pos_obsolete "

get obsolete positive log derivative

Parameters:
-----------

obs:  observation

pos:  position

positive log derivative ";

%feature("docstring")
shogun::CPluginEstimate::log_derivative_neg_obsolete "

get obsolete negative log derivative

Parameters:
-----------

obs:  observation

pos:  position

negative log derivative ";

%feature("docstring")  shogun::CPluginEstimate::get_model_params "

get model parameters

Parameters:
-----------

pos_params:  parameters of positive model

neg_params:  parameters of negative model

seq_length:  sequence length

num_symbols:  numbe of symbols

if operation was successful ";

%feature("docstring")  shogun::CPluginEstimate::set_model_params "

set model parameters

Parameters:
-----------

pos_params:  parameters of positive model

neg_params:  parameters of negative model

seq_length:  sequence length

num_symbols:  numbe of symbols ";

%feature("docstring")  shogun::CPluginEstimate::get_num_params "

get number of parameters

number of parameters ";

%feature("docstring")  shogun::CPluginEstimate::check_models "

check models

if one of the two models is invalid ";

%feature("docstring")  shogun::CPluginEstimate::get_name "

object name ";


// File: classshogun_1_1CQPBSVMLib.xml
%feature("docstring") shogun::CQPBSVMLib "

class QPBSVMLib

C++ includes: qpbsvmlib.h ";

%feature("docstring")  shogun::CQPBSVMLib::CQPBSVMLib "

constructor

Parameters:
-----------

H:  symmetric matrix of size n x n

n:  size of H's matrix

f:  is vector of size m

m:  size of vector f

UB:  UB ";

%feature("docstring")  shogun::CQPBSVMLib::solve_qp "

result has to be allocated & zeroed ";

%feature("docstring")  shogun::CQPBSVMLib::set_solver "

set solver

Parameters:
-----------

solver:  new solver ";

%feature("docstring")  shogun::CQPBSVMLib::~CQPBSVMLib "";


// File: classshogun_1_1CScatterSVM.xml
%feature("docstring") shogun::CScatterSVM "

ScatterSVM - Multiclass SVM.

The ScatterSVM is an unpublished experimental true multiclass SVM.
Details are availabe in the following technical report.

Robert Jenssen and Marius Kloft and Alexander Zien and S\\\\\"oren
Sonnenburg and            Klaus-Robert M\\\\\"{u}ller, A Multi-Class
Support Vector Machine Based on Scatter Criteria, TR 014-2009 TU
Berlin, 2009

C++ includes: ScatterSVM.h ";

%feature("docstring")  shogun::CScatterSVM::CScatterSVM "

constructor ";

%feature("docstring")  shogun::CScatterSVM::CScatterSVM "

constructor

Parameters:
-----------

C:  constant C

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CScatterSVM::~CScatterSVM "

default destructor ";

%feature("docstring")  shogun::CScatterSVM::train "

train SVM classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CScatterSVM::get_classifier_type "

get classifier type

classifier type LIBSVM ";

%feature("docstring")  shogun::CScatterSVM::classify_example "

classify one example

Parameters:
-----------

num:  number of example to classify

resulting classification ";

%feature("docstring")  shogun::CScatterSVM::classify_one_vs_rest "

classify one vs rest

resulting labels ";

%feature("docstring")  shogun::CScatterSVM::get_name "

object name ";


// File: classshogun_1_1CSubGradientSVM.xml
%feature("docstring") shogun::CSubGradientSVM "

class SubGradientSVM

C++ includes: SubGradientSVM.h ";

%feature("docstring")  shogun::CSubGradientSVM::CSubGradientSVM "

default constructor ";

%feature("docstring")  shogun::CSubGradientSVM::CSubGradientSVM "

constructor

Parameters:
-----------

C:  constant C

traindat:  training features

trainlab:  labels for training features ";

%feature("docstring")  shogun::CSubGradientSVM::~CSubGradientSVM "";

%feature("docstring")  shogun::CSubGradientSVM::get_classifier_type "

get classifier type

classifier type SUBGRADIENTSVM ";

%feature("docstring")  shogun::CSubGradientSVM::train "

train SVM classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CSubGradientSVM::set_C "

set C

Parameters:
-----------

c1:  new C1

c2:  new C2 ";

%feature("docstring")  shogun::CSubGradientSVM::get_C1 "

get C1

C1 ";

%feature("docstring")  shogun::CSubGradientSVM::get_C2 "

get C2

C2 ";

%feature("docstring")  shogun::CSubGradientSVM::set_bias_enabled "

set if bias shall be enabled

Parameters:
-----------

enable_bias:  if bias shall be enabled ";

%feature("docstring")  shogun::CSubGradientSVM::get_bias_enabled "

check if bias is enabled

if bias is enabled ";

%feature("docstring")  shogun::CSubGradientSVM::set_epsilon "

set epsilon

Parameters:
-----------

eps:  new epsilon ";

%feature("docstring")  shogun::CSubGradientSVM::get_epsilon "

get epsilon

epsilon ";

%feature("docstring")  shogun::CSubGradientSVM::set_qpsize "

set qpsize

Parameters:
-----------

q:  new qpsize ";

%feature("docstring")  shogun::CSubGradientSVM::get_qpsize "

get qpsize

qpsize ";

%feature("docstring")  shogun::CSubGradientSVM::set_qpsize_max "

set qpsize_max

Parameters:
-----------

q:  new qpsize_max ";

%feature("docstring")  shogun::CSubGradientSVM::get_qpsize_max "

get qpsize_max

qpsize_max ";


// File: classshogun_1_1CSVM.xml
%feature("docstring") shogun::CSVM "

A generic Support Vector Machine Interface.

A support vector machine is defined as \\\\[ f({\\\\bf
x})=\\\\sum_{i=0}^{N-1} \\\\alpha_i k({\\\\bf x}, {\\\\bf x_i})+b
\\\\]

where $N$ is the number of training examples $\\\\alpha_i$ are the
weights assigned to each training example $k(x,x')$ is the kernel and
$b$ the bias.

Using an a-priori choosen kernel, the $\\\\alpha_i$ and bias are
determined by solving the following quadratic program

\\\\begin{eqnarray*} \\\\max_{\\\\bf \\\\alpha} && \\\\sum_{i=0}^{N-1}
\\\\alpha_i - \\\\sum_{i=0}^{N-1}\\\\sum_{j=0}^{N-1} \\\\alpha_i y_i
\\\\alpha_j y_j k({\\\\bf x_i}, {\\\\bf x_j})\\\\\\\\ \\\\mbox{s.t.}
&& 0\\\\leq\\\\alpha_i\\\\leq C\\\\\\\\ && \\\\sum_{i=0}^{N-1}
\\\\alpha_i y_i=0\\\\\\\\ \\\\end{eqnarray*} here C is a pre-specified
regularization parameter.

C++ includes: SVM.h ";

%feature("docstring")  shogun::CSVM::CSVM "

Create an empty Support Vector Machine Object

Parameters:
-----------

num_sv:  with num_sv support vectors ";

%feature("docstring")  shogun::CSVM::CSVM "

Create a Support Vector Machine Object from a trained SVM

Parameters:
-----------

C:  the C parameter

k:  the Kernel object

lab:  the Label object ";

%feature("docstring")  shogun::CSVM::~CSVM "";

%feature("docstring")  shogun::CSVM::set_defaults "

set default values for members a SVM object ";

%feature("docstring")  shogun::CSVM::get_linear_term "

get linear term

lin the linear term ";

%feature("docstring")  shogun::CSVM::set_linear_term "

set linear term of the QP

Parameters:
-----------

lin:  the linear term ";

%feature("docstring")  shogun::CSVM::load "

load a SVM from file

Parameters:
-----------

svm_file:  the file handle ";

%feature("docstring")  shogun::CSVM::save "

write a SVM to a file

Parameters:
-----------

svm_file:  the file handle ";

%feature("docstring")  shogun::CSVM::set_nu "

set nu

Parameters:
-----------

nue:  new nu ";

%feature("docstring")  shogun::CSVM::set_C "

set C

Parameters:
-----------

c1:  new C constant for negatively labelled examples

c2:  new C constant for positively labelled examples

Note that not all SVMs support this (however at least CLibSVM and
CSVMLight do) ";

%feature("docstring")  shogun::CSVM::set_epsilon "

set epsilon

Parameters:
-----------

eps:  new epsilon ";

%feature("docstring")  shogun::CSVM::set_tube_epsilon "

set tube epsilon

Parameters:
-----------

eps:  new tube epsilon ";

%feature("docstring")  shogun::CSVM::set_qpsize "

set qpsize

Parameters:
-----------

qps:  new qpsize ";

%feature("docstring")  shogun::CSVM::get_epsilon "

get epsilon

epsilon ";

%feature("docstring")  shogun::CSVM::get_nu "

get nu

nu ";

%feature("docstring")  shogun::CSVM::get_C1 "

get C1

C1 ";

%feature("docstring")  shogun::CSVM::get_C2 "

get C2

C2 ";

%feature("docstring")  shogun::CSVM::get_qpsize "

get qpsize

qpsize ";

%feature("docstring")  shogun::CSVM::set_shrinking_enabled "

set state of shrinking

Parameters:
-----------

enable:  if shrinking will be enabled ";

%feature("docstring")  shogun::CSVM::get_shrinking_enabled "

get state of shrinking

if shrinking is enabled ";

%feature("docstring")  shogun::CSVM::compute_svm_dual_objective "

compute svm dual objective

computed dual objective ";

%feature("docstring")  shogun::CSVM::compute_svm_primal_objective "

compute svm primal objective

computed svm primal objective ";

%feature("docstring")  shogun::CSVM::set_objective "

set objective

Parameters:
-----------

v:  objective ";

%feature("docstring")  shogun::CSVM::get_objective "

get objective

objective ";

%feature("docstring")  shogun::CSVM::set_callback_function "

set callback function svm optimizers may call when they have a new
(small) set of alphas

Parameters:
-----------

m:  pointer to mkl object

cb:  callback function ";

%feature("docstring")  shogun::CSVM::get_name "

object name ";


// File: classshogun_1_1CSVMLin.xml
%feature("docstring") shogun::CSVMLin "

class SVMLin

C++ includes: SVMLin.h ";

%feature("docstring")  shogun::CSVMLin::CSVMLin "

default constructor ";

%feature("docstring")  shogun::CSVMLin::CSVMLin "

constructor

Parameters:
-----------

C:  constant C

traindat:  training features

trainlab:  labels for features ";

%feature("docstring")  shogun::CSVMLin::~CSVMLin "";

%feature("docstring")  shogun::CSVMLin::get_classifier_type "

get classifier type

classifier type SVMLIN ";

%feature("docstring")  shogun::CSVMLin::train "

train SVM classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CSVMLin::set_C "

set C

Parameters:
-----------

c1:  new C1

c2:  new C2 ";

%feature("docstring")  shogun::CSVMLin::get_C1 "

get C1

C1 ";

%feature("docstring")  shogun::CSVMLin::get_C2 "

get C2

C2 ";

%feature("docstring")  shogun::CSVMLin::set_bias_enabled "

set if bias shall be enabled

Parameters:
-----------

enable_bias:  if bias shall be enabled ";

%feature("docstring")  shogun::CSVMLin::get_bias_enabled "

get if bias is enabled

if bias is enabled ";

%feature("docstring")  shogun::CSVMLin::set_epsilon "

set epsilon

Parameters:
-----------

eps:  new epsilon ";

%feature("docstring")  shogun::CSVMLin::get_epsilon "

get epsilon

epsilon ";

%feature("docstring")  shogun::CSVMLin::get_name "

object name ";


// File: classshogun_1_1CSVMOcas.xml
%feature("docstring") shogun::CSVMOcas "

class SVMOcas

C++ includes: SVMOcas.h ";

%feature("docstring")  shogun::CSVMOcas::CSVMOcas "

constructor

Parameters:
-----------

type:  a E_SVM_TYPE ";

%feature("docstring")  shogun::CSVMOcas::CSVMOcas "

constructor

Parameters:
-----------

C:  constant C

traindat:  training features

trainlab:  labels for training features ";

%feature("docstring")  shogun::CSVMOcas::~CSVMOcas "";

%feature("docstring")  shogun::CSVMOcas::get_classifier_type "

get classifier type

classifier type SVMOCAS ";

%feature("docstring")  shogun::CSVMOcas::train "

train SVM classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CSVMOcas::set_C "

set C

Parameters:
-----------

c1:  new C1

c2:  new C2 ";

%feature("docstring")  shogun::CSVMOcas::get_C1 "

get C1

C1 ";

%feature("docstring")  shogun::CSVMOcas::get_C2 "

get C2

C2 ";

%feature("docstring")  shogun::CSVMOcas::set_epsilon "

set epsilon

Parameters:
-----------

eps:  new epsilon ";

%feature("docstring")  shogun::CSVMOcas::get_epsilon "

get epsilon

epsilon ";

%feature("docstring")  shogun::CSVMOcas::set_bias_enabled "

set if bias shall be enabled

Parameters:
-----------

enable_bias:  if bias shall be enabled ";

%feature("docstring")  shogun::CSVMOcas::get_bias_enabled "

check if bias is enabled

if bias is enabled ";

%feature("docstring")  shogun::CSVMOcas::set_bufsize "

set buffer size

Parameters:
-----------

sz:  buffer size ";

%feature("docstring")  shogun::CSVMOcas::get_bufsize "

get buffer size

buffer size ";


// File: classshogun_1_1CSVMSGD.xml
%feature("docstring") shogun::CSVMSGD "

class SVMSGD

C++ includes: SVMSGD.h ";

%feature("docstring")  shogun::CSVMSGD::CSVMSGD "

constructor

Parameters:
-----------

C:  constant C ";

%feature("docstring")  shogun::CSVMSGD::CSVMSGD "

constructor

Parameters:
-----------

C:  constant C

traindat:  training features

trainlab:  labels for training features ";

%feature("docstring")  shogun::CSVMSGD::~CSVMSGD "";

%feature("docstring")  shogun::CSVMSGD::get_classifier_type "

get classifier type

classifier type SVMOCAS ";

%feature("docstring")  shogun::CSVMSGD::train "

train classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CSVMSGD::set_C "

set C

Parameters:
-----------

c1:  new C1

c2:  new C2 ";

%feature("docstring")  shogun::CSVMSGD::get_C1 "

get C1

C1 ";

%feature("docstring")  shogun::CSVMSGD::get_C2 "

get C2

C2 ";

%feature("docstring")  shogun::CSVMSGD::set_epochs "

set epochs

Parameters:
-----------

e:  new number of training epochs ";

%feature("docstring")  shogun::CSVMSGD::get_epochs "

get epochs

the number of training epochs ";

%feature("docstring")  shogun::CSVMSGD::set_bias_enabled "

set if bias shall be enabled

Parameters:
-----------

enable_bias:  if bias shall be enabled ";

%feature("docstring")  shogun::CSVMSGD::get_bias_enabled "

check if bias is enabled

if bias is enabled ";

%feature("docstring")  shogun::CSVMSGD::set_regularized_bias_enabled "

set if regularized bias shall be enabled

Parameters:
-----------

enable_bias:  if regularized bias shall be enabled ";

%feature("docstring")  shogun::CSVMSGD::get_regularized_bias_enabled "

check if regularized bias is enabled

if regularized bias is enabled ";

%feature("docstring")  shogun::CSVMSGD::get_name "

object name ";


// File: classshogun_1_1CWDSVMOcas.xml
%feature("docstring") shogun::CWDSVMOcas "

class WDSVMOcas

C++ includes: WDSVMOcas.h ";

%feature("docstring")  shogun::CWDSVMOcas::CWDSVMOcas "

constructor

Parameters:
-----------

type:  type of SVM ";

%feature("docstring")  shogun::CWDSVMOcas::CWDSVMOcas "

constructor

Parameters:
-----------

C:  constant C

d:  degree

from_d:  from degree

traindat:  training features

trainlab:  labels for training features ";

%feature("docstring")  shogun::CWDSVMOcas::~CWDSVMOcas "";

%feature("docstring")  shogun::CWDSVMOcas::get_classifier_type "

get classifier type

classifier type WDSVMOCAS ";

%feature("docstring")  shogun::CWDSVMOcas::train "

train classifier

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CWDSVMOcas::set_C "

set C

Parameters:
-----------

c1:  new C1

c2:  new C2 ";

%feature("docstring")  shogun::CWDSVMOcas::get_C1 "

get C1

C1 ";

%feature("docstring")  shogun::CWDSVMOcas::get_C2 "

get C2

C2 ";

%feature("docstring")  shogun::CWDSVMOcas::set_epsilon "

set epsilon

Parameters:
-----------

eps:  new epsilon ";

%feature("docstring")  shogun::CWDSVMOcas::get_epsilon "

get epsilon

epsilon ";

%feature("docstring")  shogun::CWDSVMOcas::set_features "

set features

Parameters:
-----------

feat:  features to set ";

%feature("docstring")  shogun::CWDSVMOcas::get_features "

get features

features ";

%feature("docstring")  shogun::CWDSVMOcas::set_bias_enabled "

set if bias shall be enabled

Parameters:
-----------

enable_bias:  if bias shall be enabled ";

%feature("docstring")  shogun::CWDSVMOcas::get_bias_enabled "

check if bias is enabled

if bias is enabled ";

%feature("docstring")  shogun::CWDSVMOcas::set_bufsize "

set buffer size

Parameters:
-----------

sz:  buffer size ";

%feature("docstring")  shogun::CWDSVMOcas::get_bufsize "

get buffer size

buffer size ";

%feature("docstring")  shogun::CWDSVMOcas::set_degree "

set degree

Parameters:
-----------

d:  degree

from_d:  from degree ";

%feature("docstring")  shogun::CWDSVMOcas::get_degree "

get degree

degree ";

%feature("docstring")  shogun::CWDSVMOcas::classify "

classify all examples

resulting labels ";

%feature("docstring")  shogun::CWDSVMOcas::classify "

classify objects

Parameters:
-----------

data:  (test) data to be classified

classified labels ";

%feature("docstring")  shogun::CWDSVMOcas::classify_example "

classify one example

Parameters:
-----------

num:  number of example to classify

classified result ";

%feature("docstring")  shogun::CWDSVMOcas::set_normalization_const "

set normalization const ";

%feature("docstring")  shogun::CWDSVMOcas::get_normalization_const "

get normalization const

normalization const ";


// File: structshogun_1_1data.xml
%feature("docstring") shogun::data "

Data: Input examples are stored in sparse (Compressed Row Storage)
format

C++ includes: ssl.h ";


// File: classshogun_1_1Delta.xml
%feature("docstring") shogun::Delta "

used in line search

C++ includes: ssl.h ";

%feature("docstring")  shogun::Delta::Delta "

default constructor ";


// File: structshogun_1_1larank__kcache__s.xml
%feature("docstring") shogun::larank_kcache_s "";


// File: classshogun_1_1LaRankOutput.xml
%feature("docstring") shogun::LaRankOutput "";

%feature("docstring")  shogun::LaRankOutput::LaRankOutput "";

%feature("docstring")  shogun::LaRankOutput::~LaRankOutput "";

%feature("docstring")  shogun::LaRankOutput::initialize "";

%feature("docstring")  shogun::LaRankOutput::destroy "";

%feature("docstring")  shogun::LaRankOutput::computeScore "";

%feature("docstring")  shogun::LaRankOutput::computeGradient "";

%feature("docstring")  shogun::LaRankOutput::update "";

%feature("docstring")  shogun::LaRankOutput::set_kernel_buddy "";

%feature("docstring")  shogun::LaRankOutput::cleanup "";

%feature("docstring")  shogun::LaRankOutput::getKernel "";

%feature("docstring")  shogun::LaRankOutput::get_l "";

%feature("docstring")  shogun::LaRankOutput::getW2 "";

%feature("docstring")  shogun::LaRankOutput::getKii "";

%feature("docstring")  shogun::LaRankOutput::getBeta "";

%feature("docstring")  shogun::LaRankOutput::getBetas "";

%feature("docstring")  shogun::LaRankOutput::getGradient "";

%feature("docstring")  shogun::LaRankOutput::isSupportVector "";

%feature("docstring")  shogun::LaRankOutput::getSV "";


// File: classshogun_1_1LaRankPattern.xml
%feature("docstring") shogun::LaRankPattern "";

%feature("docstring")  shogun::LaRankPattern::LaRankPattern "";

%feature("docstring")  shogun::LaRankPattern::LaRankPattern "";

%feature("docstring")  shogun::LaRankPattern::exists "";

%feature("docstring")  shogun::LaRankPattern::clear "";


// File: classshogun_1_1LaRankPatterns.xml
%feature("docstring") shogun::LaRankPatterns "";

%feature("docstring")  shogun::LaRankPatterns::LaRankPatterns "";

%feature("docstring")  shogun::LaRankPatterns::~LaRankPatterns "";

%feature("docstring")  shogun::LaRankPatterns::insert "";

%feature("docstring")  shogun::LaRankPatterns::remove "";

%feature("docstring")  shogun::LaRankPatterns::empty "";

%feature("docstring")  shogun::LaRankPatterns::size "";

%feature("docstring")  shogun::LaRankPatterns::sample "";

%feature("docstring")  shogun::LaRankPatterns::getPatternRank "";

%feature("docstring")  shogun::LaRankPatterns::isPattern "";

%feature("docstring")  shogun::LaRankPatterns::getPattern "";

%feature("docstring")  shogun::LaRankPatterns::maxcount "";


// File: classshogun_1_1MKLMultiClassGLPK.xml
%feature("docstring") shogun::MKLMultiClassGLPK "

MKLMultiClassGLPK is a helper class for MKLMultiClass.

it solves the corresponding linear problem arising in SIP formulation
for MKL using glpk

C++ includes: MKLMultiClassGLPK.h ";

%feature("docstring")  shogun::MKLMultiClassGLPK::MKLMultiClassGLPK "

Class default Constructor ";

%feature("docstring")  shogun::MKLMultiClassGLPK::~MKLMultiClassGLPK "

Class default Destructor ";

%feature("docstring")  shogun::MKLMultiClassGLPK::setup "

initializes GLPK LP sover

Parameters:
-----------

numkernels2:  is the number of kernels ";

%feature("docstring")  shogun::MKLMultiClassGLPK::addconstraint "

adds a constraint to the LP arising in L1 MKL based on two parameters

Parameters:
-----------

normw2:  is the vector of $ \\\\|w_k \\\\|^2 $ for all kernels

sumofpositivealphas:  is a term depending on alphas, labels and
biases, see in the function float64_t getsumofsignfreealphas() from
MKLMultiClass.h, it depends on the formulation of the underlying
GMNPSVM. ";

%feature("docstring")  shogun::MKLMultiClassGLPK::computeweights "

computes MKL weights

Parameters:
-----------

weights2:  stores the new weights ";

%feature("docstring")  shogun::MKLMultiClassGLPK::get_name "

object name ";


// File: structshogun_1_1ocas__return__value__T.xml
%feature("docstring") shogun::ocas_return_value_T "

ocas return value

C++ includes: libocas.h ";


// File: structshogun_1_1options.xml
%feature("docstring") shogun::options "

various options user + internal optimisation

C++ includes: ssl.h ";


// File: structshogun_1_1CLaRank_1_1outputgradient__t.xml


// File: structshogun_1_1CLaRank_1_1process__return__t.xml


// File: classshogun_1_1QPproblem.xml
%feature("docstring") shogun::QPproblem "

class QProblem

C++ includes: gpdtsolve.h ";

%feature("docstring")  shogun::QPproblem::QPproblem "

constructor ";

%feature("docstring")  shogun::QPproblem::~QPproblem "";

%feature("docstring")  shogun::QPproblem::ReadSVMFile "

read SVM file

Parameters:
-----------

fInput:  input filename

an int ";

%feature("docstring")  shogun::QPproblem::ReadGPDTBinary "

read GPDT binary

Parameters:
-----------

fName:  input filename

an int ";

%feature("docstring")  shogun::QPproblem::Check2Class "

check if 2-class

an int ";

%feature("docstring")  shogun::QPproblem::Subproblem "

subproblem

Parameters:
-----------

ker:  problem kernel

len:  length

perm:  perm ";

%feature("docstring")  shogun::QPproblem::PrepMP "

PrepMP ";

%feature("docstring")  shogun::QPproblem::gpdtsolve "

solve gpdt

Parameters:
-----------

solution:

something floaty ";

%feature("docstring")  shogun::QPproblem::pgpdtsolve "

solve pgpdt

Parameters:
-----------

solution:

something floaty ";

%feature("docstring")  shogun::QPproblem::get_linadd_enabled "

check if lineadd is enabled

if lineadd is enabled ";

%feature("docstring")  shogun::QPproblem::get_name "

object name ";


// File: classshogun_1_1sKernel.xml
%feature("docstring") shogun::sKernel "

s kernel

C++ includes: gpdt.h ";

%feature("docstring")  shogun::sKernel::sKernel "

constructor

Parameters:
-----------

k:  kernel

ell:  ell ";

%feature("docstring")  shogun::sKernel::~sKernel "";

%feature("docstring")  shogun::sKernel::SetData "

set data

Parameters:
-----------

x_:  new x

ix_:  new ix

lx_:  new lx

ell:  new ell

dim:  dim ";

%feature("docstring")  shogun::sKernel::SetSubproblem "

set subproblem

Parameters:
-----------

ker:  kernel

len:  len

perm:  perm ";

%feature("docstring")  shogun::sKernel::Get "

get an item from the kernel

Parameters:
-----------

i:  index i

j:  index j

item from kernel at index i, j ";

%feature("docstring")  shogun::sKernel::Add "

add something

Parameters:
-----------

v:  v

j:  j

mul:  mul ";

%feature("docstring")  shogun::sKernel::Prod "

prod something

Parameters:
-----------

v:  v

j:  j

something floaty ";

%feature("docstring")  shogun::sKernel::get_kernel "

get kernel

kernel ";


// File: structshogun_1_1svm__model.xml
%feature("docstring") shogun::svm_model "

svm_model

C++ includes: SVM_libsvm.h ";


// File: structshogun_1_1svm__node.xml
%feature("docstring") shogun::svm_node "

SVM node

C++ includes: SVM_libsvm.h ";


// File: structshogun_1_1svm__parameter.xml
%feature("docstring") shogun::svm_parameter "

SVM parameter

C++ includes: SVM_libsvm.h ";


// File: structshogun_1_1svm__problem.xml
%feature("docstring") shogun::svm_problem "

SVM problem

C++ includes: SVM_libsvm.h ";

%feature("docstring")  shogun::svm_problem::svm_problem "

default constructor ";


// File: structshogun_1_1vector__double.xml
%feature("docstring") shogun::vector_double "

defines a vector of doubles

C++ includes: ssl.h ";


// File: structshogun_1_1vector__int.xml
%feature("docstring") shogun::vector_int "

defines a vector of ints for index subsets

C++ includes: ssl.h ";


// File: namespaceshogun.xml
%feature("docstring")  shogun::SplitParts "";

%feature("docstring")  shogun::SplitNum "";

%feature("docstring")  shogun::gpm_solver "";

%feature("docstring")  shogun::svm_ocas_solver "";

%feature("docstring")  shogun::pr_loqo "";

%feature("docstring")  shogun::qpssvm_solver "";

%feature("docstring")  shogun::initialize "";

%feature("docstring")  shogun::initialize "";

%feature("docstring")  shogun::GetLabeledData "";

%feature("docstring")  shogun::norm_square "";

%feature("docstring")  shogun::ssl_train "";

%feature("docstring")  shogun::CGLS "";

%feature("docstring")  shogun::L2_SVM_MFN "";

%feature("docstring")  shogun::line_search "";

%feature("docstring")  shogun::TSVM_MFN "";

%feature("docstring")  shogun::switch_labels "";

%feature("docstring")  shogun::DA_S3VM "";

%feature("docstring")  shogun::optimize_p "";

%feature("docstring")  shogun::optimize_w "";

%feature("docstring")  shogun::transductive_cost "";

%feature("docstring")  shogun::entropy "";

%feature("docstring")  shogun::KL "";

%feature("docstring")  shogun::svm_train "";

%feature("docstring")  shogun::svm_predict "";

%feature("docstring")  shogun::svm_destroy_model "";

%feature("docstring")  shogun::svm_check_parameter "";


// File: Classifier_8h.xml


// File: DistanceMachine_8h.xml


// File: KernelMachine_8h.xml


// File: KernelPerceptron_8h.xml


// File: KNN_8h.xml


// File: LDA_8h.xml


// File: LinearClassifier_8h.xml


// File: LPBoost_8h.xml


// File: LPM_8h.xml


// File: MKL_8h.xml


// File: MKLClassification_8h.xml


// File: MKLMultiClass_8h.xml


// File: MKLMultiClassGLPK_8h.xml


// File: MKLOneClass_8h.xml


// File: Perceptron_8h.xml


// File: PluginEstimate_8h.xml


// File: SubGradientLPM_8h.xml


// File: CPLEXSVM_8h.xml


// File: DomainAdaptationSVM_8h.xml


// File: gmnplib_8h.xml


// File: GMNPSVM_8h.xml


// File: gnpplib_8h.xml


// File: GNPPSVM_8h.xml


// File: GPBTSVM_8h.xml


// File: gpdt_8h.xml


// File: gpdtsolve_8h.xml


// File: gpm_8h.xml


// File: LaRank_8h.xml


// File: LibLinear_8h.xml


// File: libocas_8h.xml


// File: libocas__common_8h.xml


// File: LibSVM_8h.xml


// File: LibSVMMultiClass_8h.xml


// File: LibSVMOneClass_8h.xml


// File: MPDSVM_8h.xml


// File: MultiClassSVM_8h.xml


// File: pr__loqo_8h.xml


// File: qpbsvmlib_8h.xml


// File: qpssvmlib_8h.xml


// File: ScatterSVM_8h.xml


// File: ssl_8h.xml


// File: SubGradientSVM_8h.xml


// File: SVM_8h.xml


// File: SVM__libsvm_8h.xml


// File: SVM__light_8h.xml


// File: SVM__linear_8h.xml


// File: SVMLin_8h.xml


// File: SVMOcas_8h.xml


// File: SVMSGD_8h.xml


// File: Tron_8h.xml


// File: WDSVMOcas_8h.xml


// File: dir_0698d319890eaae8d621b18d62340105.xml


// File: dir_428a4cdd6f0859610feb34991eb0c136.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml


// File: dir_d4453277307ee0fb87a1e3676262504e.xml

