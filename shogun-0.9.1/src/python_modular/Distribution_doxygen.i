
// File: index.xml

// File: classshogun_1_1CDistribution.xml
%feature("docstring") shogun::CDistribution "

Base class Distribution from which all methods implementing a
distribution are derived.

Distributions are based on some general feature object and have to
implement interfaces to

train() - for learning a distribution get_num_model_parameters() - for
the total number of model parameters get_log_model_parameter() - for
the n-th model parameter (logarithmic) get_log_derivative() - for the
partial derivative wrt. to the n-th model parameter
get_log_likelihood_example() - for the likelihood for the n-th example

This way methods building on CDistribution, might enumerate over all
possible model parameters and obtain the parameter vector and the
gradient. This is used to compute e.g. the TOP and Fisher Kernel (cf.
CPluginEstimate, CHistogramKernel, CTOPFeatures and CFKFeatures ).

C++ includes: Distribution.h ";

%feature("docstring")  shogun::CDistribution::CDistribution "

default constructor ";

%feature("docstring")  shogun::CDistribution::~CDistribution "";

%feature("docstring")  shogun::CDistribution::train "

learn distribution

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CDistribution::get_num_model_parameters
"

get number of parameters in model

abstract base method

number of parameters in model ";

%feature("docstring")
shogun::CDistribution::get_num_relevant_model_parameters "

get number of parameters in model that are relevant, i.e. >
ALMOST_NEG_INFTY

number of relevant model parameters ";

%feature("docstring")  shogun::CDistribution::get_log_model_parameter
"

get model parameter (logarithmic)

abstrac base method

model parameter (logarithmic) ";

%feature("docstring")  shogun::CDistribution::get_log_derivative "

get partial derivative of likelihood function (logarithmic)

abstract base method

Parameters:
-----------

num_param:  derivative against which param

num_example:  which example

derivative of likelihood (logarithmic) ";

%feature("docstring")
shogun::CDistribution::get_log_likelihood_example "

compute log likelihood for example

abstract base method

Parameters:
-----------

num_example:  which example

log likelihood for example ";

%feature("docstring")
shogun::CDistribution::get_log_likelihood_sample "

compute log likelihood for whole sample

log likelihood for whole sample ";

%feature("docstring")  shogun::CDistribution::get_log_likelihood "

compute log likelihood for each example

Parameters:
-----------

dst:  where likelihood will be stored

num:  where number of likelihoods will be stored ";

%feature("docstring")  shogun::CDistribution::get_model_parameter "

get model parameter

Parameters:
-----------

num_param:  which param

model parameter ";

%feature("docstring")  shogun::CDistribution::get_derivative "

get partial derivative of likelihood function

Parameters:
-----------

num_param:  partial derivative against which param

num_example:  which example

derivative of likelihood function ";

%feature("docstring")  shogun::CDistribution::get_likelihood_example "

compute likelihood for example

Parameters:
-----------

num_example:  which example

likelihood for example ";

%feature("docstring")  shogun::CDistribution::set_features "

set feature vectors

Parameters:
-----------

f:  new feature vectors ";

%feature("docstring")  shogun::CDistribution::get_features "

get feature vectors

feature vectors ";

%feature("docstring")  shogun::CDistribution::set_pseudo_count "

set pseudo count

Parameters:
-----------

pseudo:  new pseudo count ";

%feature("docstring")  shogun::CDistribution::get_pseudo_count "

get pseudo count

pseudo count ";


// File: classshogun_1_1CGHMM.xml
%feature("docstring") shogun::CGHMM "

class GHMM - this class is non-functional and was meant to implement a
Generalize Hidden Markov Model (aka Semi Hidden Markov HMM).

C++ includes: GHMM.h ";

%feature("docstring")  shogun::CGHMM::CGHMM "

default constructor ";

%feature("docstring")  shogun::CGHMM::~CGHMM "";

%feature("docstring")  shogun::CGHMM::train "

learn distribution

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CGHMM::get_num_model_parameters "

get number of model parameters

number of model parameters ";

%feature("docstring")  shogun::CGHMM::get_log_model_parameter "

get logarithm of given model parameter

Parameters:
-----------

param_num:  which param

logarithm of given model parameter ";

%feature("docstring")  shogun::CGHMM::get_log_derivative "

get logarithm of one example's derivative's likelihood

Parameters:
-----------

param_num:  which example's param

num_example:  which example

logarithm of example's derivative's likelihood ";

%feature("docstring")  shogun::CGHMM::get_log_likelihood_example "

get logarithm of one example's likelihood

Parameters:
-----------

num_example:  which example

logarithm of example's likelihood ";


// File: classshogun_1_1CHistogram.xml
%feature("docstring") shogun::CHistogram "

Class Histogram computes a histogram over all 16bit unsigned integers
in the features.

Values in histogram are absolute counts (logarithmic)

C++ includes: Histogram.h ";

%feature("docstring")  shogun::CHistogram::CHistogram "

default constructor ";

%feature("docstring")  shogun::CHistogram::CHistogram "

constructor

Parameters:
-----------

f:  histogram's features ";

%feature("docstring")  shogun::CHistogram::~CHistogram "";

%feature("docstring")  shogun::CHistogram::train "

learn distribution

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CHistogram::get_num_model_parameters "

get number of model parameters

number of model parameters ";

%feature("docstring")  shogun::CHistogram::get_log_model_parameter "

get logarithm of given model parameter

Parameters:
-----------

num_param:  which param

logarithm of given model parameter ";

%feature("docstring")  shogun::CHistogram::get_log_derivative "

get logarithm of one example's derivative's likelihood

Parameters:
-----------

num_param:  which example's param

num_example:  which example

logarithm of example's derivative's likelihood ";

%feature("docstring")  shogun::CHistogram::get_log_likelihood_example
"

get logarithm of one example's likelihood

Parameters:
-----------

num_example:  which example

logarithm of example's likelihood ";

%feature("docstring")  shogun::CHistogram::set_histogram "

set histogram

Parameters:
-----------

src:  new histogram

num:  number of values in histogram ";

%feature("docstring")  shogun::CHistogram::get_histogram "

get histogram

Parameters:
-----------

dst:  where the histogram will be stored

num:  where number of values in histogram will be stored ";

%feature("docstring")  shogun::CHistogram::get_name "

object name ";


// File: classshogun_1_1CHMM.xml
%feature("docstring") shogun::CHMM "

Hidden Markov Model.

Structure and Function collection. This Class implements a Hidden
Markov Model. For a tutorial on HMMs see Rabiner et.al A Tutorial on
Hidden Markov Models and Selected Applications in Speech Recognition,
1989

Several functions for tasks such as training,reading/writing models,
reading observations, calculation of derivatives are supplied.

C++ includes: HMM.h ";

/*  model specific variables.  */

/*  these are p,q,a,b,N,M etc

*/

/*  Constructor/Destructor and helper function  */

/*  Train definitions. Encapsulates Modelparameters that are
constant/shall be learned. Consists of structures and access functions
for learning only defined transitions and constants.

*/

%feature("docstring")  shogun::CHMM::CHMM "

Constructor

Parameters:
-----------

N:  number of states

M:  number of emissions

model:  model which holds definitions of states to be learned + consts

PSEUDO:  Pseudo Value ";

%feature("docstring")  shogun::CHMM::CHMM "";

%feature("docstring")  shogun::CHMM::CHMM "";

%feature("docstring")  shogun::CHMM::CHMM "";

%feature("docstring")  shogun::CHMM::CHMM "

Constructor - Initialization from model file.

Parameters:
-----------

model_file:  Filehandle to a hmm model file (*.mod)

PSEUDO:  Pseudo Value ";

%feature("docstring")  shogun::CHMM::CHMM "

Constructor - Clone model h. ";

%feature("docstring")  shogun::CHMM::~CHMM "

Destructor - Cleanup. ";

%feature("docstring")  shogun::CHMM::train "

learn distribution

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CHMM::get_num_model_parameters "

get number of parameters in model

abstract base method

number of parameters in model ";

%feature("docstring")  shogun::CHMM::get_log_model_parameter "

get model parameter (logarithmic)

abstrac base method

model parameter (logarithmic) ";

%feature("docstring")  shogun::CHMM::get_log_derivative "

get partial derivative of likelihood function (logarithmic)

abstract base method

Parameters:
-----------

num_param:  derivative against which param

num_example:  which example

derivative of likelihood (logarithmic) ";

%feature("docstring")  shogun::CHMM::get_log_likelihood_example "

compute log likelihood for example

abstract base method

Parameters:
-----------

num_example:  which example

log likelihood for example ";

%feature("docstring")  shogun::CHMM::initialize "

initialization function - gets called by constructors.

Parameters:
-----------

model:  model which holds definitions of states to be learned + consts

PSEUDO:  Pseudo Value

model_file:  Filehandle to a hmm model file (*.mod) ";

/*  probability functions.  */

/*  forward/backward/viterbi algorithm

*/

%feature("docstring")  shogun::CHMM::forward_comp "

forward algorithm. calculates Pr[O_0,O_1, ..., O_t, q_time=S_i|
lambda] for 0<= time <= T-1 Pr[O|lambda] for time > T

Parameters:
-----------

time:  t

state:  i

dimension:  dimension of observation (observations are a matrix, where
a row stands for one dimension i.e. 0_0,O_1,...,O_{T-1} ";

%feature("docstring")  shogun::CHMM::forward_comp_old "";

%feature("docstring")  shogun::CHMM::backward_comp "

backward algorithm. calculates Pr[O_t+1,O_t+2, ..., O_T-1| q_time=S_i,
lambda] for 0<= time <= T-1 Pr[O|lambda] for time >= T

Parameters:
-----------

time:  t

state:  i

dimension:  dimension of observation (observations are a matrix, where
a row stands for one dimension i.e. 0_0,O_1,...,O_{T-1} ";

%feature("docstring")  shogun::CHMM::backward_comp_old "";

%feature("docstring")  shogun::CHMM::best_path "

calculates probability of best state sequence s_0,...,s_T-1 AND path
itself using viterbi algorithm. The path can be found in the array
PATH(dimension)[0..T-1] afterwards

Parameters:
-----------

dimension:  dimension of observation for which the most probable path
is calculated (observations are a matrix, where a row stands for one
dimension i.e. 0_0,O_1,...,O_{T-1} ";

%feature("docstring")  shogun::CHMM::get_best_path_state "";

%feature("docstring")  shogun::CHMM::model_probability_comp "

calculates probability that observations were generated by the model
using forward algorithm. ";

%feature("docstring")  shogun::CHMM::model_probability "

inline proxy for model probability. ";

%feature("docstring")  shogun::CHMM::linear_model_probability "

calculates likelihood for linear model on observations in MEMORY

Parameters:
-----------

dimension:  dimension for which probability is calculated

model probability ";

/*  convergence criteria  */

/*

*/

%feature("docstring")  shogun::CHMM::set_iterations "";

%feature("docstring")  shogun::CHMM::get_iterations "";

%feature("docstring")  shogun::CHMM::set_epsilon "";

%feature("docstring")  shogun::CHMM::get_epsilon "";

%feature("docstring")  shogun::CHMM::baum_welch_viterbi_train "

interface for e.g. GUIHMM to run BaumWelch or Viterbi training

Parameters:
-----------

type:  type of BaumWelch/Viterbi training ";

/*  model training  */

/*

*/

%feature("docstring")  shogun::CHMM::estimate_model_baum_welch "

uses baum-welch-algorithm to train a fully connected HMM.

Parameters:
-----------

train:  model from which the new model is estimated ";

%feature("docstring")  shogun::CHMM::estimate_model_baum_welch_trans "";

%feature("docstring")  shogun::CHMM::estimate_model_baum_welch_old "";

%feature("docstring")  shogun::CHMM::estimate_model_baum_welch_defined
"

uses baum-welch-algorithm to train the defined transitions etc.

Parameters:
-----------

train:  model from which the new model is estimated ";

%feature("docstring")  shogun::CHMM::estimate_model_viterbi "

uses viterbi training to train a fully connected HMM

Parameters:
-----------

train:  model from which the new model is estimated ";

%feature("docstring")  shogun::CHMM::estimate_model_viterbi_defined "

uses viterbi training to train the defined transitions etc.

Parameters:
-----------

train:  model from which the new model is estimated ";

/*  output functions.  */

/*

*/

%feature("docstring")  shogun::CHMM::output_model "

prints the model parameters on screen.

Parameters:
-----------

verbose:  when false only the model probability will be printed when
true the whole model will be printed additionally ";

%feature("docstring")  shogun::CHMM::output_model_defined "

performs output_model only for the defined transitions etc ";

/*  model helper functions.  */

/*

*/

%feature("docstring")  shogun::CHMM::normalize "

normalize the model to satisfy stochasticity ";

%feature("docstring")  shogun::CHMM::add_states "

increases the number of states by num_states the new a/b/p/q values
are given the value default_val where 0<=default_val<=1 ";

%feature("docstring")  shogun::CHMM::append_model "

appends the append_model to the current hmm, i.e. two extra states are
created. one is the end state of the current hmm with outputs cur_out
(of size M) and the other state is the start state of the
append_model. transition probability from state 1 to states 1 is 1 ";

%feature("docstring")  shogun::CHMM::append_model "

appends the append_model to the current hmm, here no extra states are
created. former q_i are multiplied by q_ji to give the a_ij from the
current hmm to the append_model ";

%feature("docstring")  shogun::CHMM::chop "

set any model parameter with probability smaller than value to ZERO ";

%feature("docstring")  shogun::CHMM::convert_to_log "

convert model to log probabilities ";

%feature("docstring")  shogun::CHMM::init_model_random "

init model with random values ";

%feature("docstring")  shogun::CHMM::init_model_defined "

init model according to const_x, learn_x. first model is initialized
with 0 for all parameters then parameters in learn_x are initialized
with random values finally const_x parameters are set and model is
normalized. ";

%feature("docstring")  shogun::CHMM::clear_model "

initializes model with log(PSEUDO) ";

%feature("docstring")  shogun::CHMM::clear_model_defined "

initializes only parameters in learn_x with log(PSEUDO) ";

%feature("docstring")  shogun::CHMM::copy_model "

copies the the modelparameters from l ";

%feature("docstring")  shogun::CHMM::invalidate_model "

invalidates all caches. this function has to be called when direct
changes to the model have been made. this is necessary for the
forward/backward/viterbi algorithms to not work with old tables ";

%feature("docstring")  shogun::CHMM::get_status "

get status true if everything is ok, else false ";

%feature("docstring")  shogun::CHMM::get_pseudo "

returns current pseudo value ";

%feature("docstring")  shogun::CHMM::set_pseudo "

sets current pseudo value ";

%feature("docstring")  shogun::CHMM::set_observations "

observation functions set/get observation matrix set new observations
sets the observation pointer and initializes observation-dependent
caches if hmm is given, then the caches of the model hmm are used ";

%feature("docstring")  shogun::CHMM::set_observation_nocache "

set new observations only set the observation pointer and drop caches
if there were any ";

%feature("docstring")  shogun::CHMM::get_observations "

return observation pointer ";

/*  load/save functions.  */

/*  for observations/model/traindefinitions

*/

%feature("docstring")  shogun::CHMM::load_definitions "

read definitions file (learn_x,const_x) used for training. -format
specs: definition_file (train.def) % HMM-TRAIN - specification %
learn_a - elements in state_transition_matrix to be learned % learn_b
- elements in oberservation_per_state_matrix to be learned % note:
each line stands for % state, observation(0),
observation(1)...observation(NOW) % learn_p - elements in initial
distribution to be learned % learn_q - elements in the end-state
distribution to be learned % % const_x - specifies initial values of
elements % rest is assumed to be 0.0 % % NOTE: IMPLICIT DEFINES: %
define A 0 % define C 1 % define G 2 % define T 3

learn_a=[ [int32_t,int32_t]; [int32_t,int32_t]; [int32_t,int32_t];
........ [int32_t,int32_t]; [-1,-1]; ];

learn_b=[ [int32_t,int32_t,int32_t,...,int32_t];
[int32_t,int32_t,int32_t,...,int32_t];
[int32_t,int32_t,int32_t,...,int32_t]; ........
[int32_t,int32_t,int32_t,...,int32_t]; [-1,-1]; ];

learn_p= [ int32_t, ... , int32_t, -1 ];

learn_q= [ int32_t, ... , int32_t, -1 ];

const_a=[ [int32_t,int32_t,float64_t]; [int32_t,int32_t,float64_t];
[int32_t,int32_t,float64_t]; ........ [int32_t,int32_t,float64_t];
[-1,-1,-1]; ];

const_b=[ [int32_t,int32_t,int32_t,...,int32_t,float64_t];
[int32_t,int32_t,int32_t,...,int32_t,float64_t];
[int32_t,int32_t,int32_t,...,int32_t,<DOUBLE]; ........
[int32_t,int32_t,int32_t,...,int32_t,float64_t]; [-1,-1,-1]; ];

const_p[]=[ [int32_t, float64_t], ... , [int32_t,float64_t], [-1,-1]
]; const_q[]=[ [int32_t, float64_t], ... , [int32_t,float64_t],
[-1,-1] ];

Parameters:
-----------

file:  filehandle to definitions file

verbose:  true for verbose messages

initialize:  true to initialize to underlying HMM ";

%feature("docstring")  shogun::CHMM::load_model "

read model from file. -format specs: model_file (model.hmm) % HMM -
specification % N - number of states % M - number of
observation_tokens % a is state_transition_matrix % size(a)= [N,N] % %
b is observation_per_state_matrix % size(b)= [N,M] % % p is initial
distribution % size(p)= [1, N]

N=int32_t; M=int32_t;

p=[float64_t,float64_t...float64_t];
q=[float64_t,float64_t...float64_t];

a=[ [float64_t,float64_t...float64_t];
[float64_t,float64_t...float64_t]; [float64_t,float64_t...float64_t];
[float64_t,float64_t...float64_t]; [float64_t,float64_t...float64_t];
];

b=[ [float64_t,float64_t...float64_t];
[float64_t,float64_t...float64_t]; [float64_t,float64_t...float64_t];
[float64_t,float64_t...float64_t]; [float64_t,float64_t...float64_t];
];

Parameters:
-----------

file:  filehandle to model file ";

%feature("docstring")  shogun::CHMM::save_model "

save model to file.

Parameters:
-----------

file:  filehandle to model file ";

%feature("docstring")  shogun::CHMM::save_model_derivatives "

save model derivatives to file in ascii format.

Parameters:
-----------

file:  filehandle ";

%feature("docstring")  shogun::CHMM::save_model_derivatives_bin "

save model derivatives to file in binary format.

Parameters:
-----------

file:  filehandle ";

%feature("docstring")  shogun::CHMM::save_model_bin "

save model in binary format.

Parameters:
-----------

file:  filehandle ";

%feature("docstring")  shogun::CHMM::check_model_derivatives "

numerically check whether derivates were calculated right ";

%feature("docstring")  shogun::CHMM::check_model_derivatives_combined
"";

%feature("docstring")  shogun::CHMM::get_path "

get viterbi path and path probability

Parameters:
-----------

dim:  dimension for which to obtain best path

prob:  likelihood of path

viterbi path ";

%feature("docstring")  shogun::CHMM::save_path "

save viterbi path in ascii format

Parameters:
-----------

file:  filehandle ";

%feature("docstring")  shogun::CHMM::save_path_derivatives "

save viterbi path in ascii format

Parameters:
-----------

file:  filehandle ";

%feature("docstring")  shogun::CHMM::save_path_derivatives_bin "

save viterbi path in binary format

Parameters:
-----------

file:  filehandle ";

%feature("docstring")  shogun::CHMM::save_likelihood_bin "

save model probability in binary format

Parameters:
-----------

file:  filehandle ";

%feature("docstring")  shogun::CHMM::save_likelihood "

save model probability in ascii format

Parameters:
-----------

file:  filehandle ";

/*  access functions for model parameters  */

/*  for all the arrays a,b,p,q,A,B,psi and scalar model parameters
like N,M

*/

%feature("docstring")  shogun::CHMM::get_N "

access function for number of states N ";

%feature("docstring")  shogun::CHMM::get_M "

access function for number of observations M ";

%feature("docstring")  shogun::CHMM::set_q "

access function for probability of end states

Parameters:
-----------

offset:  index 0...N-1

value:  value to be set ";

%feature("docstring")  shogun::CHMM::set_p "

access function for probability of first state

Parameters:
-----------

offset:  index 0...N-1

value:  value to be set ";

%feature("docstring")  shogun::CHMM::set_A "

access function for matrix A

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...N-1

value:  value to be set ";

%feature("docstring")  shogun::CHMM::set_a "

access function for matrix a

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...N-1

value:  value to be set ";

%feature("docstring")  shogun::CHMM::set_B "

access function for matrix B

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...M-1

value:  value to be set ";

%feature("docstring")  shogun::CHMM::set_b "

access function for matrix b

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...M-1

value:  value to be set ";

%feature("docstring")  shogun::CHMM::set_psi "

access function for backtracking table psi

Parameters:
-----------

time:  time 0...T-1

state:  state 0...N-1

value:  value to be set

dimension:  dimension of observations 0...DIMENSION-1 ";

%feature("docstring")  shogun::CHMM::get_q "

access function for probability of end states

Parameters:
-----------

offset:  index 0...N-1

value at offset ";

%feature("docstring")  shogun::CHMM::get_p "

access function for probability of initial states

Parameters:
-----------

offset:  index 0...N-1

value at offset ";

%feature("docstring")  shogun::CHMM::get_A "

access function for matrix A

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...N-1

value at position line colum ";

%feature("docstring")  shogun::CHMM::get_a "

access function for matrix a

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...N-1

value at position line colum ";

%feature("docstring")  shogun::CHMM::get_B "

access function for matrix B

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...M-1

value at position line colum ";

%feature("docstring")  shogun::CHMM::get_b "

access function for matrix b

Parameters:
-----------

line_:  row in matrix 0...N-1

column:  column in matrix 0...M-1

value at position line colum ";

%feature("docstring")  shogun::CHMM::get_psi "

access function for backtracking table psi

Parameters:
-----------

time:  time 0...T-1

state:  state 0...N-1

dimension:  dimension of observations 0...DIMENSION-1

state at specified time and position ";

/*  functions for observations  */

/*  management and access functions for observation matrix

*/

%feature("docstring")  shogun::CHMM::state_probability "

calculates probability of being in state i at time t for dimension ";

%feature("docstring")  shogun::CHMM::transition_probability "

calculates probability of being in state i at time t and state j at
time t+1 for dimension ";

/*  derivatives of model probabilities.  */

/*  computes log dp(lambda)/d lambda_i

Parameters:
-----------

dimension:  dimension for that derivatives are calculated

i:  j:  parameter specific

*/

%feature("docstring")  shogun::CHMM::linear_model_derivative "

computes log dp(lambda)/d b_ij for linear model ";

%feature("docstring")  shogun::CHMM::model_derivative_p "

computes log dp(lambda)/d p_i. backward path downto time 0 multiplied
by observing first symbol in path at state i ";

%feature("docstring")  shogun::CHMM::model_derivative_q "

computes log dp(lambda)/d q_i. forward path upto time T-1 ";

%feature("docstring")  shogun::CHMM::model_derivative_a "

computes log dp(lambda)/d a_ij. ";

%feature("docstring")  shogun::CHMM::model_derivative_b "

computes log dp(lambda)/d b_ij. ";

/*  derivatives of path probabilities.  */

/*  computes d log p(lambda,best_path)/d lambda_i

Parameters:
-----------

dimension:  dimension for that derivatives are calculated

i:  j:  parameter specific

*/

%feature("docstring")  shogun::CHMM::path_derivative_p "

computes d log p(lambda,best_path)/d p_i ";

%feature("docstring")  shogun::CHMM::path_derivative_q "

computes d log p(lambda,best_path)/d q_i ";

%feature("docstring")  shogun::CHMM::path_derivative_a "

computes d log p(lambda,best_path)/d a_ij ";

%feature("docstring")  shogun::CHMM::path_derivative_b "

computes d log p(lambda,best_path)/d b_ij ";

/*  input helper functions.  */

/*  for reading model/definition/observation files

*/

%feature("docstring")  shogun::CHMM::alloc_state_dependend_arrays "

allocates memory that depends on N ";

%feature("docstring")  shogun::CHMM::free_state_dependend_arrays "

free memory that depends on N ";

%feature("docstring")  shogun::CHMM::linear_train "

estimates linear model from observations. ";

%feature("docstring")  shogun::CHMM::permutation_entropy "

compute permutation entropy ";

%feature("docstring")  shogun::CHMM::get_name "

object name ";


// File: classshogun_1_1CLinearHMM.xml
%feature("docstring") shogun::CLinearHMM "

The class LinearHMM is for learning Higher Order Markov chains.

While learning the parameters ${\\\\bf \\\\theta}$ in

\\\\begin{eqnarray*} P({\\\\bf x}|{\\\\bf \\\\theta}^\\\\pm)&=&P(x_1,
\\\\ldots, x_N|{\\\\bf \\\\theta}^\\\\pm)\\\\\\\\
&=&P(x_1,\\\\ldots,x_{d}|{\\\\bf \\\\theta}^\\\\pm)\\\\prod_{i=d+1}^N
P(x_i|x_{i-1},\\\\ldots,x_{i-d},{\\\\bf \\\\theta}^\\\\pm)
\\\\end{eqnarray*}

are determined.

A more detailed description can be found in

Durbin et.al, Biological Sequence Analysis -Probabilistic Models of
Proteins and Nucleic Acids, 1998

C++ includes: LinearHMM.h ";

%feature("docstring")  shogun::CLinearHMM::CLinearHMM "

constructor

Parameters:
-----------

f:  features to use ";

%feature("docstring")  shogun::CLinearHMM::CLinearHMM "

constructor

Parameters:
-----------

p_num_features:  number of features

p_num_symbols:  number of symbols in features ";

%feature("docstring")  shogun::CLinearHMM::~CLinearHMM "";

%feature("docstring")  shogun::CLinearHMM::train "

estimate LinearHMM distribution

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CLinearHMM::train "

alternative train distribution

Parameters:
-----------

indizes:  indices

num_indizes:  number of indices

pseudo_count:  pseudo count

if training was successful ";

%feature("docstring")  shogun::CLinearHMM::get_log_likelihood_example
"

get logarithm of one example's likelihood

Parameters:
-----------

vector:  the example

len:  length of vector

logarithm of likelihood ";

%feature("docstring")  shogun::CLinearHMM::get_likelihood_example "

get one example's likelihood

Parameters:
-----------

vector:  the example

len:  length of vector

likelihood ";

%feature("docstring")  shogun::CLinearHMM::get_log_likelihood_example
"

get logarithm of one example's likelihood

Parameters:
-----------

num_example:  which example

logarithm of example's likelihood ";

%feature("docstring")  shogun::CLinearHMM::get_log_derivative "

get logarithm of one example's derivative's likelihood

Parameters:
-----------

num_param:  which example's param

num_example:  which example

logarithm of example's derivative ";

%feature("docstring")  shogun::CLinearHMM::get_log_derivative_obsolete
"

obsolete get logarithm of one example's derivative's likelihood

Parameters:
-----------

obs:  observation

pos:  position ";

%feature("docstring")  shogun::CLinearHMM::get_derivative_obsolete "

obsolete get one example's derivative

Parameters:
-----------

vector:  vector

len:  length

pos:  position ";

%feature("docstring")  shogun::CLinearHMM::get_sequence_length "

get sequence length of each example

sequence length of each example ";

%feature("docstring")  shogun::CLinearHMM::get_num_symbols "

get number of symbols in examples

number of symbols in examples ";

%feature("docstring")  shogun::CLinearHMM::get_num_model_parameters "

get number of model parameters

number of model parameters ";

%feature("docstring")
shogun::CLinearHMM::get_positional_log_parameter "

get positional log parameter

Parameters:
-----------

obs:  observation

position:  position

positional log parameter ";

%feature("docstring")  shogun::CLinearHMM::get_log_model_parameter "

get logarithm of given model parameter

Parameters:
-----------

num_param:  which param

logarithm of given model parameter ";

%feature("docstring")  shogun::CLinearHMM::get_log_transition_probs "

get logarithm of all transition probs

Parameters:
-----------

dst:  where logarithm of transition probs will be stored

num:  where number of logarithm of transition probs will be stored ";

%feature("docstring")  shogun::CLinearHMM::set_log_transition_probs "

set logarithm of all transition probs

Parameters:
-----------

src:  new logarithms of transition probs

num:  number of logarithms of transition probs

if setting was successful ";

%feature("docstring")  shogun::CLinearHMM::get_transition_probs "

get all transition probs

Parameters:
-----------

dst:  where transition probs will be stored

num:  where number of transition probs will be stored ";

%feature("docstring")  shogun::CLinearHMM::set_transition_probs "

set all transition probs

Parameters:
-----------

src:  new transition probs

num:  number of transition probs

if setting was successful ";

%feature("docstring")  shogun::CLinearHMM::get_name "

object name ";


// File: classshogun_1_1CModel.xml
%feature("docstring") shogun::CModel "

class Model

C++ includes: HMM.h ";

/*  learn arrays.  */

/*  Everything that is to be learned is enumerated here. All values
will be inititialized with random values and normalized to satisfy
stochasticity.

*/

/*  constant arrays.  */

/*  These arrays hold constant fields. All values that are not
constant and will not be learned are initialized with 0.

*/

/*  read access functions.  */

/*  For learn arrays and const arrays

*/

%feature("docstring")  shogun::CModel::get_learn_a "

get entry out of learn_a matrix ";

%feature("docstring")  shogun::CModel::get_learn_b "

get entry out of learn_b matrix ";

%feature("docstring")  shogun::CModel::get_learn_p "

get entry out of learn_p vector ";

%feature("docstring")  shogun::CModel::get_learn_q "

get entry out of learn_q vector ";

%feature("docstring")  shogun::CModel::get_const_a "

get entry out of const_a matrix ";

%feature("docstring")  shogun::CModel::get_const_b "

get entry out of const_b matrix ";

%feature("docstring")  shogun::CModel::get_const_p "

get entry out of const_p vector ";

%feature("docstring")  shogun::CModel::get_const_q "

get entry out of const_q vector ";

%feature("docstring")  shogun::CModel::get_const_a_val "

get value out of const_a_val vector ";

%feature("docstring")  shogun::CModel::get_const_b_val "

get value out of const_b_val vector ";

%feature("docstring")  shogun::CModel::get_const_p_val "

get value out of const_p_val vector ";

%feature("docstring")  shogun::CModel::get_const_q_val "

get value out of const_q_val vector ";

/*  write access functions  */

/*  For learn and const arrays

*/

%feature("docstring")  shogun::CModel::set_learn_a "

set value in learn_a matrix ";

%feature("docstring")  shogun::CModel::set_learn_b "

set value in learn_b matrix ";

%feature("docstring")  shogun::CModel::set_learn_p "

set value in learn_p vector ";

%feature("docstring")  shogun::CModel::set_learn_q "

set value in learn_q vector ";

%feature("docstring")  shogun::CModel::set_const_a "

set value in const_a matrix ";

%feature("docstring")  shogun::CModel::set_const_b "

set value in const_b matrix ";

%feature("docstring")  shogun::CModel::set_const_p "

set value in const_p vector ";

%feature("docstring")  shogun::CModel::set_const_q "

set value in const_q vector ";

%feature("docstring")  shogun::CModel::set_const_a_val "

set value in const_a_val vector ";

%feature("docstring")  shogun::CModel::set_const_b_val "

set value in const_b_val vector ";

%feature("docstring")  shogun::CModel::set_const_p_val "

set value in const_p_val vector ";

%feature("docstring")  shogun::CModel::set_const_q_val "

set value in const_q_val vector ";

%feature("docstring")  shogun::CModel::CModel "

Constructor - initializes all variables/structures. ";

%feature("docstring")  shogun::CModel::~CModel "

Destructor - cleans up. ";

%feature("docstring")  shogun::CModel::sort_learn_a "

sorts learn_a matrix ";

%feature("docstring")  shogun::CModel::sort_learn_b "

sorts learn_b matrix ";


// File: structshogun_1_1T__ALPHA__BETA.xml
%feature("docstring") shogun::T_ALPHA_BETA "

type for alpha/beta table

C++ includes: HMM.h ";


// File: namespaceshogun.xml
/*  HMM specific types  */

/*

*/


// File: Distribution_8h.xml


// File: GHMM_8h.xml


// File: Histogram_8h.xml


// File: HMM_8h.xml


// File: LinearHMM_8h.xml


// File: dir_42ed0b19a09f15c83ba978a8ae6e74d1.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml

