
// File: index.xml

// File: classshogun_1_1CLibSVR.xml
%feature("docstring") shogun::CLibSVR "

Class LibSVR, performs support vector regression using LibSVM.

The SVR solution can be expressed as \\\\[ f({\\\\bf
x})=\\\\sum_{i=1}^{N} \\\\alpha_i k({\\\\bf x}, {\\\\bf x_i})+b \\\\]

where $\\\\alpha$ and $b$ are determined in training, i.e. using a
pre-specified kernel, a given tube-epsilon for the epsilon insensitive
loss, the follwoing quadratic problem is minimized (using sequential
minimal decomposition (SMO))

\\\\begin{eqnarray*} \\\\max_{{\\\\bf \\\\alpha},{\\\\bf \\\\alpha}^*}
&-\\\\frac{1}{2}\\\\sum_{i,j=1}^N(\\\\alpha_i-\\\\alpha_i^*)(\\\\alpha_j-\\\\alpha_j^*){\\\\bf
x}_i^T {\\\\bf x}_j
-\\\\sum_{i=1}^N(\\\\alpha_i+\\\\alpha_i^*)\\\\epsilon -
\\\\sum_{i=1}^N(\\\\alpha_i-\\\\alpha_i^*)y_i\\\\\\\\ \\\\mbox{wrt}:&
{\\\\bf \\\\alpha},{\\\\bf \\\\alpha}^*\\\\in{\\\\bf R}^N\\\\\\\\
\\\\mbox{s.t.}:& 0\\\\leq \\\\alpha_i,\\\\alpha_i^*\\\\leq C,\\\\,
\\\\forall i=1\\\\dots N\\\\\\\\
&\\\\sum_{i=1}^N(\\\\alpha_i-\\\\alpha_i^*)y_i=0 \\\\end{eqnarray*}

Note that the SV regression problem is reduced to the standard SV
classification problem by introducing artificial labels $-y_i$ which
leads to the epsilon insensitive loss constraints *
\\\\begin{eqnarray*} {\\\\bf w}^T{\\\\bf x}_i+b-c_i-\\\\xi_i\\\\leq
0,&\\\\, \\\\forall i=1\\\\dots N\\\\\\\\ -{\\\\bf w}^T{\\\\bf
x}_i-b-c_i^*-\\\\xi_i^*\\\\leq 0,&\\\\, \\\\forall i=1\\\\dots N
\\\\end{eqnarray*} with $c_i=y_i+ \\\\epsilon$ and $c_i^*=-y_i+
\\\\epsilon$

C++ includes: LibSVR.h ";

%feature("docstring")  shogun::CLibSVR::CLibSVR "

default constructor ";

%feature("docstring")  shogun::CLibSVR::CLibSVR "

constructor

Parameters:
-----------

C:  constant C

epsilon:  epsilon

k:  kernel

lab:  labels ";

%feature("docstring")  shogun::CLibSVR::~CLibSVR "";

%feature("docstring")  shogun::CLibSVR::train "

train regression

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based regressor are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CLibSVR::get_classifier_type "

get classifier type

classifie type LIBSVR ";

%feature("docstring")  shogun::CLibSVR::get_name "

object name ";


// File: classshogun_1_1CMKLRegression.xml
%feature("docstring") shogun::CMKLRegression "

Multiple Kernel Learning for regression.

Performs support vector regression while learning kernel weights at
the same time. Makes only sense if multiple kernels are used.

See:  CMKL

C++ includes: MKLRegression.h ";

%feature("docstring")  shogun::CMKLRegression::CMKLRegression "

Constructor

Parameters:
-----------

s:  SVM to use as constraint generator in MKL SILP ";

%feature("docstring")  shogun::CMKLRegression::~CMKLRegression "

Destructor ";

%feature("docstring")  shogun::CMKLRegression::compute_sum_alpha "

compute beta independent term from objective, e.g., in 2-class MKL
sum_i alpha_i etc ";


// File: namespaceshogun.xml


// File: KRR_8h.xml


// File: Regression_8h.xml


// File: LibSVR_8h.xml


// File: MKLRegression_8h.xml


// File: SVR__light_8h.xml


// File: dir_009e383d902f1bcb77ecacee31c583c2.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml


// File: dir_cae5aab850498dea6de9c5dc8a4d425e.xml

