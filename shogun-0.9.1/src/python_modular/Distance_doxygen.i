
// File: index.xml

// File: classshogun_1_1CBrayCurtisDistance.xml
%feature("docstring") shogun::CBrayCurtisDistance "

class Bray-Curtis distance

The Bray-Curtis distance (Sorensen distance) is similar to the
Manhattan distance with normalization.

\\\\[\\\\displaystyle d(\\\\bf{x},\\\\bf{x}') =
\\\\frac{\\\\sum_{i=1}^{n}|x_{i}-x'_{i}|}{\\\\sum_{i=1}^{n}|x_{i}
+x'_{i}|} \\\\quad x,x' \\\\in R^{n} \\\\]

C++ includes: BrayCurtisDistance.h ";

%feature("docstring")
shogun::CBrayCurtisDistance::CBrayCurtisDistance "

default constructor ";

%feature("docstring")
shogun::CBrayCurtisDistance::CBrayCurtisDistance "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")
shogun::CBrayCurtisDistance::~CBrayCurtisDistance "";

%feature("docstring")  shogun::CBrayCurtisDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CBrayCurtisDistance::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CBrayCurtisDistance::get_distance_type
"

get distance type we are

distance type BRAYCURTIS ";

%feature("docstring")  shogun::CBrayCurtisDistance::get_name "

get name of the distance

name Bray-Curtis distance ";


// File: classshogun_1_1CCanberraMetric.xml
%feature("docstring") shogun::CCanberraMetric "

class CanberraMetric

The Canberra distance sums up the dissimilarity (ratios) between
feature dimensions of two data points.

\\\\[\\\\displaystyle d(\\\\bf{x},\\\\bf{x'}) =
\\\\sum_{i=1}^{n}\\\\frac{|\\\\bf{x_{i}-\\\\bf{x'_{i}}}|}
{|\\\\bf{x_{i}}|+|\\\\bf{x'_{i}}|} \\\\quad \\\\bf{x},\\\\bf{x'}
\\\\in R^{n} \\\\]

A summation element has range [0,1]. Note that $d(x,0)=d(0,x')=n$ and
$d(0,0)=0$.

C++ includes: CanberraMetric.h ";

%feature("docstring")  shogun::CCanberraMetric::CCanberraMetric "

default constructor ";

%feature("docstring")  shogun::CCanberraMetric::CCanberraMetric "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CCanberraMetric::~CCanberraMetric "";

%feature("docstring")  shogun::CCanberraMetric::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CCanberraMetric::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CCanberraMetric::get_distance_type "

get distance type we are

distance type CANBERRA ";

%feature("docstring")  shogun::CCanberraMetric::get_name "

get name of the distance

name Canberra-Metric ";


// File: classshogun_1_1CCanberraWordDistance.xml
%feature("docstring") shogun::CCanberraWordDistance "

class CanberraWordDistance

C++ includes: CanberraWordDistance.h ";

%feature("docstring")
shogun::CCanberraWordDistance::CCanberraWordDistance "

default constructor ";

%feature("docstring")
shogun::CCanberraWordDistance::CCanberraWordDistance "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")
shogun::CCanberraWordDistance::~CCanberraWordDistance "";

%feature("docstring")  shogun::CCanberraWordDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CCanberraWordDistance::cleanup "

cleanup distance ";

%feature("docstring")
shogun::CCanberraWordDistance::get_distance_type "

get distance type we are

distance type CHEBYSHEW ";

%feature("docstring")  shogun::CCanberraWordDistance::get_name "

get name of the distance

name Chebyshew-Metric ";

%feature("docstring")  shogun::CCanberraWordDistance::get_dictionary "

get dictionary weights

Parameters:
-----------

dsize:  size of the dictionary

dweights:  dictionary weights are stored in here ";


// File: classshogun_1_1CChebyshewMetric.xml
%feature("docstring") shogun::CChebyshewMetric "

class ChebyshewMetric

The Chebyshev distance ( $L_{\\\\infty}$ norm) returns the maximum of
absolute feature dimension differences between two data points.

\\\\[\\\\displaystyle d(\\\\bf{x},\\\\bf{x'}) =
max|\\\\bf{x_{i}}-\\\\bf{x'_{i}}| \\\\quad x,x' \\\\in R^{n} \\\\]

See:  Wikipedia: Chebyshev distance

C++ includes: ChebyshewMetric.h ";

%feature("docstring")  shogun::CChebyshewMetric::CChebyshewMetric "

default constructor ";

%feature("docstring")  shogun::CChebyshewMetric::CChebyshewMetric "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CChebyshewMetric::~CChebyshewMetric "";

%feature("docstring")  shogun::CChebyshewMetric::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CChebyshewMetric::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CChebyshewMetric::get_distance_type "

get distance type we are

distance type CHEBYSHEW ";

%feature("docstring")  shogun::CChebyshewMetric::get_name "

get name of the distance

name Chebyshew-Metric ";


// File: classshogun_1_1CChiSquareDistance.xml
%feature("docstring") shogun::CChiSquareDistance "

class ChiSquareDistance

This implementation of $\\\\chi^{2}$ distance extends the concept of
$\\\\chi^{2}$ metric to negative values.

\\\\[\\\\displaystyle d(\\\\bf{x},\\\\bf{x'}) =
\\\\sum_{i=1}^{n}\\\\frac{(x_{i}-x'_{i})^2} {|x_{i}|+|x'_{i}|}
\\\\quad \\\\bf{x},\\\\bf{x'} \\\\in R^{n} \\\\]

See:  K. Rieck, P. Laskov. Linear-Time Computation of Similarity
Measures for Sequential Data. Journal of Machine Learning Research,
9:23-- 48,2008.

C++ includes: ChiSquareDistance.h ";

%feature("docstring")  shogun::CChiSquareDistance::CChiSquareDistance
"

default constructor ";

%feature("docstring")  shogun::CChiSquareDistance::CChiSquareDistance
"

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CChiSquareDistance::~CChiSquareDistance
"";

%feature("docstring")  shogun::CChiSquareDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CChiSquareDistance::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CChiSquareDistance::get_distance_type "

get distance type we are

distance type CHISQUARE ";

%feature("docstring")  shogun::CChiSquareDistance::get_name "

get name of the distance

name Chi-square distance ";


// File: classshogun_1_1CCosineDistance.xml
%feature("docstring") shogun::CCosineDistance "

class CosineDistance

The Cosine distance is obtained by using the Cosine similarity
(Orchini similarity, angular similarity, normalized dot product),
which measures similarity between two vectors by finding their angle.
An extension to the Cosine similarity yields the Tanimoto coefficient.

\\\\[\\\\displaystyle d(\\\\bf{x},\\\\bf{x'}) = 1 -
\\\\frac{\\\\sum_{i=1}^{n}\\\\bf{x_{i}}\\\\bf{x'_{i}}}
{\\\\sqrt{\\\\sum_{i=1}^{n} x_{i}^2 \\\\sum_{i=1}^{n} {x'}_{i}^2}}
\\\\quad x,x' \\\\in R^{n} \\\\]

See:  Wikipedia: Cosine similarity

CTanimotoDistance

C++ includes: CosineDistance.h ";

%feature("docstring")  shogun::CCosineDistance::CCosineDistance "

default constructor ";

%feature("docstring")  shogun::CCosineDistance::CCosineDistance "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CCosineDistance::~CCosineDistance "";

%feature("docstring")  shogun::CCosineDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CCosineDistance::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CCosineDistance::get_distance_type "

get distance type we are

distance type COSINE ";

%feature("docstring")  shogun::CCosineDistance::get_name "

get name of the distance

name Cosine distance ";


// File: classshogun_1_1CDistance.xml
%feature("docstring") shogun::CDistance "

class Distance

All distance classes are derived from this base class.

C++ includes: Distance.h ";

%feature("docstring")  shogun::CDistance::CDistance "

default constructor ";

%feature("docstring")  shogun::CDistance::CDistance "

init distance

Parameters:
-----------

lhs:  features of left-hand side

rhs:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CDistance::~CDistance "";

%feature("docstring")  shogun::CDistance::distance "

get distance function for lhs feature vector a and rhs feature vector
b

Parameters:
-----------

idx_a:  feature vector a at idx_a

idx_b:  feature vector b at idx_b

distance value ";

%feature("docstring")  shogun::CDistance::get_distance_matrix "

get distance matrix

Parameters:
-----------

dst:  distance matrix is stored in here

m:  dimension m of matrix is stored in here

n:  dimension n of matrix is stored in here ";

%feature("docstring")  shogun::CDistance::get_distance_matrix_real "

get distance matrix real

Parameters:
-----------

m:  dimension m

n:  dimension n

target:  target matrix

target matrix ";

%feature("docstring")
shogun::CDistance::get_distance_matrix_shortreal "

get distance matrix short real

Parameters:
-----------

m:  dimension m

n:  dimension n

target:  target matrix

target matrix ";

%feature("docstring")  shogun::CDistance::init "

init distance

make sure to check that your distance can deal with the supplied
features (!)

Parameters:
-----------

lhs:  features of left-hand side

rhs:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CDistance::cleanup "

cleanup distance

abstract base method ";

%feature("docstring")  shogun::CDistance::load "

load distance matrix from file

Parameters:
-----------

fname:  filename to load from

if loading was successful ";

%feature("docstring")  shogun::CDistance::save "

save distance matrix to file

Parameters:
-----------

fname:  filename to save to

if saving was successful ";

%feature("docstring")  shogun::CDistance::get_lhs "

get left-hand side features used in distance matrix

left-hand side features ";

%feature("docstring")  shogun::CDistance::get_rhs "

get right-hand side features used in distance matrix

right-hand side features ";

%feature("docstring")  shogun::CDistance::replace_rhs "

replace right-hand side features used in distance matrix

make sure to check that your distance can deal with the supplied
features (!)

Parameters:
-----------

rhs:  features of right-hand side

replaced right-hand side features ";

%feature("docstring")  shogun::CDistance::remove_lhs_and_rhs "

remove lhs and rhs from distance ";

%feature("docstring")  shogun::CDistance::remove_lhs "

takes all necessary steps if the lhs is removed from distance matrix
";

%feature("docstring")  shogun::CDistance::remove_rhs "

takes all necessary steps if the rhs is removed from distance matrix
";

%feature("docstring")  shogun::CDistance::get_distance_type "

get distance type we are

abstrace base method

distance type ";

%feature("docstring")  shogun::CDistance::get_feature_type "

get feature type the distance can deal with

abstrace base method

feature type ";

%feature("docstring")  shogun::CDistance::get_feature_class "

get feature class the distance can deal with

abstract base method

feature class ";

%feature("docstring")  shogun::CDistance::get_precompute_matrix "

FIXME: precompute matrix should be dropped, handling should be via
customdistance

if precompute_matrix ";

%feature("docstring")  shogun::CDistance::set_precompute_matrix "

FIXME: precompute matrix should be dropped, handling should be via
customdistance

Parameters:
-----------

flag:  if precompute_matrix ";

%feature("docstring")  shogun::CDistance::get_num_vec_lhs "

get number of vectors of lhs features

number of vectors of left-hand side ";

%feature("docstring")  shogun::CDistance::get_num_vec_rhs "

get number of vectors of rhs features

number of vectors of right-hand side ";

%feature("docstring")  shogun::CDistance::has_features "

test whether features have been assigned to lhs and rhs

true if features are assigned ";

%feature("docstring")  shogun::CDistance::lhs_equals_rhs "

test whether features on lhs and rhs are the same

true if features are the same ";


// File: classshogun_1_1CEuclidianDistance.xml
%feature("docstring") shogun::CEuclidianDistance "

class EuclidianDistance

The familiar Euclidian distance for real valued features computes the
square root of the sum of squared disparity between the corresponding
feature dimensions of two data points.

\\\\[\\\\displaystyle d({\\\\bf x},{\\\\bf x'})=
\\\\sqrt{\\\\sum_{i=0}^{n}|{\\\\bf x_i}-{\\\\bf x'_i}|^2} \\\\]

This special case of Minkowski metric is invariant to an arbitrary
translation or rotation in feature space.

The Euclidian Squared distance does not take the square root:

\\\\[\\\\displaystyle d({\\\\bf x},{\\\\bf x'})=
\\\\sum_{i=0}^{n}|{\\\\bf x_i}-{\\\\bf x'_i}|^2 \\\\]

See:   CMinkowskiMetric

Wikipedia: Distance in Euclidean space

C++ includes: EuclidianDistance.h ";

%feature("docstring")  shogun::CEuclidianDistance::CEuclidianDistance
"

default constructor ";

%feature("docstring")  shogun::CEuclidianDistance::CEuclidianDistance
"

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CEuclidianDistance::~CEuclidianDistance
"";

%feature("docstring")  shogun::CEuclidianDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CEuclidianDistance::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CEuclidianDistance::get_distance_type "

get distance type we are

distance type EUCLIDIAN ";

%feature("docstring")  shogun::CEuclidianDistance::get_feature_type "

get feature type the distance can deal with

feature type DREAL ";

%feature("docstring")  shogun::CEuclidianDistance::get_name "

get name of the distance

name Euclidian ";

%feature("docstring")  shogun::CEuclidianDistance::get_disable_sqrt "

disable application of sqrt on matrix computation the matrix can then
also be named norm squared

if application of sqrt is disabled ";

%feature("docstring")  shogun::CEuclidianDistance::set_disable_sqrt "

disable application of sqrt on matrix computation the matrix can then
also be named norm squared

Parameters:
-----------

state:  new disable_sqrt ";


// File: classshogun_1_1CGeodesicMetric.xml
%feature("docstring") shogun::CGeodesicMetric "

class GeodesicMetric

The Geodesic distance (Great circle distance) computes the shortest
path between two data points on a sphere (the radius is set to one for
the evaluation).

\\\\[\\\\displaystyle d(\\\\bf{x},\\\\bf{x'}) =
arccos\\\\sum_{i=1}^{n} \\\\frac{\\\\bf{x_{i}}\\\\cdot\\\\bf{x'_{i}}}
{\\\\sqrt{x_{i}x_{i} x'_{i}x'_{i}}} \\\\]

See:  Wikipedia: Geodesic distance

C++ includes: GeodesicMetric.h ";

%feature("docstring")  shogun::CGeodesicMetric::CGeodesicMetric "

default constructor ";

%feature("docstring")  shogun::CGeodesicMetric::CGeodesicMetric "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CGeodesicMetric::~CGeodesicMetric "";

%feature("docstring")  shogun::CGeodesicMetric::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CGeodesicMetric::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CGeodesicMetric::get_distance_type "

get distance type we are

distance type GEODESIC ";

%feature("docstring")  shogun::CGeodesicMetric::get_name "

get name of the distance

name Chebyshew-Metric ";


// File: classshogun_1_1CHammingWordDistance.xml
%feature("docstring") shogun::CHammingWordDistance "

class HammingWordDistance

C++ includes: HammingWordDistance.h ";

%feature("docstring")
shogun::CHammingWordDistance::CHammingWordDistance "

constructor

Parameters:
-----------

use_sign:  if sign shall be used ";

%feature("docstring")
shogun::CHammingWordDistance::CHammingWordDistance "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

use_sign:  if sign shall be used ";

%feature("docstring")
shogun::CHammingWordDistance::~CHammingWordDistance "";

%feature("docstring")  shogun::CHammingWordDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CHammingWordDistance::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CHammingWordDistance::get_distance_type
"

get distance type we are

distance type HAMMINGWORD ";

%feature("docstring")  shogun::CHammingWordDistance::get_name "

get name of the distance

name HammingWord ";

%feature("docstring")  shogun::CHammingWordDistance::get_dictionary "

get dictionary weights

Parameters:
-----------

dsize:  size of the dictionary

dweights:  dictionary weights are stored in here ";


// File: classshogun_1_1CJensenMetric.xml
%feature("docstring") shogun::CJensenMetric "

class JensenMetric

The Jensen-Shannon distance/divergence measures the similarity between
two data points which is based on the Kullback-Leibler divergence.

\\\\[\\\\displaystyle d(\\\\bf{x},\\\\bf{x'}) = \\\\sum_{i=0}^{n}
x_{i} log\\\\frac{x_{i}}{0.5(x_{i} +x'_{i})} + x'_{i}
log\\\\frac{x'_{i}}{0.5(x_{i}+x'_{i})} \\\\]

See:  Wikipedia: Jensen-Shannon divergence

Wikipedia: Kullback-Leibler divergence

C++ includes: JensenMetric.h ";

%feature("docstring")  shogun::CJensenMetric::CJensenMetric "

default constructor ";

%feature("docstring")  shogun::CJensenMetric::CJensenMetric "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CJensenMetric::~CJensenMetric "";

%feature("docstring")  shogun::CJensenMetric::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CJensenMetric::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CJensenMetric::get_distance_type "

get distance type we are

distance type JENSEN ";

%feature("docstring")  shogun::CJensenMetric::get_name "

get name of the distance

name Jensen-Metric ";


// File: classshogun_1_1CManhattanMetric.xml
%feature("docstring") shogun::CManhattanMetric "

class ManhattanMetric

The Manhattan distance (city block distance, $L_{1}$ norm, rectilinear
distance or taxi cab metric ) is a special case of general Minkowski
metric and computes the absolute differences between the feature
dimensions of two data points.

\\\\[\\\\displaystyle d(\\\\bf{x},\\\\bf{x'}) = \\\\sum_{i=1}^{n}
|\\\\bf{x_{i}}-\\\\bf{x'_{i}}| \\\\quad \\\\bf{x},\\\\bf{x'} \\\\in
R^{n} \\\\]

See:   CMinkowskiMetric

Wikipedia: Manhattan distance

C++ includes: ManhattanMetric.h ";

%feature("docstring")  shogun::CManhattanMetric::CManhattanMetric "

default constructor ";

%feature("docstring")  shogun::CManhattanMetric::CManhattanMetric "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CManhattanMetric::~CManhattanMetric "";

%feature("docstring")  shogun::CManhattanMetric::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CManhattanMetric::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CManhattanMetric::get_distance_type "

get distance type we are

distance type MANHATTAN ";

%feature("docstring")  shogun::CManhattanMetric::get_name "

get name of the distance

name Manhattan-Metric ";


// File: classshogun_1_1CManhattanWordDistance.xml
%feature("docstring") shogun::CManhattanWordDistance "

class ManhattanWordDistance

C++ includes: ManhattanWordDistance.h ";

%feature("docstring")
shogun::CManhattanWordDistance::CManhattanWordDistance "

default constructor ";

%feature("docstring")
shogun::CManhattanWordDistance::CManhattanWordDistance "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")
shogun::CManhattanWordDistance::~CManhattanWordDistance "";

%feature("docstring")  shogun::CManhattanWordDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CManhattanWordDistance::cleanup "

cleanup distance ";

%feature("docstring")
shogun::CManhattanWordDistance::get_distance_type "

get distance type we are

distance type MANHATTANWORD ";

%feature("docstring")  shogun::CManhattanWordDistance::get_name "

get name of the distance

name ManhattanWord ";

%feature("docstring")  shogun::CManhattanWordDistance::get_dictionary
"

get dictionary weights

Parameters:
-----------

dsize:  size of the dictionary

dweights:  dictionary weights are stored in here ";


// File: classshogun_1_1CMinkowskiMetric.xml
%feature("docstring") shogun::CMinkowskiMetric "

class MinkowskiMetric

The Minkowski metric is one general class of metrics for a
$\\\\displaystyle R^{n}$ feature space also referred as the
$\\\\displaystyle L_{k} $ norm.

\\\\[ \\\\displaystyle d(\\\\bf{x},\\\\bf{x'}) = (\\\\sum_{i=1}^{n}
|\\\\bf{x_{i}}-\\\\bf{x'_{i}}|^{k})^{\\\\frac{1}{k}} \\\\quad x,x'
\\\\in R^{n} \\\\]

special cases:  $\\\\displaystyle L_{1} $ norm: Manhattan distance
See:   CManhattanMetric

$\\\\displaystyle L_{2} $ norm: Euclidean distance See:
CEuclidianDistance  Note that the Minkowski distance tends to the
Chebyshew distance for increasing $k$.

See:  Wikipedia: Distance

C++ includes: MinkowskiMetric.h ";

%feature("docstring")  shogun::CMinkowskiMetric::CMinkowskiMetric "

constructor

Parameters:
-----------

k:  parameter k ";

%feature("docstring")  shogun::CMinkowskiMetric::CMinkowskiMetric "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

k:  parameter k ";

%feature("docstring")  shogun::CMinkowskiMetric::~CMinkowskiMetric "";

%feature("docstring")  shogun::CMinkowskiMetric::init "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CMinkowskiMetric::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CMinkowskiMetric::get_distance_type "

get distance type we are

distance type MINKOWSKI ";

%feature("docstring")  shogun::CMinkowskiMetric::get_name "

get name of the distance

name Minkowski-Metric ";


// File: classshogun_1_1CRealDistance.xml
%feature("docstring") shogun::CRealDistance "

class RealDistance

C++ includes: RealDistance.h ";

%feature("docstring")  shogun::CRealDistance::CRealDistance "

default constructor ";

%feature("docstring")  shogun::CRealDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CRealDistance::get_feature_type "

get feature type the distance can deal with

feature type DREAL ";


// File: classshogun_1_1CSimpleDistance.xml
%feature("docstring") shogun::CSimpleDistance "

template class SimpleDistance

C++ includes: SimpleDistance.h ";

%feature("docstring")  shogun::CSimpleDistance::CSimpleDistance "

default constructor ";

%feature("docstring")  shogun::CSimpleDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CSimpleDistance::get_feature_class "

get feature class the distance can deal with

feature class SIMPLE ";

%feature("docstring")  shogun::CSimpleDistance::get_feature_type "

get feature type the distance can deal with

template-specific feature type ";

%feature("docstring")  shogun::CSimpleDistance::get_feature_type "

get feature type the DREAL distance can deal with

feature type DREAL ";

%feature("docstring")  shogun::CSimpleDistance::get_feature_type "

get feature type the ULONG distance can deal with

feature type ULONG ";

%feature("docstring")  shogun::CSimpleDistance::get_feature_type "

get feature type the INT distance can deal with

feature type INT ";

%feature("docstring")  shogun::CSimpleDistance::get_feature_type "

get feature type the WORD distance can deal with

feature type WORD ";

%feature("docstring")  shogun::CSimpleDistance::get_feature_type "

get feature type the SHORT distance can deal with

feature type SHORT ";

%feature("docstring")  shogun::CSimpleDistance::get_feature_type "

get feature type the BYTE distance can deal with

feature type BYTE ";

%feature("docstring")  shogun::CSimpleDistance::get_feature_type "

get feature type the CHAR distance can deal with

feature type CHAR ";


// File: classshogun_1_1CSparseDistance.xml
%feature("docstring") shogun::CSparseDistance "

template class SparseDistance

C++ includes: SparseDistance.h ";

%feature("docstring")  shogun::CSparseDistance::CSparseDistance "

default constructor ";

%feature("docstring")  shogun::CSparseDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CSparseDistance::get_feature_class "

get feature class the distance can deal with

feature class SPARSE ";

%feature("docstring")  shogun::CSparseDistance::get_feature_type "

get feature type the distance can deal with

template-specific feature type ";

%feature("docstring")  shogun::CSparseDistance::get_feature_type "

get feature type the DREAL distance can deal with

feature type DREAL ";

%feature("docstring")  shogun::CSparseDistance::get_feature_type "

get feature type the ULONG distance can deal with

feature type ULONG ";

%feature("docstring")  shogun::CSparseDistance::get_feature_type "

get feature type the INT distance can deal with

feature type INT ";

%feature("docstring")  shogun::CSparseDistance::get_feature_type "

get feature type the WORD distance can deal with

feature type WORD ";

%feature("docstring")  shogun::CSparseDistance::get_feature_type "

get feature type the SHORT distance can deal with

feature type SHORT ";

%feature("docstring")  shogun::CSparseDistance::get_feature_type "

get feature type the BYTE distance can deal with

feature type BYTE ";

%feature("docstring")  shogun::CSparseDistance::get_feature_type "

get feature type the CHAR distance can deal with

feature type CHAR ";


// File: classshogun_1_1CSparseEuclidianDistance.xml
%feature("docstring") shogun::CSparseEuclidianDistance "

class SparseEucldianDistance

C++ includes: SparseEuclidianDistance.h ";

%feature("docstring")
shogun::CSparseEuclidianDistance::CSparseEuclidianDistance "

default constructor ";

%feature("docstring")
shogun::CSparseEuclidianDistance::CSparseEuclidianDistance "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")
shogun::CSparseEuclidianDistance::~CSparseEuclidianDistance "";

%feature("docstring")  shogun::CSparseEuclidianDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CSparseEuclidianDistance::cleanup "

cleanup distance ";

%feature("docstring")
shogun::CSparseEuclidianDistance::get_distance_type "

get distance type we are

distance type SPARSEEUCLIDIAN ";

%feature("docstring")
shogun::CSparseEuclidianDistance::get_feature_type "

get supported feature type

feature type DREAL ";

%feature("docstring")  shogun::CSparseEuclidianDistance::get_name "

get name of the distance

name SparseEuclidian ";


// File: classshogun_1_1CStringDistance.xml
%feature("docstring") shogun::CStringDistance "

template class StringDistance

C++ includes: StringDistance.h ";

%feature("docstring")  shogun::CStringDistance::CStringDistance "

default constructor ";

%feature("docstring")  shogun::CStringDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CStringDistance::get_feature_class "

get feature class the distance can deal with

feature class STRING ";

%feature("docstring")  shogun::CStringDistance::get_feature_type "

get feature type the distance can deal with

template-specific feature type ";

%feature("docstring")  shogun::CStringDistance::get_feature_type "

get feature type the DREAL distance can deal with

feature type DREAL ";

%feature("docstring")  shogun::CStringDistance::get_feature_type "

get feature type the ULONG distance can deal with

feature type ULONG ";

%feature("docstring")  shogun::CStringDistance::get_feature_type "

get feature type the INT distance can deal with

feature type INT ";

%feature("docstring")  shogun::CStringDistance::get_feature_type "

get feature type the WORD distance can deal with

feature type WORD ";

%feature("docstring")  shogun::CStringDistance::get_feature_type "

get feature type the SHORT distance can deal with

feature type SHORT ";

%feature("docstring")  shogun::CStringDistance::get_feature_type "

get feature type the BYTE distance can deal with

feature type BYTE ";

%feature("docstring")  shogun::CStringDistance::get_feature_type "

get feature type the CHAR distance can deal with

feature type CHAR ";


// File: classshogun_1_1CTanimotoDistance.xml
%feature("docstring") shogun::CTanimotoDistance "

class Tanimoto coefficient

The Tanimoto distance/coefficient (extended Jaccard coefficient) is
obtained by extending the cosine similarity.

\\\\[\\\\displaystyle d(\\\\bf{x},\\\\bf{x'}) =
\\\\frac{\\\\sum_{i=1}^{n}x_{i}x'_{i}}{
\\\\sum_{i=1}^{n}x_{i}x_{i}x'_{i}x'_{i}-x_{i}x'_{i}} /quad x,x' /in
R^{n} \\\\]

See:  Wikipedia: Tanimoto coefficient

CCosineDistance

C++ includes: TanimotoDistance.h ";

%feature("docstring")  shogun::CTanimotoDistance::CTanimotoDistance "

default constructor ";

%feature("docstring")  shogun::CTanimotoDistance::CTanimotoDistance "

constructor

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side ";

%feature("docstring")  shogun::CTanimotoDistance::~CTanimotoDistance "";

%feature("docstring")  shogun::CTanimotoDistance::init "

init distance

Parameters:
-----------

l:  features of left-hand side

r:  features of right-hand side

if init was successful ";

%feature("docstring")  shogun::CTanimotoDistance::cleanup "

cleanup distance ";

%feature("docstring")  shogun::CTanimotoDistance::get_distance_type "

get distance type we are

distance type TANIMOTO ";

%feature("docstring")  shogun::CTanimotoDistance::get_name "

get name of the distance

name Tanimoto coefficient/distance ";


// File: namespaceshogun.xml


// File: BrayCurtisDistance_8h.xml


// File: CanberraMetric_8h.xml


// File: CanberraWordDistance_8h.xml


// File: ChebyshewMetric_8h.xml


// File: ChiSquareDistance_8h.xml


// File: CosineDistance_8h.xml


// File: Distance_8h.xml


// File: EuclidianDistance_8h.xml


// File: GeodesicMetric_8h.xml


// File: HammingWordDistance_8h.xml


// File: JensenMetric_8h.xml


// File: ManhattanMetric_8h.xml


// File: ManhattanWordDistance_8h.xml


// File: MinkowskiMetric_8h.xml


// File: RealDistance_8h.xml


// File: SimpleDistance_8h.xml


// File: SparseDistance_8h.xml


// File: SparseEuclidianDistance_8h.xml


// File: StringDistance_8h.xml


// File: TanimotoDistance_8h.xml


// File: dir_5515f879a205d7f837677afc0399f837.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml

