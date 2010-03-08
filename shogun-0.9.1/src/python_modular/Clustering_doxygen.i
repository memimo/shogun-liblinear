
// File: index.xml

// File: classshogun_1_1CHierarchical.xml
%feature("docstring") shogun::CHierarchical "

Agglomerative hierarchical single linkage clustering.

Starting with each object being assigned to its own cluster clusters
are iteratively merged. Here the clusters are merged whose elements
have minimum distance, i.e. the clusters A and B that obtain

\\\\[ \\\\min\\\\{d({\\\\bf x},{\\\\bf x'}): {\\\\bf x}\\\\in {\\\\cal
A},{\\\\bf x'}\\\\in {\\\\cal B}\\\\} \\\\]

are merged.

cf e.g.http://en.wikipedia.org/wiki/Data_clustering

C++ includes: Hierarchical.h ";

%feature("docstring")  shogun::CHierarchical::CHierarchical "

default constructor ";

%feature("docstring")  shogun::CHierarchical::CHierarchical "

constructor

Parameters:
-----------

merges:  the merges

d:  distance ";

%feature("docstring")  shogun::CHierarchical::~CHierarchical "";

%feature("docstring")  shogun::CHierarchical::get_classifier_type "

get classifier type

classifier type HIERARCHICAL ";

%feature("docstring")  shogun::CHierarchical::train "

estimate hierarchical clustering

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CHierarchical::load "

load distance machine from file

Parameters:
-----------

srcfile:  file to load from

if loading was successful ";

%feature("docstring")  shogun::CHierarchical::save "

save distance machine to file

Parameters:
-----------

dstfile:  file to save to

if saving was successful ";

%feature("docstring")  shogun::CHierarchical::set_merges "

set merges

Parameters:
-----------

m:  new merges ";

%feature("docstring")  shogun::CHierarchical::get_merges "

get merges

merges ";

%feature("docstring")  shogun::CHierarchical::get_assignment "

get assignment

Parameters:
-----------

assign:  current assignment is stored in here

num:  number of assignments is stored in here ";

%feature("docstring")  shogun::CHierarchical::get_merge_distance "

get merge distance

Parameters:
-----------

dist:  current merge distance is stored in here

num:  number of merge distances is stored in here ";

%feature("docstring")  shogun::CHierarchical::get_merge_distances "

get merge distances (swig compatible)

Parameters:
-----------

dist:  current merge distances are stored in here

num:  number of merge distances are stored in here ";

%feature("docstring")  shogun::CHierarchical::get_pairs "

get pairs

Parameters:
-----------

tuples:  current pairs are stored in here

rows:  number of rows is stored in here

num:  number of pairs is stored in here ";

%feature("docstring")  shogun::CHierarchical::get_cluster_pairs "

get cluster pairs (swig compatible)

Parameters:
-----------

tuples:  current pairs are stored in here

rows:  number of rows is stored in here

num:  number of pairs is stored in here ";

%feature("docstring")  shogun::CHierarchical::classify "

classify objects using the currently set features

classified labels ";

%feature("docstring")  shogun::CHierarchical::classify "

classify objects

Parameters:
-----------

data:  (test)data to be classified

classified labels ";

%feature("docstring")  shogun::CHierarchical::get_name "

object name ";


// File: classshogun_1_1CKMeans.xml
%feature("docstring") shogun::CKMeans "

KMeans clustering, partitions the data into k (a-priori specified)
clusters.

It minimizes \\\\[ \\\\sum_{i=1}^k\\\\sum_{x_j\\\\in S_i}
(x_j-\\\\mu_i)^2 \\\\]

where $\\\\mu_i$ are the cluster centers and $S_i,\\\\;i=1,\\\\dots,k$
are the index sets of the clusters.

Beware that this algorithm obtains only a local optimum.

cf.http://en.wikipedia.org/wiki/K-means_algorithm

C++ includes: KMeans.h ";

%feature("docstring")  shogun::CKMeans::CKMeans "

default constructor ";

%feature("docstring")  shogun::CKMeans::CKMeans "

constructor

Parameters:
-----------

k:  parameter k

d:  distance ";

%feature("docstring")  shogun::CKMeans::~CKMeans "";

%feature("docstring")  shogun::CKMeans::get_classifier_type "

get classifier type

classifier type KMEANS ";

%feature("docstring")  shogun::CKMeans::train "

train k-means

Parameters:
-----------

data:  training data (parameter can be avoided if distance or kernel-
based classifiers are used and distance/kernels are initialized with
train data)

whether training was successful ";

%feature("docstring")  shogun::CKMeans::load "

load distance machine from file

Parameters:
-----------

srcfile:  file to load from

if loading was successful ";

%feature("docstring")  shogun::CKMeans::save "

save distance machine to file

Parameters:
-----------

dstfile:  file to save to

if saving was successful ";

%feature("docstring")  shogun::CKMeans::set_k "

set k

Parameters:
-----------

p_k:  new k ";

%feature("docstring")  shogun::CKMeans::get_k "

get k

the parameter k ";

%feature("docstring")  shogun::CKMeans::set_max_iter "

set maximum number of iterations

Parameters:
-----------

iter:  the new maximum ";

%feature("docstring")  shogun::CKMeans::get_max_iter "

get maximum number of iterations

maximum number of iterations ";

%feature("docstring")  shogun::CKMeans::get_radi "

get radi

Parameters:
-----------

radi:  current radi are stored in here

num:  number of radi is stored in here ";

%feature("docstring")  shogun::CKMeans::get_centers "

get centers

Parameters:
-----------

centers:  current centers are stored in here

dim:  dimensions are stored in here

num:  number of centers is stored in here ";

%feature("docstring")  shogun::CKMeans::get_radiuses "

get radiuses (swig compatible)

Parameters:
-----------

radii:  current radiuses are stored in here

num:  number of radiuses is stored in here ";

%feature("docstring")  shogun::CKMeans::get_cluster_centers "

get cluster centers (swig compatible)

Parameters:
-----------

centers:  current cluster centers are stored in here

dim:  dimensions are stored in here

num:  number of centers is stored in here ";

%feature("docstring")  shogun::CKMeans::get_dimensions "

get dimensions

number of dimensions ";


// File: namespaceshogun.xml


// File: Hierarchical_8h.xml


// File: KMeans_8h.xml


// File: dir_c3d778c97b25f2633a998f287941db3f.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml

