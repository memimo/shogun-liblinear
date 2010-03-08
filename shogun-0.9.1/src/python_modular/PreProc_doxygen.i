
// File: index.xml

// File: classshogun_1_1CDecompressString.xml
%feature("docstring") shogun::CDecompressString "

Preprocessor that decompresses compressed strings.

Each string in CStringFeatures might be stored compressed in memory.
This preprocessor decompresses these strings on the fly. This may be
especially usefull for long strings and when datasets become too large
to fit in memoryin uncompressed form but still when they are
compressed.

Then avoiding expensive disk i/o strings are on-the-fly decompressed.

C++ includes: DecompressString.h ";

%feature("docstring")  shogun::CDecompressString::CDecompressString "

constructor ";

%feature("docstring")  shogun::CDecompressString::~CDecompressString "

destructor ";

%feature("docstring")  shogun::CDecompressString::init "

initialize preprocessor from features ";

%feature("docstring")  shogun::CDecompressString::cleanup "

cleanup ";

%feature("docstring")  shogun::CDecompressString::load "

initialize preprocessor from file ";

%feature("docstring")  shogun::CDecompressString::save "

save preprocessor init-data to file ";

%feature("docstring")
shogun::CDecompressString::apply_to_string_features "

apply preproc on feature matrix result in feature matrix return
pointer to feature_matrix, i.e. f->get_feature_matrix(); ";

%feature("docstring")  shogun::CDecompressString::apply_to_string "

apply preproc on single feature vector ";


// File: classshogun_1_1CLogPlusOne.xml
%feature("docstring") shogun::CLogPlusOne "

Preprocessor LogPlusOne does what the name says, it adds one to a
dense real valued vector and takes the logarithm of each component of
it.

\\\\[ {\\\\bf x}\\\\leftarrow \\\\log({\\\\bf x}+{\\\\bf 1} \\\\] It
therefore does not need any initialization. It is most useful in
situations where the inputs are counts: When one compares differences
of small counts any difference may matter a lot, while small
differences in large counts don't. This is what this log
transformation controls for.

C++ includes: LogPlusOne.h ";

%feature("docstring")  shogun::CLogPlusOne::CLogPlusOne "

default constructor ";

%feature("docstring")  shogun::CLogPlusOne::~CLogPlusOne "";

%feature("docstring")  shogun::CLogPlusOne::init "

initialize preprocessor from features ";

%feature("docstring")  shogun::CLogPlusOne::cleanup "

cleanup ";

%feature("docstring")  shogun::CLogPlusOne::load "

initialize preprocessor from file ";

%feature("docstring")  shogun::CLogPlusOne::save "

save preprocessor init-data to file ";

%feature("docstring")  shogun::CLogPlusOne::apply_to_feature_matrix "

apply preproc on feature matrix result in feature matrix return
pointer to feature_matrix, i.e. f->get_feature_matrix(); ";

%feature("docstring")  shogun::CLogPlusOne::apply_to_feature_vector "

apply preproc on single feature vector result in feature matrix ";

%feature("docstring")  shogun::CLogPlusOne::get_name "

object name ";


// File: classshogun_1_1CNormDerivativeLem3.xml
%feature("docstring") shogun::CNormDerivativeLem3 "

Preprocessor NormDerivativeLem3, performs the normalization used in
Lemma3 in Jaakola Hausslers Fischer Kernel paper currently not
implemented.

C++ includes: NormDerivativeLem3.h ";

%feature("docstring")
shogun::CNormDerivativeLem3::CNormDerivativeLem3 "

default constructor ";

%feature("docstring")
shogun::CNormDerivativeLem3::~CNormDerivativeLem3 "";

%feature("docstring")  shogun::CNormDerivativeLem3::init "

initialize preprocessor from features ";

%feature("docstring")  shogun::CNormDerivativeLem3::cleanup "

cleanup ";

%feature("docstring")  shogun::CNormDerivativeLem3::load "

initialize preprocessor from file ";

%feature("docstring")  shogun::CNormDerivativeLem3::save "

save preprocessor init-data to file ";

%feature("docstring")
shogun::CNormDerivativeLem3::apply_to_feature_matrix "

apply preproc on feature matrix result in feature matrix return
pointer to feature_matrix, i.e. f->get_feature_matrix(); ";

%feature("docstring")
shogun::CNormDerivativeLem3::apply_to_feature_vector "

apply preproc on single feature vector result in feature matrix ";


// File: classshogun_1_1CNormOne.xml
%feature("docstring") shogun::CNormOne "

Preprocessor NormOne, normalizes vectors to have norm 1.

Formally, it computes

\\\\[ {\\\\bf x} \\\\leftarrow \\\\frac{{\\\\bf x}}{||{\\\\bf x}||}
\\\\]

It therefore does not need any initialization. It is most useful to
get data onto a ball of radius one.

C++ includes: NormOne.h ";

%feature("docstring")  shogun::CNormOne::CNormOne "

default constructor ";

%feature("docstring")  shogun::CNormOne::~CNormOne "";

%feature("docstring")  shogun::CNormOne::init "

initialize preprocessor from features ";

%feature("docstring")  shogun::CNormOne::cleanup "

cleanup ";

%feature("docstring")  shogun::CNormOne::load "

initialize preprocessor from file ";

%feature("docstring")  shogun::CNormOne::save "

save preprocessor init-data to file ";

%feature("docstring")  shogun::CNormOne::apply_to_feature_matrix "

apply preproc on feature matrix result in feature matrix return
pointer to feature_matrix, i.e. f->get_feature_matrix(); ";

%feature("docstring")  shogun::CNormOne::apply_to_feature_vector "

apply preproc on single feature vector result in feature matrix ";

%feature("docstring")  shogun::CNormOne::get_name "

object name ";


// File: classshogun_1_1CPreProc.xml
%feature("docstring") shogun::CPreProc "

Class PreProc defines a preprocessor interface.

Preprocessors are transformation functions that don't change the
domain of the input features. These functions can be applied in-place
if the input features fit in memory or can be applied on-the-fly when
(depending on features) a feature caching strategy is applied.
However, if the individual features are in $\\\\bf{R}$ they have to
stay in $\\\\bf{R}$ although the dimensionality of the feature vectors
is allowed change.

As preprocessors might need a certain initialization they may expect
that the init() function is called before anything else. The actual
preprocessing is feature type dependent and thus coordinated in the
sub-classes, cf. e.g. CSimplePreProc .

C++ includes: PreProc.h ";

%feature("docstring")  shogun::CPreProc::CPreProc "

constructor

Parameters:
-----------

name:  preprocessor's name

id:  preprocessor's id ";

%feature("docstring")  shogun::CPreProc::~CPreProc "";

%feature("docstring")  shogun::CPreProc::init "

initialize preprocessor from features ";

%feature("docstring")  shogun::CPreProc::cleanup "

cleanup ";

%feature("docstring")  shogun::CPreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";

%feature("docstring")  shogun::CPreProc::get_feature_class "

return feature class like Sparse,Simple,... ";

%feature("docstring")  shogun::CPreProc::get_name "

return the name of the preprocessor ";

%feature("docstring")  shogun::CPreProc::get_id "

return a FOUR letter id of the preprocessor ";


// File: classshogun_1_1CPruneVarSubMean.xml
%feature("docstring") shogun::CPruneVarSubMean "

Preprocessor PruneVarSubMean will substract the mean and remove
features that have zero variance.

It will optionally normalize standard deviation of features to 1 (by
dividing by standard deviation of the feature)

C++ includes: PruneVarSubMean.h ";

%feature("docstring")  shogun::CPruneVarSubMean::CPruneVarSubMean "

constructor

Parameters:
-----------

divide:  if division shall be made ";

%feature("docstring")  shogun::CPruneVarSubMean::~CPruneVarSubMean "";

%feature("docstring")  shogun::CPruneVarSubMean::init "

initialize preprocessor from features ";

%feature("docstring")  shogun::CPruneVarSubMean::cleanup "

cleanup ";

%feature("docstring")
shogun::CPruneVarSubMean::apply_to_feature_matrix "

apply preproc on feature matrix result in feature matrix return
pointer to feature_matrix, i.e. f->get_feature_matrix(); ";

%feature("docstring")
shogun::CPruneVarSubMean::apply_to_feature_vector "

apply preproc on single feature vector result in feature matrix ";

%feature("docstring")  shogun::CPruneVarSubMean::get_name "

object name ";


// File: classshogun_1_1CSimplePreProc.xml
%feature("docstring") shogun::CSimplePreProc "

Template class SimplePreProc, base class for preprocessors (cf.
CPreProc) that apply to CSimpleFeatures (i.e. rectangular dense
matrices).

Two new functions apply_to_feature_vector() and
apply_to_feature_matrix() are defined in this interface that need to
be implemented in each particular preprocessor operating on
CSimpleFeatures. For examples see e.g. CLogPlusOne or CPCACut.

C++ includes: SimplePreProc.h ";

%feature("docstring")  shogun::CSimplePreProc::CSimplePreProc "

constructor

Parameters:
-----------

name:  simple preprocessor's name

id:  simple preprocessor's id ";

%feature("docstring")  shogun::CSimplePreProc::apply_to_feature_matrix
"

apply preproc on feature matrix result in feature matrix return
pointer to feature_matrix, i.e. f->get_feature_matrix(); ";

%feature("docstring")  shogun::CSimplePreProc::apply_to_feature_vector
"

apply preproc on single feature vector result in feature matrix ";

%feature("docstring")  shogun::CSimplePreProc::get_feature_class "

return that we are simple features (just fixed size matrices) ";

%feature("docstring")  shogun::CSimplePreProc::get_feature_type "

return feature type ";

%feature("docstring")  shogun::CSimplePreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";

%feature("docstring")  shogun::CSimplePreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";

%feature("docstring")  shogun::CSimplePreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";

%feature("docstring")  shogun::CSimplePreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";

%feature("docstring")  shogun::CSimplePreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";

%feature("docstring")  shogun::CSimplePreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";


// File: classshogun_1_1CSortUlongString.xml
%feature("docstring") shogun::CSortUlongString "

Preprocessor SortUlongString, sorts the indivual strings in ascending
order.

This is useful in conjunction with the CCommUlongStringKernel and will
result in the spectrum kernel. For this to work the strings have to be
mapped into a binary higher order representation first (cf.
obtain_from_*() functions in CStringFeatures)

C++ includes: SortUlongString.h ";

%feature("docstring")  shogun::CSortUlongString::CSortUlongString "

default constructor ";

%feature("docstring")  shogun::CSortUlongString::~CSortUlongString "";

%feature("docstring")  shogun::CSortUlongString::init "

initialize preprocessor from features ";

%feature("docstring")  shogun::CSortUlongString::cleanup "

cleanup ";

%feature("docstring")  shogun::CSortUlongString::load "

initialize preprocessor from file ";

%feature("docstring")  shogun::CSortUlongString::save "

save preprocessor init-data to file ";

%feature("docstring")
shogun::CSortUlongString::apply_to_string_features "

apply preproc to feature matrix result in feature matrix return
pointer to feature_matrix, i.e. f->get_feature_matrix(); ";

%feature("docstring")  shogun::CSortUlongString::apply_to_string "

apply preproc on single feature vector result in feature matrix ";

%feature("docstring")  shogun::CSortUlongString::get_name "

object name ";


// File: classshogun_1_1CSortWordString.xml
%feature("docstring") shogun::CSortWordString "

Preprocessor SortWordString, sorts the indivual strings in ascending
order.

This is useful in conjunction with the CCommWordStringKernel and will
result in the spectrum kernel. For this to work the strings have to be
mapped into a binary higher order representation first (cf.
obtain_from_*() functions in CStringFeatures)

C++ includes: SortWordString.h ";

%feature("docstring")  shogun::CSortWordString::CSortWordString "

default constructor ";

%feature("docstring")  shogun::CSortWordString::~CSortWordString "";

%feature("docstring")  shogun::CSortWordString::init "

initialize preprocessor from features ";

%feature("docstring")  shogun::CSortWordString::cleanup "

cleanup ";

%feature("docstring")  shogun::CSortWordString::load "

initialize preprocessor from file ";

%feature("docstring")  shogun::CSortWordString::save "

save preprocessor init-data to file ";

%feature("docstring")
shogun::CSortWordString::apply_to_string_features "

apply preproc on feature matrix result in feature matrix return
pointer to feature_matrix, i.e. f->get_feature_matrix(); ";

%feature("docstring")  shogun::CSortWordString::apply_to_string "

apply preproc on single feature vector result in feature matrix ";

%feature("docstring")  shogun::CSortWordString::get_name "

object name ";


// File: classshogun_1_1CSparsePreProc.xml
%feature("docstring") shogun::CSparsePreProc "

Template class SparsePreProc, base class for preprocessors (cf.
CPreProc) that apply to CSparseFeatures.

Two new functions apply_to_sparse_feature_vector() and
apply_to_sparse_feature_matrix() are defined in this interface that
need to be implemented in each particular preprocessor operating on
CSparseFeatures.

C++ includes: SparsePreProc.h ";

%feature("docstring")  shogun::CSparsePreProc::CSparsePreProc "

constructor

Parameters:
-----------

name:  sparse preprocessor's name

id:  sparse preprocessor's id ";

%feature("docstring")
shogun::CSparsePreProc::apply_to_sparse_feature_matrix "

apply preproc on feature matrix result in feature matrix return
pointer to feature_matrix, i.e. f->get_feature_matrix(); ";

%feature("docstring")
shogun::CSparsePreProc::apply_to_sparse_feature_vector "

apply preproc on single feature vector result in feature matrix ";

%feature("docstring")  shogun::CSparsePreProc::get_feature_class "

return that we are simple minded features (just fixed size matrices)
";


// File: classshogun_1_1CStringPreProc.xml
%feature("docstring") shogun::CStringPreProc "

Template class StringPreProc, base class for preprocessors (cf.
CPreProc) that apply to CStringFeatures (i.e. strings of variable
length).

Two new functions apply_to_string() and apply_to_string_features() are
defined in this interface that need to be implemented in each
particular preprocessor operating on CStringFeatures.

C++ includes: StringPreProc.h ";

%feature("docstring")  shogun::CStringPreProc::CStringPreProc "

constructor

Parameters:
-----------

name:  string preprocessor's name

id:  string preprocessor's id ";

%feature("docstring")
shogun::CStringPreProc::apply_to_string_features "

apply preproc on feature matrix result in feature matrix return
pointer to feature_matrix, i.e. f->get_feature_matrix(); ";

%feature("docstring")  shogun::CStringPreProc::apply_to_string "

apply preproc on single feature vector ";

%feature("docstring")  shogun::CStringPreProc::get_feature_class "

return that we are string features (just fixed size matrices) ";

%feature("docstring")  shogun::CStringPreProc::get_feature_type "

return feature type ";

%feature("docstring")  shogun::CStringPreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";

%feature("docstring")  shogun::CStringPreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";

%feature("docstring")  shogun::CStringPreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";

%feature("docstring")  shogun::CStringPreProc::get_feature_type "

return feature type with which objects derived from CPreProc can deal
";


// File: namespaceshogun.xml


// File: DecompressString_8h.xml


// File: LogPlusOne_8h.xml


// File: NormDerivativeLem3_8h.xml


// File: NormOne_8h.xml


// File: PCACut_8h.xml


// File: PreProc_8h.xml


// File: PruneVarSubMean_8h.xml


// File: SimplePreProc_8h.xml


// File: SortUlongString_8h.xml


// File: SortWordString_8h.xml


// File: SparsePreProc_8h.xml


// File: StringPreProc_8h.xml


// File: dir_c7d69642032cc8b950db23443eae9e33.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml

