
// File: index.xml

// File: classshogun_1_1CPerformanceMeasures.xml
%feature("docstring") shogun::CPerformanceMeasures "

Class to implement various performance measures.

Receiver Operating Curve (ROC)

Area under the ROC curve (auROC)

Area over the ROC curve (aoROC)

Precision Recall Curve (PRC)

Area under the PRC (auPRC)

Area over the PRC (aoPRC)

Detection Error Tradeoff (DET)

Area under the DET (auDET)

Area over the DET (aoDET)

Cross Correlation coefficient (CC)

Weighted Relative Accuracy (WRAcc)

Balanced Error (BAL)

F-Measure

Accuracy

Error

based on: Fawcett, T: March 2004, ROC Graphs: Notes and Practical
Considerations for Researchers and input from Sonnenburg, S: Feburary
2008, various discussions

C++ includes: PerformanceMeasures.h ";

%feature("docstring")
shogun::CPerformanceMeasures::CPerformanceMeasures "

default constructor ";

%feature("docstring")
shogun::CPerformanceMeasures::CPerformanceMeasures "

constructor

Parameters:
-----------

true_labels:  true labels as seen in real world

output:  output labels/hypothesis from a classifier ";

%feature("docstring")
shogun::CPerformanceMeasures::~CPerformanceMeasures "";

%feature("docstring")  shogun::CPerformanceMeasures::init "

initialise performance measures

Parameters:
-----------

true_labels:  true labels as seen in real world

output:  output labels/hypothesis from a classifier ";

%feature("docstring")  shogun::CPerformanceMeasures::set_true_labels "

set true labels as seen in real world

Parameters:
-----------

true_labels:  true labels

if setting was successful ";

%feature("docstring")  shogun::CPerformanceMeasures::get_true_labels "

get true labels as seen in real world

true labels as seen in real world ";

%feature("docstring")  shogun::CPerformanceMeasures::set_output "

set output labels/hypothesis from a classifier

Parameters:
-----------

output:  output labels

if setting was successful ";

%feature("docstring")  shogun::CPerformanceMeasures::get_output "

get classifier's output labels/hypothesis

output labels ";

%feature("docstring")  shogun::CPerformanceMeasures::get_num_labels "

get number of labels in output/true labels

number of labels in output/true labels ";

%feature("docstring")  shogun::CPerformanceMeasures::get_ROC "

get Receiver Operating Curve for previously given labels (swig
compatible)

ROC point = false positives / all false labels, true positives / all
true labels

caller has to free

Parameters:
-----------

result:  where computed ROC values will be stored

num:  number of labels/examples

dim:  dimensions == 2 (false positive rate, true positive rate) ";

%feature("docstring")  shogun::CPerformanceMeasures::get_auROC "

return area under Receiver Operating Curve

calculated by adding trapezoids

area under ROC ";

%feature("docstring")  shogun::CPerformanceMeasures::get_aoROC "

return area over Reveiver Operating Curve

value is 1 - auROC

area over ROC ";

%feature("docstring")  shogun::CPerformanceMeasures::get_PRC "

get Precision Recall Curve for previously given labels (swig
compatible)

PRC point = true positives / all true labels, true positives / (true
positives + false positives)

caller has to free

Parameters:
-----------

result:  where computed ROC values will be stored

num:  number of labels/examples

dim:  dimension == 2 (recall, precision) ";

%feature("docstring")  shogun::CPerformanceMeasures::get_auPRC "

return area under Precision Recall Curve

calculated by adding trapezoids

area under PRC ";

%feature("docstring")  shogun::CPerformanceMeasures::get_aoPRC "

return area over Precision Recall Curve

value is 1 - auPRC

area over PRC ";

%feature("docstring")  shogun::CPerformanceMeasures::get_DET "

get Detection Error Tradeoff curve for previously given labels (swig
compatible)

DET point = false positives / all false labels, false negatives / all
false labels

caller has to free

Parameters:
-----------

result:  where computed DET values will be stored

num:  number of labels/examples

dim:  dimension == 2 (false positive rate, false negative rate) ";

%feature("docstring")  shogun::CPerformanceMeasures::get_auDET "

return area under Detection Error Tradeoff curve

calculated by adding trapezoids

area under DET curve ";

%feature("docstring")  shogun::CPerformanceMeasures::get_aoDET "

return area over Detection Error Tradeoff curve

value is 1 - auDET

area over DET curve ";

%feature("docstring")  shogun::CPerformanceMeasures::get_all_accuracy
"

get classifier's accuracies (swig compatible)

accuracy = (true positives + true negatives) / all labels

caller has to free

Parameters:
-----------

result:  storage of accuracies in 2 dim array: (output, accuracy),
sorted by output

num:  number of accuracy points

dim:  dimension == 2 (output, accuracy) ";

%feature("docstring")  shogun::CPerformanceMeasures::get_accuracy "

get classifier's accuracy at given threshold

Parameters:
-----------

threshold:  all values below are considered negative examples (default
0)

classifer's accuracy at threshold ";

%feature("docstring")  shogun::CPerformanceMeasures::get_all_error "

get classifier's error rates (swig compatible)

value is 1 - accuracy

caller has to free

Parameters:
-----------

result:  storage of errors in 2 dim array: (output, error), sorted by
output

num:  number of accuracy points

dim:  dimension == 2 (output, error) ";

%feature("docstring")  shogun::CPerformanceMeasures::get_error "

get classifier's error at threshold

value is 1 - accuracy0

Parameters:
-----------

threshold:  all values below are considered negative examples (default
0)

classifer's error at threshold ";

%feature("docstring")  shogun::CPerformanceMeasures::get_all_fmeasure
"

get classifier's F-measure (swig compatible)

F-measure = 2 / (1 / precision + 1 / recall)

caller has to free

Parameters:
-----------

result:  storage of F-measure in 2 dim array (output, fmeasure),
sorted by output

num:  number of accuracy points

dim:  dimension == 2 (output, fmeasure) ";

%feature("docstring")  shogun::CPerformanceMeasures::get_fmeasure "

get classifier's F-measure at threshold 0

classifer's F-measure at threshold 0 ";

%feature("docstring")  shogun::CPerformanceMeasures::get_all_CC "

get classifier's Cross Correlation coefficients (swig compatible)

CC = ( true positives * true negatives false positives * false
negatives ) / sqrt( (true positives + false positives) * (true
positives + false negatives) * (true negatives + false positives) *
(true negatives + false negatives) )

also checkhttp://en.wikipedia.org/wiki/Correlation

caller has to free

Parameters:
-----------

result:  storage of CCs in 2 dim array: (output, CC), sorted by output

num:  number of CC points

dim:  dimension == 2 (output, CC) ";

%feature("docstring")  shogun::CPerformanceMeasures::get_CC "

get classifier's Cross Correlation coefficient at threshold

classifer's CC at threshold ";

%feature("docstring")  shogun::CPerformanceMeasures::get_all_WRAcc "

get classifier's Weighted Relative Accuracy (swig compatible)

WRAcc = ( true positives / (true positives + false negatives) ) (
false positives / (false positives + true negatives) )

caller has to free

Parameters:
-----------

result:  storage of WRAccs in 2 dim array: (output, WRAcc), sorted by
output

num:  number of WRAcc points

dim:  dimension == 2 (output, WRAcc) ";

%feature("docstring")  shogun::CPerformanceMeasures::get_WRAcc "

get classifier's Weighted Relative Accuracy at threshold 0

classifer's WRAcc at threshold 0 ";

%feature("docstring")  shogun::CPerformanceMeasures::get_all_BAL "

get classifier's Balanced Error (swig compatible)

BAL = 0.5 * ( true positives / all true labels + true negatives / all
false labels )

caller has to free

Parameters:
-----------

result:  storage of BAL in 2 dim array: (output, BAL), sorted by
output

num:  number of BAL points

dim:  dimension == 2 (output, BAL) ";

%feature("docstring")  shogun::CPerformanceMeasures::get_BAL "

get classifier's Balanced Error at threshold 0

classifer's BAL at threshold 0 ";

%feature("docstring")  shogun::CPerformanceMeasures::get_name "

get the name of tghe object

name of object ";


// File: namespaceshogun.xml


// File: PerformanceMeasures_8h.xml


// File: dir_4302975e42b25bf851d3d0741920b572.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml

