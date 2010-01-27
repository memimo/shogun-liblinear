set_features TRAIN australian_scale.dat
set_labels TRAIN australian_scale.label
new_classifier LIBLINEAR_LR
svm_epsilon 0.01
svm_use_bias 0
c 1
train_classifier
