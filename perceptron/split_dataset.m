function [train_set, val_set, test_set] = split_dataset(X, Y, train_ratio, val_ratio)
%SPLIT_DATASET Split a dataset into train, validation and test sets.
%
%   Syntax
%
%   [train_set, val_set, test_set] = SPLIT_DATASET(X, Y, train_ratio, val_ratio)
%
%
%   Description
%
%   Split a dataset with `X` features and `Y` targets into a train set, a
%   validation set and test set, according to `train_ratio` the ratio
%   splitting the dataset into an experiment set and a test set, and
%   `val_ratio` the ratio splitting the experiment dataset into a training
%   set and a validation set.
%
%
%   Arguments
%
%   X -- features, of size (N, M) with N the number of samples, and M the
%       size of features
%   Y -- targets, of size (N, K) with K the size of targets
%   train_ratio -- the ratio splitting the dataset into experiment set and
%                  test set
%   val_ratio -- the ratio splitting the experiment set into the training
%                and the validation set
%
%   Returns
%
%   train_set -- array cell with first cell being the features and second
%                cell being the targets.
%   val_set -- array cell with first cell being the features and second
%              cell being the targets.
%   test_set -- array cell with first cell being the features and second
%               cell being the targets.

% Check arguments
assert(isscalar(train_ratio));
assert(isscalar(val_ratio));
assert(length(X) == length(Y));
N = length(X);

% Convert ratio to number of samples
N_train = floor(train_ratio * N);
N_test = N - N_train;
N_val = floor(val_ratio * N_train);
N_train = N_train - N_val;

% Generate random indices
ind_perm = randperm(N);
ind_train = ind_perm(1:N_train);
ind_val = ind_perm(N_train+1:N_train+N_val);
ind_test = ind_perm(N_train+N_val+1:end);

assert(length(ind_test) == N_test);

% Construct sets
train_set = {X(ind_train, :), Y(ind_train, :)};
val_set = {X(ind_val, :), Y(ind_val, :)};
test_set = {X(ind_test, :), Y(ind_test, :)};

end

