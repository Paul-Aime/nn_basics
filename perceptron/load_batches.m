function [batches] = load_batches(X, Y, batch_size, varargin)
%LOAD_BATCHES Generate randomized batches from a dataset.
%
%   Syntax
%
%   batches = LOAD_BATCHES(X, Y, batch_size)
%   batches = LOAD_BATCHES(X, Y, batch_size, drop_last)
%
%
%   Description
%
%   Generate randomized `batches` of size `batch_size`, from `X` features
%   and corresponding `Y` targets. If `drop_last` is False (default to
%   True), then a last batch of size inferior to `batch_size` will be
%   returned as the last cell of the returned cell array.
%
%   Arguments
%
%   X -- features, of size (N, M) with N the number of samples, and M the
%        size of features
%   Y -- targets, of size (N, K) with K the size of targets
%   batch_size -- number of samples per batch
%                 drop_last - boolean, whether to drop samples if the can't
%                 construct a full bacth of size `batch_size`
%                 (optional, default to True)
%
%
%   Returns
%
%   batches -- cell array containing `N // batch_size` cells (may be
%              `N // batch_size +1` if `drop_last` is set to False. Each
%              cell is a batch, represented as a cell with first element
%              being the features and the second element being the targets

% Check optional argument `drop_last`
assert(nargin <= 4)
if nargin == 3
    drop_last = 1; % Default to True
else
    drop_last = varargin{1};
end

% Shuffle dataset
assert(length(X) == length(Y))
dataset_size = length(X);
ind_perm = randperm(dataset_size);

% Compute number of full batches (of size `batch_size`)
n_full_batches = floor(dataset_size / batch_size);

% Handle last batch
if ~drop_last
    last_batch_size = dataset_size - n_full_batches * batch_size;
else
    last_batch_size = 0;
end

% Split indices into batches indices
batches_indices = reshape(ind_perm(1:n_full_batches * batch_size), batch_size, n_full_batches)'; 
batches_indices = num2cell(batches_indices, 2);

% Handle last batch
if last_batch_size
    batches_indices{n_full_batches+1} = ind_perm(end-(last_batch_size-1):end);
end

% Create batches
n_batches = length(batches_indices);
batches = cell(n_batches, 1);
for i = 1:n_batches
    batches{i} = {X(batches_indices{i}, :), Y(batches_indices{i}, :)};
end



end