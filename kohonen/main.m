clear all; close all; clc; disp(datetime);

% TODO check wether to normalize data
% TODO Check wether to take half the euclidean distance or not
% TODO check wheter to divide the dW by its norm
% TODO put (decay in radius decay) in nei_decay function


%% Generate dataset

% Two gaussians distribution
n1=100; mu1=[1, 0];  sigma1=eye(2);
n2=100; mu2=[-3, 2]; sigma2=eye(2);

X1 = mvnrnd(mu1, sigma1, n1); Y1 = zeros(n1, 1);
X2 = mvnrnd(mu2, sigma1, n2); Y2 = ones(n2, 1);

N = n1 + n2;
X = [X1; X2];
Y = [Y1; Y2];

%% Create model

n_features = size(X, 2);
d1 = 10;
d2 = 5;
lattice = randn(n_features, d1 * d2);

% X \in R^{N x n_features}, s.t. if N = 1 it is still working (stochastic)
% W \in R^{n_features x (d1*d2)}
% Returns D \in R^{N x n_features}
dist_fwd = @(X, W) reshape(sqrt(sum((repelem(X', 1, size(W, 2)) - repmat(W, 1, size(X, 1))).^2, 1)), size(W, 2), size(X, 1))';

radius_decay_t = @(epoch) - epoch;
dist_eucl = @(w1, w2) sqrt(sum((w1 - w2).^2));
nei_decay = @(epoch, w1, w2) exp(radius_decay_t(epoch) * dist_eucl(w1, w2));

lr = 5; % TODO adjust
epoch=0;
while lr > 1e-3 % TODO condition
    epoch = epoch + 1; % TODO split in batch + epoch
    
    % Compute distance between the input and each neuron, for each sample
    D = dist_fwd(X, lattice);
    
    
    % TODO doit retourner pas que l2 mais aussi l1.
    
    
    % Find the neuron with the minimum distance, for each sample (in column)
    [M, BMU] = min(D, [], 2); % BMU for Best Matching Unit
    
    % Update weights for the most activated neurons
    % (that are also the less distant from the input)
    % by shifting them towards the input that activated them
    for nrn = 1:size(lattice, 2)
        dW = lr * sum(M(BMU==nrn));
        
        for nrn_nei = 1:size(lattice, 2)
            nei_decay_coeff = nei_decay(epoch, lattice(:, nrn), lattice(:, nrn_nei));
            lattice(:, nrn_nei) = lattice(:, nrn_nei) + nei_decay_coeff * dW;
        end        
    end
    
    % Decrease the learning rate at each epoch
    lr = 0.95 * lr;
end

scatter(lattice(1, :), lattice(2, :));