clear all; close all; clc; disp(datetime)

%% Generate dataset

% Two gaussians distribution
n1=100; mu1=[3, 0];  sigma1=eye(2);
n2=100; mu2=[-3, 0]; sigma2=eye(2);

X1 = mvnrnd(mu1, sigma1, n1); Y1 = zeros(n1, 1);
X2 = mvnrnd(mu2, sigma1, n2); Y2 = ones(n2, 1);

N = n1 + n2;
X = [X1; X2];
Y = [Y1; Y2];
X = [X, -1 * ones(N, 1)]; % Add bias to inputs

train_ratio = .95;
val_ratio = .2; % wrt to train set size

[train_set, val_set, test_set] = split_dataset(X, Y, train_ratio, val_ratio);

%% Input space representation

% Colors
class1_color = [1 0 0];
class2_color = [0 0 1];
mycmap = [255, 115, 0;
          255, 153, 69;
          255, 192, 140;
          255, 255, 255;
          173, 179, 255;
          102, 113, 255;
          36,  51,  255]./255;

% Get data range
x1lims = [floor(min(X(:, 1))) - 1, ceil(max(X(:, 1))) + 1]; 
x2lims = [floor(min(X(:, 2))) - 1, ceil(max(X(:, 2))) + 1];

% Configuration of axes
fig1 = figure('units','normalized','outerposition',[0 0 1 1]);
% fig1 = figure();
ax_scatter = subplot(1, 2, 2);
xlabel(ax_scatter, "x1"); ylabel(ax_scatter, "x2");
colormap(ax_scatter, mycmap);
colorbar(ax_scatter)
set(ax_scatter, 'Xlim', x1lims, ...
         'YLim', x2lims, ...
         'xlimmode','manual', ...
         'ylimmode','manual', ...
         'zlimmode','manual', ...
         'climmode','manual', ...
         'alimmode','manual', ...
         'NextPlot', 'add', ...
         'XGrid', 'on', ...
         'YGrid', 'on');

% Display the train set samples
train_sample1 = scatter(train_set{1}(train_set{2}==0, 1), train_set{1}(train_set{2}==0, 2), ...
    'MarkerEdgeColor', class1_color);
train_sample2 = scatter(train_set{1}(train_set{2}==1, 1), train_set{1}(train_set{2}==1, 2), ...
    'MarkerEdgeColor', class2_color);

% Display the val set samples
val_sample1 = scatter(val_set{1}(val_set{2}==0, 1), val_set{1}(val_set{2}==0, 2), ...
    'filled', 'MarkerFaceColor', class1_color, 'MarkerEdgeColor', [1 1 1], 'Linewidth', 1);
val_sample2 = scatter(val_set{1}(val_set{2}==1, 1), val_set{1}(val_set{2}==1, 2), ...
    'filled', 'MarkerFaceColor', class2_color, 'MarkerEdgeColor', [1 1 1], 'Linewidth', 1);

% Display the test set samples
test_sample1 = scatter(test_set{1}(test_set{2}==0, 1), test_set{1}(test_set{2}==0, 2), ...
    'filled', 'MarkerFaceColor', class1_color, 'MarkerEdgeColor', [0 0 0], 'Linewidth', 1);
test_sample2 = scatter(test_set{1}(test_set{2}==1, 1), test_set{1}(test_set{2}==1, 2), ...
    'filled', 'MarkerFaceColor', class2_color, 'MarkerEdgeColor', [0 0 0], 'Linewidth', 1);

% Initialize handles for future prediction on the train set
train_pred1 = scatter(train_set{1}(train_set{2}==0, 1), train_set{1}(train_set{2}==0, 2), 'x', ...
    'MarkerEdgeColor', class1_color);
train_pred2 = scatter(train_set{1}(train_set{2}==1, 1), train_set{1}(train_set{2}==1, 2), 'x', ...
    'MarkerEdgeColor', class2_color);

% Meshgrid the input space
grid_size = 500;
x1linspace = linspace(x1lims(1), x1lims(2), grid_size); 
x2linspace = linspace(x2lims(1), x2lims(2), grid_size); 
[mesh_x1, mesh_x2] = meshgrid(x1linspace, x2linspace);
mesh_x1 = mesh_x1(:);
mesh_x2 = mesh_x2(:);
mesh = [mesh_x1, mesh_x2, -1 * ones(length(mesh_x1), 1)];

% Initialize handle for background with dummy data
im = image('CData', [1], 'CDataMapping', 'scaled', ...
     'Parent', ax_scatter, 'AlphaData', 0.4, ...
     'XData', ax_scatter.XLim, 'YData', ax_scatter.YLim);

% Set objects order (especially input space in background)
set(ax_scatter, 'Children', ...
    [train_pred1, train_pred2, ...
    train_sample1, train_sample2, ...
    val_sample1, val_sample2, ...
    test_sample1, test_sample2, ...
    im]);

% Legend
% legend([train_pred1, train_pred2, ...
%     train_sample1, train_sample2, ...
%     val_sample1, val_sample2, ...
%     test_sample1, test_sample2], ...
%     {"train_pred1", "train_pred2", ...
%     "train_sample1", "train_sample2", ...
%     "val_sample1", "val_sample2", ...
%     "test_sample1", "test_sample2"})


%% Model parameters

n_epochs = 200;
batch_size = floor(1 * (N * train_ratio) * ( 1 - val_ratio));
lr = 1e-3;
w = randn(size(X, 2), 1);
w = [0 ; 1 ; 7];

cross_entropy = @(Yhat, Y) -Y .* log(Yhat) - (1-Y) .* log(1-Yhat);
sigmoid = @(X) 1 ./ (1 + exp(-X));
cross_entropy_derivative = @(Yhat, Y) (- Y ./ Yhat) + ((1-Y) ./ (1-Yhat));
sigmoid_derivative = @(X) sigmoid(X) .* (1 - sigmoid(X));
prediction = @(X, w) sigmoid(X * w);

%% Learning

% Prepare the plots
train_loss_history = [];  % A new value each batch
val_loss_history = [];    % A new value each epoch
train_loss_linspace = []; % Keep both history synced
val_loss_linspace = 1:n_epochs; % Not really needed
w_history = zeros(n_epochs, length(w));

ax_curves = subplot(1, 2, 1);
train_loss_curve = line(0, 0, 'Color','blue', 'LineWidth',2);  % handle for training loss curve
val_loss_curve = line(0, 0, 'Color','red', 'LineWidth',2);     % handle for validation loss curve
legend(["Training loss", "Validation loss"])
xlabel("Epoch")
ylabel("Cross entropy loss (normalized)")

% Register frames to create a movie
v = VideoWriter('./img/movie.avi');
v.FrameRate = 10;
open(v);

% Learning
for epoch = 1:n_epochs
    
    % Get data
    batches = load_batches(train_set{1}, train_set{2}, batch_size, 1);
    n_batches = length(batches);
    
    for batch_idx = 1 : n_batches
        
        % Get inputs and groud truth
        X_ = batches{batch_idx}{1};
        Y_ = batches{batch_idx}{2};
        
        % Feed-forward
        Yhat = prediction(X_, w);
        
        % Compute loss
        train_loss = mean(cross_entropy(Yhat, Y_));
        train_loss_history = [train_loss_history, train_loss];
        train_loss_linspace = [train_loss_linspace, epoch + batch_idx * (1/n_batches)];
        
        % Compute grad using chain rule
        grad = cross_entropy_derivative(Yhat, Y_) .* sigmoid_derivative(X_ * w) .*  X_;
        
        % Update weights
        w = w - lr * sum(grad)';
        
        % Plot
        train_loss_curve.XData = train_loss_linspace;
        train_loss_curve.YData = train_loss_history;
    end
    
    % Register weights
    w_history(epoch, 1:length(w)) = w;
    
    % Compute validation loss
    val_loss = mean(cross_entropy(prediction(val_set{1}, w), val_set{2}));
    val_loss_history = [val_loss_history, val_loss];
    
    % Plot loss curves
    ax_curves.XLim = [0, epoch];
    ax_curves.YLim = [0, max(quantile(train_loss_history, .95), quantile(val_loss_history, .95))];
    val_loss_curve.XData = val_loss_linspace(1:epoch);
    val_loss_curve.YData = val_loss_history;
    
    % Plot input space
    mesh_pred = prediction(mesh, w);
    mesh_pred_im = reshape(mesh_pred, grid_size, grid_size);
    set(im, 'CData', mesh_pred_im, 'XData', ax_scatter.XLim, 'YData', ax_scatter.YLim);
  
    % Plot decision made on train set
    train_pred = prediction(train_set{1}, w);
    train_pred1.XData = train_set{1}(train_pred<=0.5, 1);
    train_pred1.YData = train_set{1}(train_pred<=0.5, 2);
    train_pred2.XData = train_set{1}(train_pred>0.5, 1);
    train_pred2.YData = train_set{1}(train_pred>0.5, 2);
    
    drawnow;
    
    % Register frames to construct a movie
    writeVideo(v, getframe(fig1))
end

close(v)

figure;
plot(1:n_epochs, w_history, 'Linewidth', 2)
legend({"w1", "w2", "b"})
xlabel("Epochs")
grid on