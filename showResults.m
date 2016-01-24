
% Add sub-folders containing functions
addpath('data','evaluation');
addpath(genpath('method'));

% Load a multi-label dataset
dataset = 'scene';
load([dataset,'.mat']);

% Make experimental resutls repeatly
rng('default');

% Perform n-fold cross validation and obtain evaluation results
num_fold = 5; num_metric = 3; num_method = 5;
num_cluster = 8; model = @CCridge;
indices = crossvalind('Kfold',size(data,1),num_fold);
Results = zeros(num_metric+1,num_fold,num_method);
Final_mean = zeros(num_metric+1,num_method,num_cluster-1);
Final_std = zeros(num_metric+1,num_method,num_cluster-1);

for k = 1:(num_cluster-1)
    for i = 1:num_fold
        test = (indices == i); train = ~test;
        
        % The CC method with Ridge Regression
        tic;
        [Pre_Labels,~] = CCridge(data(train,:),target(:,train'),data(test,:));
        Results(1,i,1) = toc;
        [ExactM,HamS,~,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
        Results(2:end,i,1) = [ExactM,HamS,MicroF1];
        
        % The CBMLC method with Ridge Regression
        tic;
        [Pre_Labels,~] = CBMLC(data(train,:),target(:,train'),data(test,:),k,model);
        Results(1,i,2) = toc;
        [ExactM,HamS,~,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
        Results(2:end,i,2) = [ExactM,HamS,MicroF1];
        
        % The CLMLC method only with the first stage
        tic;
        [Pre_Labels,~] = CLMLCv1(data(train,:),target(:,train'),data(test,:),k,model);
        Results(1,i,3) = toc;
        [ExactM,HamS,~,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
        Results(2:end,i,3) = [ExactM,HamS,MicroF1];
        
        % The CLMLC method only with the second stage
        tic; percent = [0.5,0.8,0.8]; num_ite = 20;
        [Pre_Labels,~] = EnMLC(data(train,:),target(:,train'),data(test,:),percent,num_ite,model);
        Results(1,i,4) = toc;
        [ExactM,HamS,~,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
        Results(2:end,i,4) = [ExactM,HamS,MicroF1];
        
        % The CLMLC method
        tic; 
        [Pre_Labels,~] = CLMLCv1(data(train,:),target(:,train'),data(test,:),k,@EnMLC);
        Results(1,i,5) = toc;
        [ExactM,HamS,~,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
        Results(2:end,i,5) = [ExactM,HamS,MicroF1];
        
    end
    
    meanResults = squeeze(mean(Results,2));
    stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2))); 
    Final_mean(:,:,k) = meanResults;
    Final_std(:,:,k) = stdResults;
end;

% Display the experimental results
dg = [0 0.4 0];
dr = [0.8 0 0];
db = [0 0.3 0.7];
dy = [0.6 0 0.6];	
dp = [0.502 0 0];

y_mean = zeros((num_metric+1),(num_cluster-1),num_method);
y_std = zeros((num_metric+1),(num_cluster-1),num_method);
for i = 1:num_method
    y_mean(:,:,i) = Final_mean(:,i,:);
    y_std(:,:,i) = Final_std(:,i,:);
end

x_axis = 2:num_cluster;
metric_str = {'Execution time','Exact-Match', 'Hamming-Score','Micro-F1'};
for i = 1:(num_metric+1)
    figure('Position', [50 50 800 600]);
    plot(x_axis,y_mean(i,:,1),'-x', 'MarkerEdgeColor', db, 'Color', db,  'LineWidth', 3);
    hold on;
    plot(x_axis,y_mean(i,:,2),'-o','MarkerEdgeColor', dg, 'Color', dg,  'LineWidth', 3);
    hold on;
    plot(x_axis,y_mean(i,:,3),'-s', 'MarkerEdgeColor', dr, 'Color', dr,  'LineWidth', 3);
    hold on;
    plot(x_axis,y_mean(i,:,4),'-d', 'MarkerEdgeColor', dp, 'Color', dp,  'LineWidth', 3);
    hold on;
    plot(x_axis,y_mean(i,:,5),'-*', 'MarkerEdgeColor', dy, 'Color', dy,  'LineWidth', 3);
    hold on;
    
    xlabel('k','FontSize', 18);
    ylabel(metric_str(i), 'FontSize', 18);
    lgd = legend('CC','CBMLC','CLMLC-1','CLMLC-2','CLMLC');
    set(lgd,  'fontsize', 14);
end





