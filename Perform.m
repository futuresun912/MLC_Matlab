
%% Add sub-folders containing functions
addpath('data','evaluation');
addpath(genpath('method'));

%% Load a multi-label dataset
dataset = 'scene';
load([dataset,'.mat']);

%% Set parameters
num_cluster = 5;    % CBMLC,CLMLC
ratio = 0.8;         % CPLST

% Make experimental resutls repeatly
rng(1); 

% Randomly select part of data
max_num = 6000;
if size(data,1) > max_num
    nRows = size(data,1); 
    nSample = max_num;
    rndIDX = randperm(nRows);
    index = rndIDX(1:nSample);
    data = data(index, :);
    target = target(:,index);
end

%% Perform n-fold cross validation and obtain evaluation results
num_fold = 5; num_metric = 4; num_method = 5;
indices = crossvalind('Kfold',size(data,1),num_fold);
Results = zeros(num_metric+1,num_fold,num_method);
for i = 1:num_fold
    disp(['Fold ',num2str(i)]);
    test = (indices == i); train = ~test; 

    % CC with Ridge Regression
    tic; [Pre_Labels,~] = CCridge(data(train,:),target(:,train'),data(test,:));
    Results(1,i,1) = toc;
    [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
    Results(2:end,i,1) = [ExactM,HamS,MacroF1,MicroF1];

    % CBMLC with CC
    tic; [Pre_Labels,~] = CBMLC(data(train,:),target(:,train'),data(test,:),num_cluster,@CCridge);
    Results(1,i,2) = toc;
    [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
    Results(2:end,i,2) = [ExactM,HamS,MacroF1,MicroF1];
  
    % MLHLS with CC
    tic; [Pre_Labels,~] = CC_DR(data(train,:),target(:,train'),data(test,:),'OPLS');
    Results(1,i,3) = toc;
    [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
    Results(2:end,i,3) = [ExactM,HamS,MacroF1,MicroF1];

    % CPLST with CC
    tic; Pre_Labels = CPLST(data(train,:),target(:,train'),data(test,:),ratio);
    Results(1,i,4) = toc;
    [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
    Results(2:end,i,4) = [ExactM,HamS,MacroF1,MicroF1];

    % CLMLC with metaCC
    tic; [Pre_Labels,~] = CLMLC(data(train,:),target(:,train'),data(test,:),num_cluster,@metaCC);
    Results(1,i,5) = toc;
    [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
    Results(2:end,i,5) = [ExactM,HamS,MacroF1,MicroF1];

end
ignore = [];  Results(:,:,ignore) = [];
meanResults = squeeze(mean(Results,2));
stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

%% Save the evaluation results
filename=strcat('results/',dataset,'.mat');
save(filename,'meanResults','stdResults','-mat');

%% Show the experimental results
disp(dataset);
disp(meanResults);
% figure('Position', [300 300 800 500]);
% bar(meanResults);
% str1 = {'Execution time';'Exact match';'Hamming Score';'Macro F1';'Micro F1'};
% set(gca,'XTickLabel',str1);
% xlabel('Metric','FontSize', 14); ylabel('Performance','FontSize', 14);
% str2 = {'CC','CBMLC','MLHSL','CPLST','CLMLC'}; 
% str2(ignore) = [];
% legend(str2,'Location','NorthWest');
% hold on;
% title(dataset,'FontSize', 18);







