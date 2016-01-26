
% Add sub-folders containing functions
addpath('data','evaluation');
addpath(genpath('method'));

% Load a multi-label dataset
dataset = 'languagelog';
load([dataset,'.mat']);

% Make experimental resutls repeatly
rng('default'); 

% Perform n-fold cross validation and obtain evaluation results
num_fold = 5; num_metric = 4; num_method = 20;
indices = crossvalind('Kfold',size(data,1),num_fold);
Results = zeros(num_metric+1,num_fold,num_method);
num_cluster = 20;
for i = 1:num_fold
    disp(['Fold ',num2str(i)]);
    test = (indices == i); train = ~test; 
    
%     % The BR method with Ridge Regression
%     tic;
%     [Pre_Labels,~] = BRridge(data(train,:),target(:,train'),data(test,:));
%     Results(1,i,1) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,1) = [ExactM,HamS,MacroF1,MicroF1];

%     % The CC method with Ridge Regression
%     tic;
%     [Pre_Labels,~] = CCridge(data(train,:),target(:,train'),data(test,:));
%     Results(1,i,2) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,2) = [ExactM,HamS,MacroF1,MicroF1];

%     % The ensemble MLC method
%     tic; percent = [0.5,0.8,0.8]; num_ite = 10; model = @CCridge;
%     [Pre_Labels,~] = EnMLC(data(train,:),target(:,train'),data(test,:),percent,num_ite,model);
%     Results(1,i,3) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,3) = [ExactM,HamS,MacroF1,MicroF1];

%     % The CBMLC method with Ridge Regression
%     tic; model = @CCridge;
%     [Pre_Labels,~] = CBMLC(data(train,:),target(:,train'),data(test,:),num_cluster,model);
%     Results(1,i,4) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,4) = [ExactM,HamS,MacroF1,MicroF1];
  
%     % The CLMLC method with Ridge Regression
%     tic; model = @CCridge;
%     [Pre_Labels,~] = CLMLCv1(data(train,:),target(:,train'),data(test,:),num_cluster,model);
%     Results(1,i,5) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,5) = [ExactM,HamS,MacroF1,MicroF1];

%     % The BR method with PCA
%     tic; alg = 'PCA';
%     [Pre_Labels,~] = BR_DR(data(train,:),target(:,train'),data(test,:),alg);
%     Results(1,i,6) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,6) = [ExactM,HamS,MacroF1,MicroF1];

%     % The BR method with OPLS
%     tic; alg = 'OPLS';
%     [Pre_Labels,~] = BR_DR(data(train,:),target(:,train'),data(test,:),alg);
%     Results(1,i,7) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,7) = [ExactM,HamS,MacroF1,MicroF1];

%     % The CC method with HLS
%     tic; alg = 'HSL';
%     [Pre_Labels,~] = CC_DR(data(train,:),target(:,train'),data(test,:),alg);
%     Results(1,i,8) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,8) = [ExactM,HamS,MacroF1,MicroF1];
    
%     % BR with SVM
%     tic;
%     [Pre_Labels,~] = BRsvm(data(train,:),target(:,train'),data(test,:),target(:,test'));
%     Results(1,i,9) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,9) = [ExactM,HamS,MacroF1,MicroF1];

%     % CC with SVM
%     tic;
%     [Pre_Labels,~] = CCsvm(data(train,:),target(:,train'),data(test,:),target(:,test'));
%     Results(1,i,10) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,10) = [ExactM,HamS,MacroF1,MicroF1];

%     % CPLST - Label Space Dimension Reduction
%     tic; ratio = 0.4;
%     Pre_Labels = CPLST(data(train,:),target(:,train'),data(test,:),ratio);
%     Results(1,i,11) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,11) = [ExactM,HamS,MacroF1,MicroF1];
%     
%     % PLST - Label Space Dimension Reduction
%     tic; ratio = 0.5;
%     Pre_Labels = PLST(data(train,:),target(:,train'),data(test,:),ratio);
%     Results(1,i,12) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,12) = [ExactM,HamS,MacroF1,MicroF1];

%     % FaIE - Feature-aware Implicit Label Space Encoding
%     tic; ratio = 0.5;
%     Pre_Labels = FaIE(data(train,:),target(:,train'),data(test,:),ratio);
%     Results(1,i,13) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,13) = [ExactM,HamS,MacroF1,MicroF1];

%     % The CLMLC method with Ridge Regression
%     tic; 
%     num_cluster = 20; model = @EnMLC;
%     [Pre_Labels,~] = CLMLCv1(data(train,:),target(:,train'),data(test,:),num_cluster,model);
%     Results(1,i,14) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,14) = [ExactM,HamS,MacroF1,MicroF1];

%     % The CLMLC method with Ridge Regression
%     tic; model = @BRridge;
%     [Pre_Labels,~] = CLMLCv1(data(train,:),target(:,train'),data(test,:),num_cluster,model);
%     Results(1,i,15) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,15) = [ExactM,HamS,MacroF1,MicroF1];
    
%     % The CLMLC method with Ridge Regression
%     tic;  model = @EMLC;
%     [Pre_Labels,~] = CLMLCv1(data(train,:),target(:,train'),data(test,:),num_cluster,model);
%     Results(1,i,16) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,16) = [ExactM,HamS,MacroF1,MicroF1];

%     % The CC method with Ridge Regression
%     tic;
%     [Pre_Labels,~] = rCC(data(train,:),target(:,train'),data(test,:),10);
%     Results(1,i,17) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,17) = [ExactM,HamS,MacroF1,MicroF1];
    
%     % The ensemble MLC method
%     tic;  model = @CCridge;
%     [Pre_Labels,~] = EMLC(data(train,:),target(:,train'),data(test,:),5,model);
%     Results(1,i,18) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,18) = [ExactM,HamS,MacroF1,MicroF1];

%     % The ensemble MLC method
%     tic; 
%     [Pre_Labels,~] = metaCC(data(train,:),target(:,train'),data(test,:));
%     Results(1,i,19) = toc;
%     [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
%     Results(2:end,i,19) = [ExactM,HamS,MacroF1,MicroF1];

    % The CLMLC method with Ridge Regression
    tic; model = @metaCC;
    [Pre_Labels,~] = CLMLCv1(data(train,:),target(:,train'),data(test,:),num_cluster,model);
    Results(1,i,20) = toc;
    [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
    Results(2:end,i,20) = [ExactM,HamS,MacroF1,MicroF1];

end
ignore = [1:19];  Results(:,:,ignore) = [];
meanResults = squeeze(mean(Results,2));
stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

% Save the evaluation results
filename=strcat('results/',dataset,'.mat');
save(filename,'meanResults','stdResults','-mat');

% Show the experimental results
disp(dataset);
disp(meanResults);
figure('Position', [300 300 800 500]);
bar(meanResults);
str1 = {'Execution time';'Exact match';'Hamming Score';'Macro F1';'Micro F1'};
set(gca,'XTickLabel',str1);
xlabel('Metric','FontSize', 14); ylabel('Performance','FontSize', 14);
str2 = {'BR','CC','EnMLC','CBMLC','CLMLC-CC','BRPCA','BROPLS','MLHSL','BRSVM','CCSVM',...
    'CPLST','PLST','FaIE','CLMLC','CLMLC-DR','CLMLC-En','rCC','ECC','metaCC','CLMCL-meta'}; 
str2(ignore) = [];
legend(str2,'Location','NorthWest');
hold on;
title(dataset,'FontSize', 18);







