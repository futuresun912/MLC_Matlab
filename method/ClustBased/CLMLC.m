function [Pre_Labels,Outputs] = CLMLC(train_data,train_target,test_data,num_cluster,model)
%CLMLC Clustering-based Local Multi-Label Classification
%   此处显示详细说明

%% Perform FSDR on the feature space
alg = 'OPLS';
[train_data,test_data] = DRwrapper(train_data,test_data,train_target,alg);
% Remove components with low variance
[train_data,test_data] = PCA(train_data,test_data,0.8);

%% Perfrom CBMLC on the subspace
[Pre_Labels,Outputs] = CBMLC(train_data,train_target,test_data,num_cluster,model);


end


