function [Pre_Labels,Outputs] = CBMLC(train_data,train_target,test_data,num_cluster,model)
%CBMLC Clustering Based Multi-Label Classification
%   此处显示详细说明

% % kmeans++
% [trainInd,centroid] = kmeanspro(train_data',num_cluster);
% trainInd = trainInd';
% centroid = centroid';

% % Lite kmeans
% [trainInd,centroid] = litekmeans(train_data,num_cluster,'MaxIter',50);

% % landmark-based spectral clustering
% opts.r = 5;
% opts.kmMaxIter = 10;
% trainInd = LSC(train_data,num_cluster,opts);
% num_data = size(train_data,1);
% centroid = sparse(trainInd,1:num_data,1,num_cluster,num_data,num_data) * train_data;
% count = zeros(1,num_cluster);
% for i = 1:num_cluster
%     count(i) = size(trainInd(trainInd==i),1);
% end
% centroid = bsxfun(@rdivide,centroid,count');

% kmeans -- the MATLAB version
[trainInd,centroid] = kmeans(train_data,num_cluster,'EmptyAction','singleton','OnlinePhase','off','Display','off');

% Find the nearest cluster for each test instance
v1 = dot(test_data,test_data,2); 
v2 = dot(centroid,centroid,2);
D = bsxfun(@plus,v1,v2') - 2*(test_data*centroid');
[~,testInd] = min(D,[],2);

% Apply local MLC on each cluster 
Pre_Labels = zeros(size(train_target,1),size(test_data,1));
Outputs = zeros(size(train_target,1),size(test_data,1));
if any(isequal(model,@EnMLC))
    percent = [0.5,0.8,0.8];
    for i = 1:num_cluster
        local_trainInd = (trainInd == i);
        local_testInd = (testInd == i);
        [Pre_Labels(:,local_testInd'),Outputs(:,local_testInd')] = model(train_data(local_trainInd,:),...
            train_target(:,local_trainInd'),test_data(local_testInd,:),percent,20,@CCridge);
    end
else
    for i = 1:num_cluster
        local_trainInd = (trainInd == i);
        local_testInd = (testInd == i);
        [Pre_Labels(:,local_testInd'),Outputs(:,local_testInd')] = model(train_data(local_trainInd,:),...
            train_target(:,local_trainInd'),test_data(local_testInd,:));
    end
end
end
