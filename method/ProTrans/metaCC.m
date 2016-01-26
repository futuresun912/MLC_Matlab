function [Pre_Labels,Outputs] = metaCC(train_data,train_target,test_data)
%METACC Train classifier chains over meta-labels
%   此处显示详细说明

num_label = size(train_target,1);

if ( num_label < 6 )
%     [Pre_Labels,Outputs] = EMLC(train_data,train_target,test_data,3,@CCridge);
    [Pre_Labels,Outputs] = CCridge(train_data,train_target,test_data);
    return;
end

% 1. Group labels into meta-labels
% Compute the affinity matrix (MI(Y) and dist(X))

% Normalized mutual information
% A = zeros(num_label,num_label);
% for i = 1:num_label
%     for j = (i+1):num_label
%         A(i,j) = nmi(train_target(i,:),train_target(j,:));
%     end
% end

% Label similarity
A1 = pdist(train_target,'jaccard');

% Instance similarity
% mean_data = zeros(size(train_data,2),num_label);
mean_data = zeros(size(train_data,2),num_label);
for i = 1:num_label
    index = (train_target(i,:) == 1);
    mean_data(:,i) = mean(train_data(index,:));
%     mean_data(:,i,1) = mean(train_data(index,:));
%     mean_data(:,i,2) = mean(train_data(~index,:));
end

A2 = exp(-pdist(mean_data','cityblock'));
% A2 = exp(-pdist(mean_data(:,:,1)','cityblock')./pdist(mean_data(:,:,2)','cityblock'));
% th = 0.5;
% A2(A2>th) = 1;
% A2(A2<=th) = 0;
A = squareform((1-A1).*round(A2));

%  A = squareform((1-A1));


% Apply spectral clustering
meta_size = 5;
k = ceil(num_label/meta_size);
[C, ~, ~] = SpectralClustering(A, k, 2);

% % kmeans
% meta_size = 5;
% k = round(num_label/meta_size);
% C = litekmeans(train_target,k);

% 2. Build classifier chains over meta-labels
% Perform CC over labels within each meta-label
num_test = size(test_data,1);
Outputs = zeros(num_label,num_test);
Pre_Labels = zeros(num_label,num_test);
for i = 1:k
    
    meta_target = train_target((C==i),:);
    [Pre_Labels((C==i),:),Outputs((C==i),:)] = CCridge(train_data,meta_target,test_data);
%     [Pre_Labels((C==i),:),Outputs((C==i),:)] = EMLC(train_data,meta_target,test_data,3,@CCridge);
    
    train_data = [train_data,meta_target'];
    test_data = [test_data,Pre_Labels((C==i),:)'];
    
%     num = size(meta_target,1);
%     temp_data = bi2de(meta_target') ./ (2.^num);
%     train_data = [train_data,temp_data];
% %     train_data = [train_data,bi2de(meta_target')];
%     temp_temp = bi2de(Pre_Labels((C==i),:)') ./ (2.^num);
%     test_data = [test_data,temp_temp];
    
end


end

