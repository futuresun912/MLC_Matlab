function [Pre_Labels,Outputs] = metaCC(train_data,train_target,test_data)
%METACC Train classifier chains over meta-labels
%   此处显示详细说明

% 0. Check if #label is enough large
num_label = size(train_target,1);
if num_label < 6 
    [Pre_Labels,Outputs] = CCridge(train_data,train_target,test_data);
    return;
end

% 1. Group labels into meta-labels
% 1.1 Spectral clustering
% % Label similarity
% A1 = 1 - pdist(train_target,'jaccard');

% % % Instance similarity
% % mean_data = bsxfun(@rdivide,train_target*train_data,sum(train_target,2));
% % A2 = exp(-pdist(mean_data));
% % 
% % % Affinity matrix
% % A = squareform(A1.*round(A2));
% 
% A = squareform(A1);
% 
% % Apply spectral clustering 
% meta_size = 5;
% k = ceil(num_label/meta_size);
% [C, ~, ~] = SpectralClustering(A, k, 2);

% 1.2 k-means
meta_size = 5;
k = ceil(num_label/meta_size);
C = litekmeans(train_target,k);

% 2. Build classifier chains over meta-labels
% Perform CC over labels within each meta-label
num_test = size(test_data,1);
Outputs = zeros(num_label,num_test);
Pre_Labels = zeros(num_label,num_test);
for i = 1:k 
   
    meta_target = train_target((C==i),:);
    temp_size = size(meta_target,1);
    
    if temp_size == 1
        [Pre_Labels((C==i),:),Outputs((C==i),:)] = BRridge(train_data,meta_target,test_data);
    elseif temp_size == 2
        [Pre_Labels((C==i),:),Outputs((C==i),:)] = CCridge(train_data,meta_target,test_data);
    else
    [Pre_Labels((C==i),:),Outputs((C==i),:)] = EMLC(train_data,meta_target,test_data,2,@CCridge);
    end
    
%     [Pre_Labels((C==i),:),Outputs((C==i),:)] = CCridge(train_data,meta_target,test_data);
    
    train_data = [train_data,meta_target'];
    test_data = [test_data,Pre_Labels((C==i),:)'];
    
    % encode meta-label for training
  
%     
%     meta_target = train_target((C==i),:);
%     meta_size = 2^size(meta_target,1) - 1;
%        
%     if meta_size > 1
%         meta_index = bi2de(meta_target') + 1;
%         meta_num = size(meta_target,2);
%         
%         % Encoding into high-dimensional space
%         LP_target = zeros(meta_num,meta_size+1);
%         index = sub2ind(size(LP_target),1:meta_num,meta_index');
%         LP_target(index) = 1;
%         
%         % Training and prediction
%         [Pre_temp,~] = BRridge(train_data,LP_target',test_data);
%         
%         % Decoding
%         [row,column] = find(Pre_temp==1);
%         Pre_final = zeros(num_test,size(meta_target,1));
%         Pre_final(column,:) = de2bi(row-1);
%         
%         % Output
%         Pre_Labels((C==i),:) = Pre_final';
%         
%         train_data = [train_data,meta_target'];
%         test_data = [test_data,Pre_final];
%     else
%         [Pre_Labels((C==i),:),Outputs((C==i),:)] = BRridge(train_data,meta_target,test_data);
%     end
%     
    
    
    
%    
%     if meta_size > 1
%         meta_target = bi2de(meta_target');
%         
%         [~,Pre_Temp] = BRridge(train_data,meta_target',test_data);
%         
%         % decode meta-label for prediction
%         Pre_Temp = round(Pre_Temp);
%         Pre_Temp(Pre_Temp>meta_size) = meta_size;
%         Pre_Temp(Pre_Temp<0) = 0;
%         
%         Pre_Labels((C==i),:) = (de2bi(Pre_Temp'))';
%         
%         train_data = [train_data,meta_target];
%         test_data = [test_data,Pre_Temp'];
%     else
%         [Pre_Labels((C==i),:),Outputs((C==i),:)] = BRridge(train_data,meta_target,test_data);
%     end
    
   

    
%         num = size(meta_target,1);
%     temp_data = bi2de(meta_target') ./ (2.^num);
%     train_data = [train_data,temp_data];
% %     train_data = [train_data,bi2de(meta_target')];
%     temp_temp = bi2de(Pre_Labels((C==i),:)') ./ (2.^num);
%     test_data = [test_data,temp_temp];
    
end

end

