function  Pre_Labels = LPridge(train_data,train_target,test_data)
%LPRIDG Label Powerset with Ridge Regression
%  incomplete

k = 4;
C = litekmeans(train_target,k);
Pre_Labels = zeros(size(train_target,1),size(test_data,1));

for i = 1:k

    subset_target = train_target((C==i),:);
    
    [num_label,num_data] = size(subset_target);
    num_test = size(test_data,1);
    
    N = num_data;
    L = num_label;
    n = num_test;
    target = subset_target';
    
    % Encode train_target
    encode_index = bi2de(target) + 1;
    encode_target = zeros(N,2^L);
    encode_index2 = sub2ind(size(encode_target),1:N,encode_index');
    encode_target(encode_index2) = 1;
    
    % Prediction
    [pre_temp,~] = BRridge(train_data,encode_target',test_data);
    
    % Decode pre_temp
    encode_target = pre_temp';
    decode_target = zeros(n,L);
    [row,column] = find(encode_target==1);
    decode_target(row,:) = de2bi(column-1);
    
    % Output
    Pre_Labels((C==i),:) = decode_target';
end

% %1. Encoding label space (train_target)
% % Map its dimension from num_label into 2^num_label
% encode_target = zeros(2^num_label,num_data);
% encode_index = bi2de(train_target') + 1;
% encode_index = sub2ind(size(encode_target),encode_index,(1:num_data)');
% encode_target(encode_index) = 1;
% 
% %2. Apply BR in the dimensional label space
% [pre_temp,~] = BRridge(train_data,encode_target,test_data);
% 
% %3. Decoding the label space
% decode_pre = zero(num_label,num_test);
% [row,column] = find(pre_temp);
% decode_pre(:,column) = de2bi(row-1);
% 
% %4. Output
% Pre_Labels = decode_pre;

end

