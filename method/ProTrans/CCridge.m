function [Pre_Labels,Outputs] = CCridge(train_data,train_target,test_data)
%CCRIDGE 此处显示有关此函数的摘要
%   此处显示详细说明

% Ridge parameter
lambda = 0.1;

% Randomly generate a chain order
num_label = size(train_target,1);
chain = randperm(num_label);

% Ridge Regression
pa = [];
ww = cell(1,num_label);
num_test = size(test_data,1);
Outputs = zeros(num_label,num_test);
Pre_Labels = zeros(num_label,num_test);
for i = chain
    ww{i} = ridgereg(train_target(i,:)',[train_data train_target(pa,:)'],lambda);
    Outputs(i,:) = [ones(num_test,1),test_data,Pre_Labels(pa,:)'] * ww{i};
    Pre_Labels(i,:) = round(Outputs(i,:));
    pa = [pa i];
end

end

