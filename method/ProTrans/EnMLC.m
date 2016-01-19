function [Pre_Labels,Outputs] = EnMLC(train_data,train_target,test_data,percent,m,model)
%ENMLC Ensemble method on MLC
%   此处显示详细说明

% Get the size of the dataset
[num_int,num_att] = size(train_data);
num_lab = size(train_target,1);

% Get the sampleing value of d, L and N
rndNUM = round([num_att,num_lab,num_int].*percent);
d = rndNUM(1); L = rndNUM(2); N = rndNUM(3);

% Perform the ensemble of models
% Pre_Labels = zeros(num_lab,size(test_data,1));
Outputs = zeros(num_lab,size(test_data,1));
count = zeros(1,num_lab);
for i = 1:m
    rndIDX_att = randperm(num_att);
    rndIDX_lab = randperm(num_lab);
    rndIDX_int = randperm(num_int);
    [Temp_Labels,~] = model(train_data(rndIDX_int(1:N),rndIDX_att(1:d)'),...
        train_target(rndIDX_lab(1:L),rndIDX_int(1:N)'),test_data(:,rndIDX_att(1:d)'));
%     Pre_Labels(rndIDX_lab(1:L),:) = Pre_Labels(rndIDX_lab(1:L),:)+ Temp_Labels;
    Outputs(rndIDX_lab(1:L),:) = Outputs(rndIDX_lab(1:L),:)+ Temp_Labels;
%     temp_count = zeros(1,num_lab);
%     temp_count(rndIDX_lab(1:L)) = 1;
%     count = count + temp_count;
%     count(rndIDX_lab(1:L)) = bsxfun(@plus,count(rndIDX_lab(1:L)),1);
    count(rndIDX_lab(1:L)) = count(rndIDX_lab(1:L)) + ones(1,L);
end
Outputs = bsxfun(@rdivide,Outputs,count');
Pre_Labels = round(Outputs);
end

