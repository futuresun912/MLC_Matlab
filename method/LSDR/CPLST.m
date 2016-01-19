function Yt_pred = CPLST(train_data,train_target,test_data,M)
%CPLST Conditional principal label space transformation
%   此处显示详细说明

lambda = 0.1;

% CPLST encoding
[Z,Vm,shift] = cplst_encode(train_target',M,train_data,lambda);

% Linear Ridge Regression
ww = ridgereg(Z,train_data,lambda);
Zt_pred = [ones(size(test_data,1),1) test_data] * ww;

% Round-based Decoding
[Yt_pred, ~] = round_linear_decode(Zt_pred, Vm, shift);
  
end

