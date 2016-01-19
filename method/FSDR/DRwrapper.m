function [train_data,test_data] = DRwrapper(train_data,test_data,train_target,alg)
%DRWRAPPER A wrapper of several FSDR approaches
%   此处显示详细说明

% Applying a FSDR Approach
if (strcmp(alg,'HSL'))
    alg = [];
    alg.Lap_type = 'clique';
    alg.alg = '2s';
    alg.reg_2norm = 1;
    W = HSL(train_data',train_target,alg);
    train_data = train_data * W;
    test_data = test_data * W;
elseif (strcmp(alg,'OPLS'))
    alg = [];
    alg.alg = '2s';
    alg.reg_2norm = 1;
    W = OPLS(train_data',train_target,alg);
    train_data = train_data * W;
    test_data = test_data * W;
elseif (strcmp(alg,'PCA'))
    [train_data,test_data] = PCA(train_data,test_data,0.3);
else
    disp('ERROR, unavailable DR approach');
    return;
end

end

