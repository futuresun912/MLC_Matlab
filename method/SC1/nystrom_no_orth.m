function cluster_labels = nystrom_no_orth(data, num_samples, sigma, num_clusters)
%NYSTROM Spectral clusterina using the Nystrom method.
%
%   Input  : data           : N-by-D data matrix, where N is the number of data,
%                             D is the number of dimensions
%            num_samples    : number of random samples
%            sigma          : sigma value used in computing similarity
%            num_clusters   : number of clusters
%
%   Output : cluster_labels : N-by-1 vector containing cluster labels
%            evd_time       : running time for eigendecomposition
%            kmeans_time    : running time for k-means
%            total_time     : total running time

%
% Randomly select samples
%

num_rows = size(data, 1);
permed_index = randperm(num_rows);
sample_data = data(permed_index(1:num_samples), :);
other_data = data(permed_index(num_samples+1:num_rows), :);
clear data;


%
% Calculate the euclidean distance between samples themselves
%

A = euclidean(sample_data', sample_data');
A = single(A);


%
% Calculate the euclidean distance between samples and other points
%

B = euclidean(sample_data', other_data');
B = single(B);
clear sample_data other_data;


%
% Convert distance matrix to similarity matrix: S = exp^(-(dist^2 / 2*sigma^2))
%

A = single(exp(-(A.*A) ./ (2*sigma*sigma)));
B = single(exp(-(B.*B) ./ (2*sigma*sigma)));


%
% Normalize A and B using row sums of W, where W = [A B; B' B'*A^-1*B].
% Let d1 = [A B]*1, d2 = [B' B'*A^-1*B]*1, dhat = sqrt(1./[d1; d2]).
%

B_T = B';
d1 = sum(A, 2) + sum(B, 2);
d2 = sum(B_T, 2) + B_T*(pinv(A)*sum(B, 2));
dhat = sqrt(1./[d1; d2]);
A = A .* (dhat(1:num_samples)*dhat(1:num_samples)');
m = num_rows - num_samples;
B1 = dhat(1:num_samples)*dhat(num_samples+(1:m))';
B = B .* B1;
clear W d1 d2 B1 dhat;


%
% Do eigendecomposition
%

B_T = B';
W = single(zeros(size(A, 1)+size(B_T, 1), size(A, 2)));
W(1:size(A, 1), :) = A;
W(size(A, 1)+1:size(W, 1), :) = B_T;
[U L] = eig(A);
[val ind] = sort(diag(L), 'descend');
U = U(:, ind); % in decreasing order
L = L(ind, ind); % in decreasing order
clear A R BBT;
W = sqrt(num_samples/size(W, 1))*W;
V = W*U(:, 1:num_clusters)*pinv(L(1:num_clusters, 1:num_clusters));
clear W Asi L U;


%
% Do k-means
%

% Normalize each row to be of unit length
sq_sum = sqrt(sum(V.*V, 2)) + 1e-20;
U = V ./ repmat(sq_sum, 1, num_clusters);
clear sq_sum V;
cluster_labels = k_means(U, [], num_clusters);
% Restore cluster_labels in original order
cluster_labels(permed_index) = cluster_labels;
clear permed_index;





%--------------------------------------------------------------------------

function d = euclidean(a, b)
%EUCLIDEAN Compute the Euclidean distance matrix between two matrices.
%   This function is designed for processing very large data using divide-
%   and-conquer technique.
%
%   Input : a : D-by-M data matrix, where D is the number of dimensions,
%               M is the number of data
%           b : D-by-N data matrix, where D is the number of dimensions,
%               N is the number of data
%   Output: d : M-by-N matrix of the Euclidean distance between a and b.

%
% Calculate a^2, b^2, here we assume b is larger than a
%
aa = single(full(sum(a.*a, 1)));
bb = single(full(sum(b.*b, 1)));

%
% Do a*b in several steps instead of once because of memory limitation
%
two_ab = single(zeros(size(aa, 2), size(bb, 2)));
% Select at most 10000 instances of b for a*b per iteration
num_iter = ceil(size(bb, 2)/10000);
for i = 1:num_iter
  start_index = 1 + (i-1)*10000;
  end_index = min(i*10000, size(bb, 2));
  abtmp = single(full(a'*b(:, start_index:end_index)));
  two_ab(:, start_index:end_index) = 2*abtmp;
end % Now we have entire ab
clear a b abtmp;

d = bb(ones(size(aa, 2), 1), :);
d = d - two_ab; % Now we have d = b^2 - 2ab
clear two_ab;

ff = aa';
ff = ff(:, ones(size(bb, 2), 1));
d = d + ff; % Now we have d = a^2 + b^2 -2ab
clear aa bb ff;
d = sqrt(d);
