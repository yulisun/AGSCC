function  wd = WeightedDistance(dist,w)
[~,N,M] = size(dist);
wd = zeros (N,N);
for m = 1:M
    wd = dist(:,:,m) * w(m) + wd;
end
    
    
