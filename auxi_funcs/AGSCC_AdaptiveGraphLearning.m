function  [Sx,Kmat,wm_x,wm_x_rel] = AGSCC_AdaptiveGraphLearning(t1_feature,opt)
Niter_AGL = opt.Niter_AGL;
eta = opt.eta;
N = size(t1_feature,2);
M = size(t1_feature,3);
kmax = round(sqrt(N));
kmin = round(sqrt(N)/10);
wm_x = ones(1,M)/M;
for m = 1:M
    distX_temp(:,:,m) = L2_distance_1(t1_feature(:,:,m),t1_feature(:,:,m));
end
for iter = 1: Niter_AGL
    wm_old = wm_x;
    distX = WeightedDistance(distX_temp,wm_x);
    [valuex, idx] = sort(distX,2);
    valuex = valuex(:,1:kmax);
    idx = idx(:,1:kmax);
    degree = tabulate(idx(:));
    Kmat = degree(:,2);
    Kmat(Kmat>=kmax)=kmax;
    Kmat(Kmat<=kmin)=kmin;
    if length(Kmat)<N
        Kmat(length(Kmat)+1:N) = kmin;
    end
    Sx = zeros(N);
    for i = 1:size(Sx,1)
        k = Kmat(i)-1;
        id_x = idx(i,1:k+1);
        di = valuex(i,1:k+1);
        Sx(i,id_x) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end
    for m = 1:M
        gc(m) = sum(sum(Sx.*distX_temp(:,:,m)));
    end
    up = gc.^(1/(eta-1));
    dgc =  gc.^(eta/(eta-1));
    down = (sum(dgc))^(1/eta);
    wm_x = up/down;
    wm_x_rel(iter) = norm(wm_x - wm_old,2)/(norm(wm_old,2)+eps);   
end