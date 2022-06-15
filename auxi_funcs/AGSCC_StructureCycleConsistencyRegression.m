function [Xpie,Zy,delty,wm_y,RELy] = AGSCC_StructureCycleConsistencyRegression(t1_feature,t2_feature,Sx,Kmat,wm_x,opt)
M = size(t1_feature,3);
Niter_SCC = opt.Niter_SCC;
beta = opt.beta;
lambda = opt.lambda;
gamma = opt.gamma ;
eta = opt.eta;
mu = opt.mu;
N = size(t1_feature,2);
Zy = t2_feature;
Xpie = t1_feature;
delty =  zeros(size(t2_feature));
Wy = zeros(size(t2_feature));
wm_y = wm_x;
for m = 1:M
    distZy_temp(:,:,m) = L2_distance_1(Zy(:,:,m),Zy(:,:,m));
end
for iter = 1:Niter_SCC
    delty_old = delty;
    distZy = WeightedDistance(distZy_temp, wm_y);
    for m = 1:M
        distXpie_temp(:,:,m) = L2_distance_1(Xpie(:,:,m),Xpie(:,:,m));
    end
    distXpie = WeightedDistance(distXpie_temp, wm_x);
    distnewZy = distZy + distXpie - 2*beta*Sx;
    [valuezy, idzy] = sort(distnewZy,2);
    Szy = zeros(N);
    for i = 1:N
        k = Kmat(i);
        id_zy = idzy(i,1:k+1);
        value_zy = valuezy(i,1:k+1);
        Szy(i,id_zy) = (value_zy(k+1)-value_zy)/(k*value_zy(k+1)-sum(value_zy(1:k))+eps);
    end
    Txpie = 4 * LaplacianMatrix(Szy);
    Tx = 4 * LaplacianMatrix(Sx+Szy);
    for m = 1 : M
        Xpie(:,:,m) = Zupdate_PCG(2*gamma*eye(N)+wm_x(m)*Txpie,(2*gamma*t1_feature(:,:,m)));
        Zy(:,:,m) = Zupdate_PCG(mu*eye(N)+wm_y(m)*Tx,(mu*(t2_feature(:,:,m)+delty(:,:,m)) - Wy(:,:,m)));
        Qy(:,:,m) =  Zy(:,:,m)-t2_feature(:,:,m)+Wy(:,:,m)/mu;
        delty(:,:,m) = deltUpdate(Qy(:,:,m),lambda/mu,21);
        Wy(:,:,m) = Wy(:,:,m) + mu*(Zy(:,:,m)-t2_feature(:,:,m)-delty(:,:,m)); % W update
    end
    for m = 1:M
        distZy_temp(:,:,m) = L2_distance_1(Zy(:,:,m),Zy(:,:,m));
    end
    for m = 1:M
        gc(m) = sum(sum((Sx+Szy).*distZy_temp(:,:,m)));
    end
    up = gc.^(1/(eta-1));
    dgc =  gc.^(eta/(eta-1));
    down = (sum(dgc))^(1/eta);
    wm_y = up/down;
    mu = mu*1.1;
    RELy(iter) = norm(delty(:)-delty_old(:),'fro')/norm(delty(:),'fro');
    if iter>3 && RELy(iter)<1e-2
        break
    end
end
