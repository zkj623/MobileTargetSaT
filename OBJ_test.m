%%
%初始化
dt=1;
T = 3;%预测时域（时间步）
z=zeros(4,T+1);%robot state
z(:,1)=[25;15;pi/2;0];
u=[zeros(1,T);zeros(1,T)];
for ii = 1:T
    z(:,ii+1) = z(:,ii)+[z(4,ii)*cos(z(3,ii));z(4,ii)*sin(z(3,ii));u(:,ii)]*dt;
end

%target建模
tar_pos = [25;28];
f = @(x) x+[0.5;0.5*cos(0.2*x(1))];
Q = 0.25*eye(2);
%传感器建模 'ran'
h = @(x,z) sqrt(sum((x-z).^2)+0.1);
R = 1;%观测噪声方差
mdim = 1;
r = 15;%观测范围

%target move
tar_pos = f(tar_pos);

%measurement
if inFOV(tar_pos,z(1:2,1),r)
    y = h(tar_pos,z(1:2,1))+normrnd(0,sqrt(R));
else
    y = -100;
end


%生成粒子
xMin=0;xMax=50;yMin=0;yMax=50;
%[X,Y] = meshgrid((xMin+1):2:(xMax-1),(yMin+1):2:(yMax-1));%返回二维网格坐标（每一个坐标代表一个particle）,25^2=625个particles
%particles = [X(:),Y(:)]';%2*N的矩阵，particles(:,i)表示第i个particle的坐标(x,y)
particles = mvnrnd(tar_pos,10*eye(2),625)';
N = size(particles,2);%粒子的个数
w = zeros(N,1);

%particle filter
particles = f(particles);
particles = (mvnrnd(particles',Q))';
for jj = 1:N
    if y == -100
        % if the target is outside FOV.
        if inFOV(particles(:,jj),z(1:2,1),r)
            w(jj) = 10^-20;
        else
            w(jj) = 1;
        end
    else
        if inFOV(particles(:,jj),z(1:2,1),r)
            w(jj) = normpdf(y,h(z(1:2,1),particles(:,jj)),sqrt(R));
        else
            w(jj) = 10^-20;
        end
    end
end
w = w./sum(w);%归一化的粒子权重
%重采样
M = 1/N;
U = rand(1)*M;
new_particles = zeros(2,N);
tmp_w = w(1);
ii = 1;
jj = 1;
while (jj <= N)
    while (tmp_w < U+(jj-1)*M)
        ii = ii+1;
        tmp_w = tmp_w+w(ii);
    end
    new_particles(:,jj) = particles(:,ii);
    jj = jj + 1;
end
particles = new_particles;
w = repmat(1/N, N, 1);

B = zeros(2,N,T+1);  %target的位置预测，第一列为当前的位置，2~（T+1)为预测的未来T个时间步的位置。每一列包括所有粒子的位置
B(:,:,1) = particles;
for ii = 2:T+1
    B(:,:,ii) = f(B(:,:,ii-1));
end

%将particles和target可视化
hold on
axis([0,50,0,50]);
axis equal;
plot(particles(1,:),particles(2,:),'r.');
plot(tar_pos(1),tar_pos(2),'g+');
plot(z(1,1),z(2,1),'r*');
plot(z(1,2:end),z(2,2:end),'b*')
theta=linspace(0,2*pi);
plot(z(1,1)+r*cos(theta),z(2,1)+r*sin(theta),'r');
plot(z(1,end)+r*cos(theta),z(2,end)+r*sin(theta),'b');


val = zeros(6,2);%存储互信息计算结果，val(1)和val(2)为单维度下的sigma点采样和随机采样，val(3)和val(4)为多维度下的sigma点采样和随机采样
%组会上说的互信息极小是因为条件熵H_cond和观测熵val(3)相差很小造成的 


%条件熵
H_cond = 0;
H0 = 0.5*(log(2*pi)+1)+0.5*log(det(R));
for ii = 1:T
    for jj = 1:N
        %H_temp = w(jj)*gamma(z(1:2,ii+1),B(:,jj,ii+1));
        H_temp = w(jj)*inFOV(z(1:2,ii+1),B(:,jj,ii+1),r);
        H_cond = H_cond+H_temp;
    end
end
H_cond = H0*H_cond;

%观测熵
%方法一：sigma点采样(UKF)
mu = zeros(N,mdim*T);%观测的均值矩阵，第i行表示第i个粒子在未来T个时间步的观测均值
for ii = 1:N
    mu_tmp = zeros(1,mdim*T);
    for jj = 2:T+1
        mu_tmp(((jj-2)*mdim+1):(jj-1)*mdim) = h(z(1:2,jj),B(:,ii,jj))';
    end
    mu(ii,:) = mu_tmp;
end

%因为计算val(1)和val(2)需要花费久一点的时间，目前是注释掉了
%{
nx = mdim;
X = sqrtm(R);
lambda = 3-nx;
ws = zeros(1,2*nx+1);
ws(1) = lambda/(lambda+nx);
ws(2:end) = repmat(1/2/(lambda+nx),1,2*nx);

tmp1 = 0;
for jj = 1:N
    sigma = zeros(3,nx*T);
    sigma(1,:) = mu(jj,:);
    sigma(2,:) = mu(jj,:) + sqrt(lambda+nx)*X;
    sigma(3,:) = mu(jj,:) - sqrt(lambda+nx)*X;
    tmp2=0;
    for p1 = 1:(2*nx+1)
        for p2 = 1:(2*nx+1)
            for p3 = 1:(2*nx+1)
                tmp3=1;
                for m = 1:T
                    gamma_jj_m = gamma(z(1:2,m+1),B(:,jj,m+1));
                    %gamma_jj_m = inFOV(z(1:2,m+1),B(:,jj,m+1),r);
                    eval(['tp = p',num2str(m)]);%tp = pm
                    tmp3=tmp3*(1-gamma_jj_m+gamma_jj_m*ws(tp));
                end
                tmp4 = 0;
                for ss=1:N
                    tmp5=1;
                    for n=1:T
                        gamma_jj_n = gamma(z(1:2,n+1),B(:,jj,n+1));
                        %gamma_jj_n = inFOV(z(1:2,n+1),B(:,jj,n+1),r);
                        eval(['tp = p',num2str(n)]);%tp = pn
                        tmp5 = tmp5*(1-gamma_jj_n+gamma_jj_n*normpdf(sigma(tp,n),mu(ss,n),R));
                    end
                    tmp4=tmp4+w(ss)*tmp5;
                end
                tmp2=tmp2+tmp3*log(tmp4);
            end
        end
    end
    gamma_jj_sum = 0;
    for t=1:T
        gamma_jj_t= gamma(z(1:2,t+1),B(:,jj,t+1));
        %gamma_jj_t = inFOV(z(1:2,t+1),B(:,jj,t+1),r);
        gamma_jj_sum = gamma_jj_sum+gamma_jj_t;
    end
    tmp1 = tmp1 + w(jj) * tmp2 / 3^(T-gamma_jj_sum);
end
%观测熵
H_sigma=-tmp1;

%总的熵
val_sigma=H_sigma-H_cond;
val(1) = H_sigma;
toc

title(['H\_cond=',num2str(H_cond),'   H\_sigma=' ,num2str(H_sigma),'   I=',num2str(val_sigma)]);
%title(['H\_sigma=',num2str(H_sigma)]);

%方法二：随机采样
%随机采样
M=10;%采样点个数

tmp1 = 0;
for jj = 1:N
    flag = zeros(T,1);
    flag_num = 0;
    random = zeros(M,nx*T);
    for kk = 1:T
        if inFOV(z(1:2,kk+1),B(:,jj,kk+1),r)==1
            flag(kk) = 1;
            flag_num = flag_num + 1;
            random(:,kk) = normrnd(mu(jj,kk),R,M,1);
        end
    end
    tmp2=0;
    for p1 = 1:M
        for p2 = 1:M
            for p3 = 1:M
                tmp4 = 0;
                for ss=1:N
                    tmp5=1;
                    for n=1:T
                        eval(['tp = p',num2str(n)]);%tp = pn
                        if random(tp,n) == 0
                            continue;
                        end
                        tmp5=tmp5*normpdf(random(tp,n),mu(ss,n),R);
                    end
                    tmp4=tmp4+w(ss)*tmp5;
                end
                tmp2=tmp2+(1/M)^3*log(tmp4);
            end
        end
    end
    tmp1 = tmp1 + w(jj) * tmp2;
end
%观测熵
H_random=-tmp1;

%总的熵
val_random=H_random-H_cond;
val(2) = H_random;
%}

tic
tmp1 = 0;
flag = zeros(N,T);
for jj = 1:N
    for kk = 1:T
        if inFOV(z(1:2,kk+1),B(:,jj,kk+1),r)==1
            flag(jj,kk)=1;
        end
    end
end
for jj = 1:N
    nj = 0;
    mu_tmp = mu(jj,:);
    num = 1;
    for kk = 1:T
        if inFOV(z(1:2,kk+1),B(:,jj,kk+1),r)==1
            nj = nj + 1;
            num = num + 1;
        else
            mu_tmp(num) = [];
        end
    end
    %构造sigma点的参数
    nx = nj;
    b = repmat({R},nx,1);
    V = blkdiag(b{:});
    X = sqrtm(V);
    lambda = 2;
    ws = zeros(1,2*nx+1);
    ws(1) = lambda/(lambda+nx);
    ws(2:end) = repmat(1/2/(lambda+nx),1,2*nx);
    sigma = zeros(2*nx+1,nx);
    sigma(1,:) = mu_tmp;
    for ss= 1:nx
        sigma(2*ss,:) = mu_tmp + sqrt(lambda+nx)*X(:,ss)';
        sigma(2*ss+1,:) = mu_tmp - sqrt(lambda+nx)*X(:,ss)';
    end
    tmp2=0;
    for ll = 1:(2*nx+1)
        tmp3 = 0;
        for ss=1:N
            tmp4=1;
            num = 1;
            for n=1:T
                if flag(jj,n)~=flag(ss,n)
                    tmp4 = 0;
                    break;
                end
                if flag(jj,n)==1&&flag(ss,n)==1
                    tmp4 = tmp4*normpdf(sigma(ll,num),mu(ss,n),sqrt(R));
                    num = num + 1;
                end
            end
            tmp3=tmp3+w(ss)*tmp4;
        end
        tmp2=tmp2+ws(ll)*log(tmp3);
    end
    tmp1 = tmp1 + w(jj) * tmp2;
end
val(4,1)=-tmp1;
val(4,2)=-tmp1-H_cond;
toc
%{
%方法二：随机采样
M=100;%采样点个数

tic
tmp1 = 0;
for jj = 1:N
    nj = 0;
    mu_tmp = mu(jj,:);
    num = 1;
    for kk = 1:T
        if inFOV(z(1:2,kk+1),B(:,jj,kk+1),r)==1
            nj = nj + 1;
            num = num + 1;
        else
            mu_tmp(num) = [];
        end
    end
    random1 = zeros(M,nj);
    for kk = 1:nj
        random1(:,kk) = normrnd(mu_tmp(kk),sqrt(R),M,1);
    end
    tmp2=0;
    for ll = 1:M
        tmp3 = 0;
        for ss=1:N
            tmp4=1;
            num = 1;
            for n=1:T
                if flag(jj,n)~=flag(ss,n)
                    tmp4 = 0;
                    break;
                end
                if flag(jj,n)==1&&flag(ss,n)==1
                    tmp4 = tmp4*normpdf(random1(ll,num),mu(ss,n),sqrt(R));
                    num = num + 1;
                end
            end
            tmp3=tmp3+w(ss)*tmp4;
        end
        tmp2=tmp2+1/M*log(tmp3);
    end
    tmp1 = tmp1 + w(jj) * tmp2;
end
val(3,1)=-tmp1;
val(3,2)=-tmp1-H_cond;
toc
%
b = repmat({R},T,1);
R_tmp = blkdiag(b{:});
gm = gmdistribution(mu,R_tmp);

tic
tmp1 = 0;
M = 500;
X = random(gm,M);
for jj = 1:M
    tmp = 0;
    for kk = 1:N
        a = mvnpdf(X(jj,:),mu(kk,:),R_tmp);
        tmp = tmp + w(kk)*a;
    end
    tmp1 = tmp1 - 1/M*log(tmp);
end

val(1,1)=tmp1;
val(1,2)=tmp1-T*(0.5*(log(2*pi)+1)+0.5*log(det(R)));
toc

tic
H = 0;
nx = mdim*T;
b = repmat({R},T,1);
V = blkdiag(b{:});
X = sqrtm(V);
lambda = 2;
ws = zeros(1,2*nx+1);
ws(1)=lambda/(lambda+nx);
ws(2:end) = repmat(1/2/(lambda+nx),1,2*nx);
for jj = 1:N
    tmp1 = 0;
    sigma = zeros(2*nx+1,nx);
    sigma(1,:) = mu(jj,:);
    for ss= 1:nx
        sigma(2*ss,:) = mu(jj,:) + sqrt(lambda+nx)*X(:,ss)';
        sigma(2*ss+1,:) = mu(jj,:) - sqrt(lambda+nx)*X(:,ss)';
    end
    
    for kk = 1:2*nx+1
        tmp2 = 0;
        for ll = 1:N
            P = mvnpdf(sigma(kk,:),mu(ll,:),R_tmp);
            tmp2 = tmp2 + w(ll)*P;
        end
        tmp1 = tmp1 + ws(kk)*log(tmp2);
    end
    H = H - w(jj)*tmp1;
end
val(2,1)=H;
val(2,2)=H-T*(0.5*(log(2*pi)+1)+0.5*log(det(R)));
toc
%}
%%
%初始化
dt=1;
T = 1;%预测时域（时间步）
z=zeros(4,T+1);%robot state
z(:,1)=[60;50;pi/2;3];
u=[zeros(1,T);0];
for ii = 1:T
    z(:,ii+1) = z(:,ii)+[z(4,ii)*cos(z(3,ii));z(4,ii)*sin(z(3,ii));u(:,ii)]*dt;
end

%target建模
tar_pos = [50;50];
%tar = this.target;
f = @(x) x+[0.5;0.5*cos(0.2*x(1))];
Q = 0.25*eye(2);
%传感器建模 'ran'
%h = @(x,z) sqrt(sum((x-z(1:2)).^2)+0.1);
h = @(x,z) atan2(x(2,:)-z(2),x(1,:)-z(1))-z(3);
R = 0.1;%观测噪声方差
mdim = 1;
r = 30;%观测范围
if inFOV(tar_pos,z(1:2,1),r)
    y = h(tar_pos,z(:,1))+normrnd(0,sqrt(R));
else
    y = -100;
end


%生成粒子
% xMin=tar_pos(1)-20;xMax=tar_pos(1)+20;yMin=tar_pos(2)-20;yMax=tar_pos(2)+20;
% [X,Y] = meshgrid((xMin+1):2:(xMax-1),(yMin+1):2:(yMax-1));%返回二维网格坐标（每一个坐标代表一个particle）,25^2=625个particles
% particles = [X(:),Y(:)]';%2*N的矩阵，particles(:,i)表示第i个particle的坐标(x,y)
particles = mvnrnd(tar_pos,10*eye(2),625)';
N = size(particles,2);%粒子的个数
w = zeros(N,1);
%particles = f(particles);
particles = (mvnrnd(particles',Q))';
P = normpdf(y,h(particles,z(:,1)),sqrt(R));
P1 = normpdf(y,h(particles,z(:,1))+2*pi,sqrt(R));
P2 = normpdf(y,h(particles,z(:,1))-2*pi,sqrt(R));
for jj = 1:N
    if y == -100
        % if the target is outside FOV.
        if inFOV(particles(:,jj),z(1:2,1),r)
            w(jj) = 10^-20;
        else
            w(jj) = 1;
        end
    else
        if inFOV(particles(:,jj),z(1:2,1),r)
            %
            %bearing
            if h(particles(:,jj),z(:,1)) <= 0
                w(jj) = max([P(jj),P1(jj)]);
            else
                w(jj) = max([P(jj),P2(jj)]);
            end
            %}
            %range
            %w(jj) = normpdf(y,h(particles(:,jj),z(:,1)),sqrt(R));
        else
            w(jj) = 10^-20;
        end
    end
end
w = w./sum(w);%归一化的粒子权重
%重采样
M = 1/N;
U = rand(1)*M;
new_particles = zeros(2,N);
tmp_w = w(1);
ii = 1;
jj = 1;
while (jj <= N)
    while (tmp_w < U+(jj-1)*M)
        ii = ii+1;
        tmp_w = tmp_w+w(ii);
    end
    new_particles(:,jj) = particles(:,ii);
    jj = jj + 1;
end
particles = new_particles;
w = repmat(1/N, N, 1);

B = zeros(2,N,T+1);  %target的位置预测，第一列为当前的位置，2~（T+1)为预测的未来T个时间步的位置。每一列包括所有粒子的位置
B(:,:,1) = particles;
for ii = 2:T+1
    B(:,:,ii) = (B(:,:,ii-1));
end

%将particles和target可视化
hold on
axis([-30,80,-30,80]);
axis equal;
plot(particles(1,:),particles(2,:),'b.');
plot(tar_pos(1),tar_pos(2),'g+');
plot(z(1,1),z(2,1),'b*');
plot(z(1,2:end),z(2,2:end),'r*')
theta=linspace(0,2*pi);
plot(z(1,1)+r*cos(theta),z(2,1)+r*sin(theta),'b');
plot(z(1,end)+r*cos(theta),z(2,end)+r*sin(theta),'r');

val = zeros(4,2);%存储互信息计算结果

B_tmp1 = B(:,:,2);
jj=1;
ii = 1;
while(ii<=N)
    if inFOV(z(1:2,2),B_tmp1(:,jj),r) == 0
        B_tmp1(:,jj) = [];
    else
        jj = jj + 1;
    end
    ii = ii + 1;
end
B_tmp2 = zeros(2,N);
B_tmp2(:,1:size(B_tmp1,2)) = B_tmp1;
for jj = size(B_tmp1,2)+1:N
    B_tmp2(:,jj) = B_tmp1(:,randperm(size(B_tmp1,2),1));
end
B_tmp = B(:,:,2);
B(:,:,2)=B_tmp2;
%条件熵
H_cond = 0;
H0 = 0.5*(log(2*pi)+1)+0.5*log(det(R));
for ii = 1:T
    for jj = 1:N
        %H_temp = w(jj)*gamma(z(1:2,ii+1),B(:,jj,ii+1));
        H_temp = w(jj)*inFOV(z(1:2,ii+1),B(:,jj,ii+1),r);
        H_cond = H_cond+H_temp;
    end
end
H_cond = H0*H_cond;

%观测熵
%方法一：sigma点采样(UKF)
mu = zeros(N,mdim*T);%观测的均值矩阵，第i行表示第i个粒子在未来T个时间步的观测均值
for ii = 1:N
    mu_tmp = zeros(1,mdim*T);
    for jj = 2:T+1
        mu_tmp(((jj-2)*mdim+1):(jj-1)*mdim) = h(B(:,ii,jj),z(:,jj))';
    end
    mu(ii,:) = mu_tmp;
end

if range(mod(mu,2*pi))>range(rem(mu,2*pi))
    mu = rem(mu,2*pi);
else
    mu = mod(mu,2*pi);
end

tic
tmp1 = 0;
flag = zeros(N,T);
for jj = 1:N
    for kk = 1:T
        if inFOV(z(1:2,kk+1),B(:,jj,kk+1),r)==1
            flag(jj,kk)=1;
        end
    end
end
for jj = 1:N
    nj = 0;
    mu_tmp = mu(jj,:);
    num = 1;
    for kk = 1:T
        if inFOV(z(1:2,kk+1),B(:,jj,kk+1),r)==1
            nj = nj + 1;
            num = num + 1;
        else
            mu_tmp(num) = [];
        end
    end
    %构造sigma点的参数
    nx = nj;
    b = repmat({R},nx,1);
    V = blkdiag(b{:});
    X = sqrtm(V);
    lambda = 2;
    ws = zeros(1,2*nx+1);
    ws(1) = lambda/(lambda+nx);
    ws(2:end) = repmat(1/2/(lambda+nx),1,2*nx);
    sigma = zeros(2*nx+1,nx);
    sigma(1,:) = mu_tmp;
    for ss= 1:nx
        sigma(2*ss,:) = mu_tmp + sqrt(lambda+nx)*X(:,ss)';
        sigma(2*ss+1,:) = mu_tmp - sqrt(lambda+nx)*X(:,ss)';
    end
    tmp2=0;
    for ll = 1:(2*nx+1)
        tmp3 = 0;
        for ss=1:N
            tmp4=1;
            num = 1;
            for n=1:T
                if flag(jj,n)~=flag(ss,n)
                    tmp4 = 0;
                    break;
                end
                if flag(jj,n)==1&&flag(ss,n)==1
                    tmp4 = tmp4*normpdf(sigma(ll,num),mu(ss,n),sqrt(R));
                    num = num + 1;
                end
            end
            tmp3=tmp3+w(ss)*tmp4;
        end
        tmp2=tmp2+ws(ll)*log(tmp3);
    end
    tmp1 = tmp1 + w(jj) * tmp2;
end
val(3,1)=-tmp1
val(3,2)=-tmp1-H_cond
toc
%
B(:,:,2) = B_tmp;
B_tmp1 = B(:,:,2);


jj=1;
ii = 1;
while(ii<=N)
    if inFOV(z(1:2,2),B_tmp1(:,jj),r) == 0
        B_tmp1(:,jj) = [];
    else
        jj = jj + 1;
    end
    ii = ii + 1;
end
% B_tmp2 = zeros(2,N);
% B_tmp2(:,1:size(B_tmp1,2)) = B_tmp1;
% for jj = size(B_tmp1,2)+1:N
%     B_tmp2(:,jj) = B_tmp1(:,randperm(size(B_tmp1,2),1));
% end

N = size(B_tmp1,2);
w = repmat(1/N,1,N);
B(:,1:N,2) = B_tmp1;
%条件熵
H_cond = 0;
H0 = 0.5*(log(2*pi)+1)+0.5*log(det(R));
for ii = 1:T
    for jj = 1:N
        %H_temp = w(jj)*gamma(z(1:2,ii+1),B(:,jj,ii+1));
        H_temp = w(jj)*inFOV(z(1:2,ii+1),B(:,jj,ii+1),r);
        H_cond = H_cond+H_temp;
    end
end
H_cond = H0*H_cond;

%观测熵
%方法一：sigma点采样(UKF)
mu = zeros(N,mdim*T);%观测的均值矩阵，第i行表示第i个粒子在未来T个时间步的观测均值
for ii = 1:N
    mu_tmp = zeros(1,mdim*T);
    for jj = 2:T+1
        mu_tmp(((jj-2)*mdim+1):(jj-1)*mdim) = h(B(:,ii,jj),z(:,jj))';
    end
    mu(ii,:) = mu_tmp;
end
%
if range(mod(mu,2*pi))>range(rem(mu,2*pi))
    mu = rem(mu,2*pi);
else
    mu = mod(mu,2*pi);
end
%}
tic
tmp1 = 0;
flag = zeros(N,T);
for jj = 1:N
    for kk = 1:T
        if inFOV(z(1:2,kk+1),B(:,jj,kk+1),r)==1
            flag(jj,kk)=1;
        end
    end
end
for jj = 1:N
    nj = 0;
    mu_tmp = mu(jj,:);
    num = 1;
    for kk = 1:T
        if inFOV(z(1:2,kk+1),B(:,jj,kk+1),r)==1
            nj = nj + 1;
            num = num + 1;
        else
            mu_tmp(num) = [];
        end
    end
    %构造sigma点的参数
    nx = nj;
    b = repmat({R},nx,1);
    V = blkdiag(b{:});
    X = sqrtm(V);
    lambda = 2;
    ws = zeros(1,2*nx+1);
    ws(1) = lambda/(lambda+nx);
    ws(2:end) = repmat(1/2/(lambda+nx),1,2*nx);
    sigma = zeros(2*nx+1,nx);
    sigma(1,:) = mu_tmp;
    for ss= 1:nx
        sigma(2*ss,:) = mu_tmp + sqrt(lambda+nx)*X(:,ss)';
        sigma(2*ss+1,:) = mu_tmp - sqrt(lambda+nx)*X(:,ss)';
    end
    tmp2=0;
    for ll = 1:(2*nx+1)
        tmp3 = 0;
        for ss=1:N
            tmp4=1;
            num = 1;
            for n=1:T
                if flag(jj,n)~=flag(ss,n)
                    tmp4 = 0;
                    break;
                end
                if flag(jj,n)==1&&flag(ss,n)==1
                    tmp4 = tmp4*normpdf(sigma(ll,num),mu(ss,n),sqrt(R));
                    num = num + 1;
                end
            end
            tmp3=tmp3+w(ss)*tmp4;
        end
        tmp2=tmp2+ws(ll)*log(tmp3);
    end
    tmp1 = tmp1 + w(jj) * tmp2;
end
val(4,1)=-tmp1
val(4,2)=-tmp1-H_cond
%}


function gam = gamma(z,x0)
tmp = 1.5*(sum((x0-z).^2)-3^2);
if tmp > 100
    gam_den = 1+exp(30);
else
    gam_den = 1+exp(tmp);
end
gam = 1/gam_den;
end

function flag = inFOV(z,tar_pos,r)
flag = norm(tar_pos-z) <= r;
end