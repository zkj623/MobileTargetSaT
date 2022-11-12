tar_pos = [45;70];
%z=[30;20;pi/2;1];
%z=[90;90;-pi/2;1];%robot state
%z=[5;30;pi/2;1];
%z=[5;80;-pi/2;1];
z=[75;20;pi/2;1];

f = @(x) x+[0.8;0.8*cos(0.2*x(1))];
%f = @(x) x+[0.5;0];
Q = 0.25*eye(2);
%传感器建模 'ran'
h = @(x,z) sqrt(sum((x-z(1:2)).^2)+0.1);
%h = @(x,z) atan2(x(2,:)-z(2),x(1,:)-z(1))-z(3);
R = 1;%观测噪声方差
r = 20;%观测范围

%障碍物定义obstacle
num_obstacle = 7;
polyin = repmat(Obstacle,num_obstacle,1);

polyin(1).points = [10,10,20,20;0,60,60,0];
polyin(2).points = [10,10,20,20;70,90,90,70];
polyin(3).points = [35,35,40,40;35,80,80,35];
polyin(4).points = [50,50,70,70;0,30,30,0];
polyin(5).points = [50,50,70,70;55,75,75,55];
polyin(6).points = [80,80,85,85;30,55,55,30];
polyin(7).points = [80,80,85,85;65,100,100,65];

x1 = [10,10,20,20];
y1 = [0,60,60,0];
x2 = [10,10,20,20];
y2 = [70,90,90,70];
x3 = [35,35,40,40];
y3 = [35,80,80,35];
x4 = [50,50,70,70];
y4 = [0,30,30,0];
x5 = [50,50,70,70];
y5 = [55,75,75,55];
x6 = [80,80,85,85];
y6 = [30,55,55,30];
x7 = [80,80,85,85];
y7 = [65,100,100,65];
poly = polyshape({x1,x2,x3,x4,x5,x6,x7},{y1,y2,y3,y4,y5,y6,y7});

% plot(poly);
% axis equal;
% axis([0,100,0,100]);

%feasible region
region = ones(100,100);
for ii = 1:size(region,1)
    for jj = 1:size(region,2)
        for kk = 1:length(polyin)
            if isInside(polyin(kk),[ii-0.5;jj-0.5]) == 1
                region(ii,jj) = 0;%infeasible
                break
            end
        end
    end
end

%visibility judgement
%{
V = -ones(100,100,100,100);
tic
for ii = 1:size(V,1)
    for jj = 1:size(V,2)
        for mm = 1:size(V,3)
            for nn = 1:size(V,4)
                V(ii,jj,mm,nn) = 1;
                if V(mm,nn,ii,jj) ~= -1
                    V(ii,jj,mm,nn) = V(mm,nn,ii,jj);
                    continue
                end
                if mm == ii&&nn == jj
                    V(ii,jj,mm,nn) = region(ii,jj);
                    continue
                end
                if region(ii,jj)==0||region(mm,nn)==0
                    V(ii,jj,mm,nn) = 0;
                    continue
                end
                for kk = 1:length(polyin)
                    if isVisible(polyin(kk),[ii-0.5;jj-0.5],[mm-0.5,nn-0.5],region) == 0
                        V(ii,jj,mm,nn) = 0;%invisible
                        break
                    end
                end
            end
        end
    end
end
toc
%}

xMin=tar_pos(1)-20;xMax=tar_pos(1)+20;yMin=tar_pos(2)-20;yMax=tar_pos(2)+20;
[X,Y] = meshgrid((xMin+1):2:(xMax-1),(yMin+1):2:(yMax-1));%返回二维网格坐标（每一个坐标代表一个particle）,25^2=625个particles
particles = [X(:),Y(:)]';%2*N的矩阵，particles(:,i)表示第i个particle的坐标(x,y)
particles = mvnrnd(tar_pos,10*eye(2),625)';
% mu = [tar_pos';10,40;60,5];
% sigma = 5*eye(2);
% p = [0.2,0.4,0.4];
% gm = gmdistribution(mu,sigma,p);
% particles = random(gm,1000)';

N = size(particles,2);%粒子的个数
w = zeros(1,N);

myvideo = VideoWriter('MIPFT_1112_range_11.avi');
myvideo.FrameRate = 3;
open(myvideo);
simlen = 100;
planlen = 200;

%target motion model definition
F = cell(simlen+10,1);
for ii = 1:length(F)
    if ii <= 10
        F{ii} = @(x) x+[0;1];
    elseif ii <= 40
        F{ii} = @(x) x+[1;0];
    elseif ii <= 60
        F{ii} = @(x) x+[0;-1];
    elseif ii <= 75
        F{ii} = @(x) x+[1;0];
    else
        F{ii} = @(x) x+[0;-1];
    end
end

list = repmat(Node_IMPFT,simlen,500);

%% %MCAP
for ii = 1:simlen
    tic
    f = F{ii};

    %target move
    tar_pos = f(tar_pos);

    
    %generate measurement
    if inFOV(z(1:2),tar_pos,r)&&V(ceil(z(1)),ceil(z(2)),ceil(tar_pos(1)),ceil(tar_pos(2)))
        y = h(tar_pos,z)+normrnd(0,sqrt(R));
    else
        y = -100;
    end
    
    %particle filter
    [particles,w] = PF(z,particles,w,y,r,Q,R,f,h,polyin,region,V);
    
    %IMPFT
    is_tracking = 0;
    if inFOV(z(1:2),tar_pos,r)&&V(ceil(z(1)),ceil(z(2)),ceil(tar_pos(1)),ceil(tar_pos(2)))
        is_tracking = 1;
    end
    
    list_tmp = [];
    root = Node_IMPFT;
    %根节点初始化
    root.num = 1;
    root.state = z;%匀速运动v=1
    root.h = [];
    root.a = [pi/4,pi/4,pi/4,pi/4,pi/4,pi/4,0,0,0,0,0,0,-pi/4,-pi/4,-pi/4,-pi/4,-pi/4,-pi/4;-3,-1.5,-0.5,0,0.5,1.5,-3,-1.5,-0.5,0,0.5,1.5,-3,-1.5,-0.5,0,0.5,1.5];
    root.N = 0;
    root.Q = 0;
    root.children = [];
    root.children_maxnum = 18;
    root.is_terminal = 0;
    root.delete = 0;
    list_tmp = [list_tmp,root];
    
    depth = 10;
    eta = 0.9;%discount factor
    num = 1;%addtion point index
    for jj = 1:planlen
        %tic
        [list_tmp,Reward,num] = simulate(1,num,list_tmp,particles,depth,f,h,R,r,Q,eta,w,is_tracking,polyin,region,V,F,ii);
        %toc
    end
    
    max_value = -500;
    for jj = 1:length(list_tmp(1).children)
        %val = list(list(1).children(jj)).Q/list(list(1).children(jj)).N;
        val = list_tmp(list_tmp(1).children(jj)).Q;
        if val>max_value
            max_value = val;
            num = jj;
        end
    end
    opt = list_tmp(1).children(num);
    state_opt = list_tmp(opt).state;
    
    list(ii,1:length(list_tmp)) = list_tmp;
    
    hold on
    axis equal;
    axis([0,100,0,100]);
    plot(poly);
    plot(particles(1,:),particles(2,:),'.');
    plot(tar_pos(1),tar_pos(2),'g+');
    plot(z(1),z(2),'r*');
    theta=linspace(0,2*pi);
    plot(z(1)+r*cos(theta),z(2)+r*sin(theta),'r');
    z = state_opt;
    plot(z(1),z(2),'b*');
    plot(z(1)+r*cos(theta),z(2)+r*sin(theta),'b');
    text(5,95,num2str(ii));
    drawnow limitrate
    set(gcf,'position',[1000,500,1080,720]);
    frame = getframe(gcf);
    writeVideo(myvideo,frame);
    clf
    toc
end

close(myvideo);


function [list_tmp,Reward,num] = simulate(begin,num,list_tmp,B,depth,f,h,R,r,Q,eta,w,is_tracking,polyin,region,V,F,Sim)
K = 3;
alpha = 0.1;
f = F{Sim};
if depth == 0
    Reward = 0;
    return
else
    list_tmp(begin).N = list_tmp(begin).N+1;
    if length(list_tmp(begin).children) == list_tmp(begin).children_maxnum
        [begin,list_tmp,num] = best_child(begin,0.732,list_tmp,num,polyin,region,V);
    else
        num = num + 1;
        [list_tmp,begin] = expand(begin,num,list_tmp,0,1,polyin,region,V);
    end
    num_a = begin;
    list_tmp(num_a).N = list_tmp(num_a).N+1;
    state = list_tmp(num_a).state;
    B = f(B);%prediction
    % feasible particles
    jj = 1;
    for ii = 1:size(B,2)
        if any([0;0] > B(:,jj))||any([100;100] < B(:,jj))||region(ceil(B(1,jj)),ceil(B(2,jj))) == 0
            B(:,jj) = [];
            w(jj) = [];
            continue
        end
        jj = jj+1;
    end
    w = w./sum(w);
    
    N = size(B,2);
    reward = MI(list_tmp(num_a).state,N,B,w,is_tracking,V); 
    if length(list_tmp(begin).children) <= K*(list_tmp(begin).N^alpha)
        B_tmp = B;
        jj = 1;
        ii = 1;
        while(ii<=N)
            if inFOV(state(1:2),B(:,jj),r) == 0||V(ceil(state(1)),ceil(state(2)),ceil(B(1,jj)),ceil(B(2,jj))) == 0
                B_tmp(:,jj) = [];
            else
                jj = jj + 1;
            end
            ii = ii + 1;
        end
        if size(B_tmp,2) == 0
            o = -100;
        else
            mu = zeros(size(B_tmp,2),1);
            for ii = 1:size(B_tmp,2)
                mu = h(B_tmp(:,ii),state);
            end
            gm = gmdistribution(mu,R);
            o = random(gm);
        end
        num = num + 1;
        [list_tmp,begin] = expand(begin,num,list_tmp,o,2,polyin,region,V);
        flag = 1;
    else
        begin = list_tmp(begin).children(randperm(length(list_tmp(begin).children),1));
        o = list_tmp(begin).h(3,end);
        flag = 0;
    end
    num_o = begin;
    if o~=-100%这里如果不走PF可能会出现infeasible的粒子
        I = @(x) x;
        [B,w] = PF(list_tmp(num_a).state,B,w,o,r,zeros(2,2),R,I,h,polyin,region,V);
    end
    if flag == 1
        node = list_tmp(begin);
        Sim = Sim + 1;
        rollout = rollOut(node,eta,depth-1,B,w,f,h,R,r,is_tracking,polyin,region,V,F,Sim);
        Reward = reward + eta*rollout;
    else
        Sim = Sim + 1;
        [list_tmp,Reward,num] = simulate(begin,num,list_tmp,B,depth-1,f,h,R,r,Q,eta,w,is_tracking,polyin,region,V,F,Sim);
        Reward = reward + eta*Reward;
    end
    list_tmp(num_o).N = list_tmp(num_o).N+1;
    list_tmp(num_a).Q = list_tmp(num_a).Q + (Reward-list_tmp(num_a).Q)/list_tmp(num_a).N; 
end
end

function [v,list_tmp,num] = best_child(begin,c,list_tmp,num,polyin,region,V)
if isempty(list_tmp(begin).children) == 1
    list_tmp(begin).state(3) = list_tmp(begin).state(3)+pi;
    list_tmp(begin).children_maxnum = 18;
    list_tmp(begin).a = [pi/4,pi/4,pi/4,pi/4,pi/4,pi/4,0,0,0,0,0,0,-pi/4,-pi/4,-pi/4,-pi/4,-pi/4,-pi/4;-3,-1.5,-0.5,0,0.5,1.5,-3,-1.5,-0.5,0,0.5,1.5,-3,-1.5,-0.5,0,0.5,1.5];
    %num = num + 1;
    [list_tmp,v] = expand(begin,num,list_tmp,0,1,polyin,region,V);
    return
end
max = -100;
for jj = 1:length(list_tmp(begin).children)
    node = list_tmp(begin);
    tmp = node.children(jj);
    val = list_tmp(tmp).Q+2*c*(log(node.N)/list_tmp(tmp).N)^0.5;
    if val>max
        max = val;
        v = tmp;
    end
end
end

function [list_tmp,begin] = expand(begin,num,list_tmp,o,tmp,polyin,region,V)
node = list_tmp(begin); 
state = zeros(4,1);
if tmp == 1 %action
    %
    while(1)
        flag = 0;
        inregion = 1;
        if size(list_tmp(begin).a,2) == 0
            [begin,list_tmp] = best_child(begin,0.732,list_tmp,num,polyin,region,V);
            return
        end
        ii = randperm(size(list_tmp(begin).a,2),1);
        action = node.a(:,ii);
        list_tmp(begin).a(:,ii) = [];
        state(3) = node.state(3)+action(1);
        state(4) = node.state(4)+action(2);
        state(1) = node.state(1)+cos(state(3))*state(4);
        state(2) = node.state(2)+sin(state(3))*state(4);
        if any([0;0] >= state(1:2))||any([100;100] <= state(1:2))
            flag = 1;
            inregion = 0;
        end
        if inregion == 1
            if region(ceil(state(1)),ceil(state(2))) == 0||V(ceil(node.state(1)),ceil(node.state(2)),ceil(state(1)),ceil(state(2))) == 0
                flag = 1;
            end
        end
        %{
        for kk = 1:length(polyin)
            if isInside(polyin(kk),state(1:2)) == 1
                flag = 1;
                break
            end
        end
        %}
        if state(4)>=0 && flag == 0
            break
        end
        list_tmp(begin).children_maxnum = list_tmp(begin).children_maxnum-1;
    end
    %}
    %{
    while(1)
        if size(list_tmp(begin).a,2) == 0
            begin = best_child(begin,0.732,list_tmp);
            return
        end
        ii = randperm(size(list_tmp(begin).a,2),1);
        list_tmp(begin).a(:,ii) = [];
        action = node.a(:,ii);
        state(3) = node.state(3)+action(1);
        state(4) = node.state(4)+action(2);
        state(1) = node.state(1)+cos(state(3))*state(4);
        state(2) = node.state(2)+sin(state(3))*state(4);
        if state(4)>=0
            break
        end
        list_tmp(begin).children_maxnum = list_tmp(begin).children_maxnum-1;
    end
    %}
    new = Node_IMPFT;
    new.num = num;
    new.state = state;
    h = [action;0];
    new.h = [node.h,h];
else % observation
    new = node;
    new.num = num;
    new.h(3,end) = o;
end
    new.a = [pi/4,pi/4,pi/4,pi/4,pi/4,pi/4,0,0,0,0,0,0,-pi/4,-pi/4,-pi/4,-pi/4,-pi/4,-pi/4;-3,-1.5,-0.5,0,0.5,1.5,-3,-1.5,-0.5,0,0.5,1.5,-3,-1.5,-0.5,0,0.5,1.5];
    new.N = 0;
    new.Q = 0;
    new.parent = node.num;
    new.children = [];
    new.children_maxnum = 18;
    list_tmp(node.num).children = [list_tmp(node.num).children,new.num];
    new.is_terminal = 0;
    new.delete = 0;
    list_tmp(num) = new;    
    begin = num;
end

function reward = rollOut(node,eta,depth,B,w,f,h,R,r,is_tracking,polyin,region,V,F,Sim)
if depth == 0
    reward = 0;
    return
else
    f = F{Sim};
    B = f(B);
    
    jj = 1;
    for ii = 1:size(B,2)
        if any([0;0] > B(:,jj))||any([100;100] < B(:,jj))||region(ceil(B(1,jj)),ceil(B(2,jj))) == 0
            B(:,jj) = [];
            w(jj) = [];
            continue
        end
        jj = jj+1;
    end
    w = w./sum(w);
    
    action = node.a(:,randperm(length(node.a),1));
    state = node.state;
    node.state(3) = node.state(3)+action(1);
    node.state(4) = node.state(4)+action(2);
    node.state(1) = node.state(1)+cos(node.state(3))*node.state(4);
    node.state(2) = node.state(2)+sin(node.state(3))*node.state(4);
    if any([0;0] >= node.state(1:2))||any([100;100] <= node.state(1:2))%||region(ceil(node.state(1)),ceil(node.state(2))) == 0||V(ceil(node.state(1)),ceil(node.state(2)),ceil(state(1)),ceil(state(2))) == 0
        reward = -0.1;
        return
    end
    
    reward = MI(node.state,size(B,2),B,w,is_tracking,V);
%     mu = h(B,node.state)';
%     gm = gmdistribution(mu,R);
%     o = random(gm);
%     I = @(x) x;
%    [B,w] = PF(node.state,B,w,o,r,zeros(2,2),R,I,h,list_obstacle);
    reward = reward + eta*rollOut(node,eta,depth-1,B,w,f,h,R,r,is_tracking,polyin,region,V,F,Sim);
end
end

function reward = MI(state,N,particles,w,is_tracking,V)
H_cond = 0;
R = 1;
r = 20;
H0 = 0.5*(log(2*pi)+1)+0.5*log(det(R));
FOV = inFOV(state(1:2),particles,r);
%
if is_tracking == 1%&&~all(FOV)
    %
    particles_tmp1 = zeros(2,N);
    particles_tmp1(1,:) = particles(1,:).*FOV;
    particles_tmp1(2,:) = particles(2,:).*FOV;
    w = w.*FOV';
    particles_tmp1(:,any(particles_tmp1,1)==0)=[];
    w(any(w,2)==0,:)=[];
    
    N_tmp = size(particles_tmp1,2);
    jj = 1;
    for ii = 1:N_tmp
        if V(ceil(state(1)),ceil(state(2)),ceil(particles_tmp1(1,jj)),ceil(particles_tmp1(2,jj)))==0
            particles_tmp1(:,jj) = [];
            w(jj) = [];
            continue
        end  
        jj=jj+1;
    end
    w = w./sum(w);
    %}
    %{
    particles_tmp1 = particles;
    jj = 1;
    ii = 1;
    while(ii<=N)
        if FOV(jj) == 0
            particles_tmp1(:,jj) = [];
        else
            jj = jj + 1;
        end
        ii = ii + 1;
    end
    %}
    if size(particles_tmp1,2) == 0
        reward = 0;
        return;
    end
    particles = particles_tmp1;
    %{
    %没必要补全 删除不需要的即可 验证了删除和补全差别很小
    particles_tmp2 = zeros(2,N);
    particles_tmp2(:,1:size(particles_tmp1,2)) = particles_tmp1;
    for jj = size(particles_tmp1,2)+1:N
        particles_tmp2(:,jj) = particles_tmp1(:,randperm(size(particles_tmp1,2),1));
    end
    particles = particles_tmp2;
    %}
end

%}
Cidx = zeros(size(particles,2),2);
flag = zeros(100,100);
N = 0;
for mm = 1:size(particles,2)
    id1 = ceil(particles(1,mm)/2.5)+5;
    Cidx(mm,1) = id1;
    id2 = ceil(particles(2,mm)/2.5)+5;
    Cidx(mm,2) = id2;
    if flag(id1,id2) == 0
        N = N + 1;
        flag(id1,id2) = N;
    end
end
particles_tmp = particles;
w_tmp = w;
particles = zeros(2,N);
w = zeros(1,N);
for mm = 1:size(particles_tmp,2)
    w(flag(Cidx(mm,1),Cidx(mm,2))) = w(flag(Cidx(mm,1),Cidx(mm,2))) + w_tmp(mm);
end
for mm = 1:size(particles_tmp,2)
    particles(:,flag(Cidx(mm,1),Cidx(mm,2))) = particles(:,flag(Cidx(mm,1),Cidx(mm,2))) + particles_tmp(:,mm).*w_tmp(mm)./w(flag(Cidx(mm,1),Cidx(mm,2)));
end
%%是否要修改 改了有问题（好像把atan2的突变问题解决后又没问题了）
%
visibility = zeros(1,N);
for jj = 1:N   
    visibility(jj) = V(ceil(state(1)),ceil(state(2)),ceil(particles(1,jj)),ceil(particles(2,jj)));
end
FOV = inFOV(state(1:2),particles,r).*visibility;
%}
%FOV = inFOV(state(1:2),particles,r);

for jj = 1:N
    H_temp = w(jj)*FOV(jj);
    H_cond = H_cond+H_temp;
end
H_cond = H0*H_cond;

mu = zeros(N,1);%观测的均值矩阵，第i行表示第i个粒子在未来T个时间步的观测均值
for jj = 1:N
    %mu(jj) = atan2(particles(2,jj)-state(2),particles(1,jj)-state(1));%-state(3);%state(3)~=0在目前解决atan2突变的方法有问题
    mu(jj) = sqrt(sum((state(1:2)-particles(:,jj)).^2)+0.1);
end
%解决atan2突变问题
%{
if range(mod(mu,2*pi))>range(rem(mu,2*pi))
    mu = rem(mu,2*pi);
else
    mu = mod(mu,2*pi);
end
%}

%构造sigma点的参数
nx = 1;
b = repmat({R},nx,1);
V = blkdiag(b{:});
X = sqrtm(V);
lambda = 2;
ws = zeros(1,2*nx+1);
ws(1) = lambda/(lambda+nx);
ws(2:end) = repmat(1/2/(lambda+nx),1,2*nx);
tmp1 = 0;
for jj = 1:N
    sigma = zeros(2*nx+1,nx);
    sigma(1,:) = mu(jj);
    for ss= 1:nx
        sigma(2*ss,:) = mu(jj) + sqrt(lambda+nx)*X(:,ss)';
        sigma(2*ss+1,:) = mu(jj) - sqrt(lambda+nx)*X(:,ss)';
    end
    tmp2=0;
    for ll = 1:(2*nx+1)
        %{
        tmp3 = 0;
        for ss=1:N
            tmp4 = 1;
            if FOV(jj)~=FOV(ss)
                tmp4 = 0;
            end
            if FOV(jj)==1&&FOV(ss)==1
                tmp4 = normpdf(sigma(ll),mu(ss),sqrt(R));
            end
            tmp3=tmp3+w(ss)*tmp4;
        end
        %}
        if FOV(jj)==1
            tmp4 = (FOV==1)'.*(normpdf(sigma(ll),mu,sqrt(R))-1)+1;
            tmp4 = (FOV==1)'.*tmp4;
        else
            tmp4 = (FOV==0)';
        end
        tmp3 = w*tmp4;
        tmp2=tmp2+ws(ll)*log(tmp3);
    end
    tmp1 = tmp1 + w(jj) * tmp2;
end
reward = -tmp1-H_cond;
if reward < -10^-10
    error('1');
elseif reward < 0
    reward = 0;
end
end

function flag = inFOV(z,tar_pos,r)
A = tar_pos - z;
flag = sqrt(A(1,:).^2+(A(2,:).^2)) <= r;
end

function [particles,w] = PF(z,particles,w,y,r,Q,R,f,h,polyin,region,V)
N = size(particles,2);%粒子的个数
particles = f(particles);
particles = (mvnrnd(particles',Q))';
FOV = inFOV(z(1:2),particles,r);
P = normpdf(y,h(particles,z),sqrt(R));
%
P1 = normpdf(y,h(particles,z)+2*pi,sqrt(R));
P2 = normpdf(y,h(particles,z)-2*pi,sqrt(R));
%
for jj = 1:N
    if any([0;0] > particles(:,jj))||any([100;100] < particles(:,jj))
        w(jj) = 10^-20;
        continue
    end
    if region(ceil(particles(1,jj)),ceil(particles(2,jj))) == 0
        w(jj) = 10^-20;
        continue
    end
    if y == -100
        % if the target is outside FOV.
        %
        if FOV(jj)&&V(ceil(z(1)),ceil(z(2)),ceil(particles(1,jj)),ceil(particles(2,jj)))
            w(jj) = 10^-20;
        else
            w(jj) = 1;
        end
        %}
    else
        if FOV(jj)&&V(ceil(z(1)),ceil(z(2)),ceil(particles(1,jj)),ceil(particles(2,jj)))
            %bearing
            %{
            if h(particles(:,jj),z) +z(3) <= 0
                w(jj) = max([P(jj),P1(jj)]);
            else
                w(jj) = max([P(jj),P2(jj)]);
            end
            %}
            %range
            w(jj) = P(jj);
        else
            w(jj) = 10^-20;
        end
    end
%{
    for kk = 1:length(polyin)
        if isInside(polyin(kk),particles(:,jj)) == 1
            w(jj) = 0;
            break
        end
    end
%}  
end
%     for jj = 1:N
%         if any([0;0] > particles(:,jj))||any([50;50] < particles(:,jj))
%             w(ii) = 10^-20;
%         end
%     end
w = w./sum(w);%归一化的粒子权重
%重采样
M = 1/N;
U = rand(1)*M;
new_particles = zeros(2,N);
tmp_w = w(1);
i = 1;
jj = 1;
while (jj <= N)
    while (tmp_w < U+(jj-1)*M)
        i = i+1;
        tmp_w = tmp_w+w(i);
    end
    new_particles(:,jj) = particles(:,i);
    jj = jj + 1;
end
particles = new_particles;
w = repmat(1/N, N, 1);
end

function flag = isInside(obstacle,state)
flag = 0;
points = obstacle.points;
if state(1)<min(points(1,:))||state(1)>max(points(1,:))||state(2)<min(points(2,:))||state(2)>max(points(2,:))
    return
end
n = size(points,2);
jj = n;
%射线法
%
for ii = 1:n
    if ((points(2,ii)>state(2))~=(points(2,jj)>state(2)))&&(state(1)-points(1,ii)<(points(1,jj)-points(1,ii))*(state(2)-points(2,ii))/(points(2,jj)-points(2,ii)))
        flag = ~flag;
    end
    jj = ii;
end
%}
%叉积法
%{
for ii = 1:n
    if (state(1)-points(1,jj))*(points(2,ii)-points(2,jj))<(state(2)-points(2,jj))*(points(1,ii)-points(1,jj))
        return
    end
    jj = ii;
end
flag = 1;
%}
%Ax>=b
%{
A = zeros(n,2);
b = zeros(n,1);
for ii = 1:n
    A(ii,:) = [points(2,ii)-points(2,jj) points(1,jj)-points(1,ii)];
    b(ii) = points(1,jj)*A(ii,1)+points(2,jj)*A(ii,2);
    jj = ii;
end
flag = all(A*state>=b);
%}
end

function flag = isVisible(obstacle,state1,state2,region)
flag = 1;
points = obstacle.points;
if region(ceil(state1(1)),ceil(state1(2))) == 0 || region(ceil(state2(1)),ceil(state2(2))) == 0
    flag = 0;
    return
end
n = size(points,2);
jj = n;
for ii = 1:n
    if ((state1(1)-points(1,jj))*(points(2,ii)-points(2,jj))-(state1(2)-points(2,jj))*(points(1,ii)-points(1,jj)))*((state2(1)-points(1,jj))*(points(2,ii)-points(2,jj))-(state2(2)-points(2,jj))*(points(1,ii)-points(1,jj)))<0 ...
    &&((points(1,jj)-state1(1))*(state2(2)-state1(2))-(points(2,jj)-state1(2))*(state2(1)-state1(1)))*((points(1,ii)-state1(1))*(state2(2)-state1(2))-(points(2,ii)-state1(2))*(state2(1)-state1(1)))<0
        flag = 0;
        break
    end
end
end