%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Class for the mobile robot
% ver 1.0, Kangjie Zhou, 2023/9
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef RobotClass
    properties
        % motion specs
        traj;
        state; % [x;y;theta;v]
        a_lb;
        a_ub;
        w_lb;
        w_ub;
        v_lb;
        v_ub;
        Qr; % noise of motion

        % sensor specs
        sensor_type;
        theta0; % sensor range in angle
        rmin; % sensor range minimum
        rmax; % sensor range maximum
        inFOV_hist; 
        is_tracking;

        % for general nonlinear sensors
        h; % y=h(x)
        del_h; % gradient of h
        R; % variance(1-d)/covariance(high-d) for sensor model
        mdim; % dimension of sensor measurement

        % observation
        y; % observation measurement

        % target
        target;
        A;
        sdim; % dimension of state

        % filtering
        % xKF
        P; % estimation covariance
        P_hist;
        % PF
        particles; % array of particle positions [x;y]
        w; % particles' weight
        N; % particles' number
        
        % map
        map;

        est_pos; % estimated target position
        est_pos_hist; % estimation history

        % path planning
        mpc_hor;
        dt;
        optu;

        % configuration of optimization
        snum;

        % action space
        a_S; % search phase
        a_T; % tracking phase
        int_pts_S;
        int_pts_T;

        value_max;


        % performance metrics
        ml_pos;
        ent_pos;
    end

    methods
        function this = RobotClass(inPara)
            this.state = inPara.state;
            this.traj = inPara.traj;
            this.a_lb = inPara.a_lb;
            this.a_ub = inPara.a_ub;
            this.w_lb = inPara.w_lb;
            this.w_ub = inPara.w_ub;
            this.v_lb = inPara.v_lb;
            this.v_ub = inPara.v_ub;
            this.Qr = inPara.Qr;

            % sensor specs
            this.R = inPara.R;
            this.h = inPara.h;
            this.del_h = inPara.del_h;
            this.theta0 = inPara.theta0;
            this.rmin = inPara.minrange;
            this.rmax = inPara.maxrange;
            this.mdim = inPara.mdim;

            this.inFOV_hist = inPara.inFOV_hist; 
            this.is_tracking = inPara.is_tracking;

            % target
            this.target = inPara.target;
            this.A = [];
            this.sdim = inPara.sdim;

            % filtering
            this.sensor_type = inPara.sensor_type;
            % xKF
            this.est_pos = inPara.est_pos;
            this.P = inPara.P;
            this.est_pos_hist = [];
            this.P_hist = [];
            this.particles = inPara.particles;
            this.w = inPara.w;
            this.N = inPara.N;

            this.map = inPara.map;

            this.a_S = inPara.a_S;
            this.a_T = inPara.a_T;
            this.int_pts_S = inPara.int_pts_S;
            this.int_pts_T = inPara.int_pts_T;

            this.mpc_hor = inPara.mpc_hor;
            this.dt = inPara.dt;
            this.optu = [];
        end

        %% sensor modeling
        % approximate straightline edge of sensor FOV based on current
        % robot state
        function [a,b] = FOV(this,st)
            theta = st(3);
            x0 = st(1);
            y0 = st(2);
            alp1 = theta - this.theta0;
            alp2 = theta + this.theta0;
            a = [sin(alp1),-cos(alp1);-sin(alp2),cos(alp2)]; % [a1;a2]
            b = [x0*sin(alp1)-y0*cos(alp1);-x0*sin(alp2)+y0*cos(alp2)];%[b1;b2];
        end

        % determine if the target is in sensor FOV
%         function flag = inFOV(this,tar_pos)
%             [a,b] = this.FOV(this.state);
%             flag = (a(1,:)*tar_pos-b(1) <= 0) && (a(2,:)*tar_pos-b(2) <= 0)...
%                 && (norm(tar_pos-this.state(1:2)) <= this.r);
%         end

        function flag = inFOV(this,z,tar_pos)
            %
            A = tar_pos(1:2,:) - z(1:2);
            flag1 = sqrt(A(1,:).^2+(A(2,:).^2)) < this.rmax;
            flag2 = sqrt(A(1,:).^2+(A(2,:).^2)) > this.rmin;
            flag3 = (A(1,:)*cos(z(3))+A(2,:)*sin(z(3)))./sqrt(A(1,:).^2+(A(2,:).^2)) > cos(this.theta0/2);
            flag = flag1.*flag2.*flag3;
            %}
            %{
            A = tar_pos(1:2,:) - z(1:2);
            flag = sqrt(A(1,:).^2+(A(2,:).^2)) < r;
            %}
        end

        function flag = inFOV_red(this,z,tar_pos)
            %
            A = tar_pos(1:2,:) - z(1:2);
            flag1 = sqrt(A(1,:).^2+(A(2,:).^2)) < this.rmax-0.5;
            flag2 = sqrt(A(1,:).^2+(A(2,:).^2)) > this.rmin+0.2;
            flag3 = (A(1,:)*cos(z(3))+A(2,:)*sin(z(3)))./sqrt(A(1,:).^2+(A(2,:).^2)) > cos((this.theta0-10/180*pi)/2);
            flag = flag1.*flag2.*flag3;
            %}
            %{
            A = tar_pos(1:2,:) - z(1:2);
            flag = sqrt(A(1,:).^2+(A(2,:).^2)) < r;
            %}
        end

        %% measurement generation
        % generate a random measurement
        function y = sensorGen(this,fld)
            tar_pos = fld.target.pos;

            % draw FOV and target position. for debug purpose
            %this.drawFOV(this.state,fld,'cur',[0.9290 0.6940 0.1250])
            %hold on
            %plot(fld.target.pos(1),fld.target.pos(2),'b','markers',5,'Marker','*');

            if strcmp(this.sensor_type,'rb')
                % range-bearing sensor
                if this.inFOV(this.state,tar_pos)&&fld.map.V(ceil(this.state(1)),ceil(this.state(2)),ceil(tar_pos(1)),ceil(tar_pos(2)))
                    y = this.h(tar_pos,this.state)+(mvnrnd([0;0],this.R))';
                else
                    y = [-100;-100];
                end
            elseif strcmp(this.sensor_type,'ran')
                if this.inFOV(this.state,tar_pos)&&fld.map.V(ceil(z(1)),ceil(z(2)),ceil(tar_pos(1)),ceil(tar_pos(2)))
                    y = this.h(tar_pos,this.state)+normrnd(0,this.R);
                else
                    y = -100;
                end
            elseif strcmp(this.sensor_type,'lin')
                if this.inFOV(this.state,tar_pos)&&fld.map.V(ceil(z(1)),ceil(z(2)),ceil(tar_pos(1)),ceil(tar_pos(2)))
                    y = this.h(tar_pos,this.state)+(mvnrnd([0;0],this.R))';
                else
                    y = [-100;-100];
                end
            end
        end
        %% filtering
        % Kalman filter
        function this = KF(this,fld)
            % measurement
            y = this.y;

            % target
            tar = fld.target;
            A = tar.A;
            B = tar.B;

            Q = tar.Q;

            % current estimation
            x = this.est_pos;
            P = this.P;

            % sensor
            h = this.h;
            del_h = this.del_h;
            R = this.R;

            % prediction
            x_pred = A*x+B;
            P_pred = A*P*A'+Q;

            % update
            C = del_h(x,this.state(1:2));
            if sum(y-[-100;-100]) ~= 0
                % if an observation is obtained
                K = P_pred*C'/(C*P_pred*C'+R);
                x_next = x_pred+K*(y-h(x_pred,this.state(1:2)));%C*x_pred-(x_pred-this.state(1:2))
                P_next = P_pred-K*C*P_pred;
            else
                x_next = x_pred;
                P_next = P_pred;
            end

            this.est_pos = x_next;
            this.P = P_next;
            this.est_pos_hist = [this.est_pos_hist,x_next];
            this.P_hist = [this.P_hist,P_next];
            % this makes KF compatible with cvxPlanner_scp
            this.gmm_num = 1;
            this.wt = 1;
        end

       
        % particle filter
%         function this = PF(this,fld)
%             % target
%             tar = fld.target;
%             f = tar.f;
%             % sensor
%             h = this.h;
%             % measurement
%             y = this.y;
% 
%             particles = this.particles;
% 
%             %% particle filtering
%             np = size(particles,2); % number of particles
% 
%             % initalize particles weights
%             w = zeros(np,1);
% 
%             % state update
%             pred_par = zeros(2,np); % predicted particle state
%             for ii = 1:np
%                 pred_par(:,ii) = f(particles(:,ii));
%             end
%             pred_par = (mvnrnd(pred_par',fld.target.Q))';
% 
%             % weight update
%             for ii = 1:np
%                 if sum(y == -100) >= 1
%                     % if the target is outside FOV.
%                     if this.inFOV(pred_par(:,ii))
%                         w(ii) = 10^-20;
%                     else
%                         w(ii) = 1;
%                     end
%                 else
%                     if this.inFOV(pred_par(:,ii))
%                         if strcmp(this.sensor_type,'ran')
%                             % note: for 1-d Gaussian, normpdf accepts std
%                             w(ii) = normpdf(y,this.h(pred_par(:,ii),this.state(1:2)),sqrt(this.R));
%                         else
%                             % note: for high-d Gaussian, mvnpdf accepts
%                             % covariance
%                             w(ii) = mvnpdf(y,this.h(pred_par(:,ii),this.state(1:2)),this.R);
%                         end
%                     else
%                         w(ii) = 10^-20;
%                     end
%                 end
%             end
% 
%             % (10/4) set particles outside of field to have 0 weight
%             for ii = 1:np
%                 if any([fld.fld_cor(1);fld.fld_cor(3)] > pred_par(:,ii))||...
%                         any([fld.fld_cor(2);fld.fld_cor(4)] < pred_par(:,ii))
%                     w(ii) = 0;
%                 end
%             end
% 
%             w = w/sum(w);
% 
%             % resampling
%             % opt 1: plain random resampling
%             %{
%             idx = randsample(1:np, np, true, w);
%             new_particles = pred_par(:,idx);
%             this.particles = new_particles;
%             %}
% 
%             % opt 2: low variance sampling
%             %
%             M = 1/np;
%             U = rand(1)*M;
%             new_particles = zeros(size(pred_par));
%             tmp_w = w(1);
%             ii = 1;
%             jj = 1;
%             while (jj <= np)
%                 while (tmp_w < U+(jj-1)*M)
%                     ii = ii+1;
%                     tmp_w = tmp_w+w(ii);
%                 end
%                 new_particles(:,jj) = pred_par(:,ii);
%                 jj = jj + 1;
%             end
%             this.particles = new_particles;
%             %}
%         end

        function [particles,w] = PF(this,fld,sim,tt,ii,state,particles,w,y,flag)
            
            %particles = this.particles;
            R = this.R;
            h = this.h;
            N = size(particles,2);

            % particles = f(particles);
            % particles = (mvnrnd(particles',Q))';

            if flag == 1
                %{
            control = [fld.target.control(tt,ii,1);fld.target.control(tt,ii,2)];

            t = 1;
            particles(1:2,:) = particles(1:2,:) + [control(1).*cos(particles(3,:))*t;control(1).*sin(particles(3,:))*t];
            particles(3,:) = particles(3,:) + control(2)*t;
            particles = (mvnrnd(particles',fld.target.Q))';
                %}
                %particles(1:2,:) = particles(1:2,:) -1 + 2*[rand;rand];
                particles = (mvnrnd(particles',fld.target.Q))';
            end

            FOV = this.inFOV(state,particles(1:2,:));
            if strcmp(sim.sensor_type,'rb')
                mu = h(particles(1:2,:),state);
                P = normpdf(y(1),mu(1,:),sqrt(R(1,1)));
                %
                P0 = normpdf(y(2),mu(2,:),sqrt(R(2,2)));
                P1 = normpdf(y(2),mu(2,:)+2*pi,sqrt(R(2,2)));
                P2 = normpdf(y(2),mu(2,:)-2*pi,sqrt(R(2,2)));
            else
                P = normpdf(y,h(particles(1:2,:),z),sqrt(R));
                %
                P1 = normpdf(y,h(particles(1:2,:),z)+2*pi,sqrt(R));
                P2 = normpdf(y,h(particles(1:2,:),z)-2*pi,sqrt(R));
                %
            end
            for jj = 1:N
                if any([0;0] > particles(1:2,jj))||any([50;50] < particles(1:2,jj))
                    w(jj) = 10^-20;
                    continue
                end
                if this.map.region(ceil(particles(1,jj)),ceil(particles(2,jj))) < 0.3
                    w(jj) = 10^-20;
                    continue
                end
                if y == -100
                    % if the target is outside FOV.
                    %
                    if FOV(jj)&&fld.map.V(ceil(state(1)),ceil(state(2)),ceil(particles(1,jj)),ceil(particles(2,jj)))
                        w(jj) = 10^-20;
                    else
                        w(jj) = 1;
                    end
                    %}
                else
                    if FOV(jj)&&fld.map.V(ceil(state(1)),ceil(state(2)),ceil(particles(1,jj)),ceil(particles(2,jj)))
                        if strcmp(sim.sensor_type,'ran')
                            w(jj) = P(jj);
                        end
                        if strcmp(sim.sensor_type,'br')
                            if h(particles(1:2,jj),state) +z(3) <= 0
                                w(jj) = max([P(jj),P1(jj)]);
                            else
                                w(jj) = max([P(jj),P2(jj)]);
                            end
                        end
                        if strcmp(sim.sensor_type,'rb')
                            if mu(2,jj) + state(3) <= 0
                                w(jj) = max([P0(jj),P1(jj)]);
                            else
                                w(jj) = max([P0(jj),P2(jj)]);
                            end
                            w(jj) = w(jj)*P(jj);
                        end
                    else
                        w(jj) = 10^-20;
                    end
                end
            end
            w = w./sum(w);%归一化的粒子权重
            % resampling
            ess = 1/sum(w.^2);
            flag = 1;
            if flag == 0||flag == 1%&&ess > 0.5*N)
                M = 1/N;
                U = rand(1)*M;
                new_particles = zeros(3,N);
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
                w = repmat(1/N, 1, N);
            end
%             this.particles = particles;
%             this.w = w;
%             this.N = N;
        end

        %% planning
        function [this,optz,list_tmp] = Planner(this,fld,sim,plan_mode,list_tmp,ps,pt,tt,ii)

            is_tracking = this.is_tracking;

            if ~is_tracking
                a = this.a_S;
                interpolated_points = this.int_pts_S;
            else
                a = this.a_T;
                interpolated_points = this.int_pts_T;
            end
           
            list_tmp = [];
            root = Node_IMPFT;
            %Initialization
            root.num = 1;
            root.state = this.state;%匀速运动v=1
            root.hist = [];
            root.a = a;
            root.N = 0;
            root.Q = 0;
            root.parent = 0;
            root.children = [];
            root.children_maxnum = 18;
            root.is_terminal = 0;
            root.delete = 0;
            list_tmp = [list_tmp,root];

            if is_tracking
                max_depth = 4;
            else
                max_depth = 60;
            end
            %discount factor
            if is_tracking
                eta = 0.95;
            else
                eta = 0.7;
            end
            num = 1;% addtion point index

            if is_tracking
                planlen = 30;
            else
                planlen = 30;
            end

            B = this.particles;
            w = this.w;

            for jj = 1:planlen
                [list_tmp,Reward,num] = this.simulate(fld,sim,1,num,list_tmp,max_depth,eta,ii+1,tt,interpolated_points,a,B,w,pt,ps);
            end
            %}
            %{
    I = 1;
    if is_tracking
        I_delta = 0.2;
        while I > I_delta
            [list_tmp,Reward,num,I] = simulate(1,num,list_tmp,particles,max_depth,targetControl,h,R,r,Q,eta,w,is_tracking,polyin,region,V,ii,sensor,rb_motion_model,tt,planner,a1,G,max_depth,interpolated_points,state_estimation,a);
        end
            %}

            max_value = -10000;
            if isempty(list_tmp(1).children)
                optz = this.state;
            else
                val = zeros(length(list_tmp(1).children),1);
                for jj = 1:length(list_tmp(1).children)
                    val(jj) = list_tmp(list_tmp(1).children(jj)).Q;
                end
                [value_max,maxid] = max(val);
                if value_max == 0
                    optz = this.state;
                    this.traj = [this.traj repmat(optz(1:3),1,21)];
                else
                    opt = list_tmp(1).children(maxid);
                    optz = list_tmp(opt).state;
                    id = list_tmp(opt).a_num;
                    z = this.state;
                    if is_tracking
                        p = pt{id};
                    else
                        p = ps{id};
                    end
                    this.traj = [this.traj [p(:,1)'*sin(z(3))+p(:,2)'*cos(z(3))+z(1);-p(:,1)'*cos(z(3))+p(:,2)'*sin(z(3))+z(2);z(3)+p(:,3)']];
                end
            end
            this.value_max = value_max;

            %list(ii,1:length(list_tmp)) = list_tmp;
        end

        %% policy tree construction
        function [list_tmp,Reward,num] = simulate(this,fld,sim,begin,num,list_tmp,depth,eta,simIndex,tt,interpolated_points,a,B,w,pt,ps) 
            if this.is_tracking
                K = 3;
            else
                K = 1;
            end
            alpha = 0.1;
            control = [fld.target.control(tt,simIndex,1);fld.target.control(tt,simIndex,2)];

            if depth == 0
                Reward = 0;
                return
            else
                z = list_tmp(begin).state(1:3);

                list_tmp(begin).N = list_tmp(begin).N+1;
                %if length(list_tmp(begin).children) == list_tmp(begin).children_maxnum
                if isempty(list_tmp(begin).a)
                    if ~isempty(list_tmp(begin).children)
                        [begin,list_tmp] = this.best_child(begin,0.732,list_tmp);
                    else
                        begin_tmp = begin;
                        begin = list_tmp(begin).parent;
                        if begin ~= 0
                            list_tmp(begin).children(find(list_tmp(begin).children==begin_tmp))=[];
                        end
                        Reward = -100;
                        return
                    end
                else
                    num = num + 1;
                    [list_tmp,begin,flag2] = this.expand(fld,sim,begin,num,list_tmp,0,1,interpolated_points,a,pt,ps);
                    if flag2 == 0
                        num = num - 1;
                        if ~isempty(list_tmp(begin).children)
                            [begin,list_tmp] = this.best_child(begin,0.732,list_tmp);
                        else
                            begin_tmp = begin;
                            begin = list_tmp(begin).parent;
                            if begin ~= 0
                                list_tmp(begin).children(find(list_tmp(begin).children==begin_tmp))=[];
                            end
                            Reward = -100;
                            return
                        end
                    end
                end
                
                num_a = begin;

                %
                id = list_tmp(num_a).a_num;
                if this.is_tracking
                    p = pt{id};
                else
                    p = ps{id};
                end

                tmp = [p(:,1)'*sin(z(3))+p(:,2)'*cos(z(3))+z(1);-p(:,1)'*cos(z(3))+p(:,2)'*sin(z(3))+z(2);z(3)+p(:,3)'];

                wrong = 0;
                for jj = 1:21
                    if any([1;1] >= tmp(1:2,jj))||any([49;49] <= tmp(1:2,jj))||this.map.region(ceil(tmp(1,jj)),ceil(tmp(2,jj))) < 0.3
                        wrong = 1;
                        break
                    end
                end

                if ~this.is_tracking
                    if wrong
                        %if any([0;0] >= list_tmp(num_a).state(1:2))||any([50;50] <= list_tmp(num_a).state(1:2))||any([0;0] >= list_tmp(num_a).inter_state(1:2))||any([50;50] <= list_tmp(num_a).inter_state(1:2))||fld.map.region_exp(ceil(list_tmp(num_a).state(1)),ceil(list_tmp(num_a).state(2))) == 0||fld.map.region_exp(ceil(list_tmp(num_a).inter_state(1)),ceil(list_tmp(num_a).inter_state(2))) == 0
                        list_tmp(num_a).N = list_tmp(num_a).N+1;
                        %%%% need to be modified
                        list_tmp(num_a).Q = -100;
                        Reward = -100;
                        return
                    end
                end
                %}

                num_a = begin;
                list_tmp(num_a).N = list_tmp(num_a).N+1;

                %{
                t = 1;
                B(1:2,:) = B(1:2,:) + [control(1).*cos(B(3,:))*t;control(1).*sin(B(3,:))*t];
                B(3,:) = B(3,:) + control(2)*t;
                B = (mvnrnd(B',fld.target.Q))';
                %}
                %B(1:2,:) = B(1:2,:) -1 + 2*[rand;rand];
                B = (mvnrnd(B',fld.target.Q))';

                % feasible particles
                jj = 1;
                for ii = 1:size(B,2)
                    if any([1;1] > B(1:2,jj))||any([49;49] < B(1:2,jj))||this.map.region_exp(ceil(B(1,jj)),ceil(B(2,jj))) < 0.3
                        B(:,jj) = [];
                        w(jj) = [];
                        continue
                    end
                    jj = jj+1;
                end
                w = w./sum(w);
                N = size(B,2);

                state_estimation = B*w';

                state = list_tmp(num_a).state;
                if this.is_tracking
                    reward = this.MI(fld,sim,list_tmp(num_a).state,B,w) + this.MI(fld,sim,list_tmp(num_a).inter_state,B,w);
                else
                    reward = this.MI(fld,sim,list_tmp(num_a).state,B,w);
                end

                if length(list_tmp(begin).children) <= K*(list_tmp(begin).N^alpha)
                    B_tmp = B;
                    jj = 1;
                    ii = 1;
                    while(ii<=N)
                        if this.inFOV(state(1:3),B(:,jj)) == 0||fld.map.V(ceil(state(1)),ceil(state(2)),ceil(B(1,jj)),ceil(B(2,jj))) == 0
                            B_tmp(:,jj) = [];
                        else
                            jj = jj + 1;
                        end
                        ii = ii + 1;
                    end
                    if strcmp(sim.sensor_type,'rb')
                        if size(B_tmp,2) == 0
                            o = [-100;-100];
                        else
                            mu = zeros(2,size(B_tmp,2));
                            for ii = 1:size(B_tmp,2)
                                mu(:,ii) = this.h(B_tmp(1:2,ii),state);
                            end
                            gm = gmdistribution(mu',this.R);
                            o = random(gm);
                        end
                    else
                        if size(B_tmp,2) == 0
                            o = -100;
                        else
                            mu = zeros(size(B_tmp,2),1);
                            for ii = 1:size(B_tmp,2)
                                mu(ii) = this.h(B_tmp(1:2,ii),state);
                            end
                            gm = gmdistribution(mu,this.R);
                            o = random(gm);
                        end
                    end
                    num = num + 1;
                    [list_tmp,begin] = this.expand(fld,sim,begin,num,list_tmp,o,2,interpolated_points,a,pt,ps);
                    flag = 1;
                else
                    begin = list_tmp(begin).children(randperm(length(list_tmp(begin).children),1));
                    o = list_tmp(begin).hist(6:end,end);
                    flag = 0;
                end
                %o=-100;
                num_o = begin;
                if o~=-100%这里如果不走PF可能会出现infeasible的粒子
                    [B,w] = this.PF(fld,sim,tt,ii,list_tmp(num_a).state(:,1),B,w,o,0);
                end
                if flag == 1
                    node = list_tmp(begin);
                    simIndex = simIndex + 1;
                    rollout = this.rollOut(fld,sim,node,eta,depth-1,B,w,simIndex,tt);

%                     if this.is_tracking
%                         rollout = 0;
%                     end
                    Reward = reward + eta*rollout;
                else
                    simIndex = simIndex + 1;
                    [list_tmp,Reward,num] = this.simulate(fld,sim,begin,num,list_tmp,depth-1,eta,simIndex,tt,interpolated_points,a,B,w,pt,ps);
                    Reward = reward + eta*Reward;
                end
                list_tmp(num_o).N = list_tmp(num_o).N+1;
                list_tmp(num_a).Q = list_tmp(num_a).Q + (Reward-list_tmp(num_a).Q)/list_tmp(num_a).N;
                %     if depth == max_depth&&list_tmp(num_a).N>2
                %         I = abs((Reward-list_tmp(num_a).Q)/list_tmp(num_a).N)/(list_tmp(num_a).Q-(Reward-list_tmp(num_a).Q)/list_tmp(num_a).N);
                %     else
                %         I = 1;
                %     end
                %     I = 1;
            end
        end

        function [v,list_tmp] = best_child(this,begin,c,list_tmp)
            max = -10000;
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

        function [list_tmp,begin,flag2] = expand(this,fld,sim,begin,num,list_tmp,o,tmp,interpolated_points,a,pt,ps)
            t = 1;
            flag2 = 1;
            node = list_tmp(begin);
            state = zeros(4,1);
            if tmp == 1 %action
                %
                ii = randperm(size(list_tmp(begin).a,2),1);
                action = list_tmp(begin).a(:,ii);
                list_tmp(begin).a(:,ii) = [];

                state(1) = node.state(1)+action(1)*sin(node.state(3))*t+action(2)*cos(node.state(3))*t;
                state(2) = node.state(2)-action(1)*cos(node.state(3))*t+action(2)*sin(node.state(3))*t;
                state(3) = node.state(3)+action(3)*t;
                state(4) = action(4);

%                 inter_state = [interpolated_points(action(5),1,1)*sin(node.state(3))+interpolated_points(action(5),1,2)*cos(node.state(3))+node.state(1);
%                     -interpolated_points(action(5),1,1)*cos(node.state(3))+interpolated_points(action(5),1,2)*sin(node.state(3))+node.state(2);
%                     node.state(3)+interpolated_points(action(5),1,3)];

                list_tmp(begin).children_maxnum = list_tmp(begin).children_maxnum-1;
                %}
                %{
                while(1)
                    flag = 0;
                    inregion = 1;
                    if isempty(list_tmp(begin).a)%&&isempty(list_tmp(begin).children)
                        flag2 = 0;
                        return
                    end
                    ii = randperm(size(list_tmp(begin).a,2),1);
                    action = list_tmp(begin).a(:,ii);
                    list_tmp(begin).a(:,ii) = [];

                    state(1) = node.state(1)+action(1)*sin(node.state(3))*t+action(2)*cos(node.state(3))*t;
                    state(2) = node.state(2)-action(1)*cos(node.state(3))*t+action(2)*sin(node.state(3))*t;
                    state(3) = node.state(3)+action(3)*t;
                    state(4) = action(4);

                    inter_state = [interpolated_points(action(5),1,1)*sin(node.state(3))+interpolated_points(action(5),1,2)*cos(node.state(3))+node.state(1);
                        -interpolated_points(action(5),1,1)*cos(node.state(3))+interpolated_points(action(5),1,2)*sin(node.state(3))+node.state(2);
                        node.state(3)+interpolated_points(action(5),1,3)];

                    %                     if norm(state(1:2)-state_estimation(1:2)) < 1%&&is_tracking
                    %                         continue
                    %                     end

                    id = action(5);
                    if this.is_tracking
                        p = pt{id};
                    else
                        p = ps{id};
                    end

                    z = state(1:3);
                    tmp = [p(:,1)'*sin(z(3))+p(:,2)'*cos(z(3))+z(1);-p(:,1)'*cos(z(3))+p(:,2)'*sin(z(3))+z(2);z(3)+p(:,3)'];

                    wrong = 0;
                    for jj = 1:21
                        if any([1;1] >= tmp(1:2,jj))||any([49;49] <= tmp(1:2,jj))||this.map.region(ceil(tmp(1,jj)),ceil(tmp(2,jj))) < 0.3
                            wrong = 1;
                            break
                        end
                    end

                    if wrong
                        break
                    end

                    %{
                    if any([0;0] >= state(1:2))||any([50;50] <= state(1:2))||any([0;0] >= inter_state(1:2))||any([50;50] <= inter_state(1:2))
                        flag = 1;
                        inregion = 0;
                        %list_tmp(begin).Q = list_tmp(begin).Q - 1000;
                    end
                    if inregion == 1
                        if fld.map.region_exp(ceil(state(1)),ceil(state(2))) == 0||fld.map.V(ceil(node.state(1)),ceil(node.state(2)),ceil(state(1)),ceil(state(2))) == 0%||region(ceil(inter_state(1)),ceil(inter_state(2))) == 0
                            flag = 1;
                            %list_tmp(begin).Q = list_tmp(begin).Q - 1000;
                        end
                    end

                    if state(4)>=0&&state(4)<=50&&flag == 0&&this.is_tracking==0
                        break
                    end
                    if state(4)>=0&&state(4)<=30&&flag==0&&this.is_tracking==1%&&abs(action(3))<0.4
                        break
                    end
                    %}
                    list_tmp(begin).children_maxnum = list_tmp(begin).children_maxnum-1;
                end
                %}
                new = Node_IMPFT;
                new.num = num;
                new.a_num = action(5);
                new.state = state;
                inter_state = [interpolated_points(action(5),1,1)*sin(node.state(3))+interpolated_points(action(5),1,2)*cos(node.state(3))+node.state(1);
                    -interpolated_points(action(5),1,1)*cos(node.state(3))+interpolated_points(action(5),1,2)*sin(node.state(3))+node.state(2);
                    node.state(3)+interpolated_points(action(5),1,3)];
                new.inter_state = inter_state;
                if strcmp(sim.sensor_type,'rb')
                    hist = [action;0;0];
                else
                    hist = [action;0];
                end
                new.hist = [node.hist,hist];
            else % observation
                new = node;
                new.num = num;
                new.hist(6:end,end) = o;
            end
            new.a = a;
            new.N = 0;
            new.Q = 0;
            new.parent = begin;
            new.children = [];
            new.children_maxnum = 18;
            list_tmp(begin).children = [list_tmp(begin).children,new.num];
            new.is_terminal = 0;
            new.delete = 0;
            list_tmp(num) = new;
            begin = num;
        end

        function reward = rollOut(this,fld,sim,node,eta,depth,B,w,simIndex,tt)
            %{
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
        reward = 0;
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
            %}
            if depth == 0
                reward = 0;
                return
            else
                %{
                if simIndex <= 210
                    control = [fld.target.control(tt,simIndex,1);fld.target.control(tt,simIndex,2)];
                else
                    control = [0;0];
                end

                Q = fld.target.Q;
                t = 1;
                B(1:2,:) = B(1:2,:) + [control(1).*cos(B(3,:))*t;control(1).*sin(B(3,:))*t];
                B(3,:) = B(3,:) + control(2)*t;
                B = (mvnrnd(B',Q))';
                %}
                %B(1:2,:) = B(1:2,:) -1 + 2*[rand;rand];
                B = (mvnrnd(B',fld.target.Q))';

                B_tmp1 = B;
                %{
    jj = 1;
    for ii = 1:size(B_tmp1,2)
        if any([0;0] > B_tmp1(1:2,jj))||any([100;100] < B_tmp1(1:2,jj))||region(ceil(B_tmp1(1,jj)),ceil(B_tmp1(2,jj))) == 0
            B_tmp1(:,jj) = [];
            %w(jj) = [];
            continue
        end
        jj = jj+1;
    end
                %}
                flag1 = ~any(zeros(2,size(B,2))>B_tmp1(1:2,:));
                flag2 = ~any(50*ones(2,size(B,2))<B_tmp1(1:2,:));
                flag = flag1.*flag2;
                B_tmp1(1,:) = B_tmp1(1,:).*flag;
                B_tmp1(2,:) = B_tmp1(2,:).*flag;
                B_tmp1(3,:) = B_tmp1(3,:).*flag;
                B_tmp1(:,any(B_tmp1,1)==0)=[];

                % 得加
                %%%% need to be modified
                %{
                if ~this.is_tracking&&norm([B(1,:)*w';B(2,:)*w']-node.state(1:2))<10
                %if norm([B(1,:)*w';B(2,:)*w']-node.state(1:2))<50
                    flag3 = fld.map.region_exp(ceil(B_tmp1(1,:)),ceil(B_tmp1(2,:)));
                    flag3 = diag(flag3)';
                    B_tmp1(1,:) = B_tmp1(1,:).*flag3;
                    B_tmp1(2,:) = B_tmp1(2,:).*flag3;
                    B_tmp1(3,:) = B_tmp1(3,:).*flag3;
                    B_tmp1(:,any(B_tmp1,1)==0)=[];
                end
                %}

                if isempty(B_tmp1)
                    reward = 0;
                    return
                end

                B_tmp2 = zeros(3,size(B,2));
                B_tmp2(:,1:size(B_tmp1,2)) = B_tmp1;
                for jj = size(B_tmp1,2)+1:size(B,2)
                    B_tmp2(:,jj) = B_tmp1(:,randperm(size(B_tmp1,2),1));
                end
                B = B_tmp2;
                %w = w./sum(w);

                %
                if this.is_tracking
                    threshold = 0.9;
                else
                    threshold = 0.4;
                end

                if rand < threshold
                    if isempty(node.a)
                        reward = 0;
                    else
                        id = randperm(size(node.a,2),1);
                        action = node.a(:,id);
                        state = node.state;
                        state(1) = node.state(1)+action(1)*sin(node.state(3))+action(2)*cos(node.state(3));
                        state(2) = node.state(2)-action(1)*cos(node.state(3))+action(2)*sin(node.state(3));
                        state(3) = node.state(3)+action(3);
                        state(4) = action(4);
                        if any([0;0] >= state(1:2))||any([50;50] <= state(1:2))||this.map.region_exp(ceil(state(1)),ceil(state(2))) < 0.3||fld.map.V(ceil(node.state(1)),ceil(node.state(2)),ceil(state(1)),ceil(state(2))) == 0||state(4)<0%||state(4)>5
                            reward = 0;
                            node.a(:,id) = [];
                        else
                            reward = MI(this,fld,sim,state,B,w);
                            node.state = state;
                        end
                    end
                else
                    action_opt = [];
                    target = [B(1,:)*w';B(2,:)*w'];
                    mindis = 100000;
                    for jj = 1:size(node.a,2)
                        action = node.a(:,jj);
                        state = node.state;
                        state(1) = node.state(1)+action(1)*sin(node.state(3))+action(2)*cos(node.state(3));
                        state(2) = node.state(2)-action(1)*cos(node.state(3))+action(2)*sin(node.state(3));
                        state(3) = node.state(3)+action(3);
                        state(4) = action(4);
                        if any([0;0] >= state(1:2))||any([50;50] <= state(1:2))||this.map.region_exp(ceil(state(1)),ceil(state(2))) < 0.3||fld.map.V(ceil(node.state(1)),ceil(node.state(2)),ceil(state(1)),ceil(state(2))) == 0||state(4)<0%||state(4)>5
                            continue
                        end
                        %
                        if norm(state(1:2)-target) < mindis
                            mindis = norm(state(1:2)-target);
                            action_opt = action;
                        end
                    end
                    if isempty(action_opt)
                        reward = 0;
                    else
                        node.state(1) = node.state(1)+action_opt(1)*sin(node.state(3))+action_opt(2)*cos(node.state(3));
                        node.state(2) = node.state(2)-action_opt(1)*cos(node.state(3))+action_opt(2)*sin(node.state(3));
                        node.state(3) = node.state(3)+action_opt(3);
                        node.state(4) = action_opt(4);
                        reward = MI(this,fld,sim,state,B,w);
                    end
                end
                %{
                if ~this.is_tracking
                    sim.plotSim(this,fld,state,B,tt,simIndex);
                    %pause(0.01);
                    clf
                end
                %}
                if reward>0.1 &&~this.is_tracking
                    return
                end
                reward = reward + eta*this.rollOut(fld,sim,node,eta,depth-1,B,w,simIndex+1,tt);
            end
        end

        %% objective function
        
        function reward = MI(this,fld,sim,state,particles,w)
            %
            if any([0;0] >= state(1:2))||any([50;50] <= state(1:2))||this.map.region_exp(ceil(state(1)),ceil(state(2))) < 0.3
                reward = -1000;
                return
            end
            %}
            R = this.R;
            H_cond = 0;
            H0 = 0.5*size(R,1)*(log(2*pi)+1)+0.5*log(det(R));
            %H0 = 0.5*(log(2*pi)+1)+0.5*log(det(R));
            N = size(particles,2);
            visibility = zeros(1,N);
            for jj = 1:N
                visibility(jj) = fld.map.V(ceil(state(1)),ceil(state(2)),ceil(particles(1,jj)),ceil(particles(2,jj)));
            end

            if this.is_tracking
                FOV = this.inFOV_red(state(1:3),particles).*visibility;
            else
                FOV = this.inFOV(state(1:3),particles).*visibility;
            end
            %FOV = inFOV(state(1:3),particles,1,r,pi/2).*visibility;

            if ~any(FOV)
                reward = 0;
                return;
            end

            ratio = 1;

            %{
            if is_tracking == 1%&&~all(FOV)
                particles_tmp1 = zeros(3,N);
                particles_tmp1(1,:) = particles(1,:).*FOV;
                particles_tmp1(2,:) = particles(2,:).*FOV;
                particles_tmp1(3,:) = zeros(1,N);
                w = w.*FOV;
                particles_tmp1(:,any(particles_tmp1,1)==0)=[];
                w(:,any(w,1)==0)=[];

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
                ratio = length(w)/N;
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
                N = length(w);
                %FOV = ones(1,N);
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

            % particle simplification
            Cidx = zeros(size(particles,2),2);
            flag = zeros(200,200);
            N = 0;
            if this.is_tracking
                grid_size = 0.5;
            else
                grid_size = 2.5;
            end
            for mm = 1:size(particles,2)
                id1 = ceil(particles(1,mm)/grid_size)+5;
                Cidx(mm,1) = id1;
                id2 = ceil(particles(2,mm)/grid_size)+5;
                Cidx(mm,2) = id2;
                if flag(id1,id2) == 0
                    N = N + 1;
                    flag(id1,id2) = N;
                end
            end
            %N
            particles_tmp = particles;
            w_tmp = w;
            particles = zeros(3,N);
            w = zeros(1,N);
            for mm = 1:size(particles_tmp,2)
                w(flag(Cidx(mm,1),Cidx(mm,2))) = w(flag(Cidx(mm,1),Cidx(mm,2))) + w_tmp(mm);
            end
            for mm = 1:size(particles_tmp,2)
                particles(:,flag(Cidx(mm,1),Cidx(mm,2))) = particles(:,flag(Cidx(mm,1),Cidx(mm,2))) + particles_tmp(:,mm).*w_tmp(mm)./w(flag(Cidx(mm,1),Cidx(mm,2)));
            end
            %
            %%是否要修改 改了有问题（好像把atan2的突变问题解决后又没问题了）
            %
            visibility = zeros(1,N);
            for jj = 1:N
                visibility(jj) = fld.map.V(ceil(state(1)),ceil(state(2)),ceil(particles(1,jj)),ceil(particles(2,jj)));
            end
            %
            if this.is_tracking
                FOV = this.inFOV_red(state(1:3),particles).*visibility;
            else
                FOV = this.inFOV(state(1:3),particles).*visibility;
            end
            %}
            %FOV = inFOV(state(1:3),particles,1,r,pi/2).*visibility;
            %}
            %FOV = inFOV(state(1:3),particles,r);

            for jj = 1:N
                H_temp = w(jj)*FOV(jj);
                H_cond = H_cond+H_temp;
            end
            H_cond = H0*H_cond;

            mu = zeros(N,1);%观测的均值矩阵，第i行表示第i个粒子在未来T个时间步的观测均值
            if strcmp(sim.sensor_type,'rb')
                mu = zeros(N,2);
            end
            for jj = 1:N
                if strcmp(sim.sensor_type,'ran')
                    mu(jj) = sqrt(sum((state(1:2)-particles(1:2,jj)).^2)+0.1);
                end
                if strcmp(sim.sensor_type,'br')
                    mu(jj) = atan2(particles(2,jj)-state(2),particles(1,jj)-state(1));%-state(3);%state(3)~=0在目前解决atan2突变的方法有问题
                end
                if strcmp(sim.sensor_type,'rb')
                    mu(jj,:) = [sqrt(sum((state(1:2)-particles(1:2,jj)).^2)+0.1),atan2(particles(2,jj)-state(2),particles(1,jj)-state(1))];%-state(3);%state(3)~=0在目前解决atan2突变的方法有问题
                end
            end
            %解决atan2突变问题
            if strcmp(sim.sensor_type,'br')
                if range(mod(mu,2*pi))>range(rem(mu,2*pi))
                    mu = rem(mu,2*pi);
                else
                    mu = mod(mu,2*pi);
                end
            end
            if strcmp(sim.sensor_type,'rb')
                if range(mod(mu(:,2),2*pi))>range(rem(mu(:,2),2*pi))
                    mu(:,2) = rem(mu(:,2),2*pi);
                else
                    mu(:,2) = mod(mu(:,2),2*pi);
                end
            end
            %
            %构造sigma点的参数
            nx = 1;
            if strcmp(sim.sensor_type,'rb')
                nx = 2;
            end
            
            %b = repmat({R},nx,1);
           
            b = repmat({R},1,1);
            V = blkdiag(b{:});
            X = sqrtm(V);
            lambda = 2;
            ws = zeros(1,2*nx+1);
            ws(1) = lambda/(lambda+nx);
            ws(2:end) = repmat(1/2/(lambda+nx),1,2*nx);
            tmp1 = 0;
            for jj = 1:N
                sigma = zeros(2*nx+1,nx);
                sigma(1,:) = mu(jj,:);
                for ss= 1:nx
                    sigma(2*ss,:) = mu(jj,:) + sqrt(lambda+nx)*X(:,ss)';
                    sigma(2*ss+1,:) = mu(jj,:) - sqrt(lambda+nx)*X(:,ss)';
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
                    %
                    if FOV(jj)==1
                        if strcmp(sim.sensor_type,'rb')
                            tmp4 = (FOV==1)'.*normpdf(sigma(ll,1),mu(:,1),sqrt(R(1,1))).*normpdf(sigma(ll,2),mu(:,2),sqrt(R(2,2)));
%                             tmp4 = (FOV==1)'.*mvnpdf(sigma(ll,:),mu,R);
                        else
                            tmp4 = (FOV==1)'.*normpdf(sigma(ll),mu,sqrt(R));
                        end
                    else
                        tmp4 = (FOV==0)';
                    end
                    tmp3 = w*tmp4;
                    %
                    tmp2=tmp2+ws(ll)*log(tmp3);
                end
                tmp1 = tmp1 + w(jj) * tmp2;
            end
            %}
            %Taylor series expansion (Charrow)
            %{
            tmp1 = 0;
            for jj = 1:N
                tmp2 = 0;
                for kk = 1:N
                    if FOV(jj)==1&&FOV(kk)==1
                        tmp2 = tmp2+w(kk)*normpdf(mu(jj),mu(kk),sqrt(R));
                    elseif  FOV(jj)==FOV(kk)
                        tmp2 = tmp2+w(kk);
                    end
                end
                tmp1 = tmp1 + w(jj)*log(tmp2);
            end
            %}
            reward = -tmp1-H_cond;
            reward = reward*ratio;
            %reward = -tmp1;
            if reward < -10^1
                error('1');
            elseif reward < 1e-10
                reward = 0;
            end
        end

        %% robot state updating
        function this = updState(this,u)
            % update robot's actual state using control input
            st = this.state;

            if u(1) > this.w_ub
                fprintf('[main loop] Robot.m, line %d, u(1)=%d> w_ub. Adjusted to upper bound\n',MFileLineNr(),u(1))
                u(1) = this.w_ub;
            elseif u(1) < this.w_lb
                fprintf('[main loop] Robot.m, line %d, u(1)=%d< w_lb. Adjusted to lower bound\n',MFileLineNr(),u(1))
                u(1) = this.w_lb;
            end

            if u(2) > this.a_ub
                fprintf('[main loop] Robot.m, line %d, u(2)=%d> a_ub. Adjusted to upper bound\n',MFileLineNr(),u(2))
                u(2) = this.a_ub;
            elseif u(2) < this.a_lb
                fprintf('[main loop] Robot.m, line %d, u(2)=%d< a_lb. Adjusted to upper bound\n',MFileLineNr(),u(2))
                u(2) = this.a_lb;
            end

            this.optu = [this.optu,u(:,1)];
            dt = this.dt;
            this.state = st+[st(4)*cos(st(3));st(4)*sin(st(3));u(:,1)]*dt;
            % (10/9) added noise in motion model
            new_state = mvnrnd(this.state',this.Qr)';
            this.state = new_state;

            if this.state(4) > this.v_ub
                fprintf('[main loop] Robot.m, line %d, z(4)=%d> v_ub. Adjusted to upper bound\n',MFileLineNr(),this.state(4))
                this.state(4) = this.v_ub;
            elseif this.state(4) < this.v_lb
                fprintf('[main loop] Robot.m, line %d, z(4)=%d< v_lb. Adjusted to upper bound\n',MFileLineNr(),this.state(4))
                this.state(4) = this.v_lb;
            end

            if this.state(1) > 50
                fprintf('[main loop] Robot.m, line %d, z(1)=%d> 50. Adjusted to upper bound\n',MFileLineNr(),this.state(1))
                this.state(1) = 50;
            elseif this.state(1) < 0
                fprintf('[main loop] Robot.m, line %d, z(1)=%d< 0. Adjusted to upper bound\n',MFileLineNr(),this.state(1))
                this.state(1) = 0;
            end

            if this.state(2) > 50
                fprintf('[main loop] Robot.m, line %d, z(1)=%d> 50. Adjusted to upper bound\n',MFileLineNr(),this.state(2))
                this.state(2) = 50;
            elseif this.state(2) < 0
                fprintf('[main loop] Robot.m, line %d, z(1)=%d< 0. Adjusted to upper bound\n',MFileLineNr(),this.state(2))
                this.state(2) = 0;
            end

            %%%%% there should be psd checker for P. May fill this part
            %%%%% later

            this.traj = [this.traj,this.state];
        end

        function z = simState(this,u,z0)
            % used in optimizer for simulating state given input u
            if nargin > 2
                st = z0;
            else
                st = this.state;
            end

            dt = this.dt;
            len = size(u,2);
            z = zeros(length(st),len+1);
            z(:,1) = st;
            for ii = 1:len
                z(:,ii+1) = z(:,ii)+[z(4,ii)*cos(z(3,ii));z(4,ii)*sin(z(3,ii));u(:,ii)]*dt;
            end
        end

        %% %%%%% visualization for debug purpose

        function drawFOV(this,z,fld,fov_mode,color)
            if strcmp(fov_mode,'plan')
                % draw the FOV for planned robot state
                N = this.mpc_hor;
                for ii = 1:N+1
                    a1 = z(3,ii)-this.theta0;  % A random direction
                    a2 = z(3,ii)+this.theta0;
                    t = linspace(a1,a2,50);
                    x0 = z(1,ii);
                    y0 = z(2,ii);
                    x1 = z(1,ii) + this.r*cos(t);
                    y1 = z(2,ii) + this.r*sin(t);
                    plot([x0,x1,x0],[y0,y1,y0],'r--','LineWidth',0.5)
                end
            elseif strcmp(fov_mode,'cur')
                % draw the FOV for current robot state
                rmin = this.rmin;
                rmax = this.rmax;
                theta0 = this.theta0;
                plot([z(1)+rmin*cos(z(3)+theta0/2),z(1)+rmax*cos(z(3)+theta0/2)],[z(2)+rmin*sin(z(3)+theta0/2),z(2)+rmax*sin(z(3)+theta0/2)],'Color',color,'LineStyle','--',LineWidth=2);
                plot([z(1)+rmin*cos(z(3)-theta0/2),z(1)+rmax*cos(z(3)-theta0/2)],[z(2)+rmin*sin(z(3)-theta0/2),z(2)+rmax*sin(z(3)-theta0/2)],'Color',color,'LineStyle','--',LineWidth=2);
                theta=linspace(z(3)-theta0/2,z(3)+theta0/2);
                plot(z(1)+rmin*cos(theta),z(2)+rmin*sin(theta),'Color',color,'LineStyle','--',LineWidth=1);
                plot(z(1)+rmax*cos(theta),z(2)+rmax*sin(theta),'Color',color,'LineStyle','--',LineWidth=1);
            end
%             xlim([fld.fld_cor(1),fld.fld_cor(2)]);
%             ylim([fld.fld_cor(3),fld.fld_cor(4)]);
%             box on
%             axis equal
%             drawnow
        end

        function drawFOV_red(this,z,fld,fov_mode,color)
            if strcmp(fov_mode,'plan')
                % draw the FOV for planned robot state
                N = this.mpc_hor;
                for ii = 1:N+1
                    a1 = z(3,ii)-this.theta0;  % A random direction
                    a2 = z(3,ii)+this.theta0;
                    t = linspace(a1,a2,50);
                    x0 = z(1,ii);
                    y0 = z(2,ii);
                    x1 = z(1,ii) + this.r*cos(t);
                    y1 = z(2,ii) + this.r*sin(t);
                    plot([x0,x1,x0],[y0,y1,y0],'r--','LineWidth',0.5)
                end
            elseif strcmp(fov_mode,'cur')
                % draw the FOV for current robot state
                rmin = this.rmin+0.2;
                rmax = this.rmax-0.5;
                theta0 = this.theta0-10/180*pi;
                plot([z(1)+rmin*cos(z(3)+theta0/2),z(1)+rmax*cos(z(3)+theta0/2)],[z(2)+rmin*sin(z(3)+theta0/2),z(2)+rmax*sin(z(3)+theta0/2)],'Color',color,'LineStyle','--',LineWidth=2);
                plot([z(1)+rmin*cos(z(3)-theta0/2),z(1)+rmax*cos(z(3)-theta0/2)],[z(2)+rmin*sin(z(3)-theta0/2),z(2)+rmax*sin(z(3)-theta0/2)],'Color',color,'LineStyle','--',LineWidth=2);
                theta=linspace(z(3)-theta0/2,z(3)+theta0/2);
                plot(z(1)+rmin*cos(theta),z(2)+rmin*sin(theta),'Color',color,'LineStyle','--',LineWidth=1);
                plot(z(1)+rmax*cos(theta),z(2)+rmax*sin(theta),'Color',color,'LineStyle','--',LineWidth=1);
            end

        end
    end
end