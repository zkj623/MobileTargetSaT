%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize parameters for the simulation
% Kangjie Zhou, 2023/9
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Simulation Setup
importfile('APFT_small_scene1.mat');
importfile('structured_map.mat');

traj_rbt = cell(5,50);
particles_all = cell(5,50,200);
est_all = cell(5,50,200);
obs_all = cell(5,50,200);

% addpath('C:\Program Files\MATLAB\cvx\functions\vec_') % some issue occurs when vec function is called (in my code, it happens when using log_det)
scale = 0.5; % scale the size of the field
%set(0,'DefaultFigureWindowStyle','docked');% docked
set(0,'DefaultFigureWindowStyle','normal');

sim_len = 200; % simulation duration
dt = 1;
plan_mode = 'ASPIRe'; % choose the mode of simulation: NBV, sampling, ASPIRe

% parameter for target motion model: 
% 'static', 'lin'(ear), 'cir'(cular), 'sin'(usoidal), 'ped'(estrian)
% The characters in parentheses should not be included when speficying the parameters
tar_model = 'sin'; 

% optimization solver:
% only 'sqp' (sequential quadratic programming) is supported now
solver = 'sqp';

% sensor type:
% 'ran': range-only sensor
% 'br': bearing-only sensor
% 'lin': linear measurement model
% 'rb': range-bearing sensor
sensor_type = 'rb';

% prior distribution:
% 'unimodal': Gaussian distribution
% 'multimodal': GMM
prior_case = 'unimodal';

inPara_sim = struct('dt',dt,'sim_len',sim_len,'sensor_type',sensor_type,'plan_mode',plan_mode,'tar_model',tar_model,'solver',solver);
sim = SimClass(inPara_sim);

save_video = true;

%% Set up testing environment %%%
% target info

% target.pos = [10;20];
%target.pos = [targetPose(tt,1,1);targetPose(tt,1,2);targetPose(tt,1,3)];
target.pos = [45;46;0];

switch tar_model
    case 'static'
        %%% setup for static target
        target.f = @(x) x;
        target.del_f = @(x) eye(2);
        target.Q = 0.0001*eye(2); % Covariance of process noise model for the target
    
    case 'lin'
        %%% setup for moving target: linear model
        target.f = @(x) x+[-0.5;0.5];
        target.del_f = @(x) eye(2);
        target.Q = 0.25*eye(2); %0.04 % Covariance of process noise model for the target
    
    case 'sin'
        %%% setup for moving target: sinusoidal model
        %
        u = [0.5;0.5];
        target.f = @(x) x+[u(1);u(2)*cos(0.2*x(1))];
        target.del_f = @(x) [1 0; -u(2)*sin(x(1)) 1];
        target.Q = 0.25*eye(2); %0.04 % Covariance of process noise model for the target
        %}
end
target.Q = [0.5 0 0;0 0.5 0;0 0 0.01];

target.model_idx = 1;
target.traj = target.pos';
target.control = targetControl;
target.target_model = tar_model;

% region info
xLength = 50;%*scale; 
yLength = 50;%*scale; 
xMin = 0;
yMin = 0;
xMax = xMin+xLength;
yMax = yMin+yLength;
grid_step = 0.5; % the side length of a probability cell

% obstacle info 
map.polyin = polyin;
map.poly = poly;
map.region = 1-region1;
map.region_exp = 1-region;
map.V = V;
map.occ_map = occ_map;

inPara_fld = struct('fld_cor',[xMin,xMax,yMin,yMax],'target',target,'dt',dt,'map',map);
fld = FieldClass(inPara_fld);

%% Set Robot %%%
% Robot
inPara_rbt = struct;
% robot state
% initial state [x,y,heading,velocity]
% inPara_rbt.state = [rbt_state(tt,:)';1];
% inPara_rbt.traj = rbt_state(tt,:)';
inPara_rbt.state = [3;3;0;1];
inPara_rbt.state = [5;17;pi;1];
inPara_rbt.state = [15;5;0;1];
%inPara_rbt.state = [3;48;-pi/2;1];
inPara_rbt.traj = inPara_rbt.state(1:3);
% z(3) = pi/2;
%z=[48;48;pi;1];
%z=[3;3;pi/2;1];
inPara_rbt.sdim = length(inPara_rbt.state); %%%%% this is incorrect, in fact, sdim was supposed for target state dim
% input constraint
% bounds for acceleration, angular velocity, and linear velocity
inPara_rbt.a_lb = -3;
inPara_rbt.a_ub = 1;
inPara_rbt.w_lb = -pi/4;
inPara_rbt.w_ub = pi/4;
inPara_rbt.v_lb = 0;
inPara_rbt.v_ub = 3;
% robot kinematics
inPara_rbt.g = @(z,u) z+u*dt;
inPara_rbt.Qr = blkdiag(0.09*eye(2),[0.01,0;0,0.04]);
inPara_rbt.del_g = @(z,u) z+u*dt;
% target defintion
inPara_rbt.target = target;

map = {};
map.region = zeros(50,50);
map.region_exp = zeros(50,50);
map.V = zeros(50,50,50,50);
map.occ_map = occupancyMap(50,50,1);
inPara_rbt.map = map;

% sensor
% (TODO:changliu) needs further revision
inPara_rbt.sensor_type = sensor_type;
inPara_rbt.theta0 = pi/2; 
inPara_rbt.minrange = 1;%15 4.5;
inPara_rbt.maxrange = 8;

inPara_rbt.inFOV_hist = [];
inPara_rbt.is_tracking = 0;

switch sensor_type 
    case 'rb'
        % range-bearing sensor
        inPara_rbt.h = @(x,z) [sqrt(sum((x(1:2,:)-z(1:2)).^2)+0.1);atan2(x(2,:)-z(2),x(1,:)-z(1))-z(3)];
        inPara_rbt.del_h = @(x,z) [2*x(1) 0; 0 2*x(2)]; % z is the robot state.
        inPara_rbt.R = [0.1 0;0 0.01];
        inPara_rbt.mdim = 2;
    case 'ran'
        % range-only sensor
        inPara_rbt.h = @(x,z) sqrt(sum((x-z).^2)+0.1);
        inPara_rbt.del_h = @(x,z) (x-z)'/sqrt(sum((x-z).^2)+0.1);
        inPara_rbt.R = 1;
        inPara_rbt.mdim = 1;
    case 'br'
        inPara_rbt.h = @(x,z) atan2(x(2,:)-z(2),x(1,:)-z(1))-z(3);
        inPara_rbt.R = 0.1;
        inPara_rbt.mdim = 1;
    case 'lin'
        % linear sensor model for KF use
        inPara_rbt.h = @(x,z) x-z;
        inPara_rbt.del_h = @(x,z) eye(2); % z is the robot state.
        inPara_rbt.R = eye(2);
        inPara_rbt.mdim = 2;
end

% estimation initialization

inPara_rbt.N = 100;
% inPara_rbt.w = ones(1,inPara_rbt.N)/inPara_rbt.N;

switch prior_case
    case 'unimodal'
        %{
        xMin=30;xMax=50;yMin=20;yMax=40;
        [X,Y] = meshgrid((xMin+1):2:(xMax-1),(yMin+1):2:(yMax-1));%返回二维网格坐标（每一个坐标代表一个particle）,25^2=625个particles
        inPara_rbt.particles = [X(:),Y(:)]';%2*N的矩阵，particles(:,i)表示第i个particle的坐标(x,y)
        inPara_rbt.particles = [inPara_rbt.particles;zeros(1,size(inPara_rbt.particles,2))];
        %}
        inPara_rbt.particles = mvnrnd(target.pos,[100 0 0;0 100 0;0 0 0.005],inPara_rbt.N)';
    case 'multimodal'
        noise1 = zeros(1,3);
        noise1(1) = noise_point(tt,1,1);
        noise1(2) = noise_point(tt,1,2);
        noise1(3) = noise_point(tt,1,3);
        noise2 = zeros(1,3);
        noise2(1) = noise_point(tt,2,1);
        noise2(2) = noise_point(tt,2,2);
        noise2(3) = noise_point(tt,2,3);
        % noise1 = [7 2 0];
        % noise2 = [35 35 -pi/2];
        mu = [target.pos';noise1;noise2];
        sigma = [3 0 0;0 3 0;0 0 0.005];
        p = [0.2,0.4,0.4];
        gm = gmdistribution(mu,sigma,p);
        inPara_rbt.particles = random(gm,inPara_rbt.N)';
end
inPara_rbt.N = size(inPara_rbt.particles,2);
inPara_rbt.w = ones(1,inPara_rbt.N)/inPara_rbt.N;
inPara_rbt.est_pos = inPara_rbt.particles*inPara_rbt.w';
%     error(ii) = norm(state_estimation(1:2)-tar_pos(1:2));
inPara_rbt.P = {[100 0; 0 100];[100 0; 0 100];[100 0; 0 100]};

% motion primitives generated by unicycle vehicle model]
kinematicModel = unicycleKinematics;
initialState = [0 0 pi/2];
tspan = 0:0.05:1;

% search
a = zeros(5,7); % action space
interpolated_points = zeros(7,2,3);
ps = {};
inputs = [30 pi/4;30 pi/8;30 -pi/4;30 -pi/8;30 0;0 pi/4;0 -pi/4];

for ii = 1:size(a,2)
[~,ps{ii}] = ode45(@(t,y)derivative(kinematicModel,y,inputs(ii,:)),tspan,initialState);
a(:,ii) = [ps{ii}(end,1);ps{ii}(end,2);ps{ii}(end,3)-pi/2;inputs(ii,1);ii];
end

for mm = 1:size(interpolated_points,1)
    p = ps{mm};
    for jj = 1:2
        interpolated_points(mm,jj,1) = p(10*jj+1,1);
        interpolated_points(mm,jj,2) = p(10*jj+1,2);
        interpolated_points(mm,jj,3) = p(10*jj+1,3);
    end
end

inPara_rbt.a_S = a;
inPara_rbt.int_pts_S = interpolated_points;

% tracking
a = zeros(5,19); % action space
interpolated_points = zeros(19,2,3);
pt = {};

inputs =[15 pi/16;
    10 pi/16;
    5 pi/16;
    15 pi/8;
    10 pi/8;
    5 pi/8;
    15 -pi/16;
    10 -pi/16;
    5 -pi/16;
    15 -pi/8;
    10 -pi/8;
    5 -pi/8;
    15 0;
    10 0;
    5 0;
    0 pi/16;
    0 pi/8;
    0 -pi/16;
    0 -pi/8];

% inputs =[20 pi/6;
%     10 pi/16;
%     5 pi/16;
%     20 pi/4;
%     10 pi/8;
%     5 pi/8;
%     20 -pi/6;
%     10 -pi/16;
%     5 -pi/16;
%     20 -pi/4;
%     10 -pi/8;
%     5 -pi/8;
%     20 0;
%     10 0;
%     5 0;
%     0 pi/16;
%     0 pi/8;
%     0 -pi/16;
%     0 -pi/8];


% inputs =[15 pi/16;
%     10 pi/16;
%     15 pi/8;
%     10 pi/8;
%     15 3*pi/16;
%     10 3*pi/16;
%     15 -pi/16;
%     10 -pi/16;
%     15 -pi/8;
%     10 -pi/8;
%     15 -3*pi/16;
%     10 -3*pi/16;
%     15 0;
%     10 0;
%     5 0;
%     0 pi/16;
%     0 pi/8;
%     0 -pi/16;
%     0 -pi/8];

for ii = 1:size(a,2)
[~,pt{ii}] = ode45(@(t,y)derivative(kinematicModel,y,inputs(ii,:)),tspan,initialState);
a(:,ii) = [pt{ii}(end,1);pt{ii}(end,2);pt{ii}(end,3)-pi/2;inputs(ii,1);ii];
end

for mm = 1:size(interpolated_points,1)
    p = pt{mm};
    for jj = 1:2
        interpolated_points(mm,jj,1) = p(10*jj+1,1);
        interpolated_points(mm,jj,2) = p(10*jj+1,2);
        interpolated_points(mm,jj,3) = p(10*jj+1,3);
    end
end

inPara_rbt.a_T = a;
inPara_rbt.int_pts_T = interpolated_points;

% planning setup
% mpc planning horizon
inPara_rbt.mpc_hor = 3;
inPara_rbt.dt = dt;

% simulation parameters
inPara_rbt.max_step = sim_len;

rbt = RobotClass(inPara_rbt);
    