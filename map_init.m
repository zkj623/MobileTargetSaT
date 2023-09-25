close all

num_obstacle = 16;
polyin = repmat(struct,num_obstacle,1);

%
polyin(1).points = [1,1,5,5;5,6,6,5];
polyin(2).points = [20,20,11,11,21,21;1,8,8,9,9,1];
polyin(3).points = [8,8,9,9;42,49,49,42];
polyin(4).points = [16,16,10,10,16,16,17,17,23,23,17,17;28,34,34,35,35,41,41,35,35,34,34,28];
polyin(4).points(1,:) = polyin(4).points(1,:)-2;
polyin(5).points = [37,36,36,30,30,36,36,37;4,4,8,8,9,9,13,13];
polyin(6).points = [49,42,42,43,43,49;13,13,19,19,14,14];
polyin(7).points = [49,44,44,49;43,43,44,44];
polyin(8).points = [33,20,20,26,26,20,20,27,27,33;16,16,17,17,23,23,24,24,17,17];
polyin(8).points(1,:) = polyin(8).points(1,:)-2;
polyin(8).points(2,:) = polyin(8).points(2,:)-1;
polyin(9).points = [41,32,32,33,33,40,40,41;32,32,38,38,33,33,38,38];
polyin(10).points = [36,36,37,37;43,49,49,43];
polyin(11).points = [1,0,0,50,50,1;0,0,50,50,49,49];
polyin(12).points = [50,1,1,49,49,50;0,0,1,1,49,49];
polyin(13).points = [1,1,9,9,10,10,9,9;20,21,21,27,27,14,14,20];
polyin(14).points = [23,22,22,23,23,29,29,23;40,40,45,45,43,43,42,42];
polyin(15).points = [33,32,32,40,40,33;22,22,28,28,27,27];
polyin(15).points(1,:) = polyin(15).points(1,:)+1;
polyin(15).points(2,:) = polyin(15).points(2,:)-2;
polyin(16).points = [25,25,26,26;27,32,32,27];
%}
%
region1 = zeros(50,50);
for ii = 1:size(region1,1)
    for jj = 1:size(region1,2)
        for kk = 1:length(polyin)
            if isInside(polyin(kk),[ii-0.8;jj-0.8]) == 1||isInside(polyin(kk),[ii-0.2;jj-0.2]) == 1||isInside(polyin(kk),[ii-0.2;jj-0.8]) == 1||isInside(polyin(kk),[ii-0.8;jj-0.2]) == 1
            %if isInside(polyin(kk),[ii-0.5;jj-0.5]) == 1
                region1(ii,jj) = 1;%infeasible
                break
            end
        end
    end
end

region_tmp = region1;

region_tmp1 = region1';
for ii = 1:size(region_tmp1,2)
    region1(ii,:) = region_tmp1(size(region_tmp1,2)-ii+1,:);
end
occ_map = occupancyMap(region1,1);
region1 = region_tmp;

% map.ProbabilitySaturation = [0.0001,0.9999];

show(occ_map)
set(gcf,'position',[600,200,1920,1080]);

% pose = [5,5,pi/4];
% ranges = 6*ones(100,1);
% angles = linspace(-pi/4,pi/4,100);
% maxrange = 6;
% 
% scan = lidarScan(ranges,angles);
% 
% insertRay(map,pose,scan,maxrange);
% 
% show(map)

%{
expand_rad = 0.3;
er = expand_rad;
polyin(1).points = [7-er,7-er,10+er,10+er;5-er,35+er,35+er,5-er];
polyin(2).points = [14-er,14-er,21+er,21+er;18-er,25+er,25+er,18-er];
polyin(3).points = [33-er,33-er,41+er,41+er;38-er,46+er,46+er,38-er];
polyin(4).points = [17-er,17-er,33+er,33+er;7-er,10+er,10+er,7-er];
polyin(5).points = [5-er,5-er,25+er,25+er;42-er,45+er,45+er,42-er];
polyin(6).points = [40,24.5,27,42.5;17.5,33,35.5,20];
polyin(7).points = [16-er,16-er,20+er,20+er;33-er,37+er,37+er,33-er];
polyin(8).points = [40-er,40-er,45+er,45+er;7-er,12+er,12+er,7-er];
polyin(9).points = [26-er,26-er,31+er,31+er;15-er,20+er,20+er,15-er];
polyin(10).points = [40,38.7,45,46.3;27.7,29,35.3,34];
%}

X = {};
Y = {};
for ii = 1:num_obstacle
    X{ii} = polyin(ii).points(1,:);
    Y{ii} = polyin(ii).points(2,:);
end
%poly = polyshape({X{1} X{2} X{3} X{4} X{5} X{6} X{7} X{8} X{9} X{10} X{11} X{12} X{13}},{Y{1} Y{2} Y{3} Y{4} Y{5} Y{6} Y{7} Y{8} Y{9} Y{10} Y{11} Y{12} Y{13}});

%feasible region
% region = ones(50,50);
% for ii = 1:size(region,1)
%     for jj = 1:size(region,2)
%         for kk = 1:length(polyin)
%             if isInside(polyin(kk),[ii-0.8;jj-0.8]) == 1||isInside(polyin(kk),[ii-0.2;jj-0.2]) == 1||isInside(polyin(kk),[ii-0.2;jj-0.8]) == 1||isInside(polyin(kk),[ii-0.8;jj-0.2]) == 1
%             %if isInside(polyin(kk),[ii-0.5;jj-0.5]) == 1
%                 region(ii,jj) = 0;%infeasible
%                 break
%             end
%         end
%     end
% end

region = 1-region1;

% for kk = 1:length(polyin)
%     if isVisible(polyin(kk),[13-0.5;13-0.5],[15-0.5,12-0.5],region) == 0
%         V_fl = 0;%invisible
%         break
%     end
% end

%visibility judgement
%{
tic
V = -ones(50,50,50,50);
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
%
V = -ones(50,50,50,50);
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
                [endPts,midPts] = raycast(occ_map,[ii-0.5 jj-0.5],[mm-0.5 nn-0.5]);
                midPts = [midPts;endPts];
                for kk = 1:size(midPts,1)
                    xx = midPts(kk,2);
                    yy = 50-midPts(kk,1)+1;
                    if region(xx,yy) == 0
                        V(ii,jj,mm,nn) = 0;
                        break
                    end
                end
            end
        end
    end
end
%}

region = region1;

% for debug only
%{
close all
%plot(poly,'FaceColor',[.5 .5 .5],'LineStyle','none');
axis equal;
axis([0,50,0,50]);
set(gcf,'position',[600,200,1920,1080]);
%}

save(sprintf("structured_map"),"region","region1","V","polyin","occ_map");

function flag = isInside(obstacle,state)
flag = 0;
points = obstacle.points;
if state(1)<min(points(1,:))||state(1)>max(points(1,:))||state(2)<min(points(2,:))||state(2)>max(points(2,:))
    return
end
n = size(points,2);
jj = n;
for ii = 1:n
    if ((points(2,ii)>state(2))~=(points(2,jj)>state(2)))&&(state(1)-points(1,ii)<(points(1,jj)-points(1,ii))*(state(2)-points(2,ii))/(points(2,jj)-points(2,ii)))
        flag = ~flag;
    end
    jj = ii;
end
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