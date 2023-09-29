%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation class, used for rendering target and robot states
% ver 1.0, Kangjie Zhou, 2023/9
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef SimClass
    properties
        dt; % discretization time interval
        sim_len;
        sensor_type;
        fig_cnt; % counter for figure
    end
    
    methods
        function this = SimClass(inPara)
            this.dt = inPara.dt;
            this.sim_len = inPara.sim_len;     
            this.sensor_type = inPara.sensor_type;
            this.fig_cnt = 1;
        end
        
        function plotFilter(this,rbt,fld,tt,ii)
            % Plotting for particles
            hold on
            set(gcf,'position',[600,200,1920,1080]);
            axis equal;
            axis([0,50,0,50]);
            set(gca,'FontSize',40);
            box on;
           
            plot(fld.map.poly,'FaceColor',[.5 .5 .5],'LineStyle','none');
            plot(rbt.particles(1,:),rbt.particles(2,:),'.','Color',[0 1 0.5]);
            plot(rbt.est_pos(1),rbt.est_pos(2),"^",'Color','r','MarkerFaceColor','r',MarkerSize=15);

            plot(fld.target.traj(tt,1:ii+1,1),fld.target.traj(tt,1:ii+1,2),'-','Color',[0.13333 0.5451 0.13333],LineWidth=3);
            plot(rbt.traj(1,1:end-1),rbt.traj(2,1:end-1),'-','Color','r',MarkerSize=0.1,LineWidth=3);

            plot(fld.target.traj(tt,ii+1,1),fld.target.traj(tt,ii+1,2),"pentagram",'Color',[0.13333 0.5451 0.13333],'MarkerFaceColor',[0.13333 0.5451 0.13333],MarkerSize=15);
            plot(rbt.state(1),rbt.state(2),'ro',MarkerFaceColor='r',MarkerSize=15);

            rbt.drawFOV(rbt.state,fld,'cur',[0.9290 0.6940 0.1250]);
            rbt.drawFOV_red(rbt.state,fld,'cur',[1 0 1]);

            xticks(0:10:50);
            yticks(0:10:50);
            text(1,49,strcat('t=',num2str(ii)),"FontSize",30,"FontName",'Times New Roman');

            drawnow limitrate
        end

        function plot_rbt_map(this,rbt,fld,tt,ii)
            hold on
            set(gcf,'position',[600,100,1300,1280]);

            %% plot ground truth
            %t = subplot(3,2,2);
            t = tiledlayout(3,4);
            nexttile(5,[2 2]);
            hold on
            axis([0,50,0,50]);
            show(fld.map.occ_map)

            %plot(rbt.particles(1,:),rbt.particles(2,:),'.','Color',[0 1 0.5]);
            plot(fld.target.traj(1:ii+1,1),fld.target.traj(1:ii+1,2),'-','Color',[0.13333 0.5451 0.13333],LineWidth=3);
            plot(fld.target.traj(ii+1,1),fld.target.traj(ii+1,2),"pentagram",'Color',[0.13333 0.5451 0.13333],'MarkerFaceColor',[0.13333 0.5451 0.13333],MarkerSize=15);
            for jj = 1:3
            %plot(rbt{jj}.est_pos(1),rbt{jj}.est_pos(2),"^",'Color','r','MarkerFaceColor','r',MarkerSize=15);

            plot(rbt{jj}.traj(1,1:end-1),rbt{jj}.traj(2,1:end-1),'-','Color','r',MarkerSize=0.1,LineWidth=3);
            plot(rbt{jj}.state(1),rbt{jj}.state(2),'ro',MarkerFaceColor='r',MarkerSize=15);

            rbt{jj}.drawFOV(rbt{jj}.state,fld,'cur',[0.9290 0.6940 0.1250]);
            rbt{jj}.drawFOV_red(rbt{jj}.state,fld,'cur',[1 0 1]);
            end

            text(1,48,strcat('t=',num2str(ii)),"FontSize",20,"FontName",'Times New Roman');

            %% map fusion
            for jj = 1:2
                for kk = jj+1:3
                    if norm(rbt{jj}.state(1:2)-rbt{kk}.state(1:2))<100
                        reg1 = rbt{jj}.map.region;
                        reg2 = rbt{kk}.map.region;
                        for mm = 1:50
                            for nn = 1:50
                                if reg2(mm,nn)<0.5 && reg2(mm,nn)<reg1(mm,nn)
                                    reg1(mm,nn) = reg2(mm,nn);
                                elseif reg2(mm,nn)>0.5 && reg2(mm,nn)>reg1(mm,nn)
                                    reg1(mm,nn) = reg2(mm,nn);
                                end
                            end
                        end
                        rbt{jj}.map.region = reg1;
                        rbt{kk}.map.region = reg1;
                        rbt{jj}.map.region_exp = reg1;
                        rbt{kk}.map.region_exp = reg1;
                        reg_tmp = 1-reg1';
                        for ll = 1:size(reg_tmp,2)
                            reg(ll,:) = reg_tmp(size(reg_tmp,2)-ll+1,:);
                        end
                        rbt{jj}.map.occ_map = occupancyMap(reg,1);
                        rbt{kk}.map.occ_map = occupancyMap(reg,1);
                    end
                end
            end

            for jj = 1:3
                nexttile(jj)
                %subplot(3,2,2*jj-1);
                hold on
                axis([0,50,0,50]);

                show(rbt{jj}.map.occ_map)

                plot(rbt{jj}.particles(1,:),rbt{jj}.particles(2,:),'.','Color',[0 1 0.5]);
                plot(rbt{jj}.est_pos(1),rbt{jj}.est_pos(2),"^",'Color','r','MarkerFaceColor','r',MarkerSize=15);

                plot(fld.target.traj(1:ii+1,1),fld.target.traj(1:ii+1,2),'-','Color',[0.13333 0.5451 0.13333],LineWidth=3);
                plot(rbt{jj}.traj(1,1:end-1),rbt{jj}.traj(2,1:end-1),'-','Color','r',MarkerSize=0.1,LineWidth=3);

                plot(fld.target.traj(ii+1,1),fld.target.traj(ii+1,2),"pentagram",'Color',[0.13333 0.5451 0.13333],'MarkerFaceColor',[0.13333 0.5451 0.13333],MarkerSize=15);
                plot(rbt{jj}.state(1),rbt{jj}.state(2),'ro',MarkerFaceColor='r',MarkerSize=15);

                rbt{jj}.drawFOV(rbt{jj}.state,fld,'cur',[0.9290 0.6940 0.1250]);
                rbt{jj}.drawFOV_red(rbt{jj}.state,fld,'cur',[1 0 1]);

                text(rbt{jj}.state(1),rbt{jj}.state(2)+2,num2str(rbt{jj}.value_max));
            end


            %% plot syncretic map
            nexttile(7,[2 2]);
            %subplot(3,2,4);
            hold on
            axis([0,50,0,50]);

            show(rbt{1}.map.occ_map)

            %plot(rbt.particles(1,:),rbt.particles(2,:),'.','Color',[0 1 0.5]);
            %plot(rbt.est_pos(1),rbt.est_pos(2),"^",'Color','r','MarkerFaceColor','r',MarkerSize=15);

            plot(fld.target.traj(1:ii+1,1),fld.target.traj(1:ii+1,2),'-','Color',[0.13333 0.5451 0.13333],LineWidth=3);
            plot(fld.target.traj(ii+1,1),fld.target.traj(ii+1,2),"pentagram",'Color',[0.13333 0.5451 0.13333],'MarkerFaceColor',[0.13333 0.5451 0.13333],MarkerSize=15);
            for jj = 1:3
                %plot(rbt{jj}.est_pos(1),rbt{jj}.est_pos(2),"^",'Color','r','MarkerFaceColor','r',MarkerSize=15);

                plot(rbt{jj}.traj(1,1:end-1),rbt{jj}.traj(2,1:end-1),'-','Color','r',MarkerSize=0.1,LineWidth=3);
                plot(rbt{jj}.state(1),rbt{jj}.state(2),'ro',MarkerFaceColor='r',MarkerSize=15);

                rbt{jj}.drawFOV(rbt{jj}.state,fld,'cur',[0.9290 0.6940 0.1250]);
                rbt{jj}.drawFOV_red(rbt{jj}.state,fld,'cur',[1 0 1]);
            end

            t.Padding = 'tight';
            t.TileSpacing = 'tight';
            % text(rbt.state(1),rbt.state(2)+2,num2str(rbt.value_max));
            %             axis equal;
            %             set(gca,'FontSize',40);
%             box on;
% 
%             xticks(0:10:50);
%             yticks(0:10:50);

            drawnow limitrate
        end

        function plotSim(this,rbt,fld,state,B,tt,ii)
            % Plotting for particles
            hold on
            set(gcf,'position',[600,200,1920,1080]);
            axis equal;
            axis([0,50,0,50]);
            set(gca,'FontSize',40);
            box on;
           
            plot(fld.map.poly,'FaceColor',[.5 .5 .5],'LineStyle','none');
            plot(B(1,:),B(2,:),'.','Color',[0 1 0.5]);
            %plot(rbt.est_pos(1),rbt.est_pos(2),"^",'Color','r','MarkerFaceColor','r',MarkerSize=15);

            plot(fld.target.traj(tt,1:ii+1,1),fld.target.traj(tt,1:ii+1,2),'-','Color',[0.13333 0.5451 0.13333],LineWidth=3);
            plot(rbt.traj(1,1:end-1),rbt.traj(2,1:end-1),'-','Color','r',MarkerSize=0.1,LineWidth=3);

            plot(fld.target.traj(tt,ii+1,1),fld.target.traj(tt,ii+1,2),"pentagram",'Color',[0.13333 0.5451 0.13333],'MarkerFaceColor',[0.13333 0.5451 0.13333],MarkerSize=15);
            plot(state(1),state(2),'ro',MarkerFaceColor='r',MarkerSize=15);

            rbt.drawFOV(state,fld,'cur',[0.9290 0.6940 0.1250]);
            rbt.drawFOV_red(state,fld,'cur',[1 0 1]);
            

            xticks(0:10:50);
            yticks(0:10:50);
            text(1,49,strcat('t=',num2str(ii)),"FontSize",30,"FontName",'Times New Roman');

            drawnow limitrate
        end
        
        
        function plotTraj(this,rbt,fld)
            % Plotting for trajectory (path planning part)
%             shading interp
%             contourf(prob_map','LineColor','none');
%             load('MyColorMap','mymap')
%             colormap(mymap);
%             colorbar
            
            % robot traj
            hdl1 = plot(rbt.traj(1,:),rbt.traj(2,:),'r','markers',5);
            set(hdl1,'MarkerFaceColor','r');
            set(hdl1,'MarkerEdgeColor','r');
            set(hdl1,'Color','r');
            set(hdl1,'LineStyle','-');
            set(hdl1,'Marker','o');
            
            % target actual traj
            hdl2 = plot(fld.target.traj(1,:),fld.target.traj(2,:),'b','markers',5);
            set(hdl2,'MarkerFaceColor','b');
            set(hdl2,'MarkerEdgeColor','b');%
            set(hdl2,'Color','b');
            %     set(hdl2,'LineStyle','-');
            set(hdl2,'Marker','*');
            
            
            drawFOV(rbt,rbt.state,fld,'cur');

            xlim([fld.fld_cor(1),fld.fld_cor(2)]);
            ylim([fld.fld_cor(3),fld.fld_cor(4)]);
            box on
            axis equal
            drawnow
        end                
        
    end
end