classdef Node_IMPFT
    properties
        num;%标号
        state;%robot state
        h;%history
        a;%未选择的动作
        N;%The number of times node has been visited
        R;%the immediate reward
        Q;%the sum reward
        r;%the reward for the rollout
        parent;
        children;
        children_maxnum;
        is_terminal;
        delete;
    end
    methods
        function list = treepolicy(begin,num,list)
            while list(begin).is_terminal == 0
                if length(list(begin).children) == 3
                    begin = best_child(begin,0.5,list);
                else
                    list = expand(begin,num,list);
                    return;
                end
            end
        end
        
        function list = expand(begin,num,list)
            node = list(begin);
            ii = randperm(length(node.a),1);
            action = node.a(ii);
            list(node.num).a(ii) = [];
            new = Node;
            new.num = num;
            new.state(3) = node.state(3)+action;
            new.state(1) = node.state(1)+cos(node.state(3));
            new.state(2) = node.state(2)+sin(node.state(3));
            new.a = [pi/4,0,-pi/4];
            new.N = 0;
            new.Q = 0;
            new.parent = node.num;
            new.children = [];
            list(node.num).children = [list(node.num).children,new.num];
            new.is_terminal = 0;
            list = [list,new];
        end
        
        function v = best_child(begin,c,list)
            max = 0;
            for ii = 1:3
                node = list(begin);
                tmp = node.children(ii);
                val = list(tmp).Q/list(tmp).N+c*(2*log(node.N)/list(tmp).N)^0.5;
                if val>max
                    max = val;
                    v = tmp;
                end
            end
        end
        
        function reward = simulate(node,tar_pos)
            % for ii = 1:5
            %     action = node.a(randperm(length(node.a),1));
            %     node.state(3) = node.state(3)+action;
            %     node.state(1) = node.state(1)+cos(node.state(3));
            %     node.state(2) = node.state(2)+sin(node.state(3));
            % end
            reward = -norm(node.state(1:2)-tar_pos);
        end
        
        function list = backup(num,reward,list)
            while  1
                list(num).N = list(num).N + 1;
                list(num).Q = list(num).Q+reward;
                if num == 1
                    return;
                end
                tmp = list(num).parent;
                num = tmp;
            end
        end
    end
end