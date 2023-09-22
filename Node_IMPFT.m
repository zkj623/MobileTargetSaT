classdef Node_IMPFT
    properties
        num;%标号
        state;%robot state
        inter_state;
        hist;%history
        a;%未选择的动作
        a_num;%从哪个action来
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
end