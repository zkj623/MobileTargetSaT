function LineNr = MFileLineNr()
% MFILELINENR returns the current linenumber
% credit: Albert (2021). MFileLineNr 
% (https://www.mathworks.com/matlabcentral/fileexchange/26262-mfilelinenr),
% MATLAB Central File Exchange. Retrieved November 30, 2021.
    Stack  = dbstack;
    LineNr = Stack(2).line;   % the line number of the calling function
end