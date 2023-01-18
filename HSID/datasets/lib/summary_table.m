function [ ] = summary_table( res_table, rowLabels, title, version )
%SUMMARY_TABLE generate texfile of summary table    
    format = {'%-6.2f', '%-6.3f', '%-6.3f', '%-6.3f', '%-6.3f'};
    mean_matrix = squeeze(res_table(:,:,1));
    std_matrix = squeeze(res_table(:,:,2));
%     array2table(mean_matrix, 'VariableNames', columnLabels, 'RowNames', rowLabels) 
    %%% generate latex code
    rs = size(mean_matrix, 1);
    cs = size(mean_matrix, 2);
    matrix = cell(rs, cs);
    [~, bold_ind] = max(mean_matrix(1:2,:), [], 2);
    [~, ind_1] = sort(mean_matrix(1:3,:), 2, 'descend');
    [~, I] = min(mean_matrix(3:end,:), [], 2);
    [~, ind_2] = sort(mean_matrix(4:end,:), 2);
    
    bold_ind = [bold_ind;I];
    ind = [ind_1;ind_2];
    
    for r = 1:rs
        for c = 1:cs  
            if version == 2
                matrix{r,c} = ['$',num2str(mean_matrix(r,c),format{r}), ' \pm ', num2str(std_matrix(r,c),format{r}), '$']; 
            elseif version == 1                    
                matrix{r,c} = ['$',num2str(mean_matrix(r,c),format{r}), '$']; 
            end
            %%% color
%             if c == ind(r, 1) % best
%                 matrix{r,c} = ['\textcolor{red}{$', num2str(mean_matrix(r,c),format{r}),'$}']; 
%             elseif c == ind(r, 2)
%                 matrix{r,c} = ['\textcolor{blue}{$', num2str(mean_matrix(r,c),format{r}),'$}']; 
%             else
%                 matrix{r,c} = ['$',num2str(mean_matrix(r,c),format{r}), '$']; 
%             end
            %%% bold
%             if c == bold_ind(r)
%                 if version == 2
%                     matrix{r,c} = ['$\bm{', num2str(mean_matrix(r,c),format{r}), ' \pm ', num2str(std_matrix(r,c),format{r}), '}$']; 
%                 elseif version == 1
%                     matrix{r,c} = ['$\bm{', num2str(mean_matrix(r,c),format{r}),'}$']; 
%                 end
%             else
%                 if version == 2
%                     matrix{r,c} = ['$',num2str(mean_matrix(r,c),format{r}), ' \pm ', num2str(std_matrix(r,c),format{r}), '$']; 
%                 elseif version == 1                    
%                     matrix{r,c} = ['$',num2str(mean_matrix(r,c),format{r}), '$']; 
%                 end
%             end
        end
    end
    
    if version == 2
        matrix = matrix';
    end
    matrix2latex(matrix, 'table.tex', 'rowLabels', rowLabels, 'columnLabels', {}, 'alignment', 'c', 'format', format, 'title', title);

end

function matrix2latex(matrix, filename, varargin)
    rowLabels = [];
    colLabels = [];
    alignment = 'l';
    format = [];
    textsize = [];
    sigma = '0';
    
    if (rem(nargin,2) == 1 || nargin < 2)
        error('matrix2latex: ', 'Incorrect number of arguments to %s.', mfilename);
    end

    okargs = {'rowlabels','columnlabels', 'alignment', 'format', 'size', 'title'};
    for j=1:2:(nargin-2)
        pname = varargin{j};
        pval = varargin{j+1};
        k = strmatch(lower(pname), okargs);
        if isempty(k)
            error('matrix2latex: ', 'Unknown parameter name: %s.', pname);
        elseif length(k)>1
            error('matrix2latex: ', 'Ambiguous parameter name: %s.', pname);
        else
            switch(k)
                case 1  % rowlabels
                    rowLabels = pval;
                    if isnumeric(rowLabels)
                        rowLabels = cellstr(num2str(rowLabels(:)));
                    end
                case 2  % column labels
                    colLabels = pval;
                    if isnumeric(colLabels)
                        colLabels = cellstr(num2str(colLabels(:)));
                    end
                case 3  % alignment
                    alignment = lower(pval);
                    if alignment == 'right'
                        alignment = 'r';
                    end
                    if alignment == 'left'
                        alignment = 'l';
                    end
                    if alignment == 'center'
                        alignment = 'c';
                    end
                    if alignment ~= 'l' && alignment ~= 'c' && alignment ~= 'r'
                        alignment = 'l';
                        warning('matrix2latex: ', 'Unkown alignment. (Set it to \''left\''.)');
                    end
                case 4  % format
                    format = lower(pval);
                case 5  % textsize
                    textsize = pval;
                case 6  % title
                    title = pval;
            end
        end
    end

    fid = fopen(filename, 'a');
    
    width = size(matrix, 2);
    height = size(matrix, 1);

    if isnumeric(matrix)
        matrix = num2cell(matrix);
        for h=1:height
            for w=1:width
                if(~isempty(format))
                    matrix{h, w} = num2str(matrix{h, w}, format);
                else
                    matrix{h, w} = num2str(matrix{h, w});
                end
            end
        end
    end
    
    fprintf(fid, '\\multirow{%d}{*}{%s} &', length(rowLabels), title);
    
    if(~isempty(colLabels))
        if(~isempty(rowLabels))
            fprintf(fid, '&');
        end
        for w=1:width-1
            fprintf(fid, '%s&', colLabels{w});
        end
        fprintf(fid, '%s\\\\\\hline\r\n', colLabels{width});
    end
    
    for h=1:height
        if(~isempty(rowLabels))
            fprintf(fid, '%s&', rowLabels{h});
        end
        for w=1:width-1
            fprintf(fid, '%s&', matrix{h, w});
        end
        if h ~= height
            fprintf(fid, '%s\\\\\\cline{2-%d}\r\n&', matrix{h, width}, 2+width);
        end
    end
    
    fprintf(fid, '%s\\\\\\hline\r\n', matrix{height, width});

%     fprintf(fid, '\\end{tabular}\r\n');
    
%     if(~isempty(textsize))
%         fprintf(fid, '\\end{%s}', textsize);
%     end

    fclose(fid);
end