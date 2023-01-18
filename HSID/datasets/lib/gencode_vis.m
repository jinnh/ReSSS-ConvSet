function [] = gencode_vis( tex, fn, method, index )
%GENCODE_VIS generate latex code for visualization 
    fid = fopen(tex, 'a');
    [~, imgname] = fileparts(fn);
%     fprintf(fid, '\\begin{figure}[htbp]\r\n\\centering\r\n');  
    fprintf(fid, '\\begin{subfigure}[b]{.18\\linewidth}\r\n\\centering\r\n');
    
%     fprintf(fid, '\\begin{tabular}{c}\r\n');
    fprintf(fid, '\\includegraphics[height=1\\linewidth,clip,keepaspectratio]{./figures/%s/%s}\r\n', imgname, [method '.png']);  
%     fprintf(fid, '\\includegraphics[width=1in,clip,keepaspectratio]{./figures/%s/1/%s}\r\n\\\\\r\n', imgname, [method '.png']);    
%     fprintf(fid, '\\includegraphics[width=1in,clip,keepaspectratio]{./figures/%s/109/%s}\r\n', imgname, [method '.png']);
    if exist('index', 'var')         
        fprintf(fid, '\\caption{%s \\\\\\hspace{\\textwidth}\\centering (%2.2f)}\r\n', method, index);
    else
        fprintf(fid, '\\caption{%s \\\\\\hspace{\\textwidth}\\centering (+$\\infty$)}\r\n', method);
    end
%     fprintf(fid, '\\caption{%s \\\\\\hspace{\\textwidth}\\centering (%2.2f, %2.2f)}\r\n', method, index(1), index(2));
%     fprintf(fid, '\\end{tabular}');
    
%     fprintf(fid, '\\caption{%s}\r\n', method);
    fprintf(fid, '\\end{subfigure}\r\n');
%     fprintf(fid, ['\\label{fig:Gauss}\r\n\\end{figure}\r\n\r\n']);
    fclose(fid);
end
