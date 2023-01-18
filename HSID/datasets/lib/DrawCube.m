function [cube] = DrawCube( hsi )
%DRAWCUBE 
    [h, w, n] = size(hsi);
    cube = ones(h+n-1, w+n-1);

    ph = 0;
    pw = n+1;
    for i = n:-1:1
        band = hsi(:,:,i);
        ph = ph + 1;
        pw = pw - 1;
        cube(ph:ph+h-1, pw:pw+w-1) = imadjust(band);
    end
    
end

