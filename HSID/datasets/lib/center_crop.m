function [ crop_img ] = center_crop( img, cropx, cropy )
%CENTER_CROP Summary of this function goes here
    [y,x,~] = size(img);
    startx = max(floor(x/2)-floor(cropx/2), 1);
    starty = floor(y/2)-floor(cropy/2);
    crop_img = img(starty:starty+cropy-1,startx:startx+cropx-1, :);
%     crop_img = img(end-512+1:end,startx:startx+cropx-1, :); % for _meta_*_2
end