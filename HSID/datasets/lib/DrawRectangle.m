function I_rgb = DrawRectangle(I, LeftUpPoint, RightBottomPoint,LineWidth, Color)  
% example  I_rgb = ShowEnlargedRectangle(I, [10,20], [50,60], 1)  
  
if ~exist('Color','var')
    Color = [255,0,0];
end

if size(I,3)==1  
    I_rgb(:,:,1) = I;  
    I_rgb(:,:,2) = I;  
    I_rgb(:,:,3) = I;  
else  
    I_rgb = I;  
end  
  
if ~exist('LineWidth','var')  
    LineWidth = 1;  
end  
  
UpRow = LeftUpPoint(1);  
LeftColumn = LeftUpPoint(2);  
BottomRow = RightBottomPoint(1);  
RightColumn = RightBottomPoint(2);  

c1=Color(1);c2=Color(2);c3=Color(3);

% 上面线  
I_rgb(UpRow:UpRow + LineWidth - 1,LeftColumn:RightColumn,1) = c1;  
I_rgb(UpRow:UpRow + LineWidth - 1,LeftColumn:RightColumn,2) = c2;  
I_rgb(UpRow:UpRow + LineWidth - 1,LeftColumn:RightColumn,3) = c3;  
% 下面线  
I_rgb(BottomRow:BottomRow + LineWidth - 1,LeftColumn:RightColumn,1) = c1;  
I_rgb(BottomRow:BottomRow + LineWidth - 1,LeftColumn:RightColumn,2) = c2;  
I_rgb(BottomRow:BottomRow + LineWidth - 1,LeftColumn:RightColumn,3) = c3;  
% 左面线  
I_rgb(UpRow:BottomRow,LeftColumn:LeftColumn + LineWidth - 1,1) = c1;  
I_rgb(UpRow:BottomRow,LeftColumn:LeftColumn + LineWidth - 1,2) = c2;  
I_rgb(UpRow:BottomRow,LeftColumn:LeftColumn + LineWidth - 1,3) = c3;  
% 右面线  
I_rgb(UpRow:BottomRow,RightColumn:RightColumn + LineWidth - 1,1) = c1;  
I_rgb(UpRow:BottomRow,RightColumn:RightColumn + LineWidth - 1,2) = c2;  
I_rgb(UpRow:BottomRow,RightColumn:RightColumn + LineWidth - 1,3) = c3;  
  
end  