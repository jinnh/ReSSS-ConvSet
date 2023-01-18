function I_rgb = ShowEnlargedRectangle(I, LeftUpPoint, RightBottomPoint, Enlargement_Factor, pos, LineWidth, Color, gap)  
% example  I_rgb = ShowEnlargedRectangle(I, [10,20], [50,60], 1.5, 1)  
 
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
  
if ~exist('Enlargement_Factor','var')  
    Enlargement_Factor = 1.5;  
end  
  
if ~exist('gap','var') %�����·�����  
    gap = 1;  
end

if ~exist('pos', 'var')
    pos = 1;
end

if ~exist('Color','var')
%     Color = [255,0,0];
    Color = [255,255,255];
end

pos = mod(pos, 4);

%%%%%
%0 2%
%1 3%
%%%%%
  
%% ������  
I_rgb = DrawRectangle(I_rgb, LeftUpPoint, RightBottomPoint, LineWidth, Color);  
  
%% ��ȡ����������ͼ��  
UpRow = LeftUpPoint(1);  
LeftColumn = LeftUpPoint(2);  
BottomRow = RightBottomPoint(1);  
RightColumn = RightBottomPoint(2);  

for i = 1 : size(I_rgb,3)  
    Patch(:,:,i) = I_rgb(UpRow + LineWidth:BottomRow  - LineWidth,LeftColumn  + LineWidth:RightColumn  - LineWidth,i);   
end

  
%% ����ȡ����������зŴ�? 
% Enlargement_Factor = 0.5;  
% Interpolation_Method = 'bilinear'; %bilinear,bicubic  
Interpolation_Method = 'bicubic';
Enlarged = imresize(Patch,Enlargement_Factor,Interpolation_Method);
% Enlarged = imadjust(Enlarged, [0 1], [0.3 0.7], 0.5);  
% hsv_Enlarged = rgb2hsv(Enlarged);
% hsv_Enlarged(:,:,3) = histeq(hsv_Enlarged(:,:,3));
% Enlarged = hsv2rgb(hsv_Enlarged);
  
%% �ԷŴ�����������ʾ  
[m, n, c] = size(Enlarged);  
[row, column, ~] = size(I_rgb);  
% EnlargedShowStartRow = row - 1 - LineWidth;  
% EnlargedShowStartColumn = 2 + LineWidth;  
% for j = 1 : c  
%     I_rgb(EnlargedShowStartRow - m + 1:EnlargedShowStartRow,EnlargedShowStartColumn:EnlargedShowStartColumn + n - 1,j) = Enlarged(:,:,j);   
% end  

switch pos
    case 0  % left up
        EnlargedShowStartRow = 1 + gap + LineWidth + m;
        EnlargedShowStartColumn = 1 + gap + LineWidth;  
    case 1  % left bottom
        EnlargedShowStartRow = row - gap - LineWidth;  
        EnlargedShowStartColumn = 1 + gap + LineWidth;  
    case 2  % right up
        EnlargedShowStartRow = 1 + gap + LineWidth + m;
        EnlargedShowStartColumn = column - gap - LineWidth - n + 1;
    case 3  % right bottom
        EnlargedShowStartRow = row - gap - LineWidth;  
        EnlargedShowStartColumn = column - gap - LineWidth - n + 1;
    otherwise 
        error('Invalid Type: pos');
end


for j = 1 : c  
    I_rgb(EnlargedShowStartRow - m + 1:EnlargedShowStartRow,EnlargedShowStartColumn:EnlargedShowStartColumn + n - 1,j) = Enlarged(:,:,j);   
end  
  
%% �ԷŴ���ʾ������򻭾���? 
Point1 = [EnlargedShowStartRow - m + 1 - LineWidth,EnlargedShowStartColumn - LineWidth];  
Point2 = [EnlargedShowStartRow + 1,EnlargedShowStartColumn + n -1 + 1];  
I_rgb = DrawRectangle(I_rgb, Point1, Point2, LineWidth, Color);  
  
end  