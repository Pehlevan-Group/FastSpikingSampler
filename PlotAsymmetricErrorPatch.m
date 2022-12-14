function [ g ] = PlotAsymmetricErrorPatch(x,y,el,eu,corder,lineStyle)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if nargin < 5
    corder = lines(size(x,2));
end
if nargin < 6
    lineStyle = '-';
end

%Make the bottom run the opposite direction to plot around the eventual
%shape of the error patch clockwise
el=el(end:-1:1,:);
ye=[eu;el];

%Similarily run the x back
xe = [x; x(end:-1:1,:)];
xe = repmat(xe, [1 size(ye,2)/size(xe,2)]);
x = repmat(x,[1 size(y,2)/size(x,2)]);

corder = repmat(corder,[ceil(size(x,2)/size(corder,1)) 1]);
corder = corder(1:size(x,2),:);

% Get the current hold status
hStat = ishold;
hold on;

if min(size(y)) > 1
    set(gca, 'ColorOrder', corder);
end

if size(x,1) < 5 % Previously <50
    if min(size(y)) > 1
        g=errorbar(x,y,y-el(end:-1:1,:),eu-y,'marker','o','LineWidth',2,'MarkerSize',8, 'lineStyle', lineStyle);
    else
        g=errorbar(x,y,y-el(end:-1:1,:),eu-y,'marker','o','LineWidth',2,'MarkerSize',8, 'lineStyle', lineStyle, 'color', corder);
    end
else
    if min(size(y)) > 1
        g=plot(x,y,'LineWidth',2, 'LineStyle', lineStyle);
    else
        g=plot(x,y,'LineWidth',2, 'LineStyle', lineStyle, 'color', corder);
    end
end

if all(ye==0)
    return;
end


if any(el(:)) || any(eu(:))
    
    if min(size(y)) > 1
        colormap(corder);
        h = fill(xe,ye,repmat(0:size(xe,2)-1,[size(xe,1) 1]),'linestyle','none','FaceAlpha',0.25, 'FaceColor', 'flat');
        %         h = patch(xe, ye, permute(corder, [1 3 2]), 'FaceAlpha', 0.25, 'LineStyle','None', 'FaceColor', 'flat');
    else
        h = fill(xe,ye,repmat(0:size(xe,2)-1,[size(xe,1) 1]),'linestyle','none','FaceAlpha',0.25, 'FaceColor', corder);
        %         h = patch(xe, ye, permute(corder, [1 3 2]), 'FaceAlpha', 0.25, 'LineStyle','None', 'FaceColor', 'flat');
        
    end
    
    hAnnotation = get(h,'Annotation');
    
    if ~iscell(hAnnotation)
        hAnnotation = {hAnnotation};
    end
    
    for ii = 1:length(h)
        hLegendEntry = get(hAnnotation{ii},'LegendInformation');
        set(hLegendEntry,'IconDisplayStyle','off');
    end
end

if ~hStat
    hold off;
end

end