
N = 100;
x = linspace(100, 1000, N);
y = linspace(1, 5, N);
[x, y] = meshgrid(x, y);
%z = interp2(1:10, 1:8, a, x, y, 'spline');
z = interp2(100:100:1000, 1:5, a, x, y, 'linear');
contourf(x, y, z);
%pause;
colormap(flipud(copper));
colorbar;

set(gca,'FontSize',30)
xlhand = get(gca,'xlabel')
set(xlhand,'string','M','fontsize',30)

ylhand = get(gca,'ylabel')
set(ylhand,'string','L','fontsize',30)
