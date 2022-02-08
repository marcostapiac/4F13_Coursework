data = load('cw1a.mat');
x = data.x;
y = data.y;

N = 100;
a = 3.5;
Xs = linspace(-a,a,N);
Ys = linspace(-2*a,2*a,N);

Z = zeros(N,N);

for i = 1:N
    for j = 1:N
              
    meanf = [];                    
    covf = @covSEiso;              
    likf = @likGauss;              

    hyp = struct('mean', [], 'cov', [Xs(i) 0], 'lik', Ys(j));
    
    [nlZ, dnlZ] = gp(hyp, @infGaussLik, meanf, covf, likf, x, y);
    
    Z(j,i) = log(nlZ);
        
    end
end

figure
contourf(Xs,Ys,Z,15)
colormap(hot)
cb = colorbar; % create and label the colorbar
cb.Label.String = 'Negative Log Marginal Likelihood';

minMatrix1 = min(Z(:));
[row1,col1] = find(Z==minMatrix1);

minMatrix2 = min(min(Z(:,50:100)));
[row2,col2] = find(Z==minMatrix2);

hold on; plot(Xs(col1), Ys(row1), 'g*', 'LineWidth', 1.0); 
plot(Xs(col2), Ys(row2), 'g*', 'LineWidth', 1.0);

set(gca,'fontsize',17);
xlabel('Log Length Scale');
ylabel('Log Signal Amplitude');

cb = colorbar; % create and label the colorbar
cb.Label.String = 'Negative Log Marginal Likelihood';
