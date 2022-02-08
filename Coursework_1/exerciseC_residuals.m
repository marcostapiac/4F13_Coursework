data = load('cw1a.mat');
% GP parameter initialisation
meanf = [];
likf = @likGauss;
covf = @covPeriodic;

hyp = struct('mean',[], 'cov', [ 0.0337,  -0.0012,  0.1587], 'lik', -2.2087);
[opt_hyp, nlml] = minimize(hyp, @gp, -100, @infGaussLik, meanf, covf, likf, data.x, data.y);
% Making predictions
[pred_mean, pred_std] = gp(opt_hyp, @infGaussLik, meanf, covf, likf, data.x, data.y, data.x);

rs =(data.y - pred_mean);
rs_std = std(rs);
r = (data.y - pred_mean)/rs_std;
[h, p] = kstest(r)

h = histogram(r, 'Normalization', 'countdensity');
hold on;
xvals = linspace(-4,4,100);
pdf = normpdf(xvals);
pdf = pdf * (max(h.Values)/max(pdf));
g =plot(xvals, pdf, 'LineWidth', 2, 'DisplayName', 'Normal Gaussian PDF');
legend(g);
xlim([-4,4]);