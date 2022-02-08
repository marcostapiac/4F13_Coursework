data = load('cw1e.mat');
N = 60;
xs = [data.x(N+1:end,1) data.x(N+1:end,2)];
x = [data.x(1:N,1) data.x(1:N,2)];
ys = data.y(N+1:end);
y = data.y(1:N);

meanf = [];
covf = @covSEard;
likf = @likGauss;

% RANDOM initialiation for hyperparameter tuning
N =500;
nlmls = zeros(N, 1);
hyps = struct('mean',[], 'cov',[], 'lik', []);
for t = 1:N
    hyp = struct('mean',[], 'cov', 0.5*randn(3,1), 'lik', 0.5*randn(1));
    [opt_hyp, nlml] = minimize(hyp, @gp, -100, @infGaussLik, meanf, covf, likf, x, y);
    nlmls(t) = nlml(end);
    hyps(t) = opt_hyp;
    disp(size(hyps));
end

[opt_nlml, opt_s] = min(nlmls);
opt_hyp = hyps(opt_s);
% Making predictions
[pred_mean, pred_std] = gp(opt_hyp, @infGaussLik, meanf, covf, likf, x, y, xs);


rs =(ys - pred_mean);
rs_mean = mean(rs)
rs_std = std(rs)

