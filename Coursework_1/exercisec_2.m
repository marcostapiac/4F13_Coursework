data = load('cw1a.mat');
test_x = linspace(floor(min(data.x)) - 0.5, ceil(max(data.x)) + 0.5, 100)';
% GP parameter initialisation
meanf = [];
likf = @likGauss;
covf = @covPeriodic;

% RANDOM initialiation for hyperparameter tuning
N =500;
nlmls = zeros(N, 1);
hyps = struct('mean',[], 'cov',[], 'lik', []);
for t = 1:N
    hyp = struct('mean',[], 'cov', 2*randn(3,1), 'lik', 2*randn(1));
    [opt_hyp, nlml] = minimize(hyp, @gp, -100, @infGaussLik, meanf, covf, likf, data.x, data.y);
    nlmls(t) = nlml(end);
    hyps(t) = opt_hyp;
    disp(size(hyps));
end

[opt_nlml, opt_s] = min(nlmls);
opt_hyp = hyps(opt_s);
% Making predictions
[pred_mean, pred_std] = gp(opt_hyp, @infGaussLik, meanf, covf, likf, data.x, data.y, test_x);
% Predictive plot
f = [pred_mean+1.96*sqrt(pred_std); flip(pred_mean-1.96*sqrt(pred_std),1)];

disp(opt_hyp.cov);
disp(opt_hyp.lik);
disp(opt_nlml);
fill([test_x; flip(test_x,1)], f, [7 7 7]/8);
hold on;
plot(test_x, pred_mean); 
plot(data.x, data.y, 'r+');
title("Predictive Mean with 95% Error Bars");
xlabel("Test data points");
ylabel("Predictive value, y");

