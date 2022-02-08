data = load('cw1a.mat');
test_x = linspace(floor(min(data.x)) - 0.5, ceil(max(data.x)) + 0.5, 100)';
disp(floor(min(data.x)) - 0.5);
disp(ceil(max(data.x)) + 0.5);
% GP parameter initialisation
meanf = [];
likf = @likGauss;
covf = @covPeriodic;

hyp = struct('mean',[], 'cov', [-2, 0, 0], 'lik', 0);
[opt_hyp, nlml] = minimize(hyp, @gp, -100, @infGaussLik, meanf, covf, likf, data.x, data.y);
disp(nlml(end));

% Making predictions
[pred_mean, pred_std] = gp(opt_hyp, @infGaussLik, meanf, covf, likf, data.x, data.y, test_x);
% Predictive plot
f = [pred_mean+1.96*sqrt(pred_std); flip(pred_mean-1.96*sqrt(pred_std),1)];

fill([test_x; flip(test_x,1)], f, [7 7 7]/8);
hold on;
plot(test_x, pred_mean, 'LineWidth', 1.5); 
g =plot(data.x, data.y, 'b+', 'DisplayName', 'Training Data Points');
legend(g);
title("Predictive Mean with 95% Error Bars");
xlabel("Input, x");
ylabel("Predictive value, y");
disp(opt_hyp);

