data = load('cw1e.mat');
test_x = [linspace(-8, 8, 121)', linspace(-8, 8, 121)'];
disp(size(test_x));
disp(data.x);
meanf = [];
covf = @covSEard;
likf = @likGauss;
hyp = struct('mean', [], 'cov', randn(3,1), 'lik', 0);

[opt_hyp, nlml] = minimize(hyp, @gp, -100, @infGaussLik, meanf, covf, likf, data.x, data.y);
% Making predictions
[pred_mean, pred_std] = gp(opt_hyp, @infGaussLik, meanf, covf, likf, data.x, data.y, test_x);
% Predictive plot
f = [pred_mean+1.96*sqrt(pred_std); flip(pred_mean-1.96*sqrt(pred_std),1)];

subplot(2,1,1);
fill([test_x; flip(test_x,1)], f, [7 7 7]/8);
hold on;
plot(test_x(:,1), pred_mean, 'r+');
ylabel("Predictive value, y");
xlabel("First Dimension of Test Points");
subplot(2,1,2);
fill([test_x; flip(test_x,1)], f, [7 7 7]/8);
hold on;
plot(test_x(:,2), pred_mean,'b+');
sgtitle("Predictive Mean with 95% Error Bars");
xlabel("Second Dimension of Test Points");
ylabel("Predictive value, y");
disp(opt_hyp.cov);
disp(opt_hyp.lik);
disp(nlml(end));