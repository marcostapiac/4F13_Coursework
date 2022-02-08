data = load('cw1e.mat');
a =8;
T = ((2*a)/0.1 + 1);
[t1 t2] = meshgrid(-a:0.1:a, -a:0.1:a);

test_x = [t1(:), t2(:)];
disp(size(test_x));
meanf = [];
covf = @covSEard;
likf = @likGauss;

% RANDOM initialiation for hyperparameter tuning
N =1;
nlmls = zeros(N, 1);
hyps = struct('mean',[], 'cov',[], 'lik', []);
for t = 1:N
    hyp = struct('mean',[], 'cov', [0.4131, 0.2515, 0.1019], 'lik', -2.2765);
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

h1=scatter3(data.x(:,1),data.x(:,2), data.y, 'r', 'filled');
hold on;
h2=mesh(t1,t2,reshape(pred_mean,T,T));
%h3=mesh(t1,t2,reshape(pred_mean -1.96*pred_std,T,T));
%h4=mesh(t1,t2,reshape(pred_mean+1.96*pred_std, T,T));
%legend([h1,h2, h3, h4], ["Training data points", "Predictive Plot", 'Lower 95% Confidence Surface','Upper 95% Confidence Surface' ]);
xlabel('Input First Dim')
ylabel('Input Second Dim', 'Rotation',-0)
zlabel('Predictive Output')
%{
subplot(2,1,1);
%fill([test_x; flip(test_x,1)], f, [7 7 7]/8);
hold on;
plot(test_x(:,1), pred_mean, 'r');
ylabel("Predictive value, y");
xlabel("First Dimension of Test Points");
subplot(2,1,2);
%fill([test_x; flip(test_x,1)], f, [7 7 7]/8);
hold on;
plot(test_x(:,2), pred_mean,'b');
sgtitle("Predictive Mean with 95% Error Bars");
xlabel("Second Dimension of Test Points");
ylabel("Predictive value, y");
disp(opt_hyp.cov);
disp(opt_hyp.lik);
disp(nlml(end));
%}

