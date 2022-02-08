data = load('cw1a.mat');
N = 200;
test_x = linspace(-5,5,N)';
% GP parameter initialisation
meanf = [];
likf = @likGauss;
covf = {@covProd, {@covPeriodic, @covSEiso}};
hyp = struct('mean', [], 'cov', [-0.5, 0, 0, 2, 0], 'lik', []); % Noise-free = no likelihood variance

% Need to generate covariance matrix
K_prod = feval(covf{:}, [-0.5, 0, 0, 2, 0], test_x);
K_iso = feval(@covSEiso, [2,0], test_x);
K_periodic = feval(@covPeriodic, [-0.5, 0, 0], test_x);
for i=1:4
    z = randn(N,1);
    y_prod = chol(K_prod + 1e-6*eye(N))'*z;
    y_iso = chol(K_iso + 1e-6*eye(N))'*z;
    y_periodic = chol(K_periodic + 1e-6*eye(N))'*z;
    subplot(2,2,i)
    plot(test_x, y_prod, 'Color',[0.8500 0.3250 0.0980], 'DisplayName',"Product of Covariance Functions", 'LineWidth', 1);
    hold on;
    plot(test_x, y_iso, 'Color', 'k', 'DisplayName',"SEIso Covariance Function", 'LineWidth', 1 );
    plot(test_x, y_periodic, 'Color',[0 0.4470 0.7410] , 'DisplayName',"Periodic Covariance Function", 'LineWidth', 1) ;
end
sgtitle("Sample functions drawn from GP");
Lgnd = legend('show');
Lgnd.Position(1) = 0.01;
Lgnd.Position(2) = 0.45;


