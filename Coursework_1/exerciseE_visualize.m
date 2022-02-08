data = load('cw1e.mat');
mesh(reshape(data.x(:,1),11,11),reshape(data.x(:,2),11,11),reshape(data.y,11,11));
xlabel('First Dimension of Input Data')
ylabel('Second Dimension of Input Data', 'Rotation',-10)
zlabel('Output Data')
