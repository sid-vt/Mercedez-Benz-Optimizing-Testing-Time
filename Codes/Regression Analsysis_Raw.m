%%
clear;clc;
%% Loading Data

T = table2array(readtable('data1.csv'));

X_train = T(1:3000,4:379);
Y_train = T(1:3000,3);
X_test = T(3001:4209,4:379);
Y_test = T(3001:4209,3);

a= [X_train;X_test];


%% 2. Linear Regression
tic;
mdl = fitlm(X_train,Y_train);
predicted_values=predict(mdl,X_test);
mse2=sqrt(mean((predicted_values-Y_test).^2));
elasped1 = toc;

plot(predicted_values(50:80,:),'r','markersize',20,'DisplayName','Predicted values'), xlabel('Sample'), ylabel('Testing time (secs)'), legend('Predicted values'), title('Predicted vs Actual output'), grid on
hold on
plot(Y_test(50:80,:),'b','markersize',20, 'DisplayName', 'Actual values')
hold off

%% PCA + Linear Regression
tic;
coeff = pca(a);
comp = 100;
X_tr_red = a*coeff(:,1:comp);
X_train_r = X_tr_red(1:3000,:);
X_test_r = X_tr_red(3001:4209,:);

mdlu = fitlm(X_train_r,Y_train);
predicted_values1=predict(mdlu,X_test_r);
mse21=sqrt(mean((predicted_values1-Y_test).^2));
elasped11 = toc;

plot(predicted_values1(50:80,:),'r','markersize',20,'DisplayName','Predicted values'), xlabel('Sample'), ylabel('Testing time (secs)'), legend('Predicted values'), title('Predicted vs Actual output'), grid on
hold on
plot(Y_test(50:80,:),'b','markersize',20, 'DisplayName', 'Actual values')
hold off

%% Relieff Algorithm + Linear Regression
tic;
[idx,weights] = relieff(X_train,Y_train,10,'categoricalx','on');
bar(weights(idx(1:25)));
xlabel('Predictor rank');
ylabel('Predictor importance weight');
idx(1:5)

X_train_r1 = [];
X_test_r1 = [];

k=20;
for i=1:k
X_train_r1 = [X_train_r1, X_train(:,idx(i))];
X_test_r1 = [X_test_r1, X_test(:,idx(i))];  
end
% 

mdl_4 = fitlm(X_train_r1,Y_train);
predicted_values4=predict(mdl_4,X_test_r1);
mse4=sqrt(mean((predicted_values4-Y_test).^2));
elaspedz = toc;

plot(predicted_values4(50:80,:),'r','markersize',20,'DisplayName','Predicted values'), xlabel('Sample'), ylabel('Testing time (secs)'), legend('Predicted values'), title('Predicted vs Actual output'), grid on
hold on
plot(Y_test(50:80,:),'b','markersize',20, 'DisplayName', 'Actual values')
hold off

%% Gaussian Process Model
tic;
gprMdl = fitrgp(X_train,Y_train,'CategoricalPredictors','all');
ypredt = resubPredict(gprMdl);
L = loss(gprMdl,X_test,Y_test);
msea=sqrt(mean((ypredt-Y_train).^2));
elaspedf = toc;

plot(ypredt(50:80,:),'r','markersize',20,'DisplayName','Predicted values'), xlabel('Sample'), ylabel('Testing time'), legend('Predicted values'), title('Predicted vs Actual output'), grid on
hold on
plot(Y_train(50:80,:),'b','markersize',20, 'DisplayName', 'Actual values')
hold off

%% Ensemble Model
tic;
Mdlq = fitrensemble(X_train,Y_train,'CategoricalPredictors','all','NumLearningCycles',50);
ypredq = predict(Mdlq,X_test);
mseq=sqrt(mean((ypredq-Y_test).^2));
elaspedq = toc;
ypreds = predict(Mdlq,X_train);
mseat =sqrt(mean((ypreds-Y_train).^2));

plot(ypredq(50:80,:),'r','markersize',20,'DisplayName','Predicted values'), xlabel('Sample'), ylabel('Testing time'), legend('Predicted values'), title('Predicted vs Actual output'), grid on
hold on
plot(Y_test(50:80,:),'b','markersize',20, 'DisplayName', 'Actual values')
hold off
