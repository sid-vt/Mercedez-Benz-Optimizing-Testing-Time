%%
clear;clc;
%% Loading Data

T = table2array(readtable('trainData_clean.csv'));

X_train = T(1:3000,3:366);
Y_train = T(1:3000,2);
X_test = T(3001:4208,3:366);
Y_test = T(3001:4208,2);

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
X_test_r = X_tr_red(3001:4208,:);


mdlp = fitlm(X_train_r,Y_train);
predicted_valuesp=predict(mdlp,X_test_r);
msep=sqrt(mean((predicted_valuesp-Y_test).^2));
elaspedp = toc;

plot(predicted_valuesp(50:80,:),'r','markersize',20,'DisplayName','Predicted values'), xlabel('Sample'), ylabel('Testing time (secs)'), legend('Predicted values'), title('Predicted vs Actual output'), grid on
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

X_train_rr = [];
X_test_rr = [];

k=10;
for i=1:k
X_train_rr = [X_train_rr, X_train(:,idx(i))];
X_test_rr = [X_test_rr, X_test(:,idx(i))];  
end
% 

mdl_r = fitlm(X_train_rr,Y_train);
predicted_valuesr=predict(mdl_r,X_test_rr);
mser=mean((predicted_valuesr-Y_test).^2);
elaspedr = toc;

plot(predicted_valuesr(50:80,:),'r','markersize',20,'DisplayName','Predicted values'), xlabel('Sample'), ylabel('Testing time (secs)'), legend('Predicted values'), title('Predicted vs Actual output'), grid on
hold on
plot(Y_test(50:80,:),'b','markersize',20, 'DisplayName', 'Actual values')
hold off

%% Gaussian Process Model
tic;
gprMdl = fitrgp(X_train,Y_train,'CategoricalPredictors','all');
ypredg = resubPredict(gprMdl);
L = loss(gprMdl,X_test,Y_test);
msea=sqrt((mean((ypredg-Y_train).^2)));
elaspedg = toc;

plot(ypredg(50:80,:),'r','markersize',20,'DisplayName','Predicted values'), xlabel('Sample'), ylabel('Testing time (secs)'), legend('Predicted values'), title('Predicted vs Actual output'), grid on
hold on
plot(Y_test(50:80,:),'b','markersize',20, 'DisplayName', 'Actual values')
hold off

%% Ensemble Model
tic;
Mdle = fitrensemble(X_train,Y_train,'CategoricalPredictors','all','NumLearningCycles',50);
yprede = predict(Mdle,X_train);
msee=sqrt(mean((yprede-Y_train).^2));
elaspede = toc;

plot(yprede(50:80,:),'r','markersize',20,'DisplayName','Predicted values'), xlabel('Sample'), ylabel('Testing time (secs)'), legend('Predicted values'), title('Predicted vs Actual output'), grid on
hold on
plot(Y_test(50:80,:),'b','markersize',20, 'DisplayName', 'Actual values')
hold off
