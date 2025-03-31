
clear; clc; close all;

%% === Load and Preprocess Data ===
% Load input features and target discharge data
input1 = readmatrix('input1.xlsx');
input2 = readmatrix('input2.xlsx');
input_comb = readmatrix('combination.xlsx');
Q_delta = readmatrix('Qdelta.xlsx');
Q_target = readmatrix('target.xlsx');

% Smooth and fill outliers in the target variable
Q_target = filloutliers(Q_target, "linear", "mean");
Q_target = smoothdata(Q_target, 'gaussian', 6);

% Normalize inputs
[input1_norm, PS1] = mapminmax(input1');
[input2_norm, PS2] = mapminmax(input2');
[input_comb_norm, PS3] = mapminmax(input_comb');
[Q_data_norm, PS_delta] = mapminmax(Q_delta');
[Q_target_norm, PS_out] = mapminmax(Q_target');


% Encode features using autoencoders
autoenc1 = trainAutoencoder(input1_norm, 1, 'L2WeightRegularization', 0.001);
autoenc2 = trainAutoencoder(input2_norm, 1, 'L2WeightRegularization', 0.001);
feature1 = encode(autoenc1, input1_norm)';
feature2 = encode(autoenc2, input2_norm)';
feature_comb = input_comb_norm';

% Merge all features
features = [feature1, feature2, feature_comb];

% Construct sequential input with 6 historical time steps
numTimeSteps = 6;
numTrain_sub = 216; 
X_seq = {};
Y_seq = [];
for i = 1:size(features, 1) - numTimeSteps
    X_seq{end+1} = features(i:i+numTimeSteps-1, :)';
    Y_seq(end+1) = Q_target_norm(i + numTimeSteps);
end

% Split training and testing
split_index = numTrain_sub - numTimeSteps; % training length
XTrain = X_seq(1:split_index);
YTrain = Y_seq(1:split_index);
XTest = X_seq(split_index+1:end);
YTest = Y_seq(split_index+1:end);

%% === Build LSTM Network ===
numFeatures= 4; % Input node
numHiddenUnitsShared = 24; % neurons in hidden layers
numResponses = 1; % Output node
rng(0);

QDELTA = Q_data_norm(1,1:numTrain_sub);
QDELTA=QDELTA';

val_size = 36; 
XValid = XTrain(end - val_size + 1:end);
YValid = YTrain(end - val_size + 1:end);

customRegressionLayerObj = customRegressionLayer('custom_regression', QDELTA);

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnitsShared, 'Name', 'lstm1')
    fullyConnectedLayer(numHiddenUnitsShared)
    lstmLayer(numHiddenUnitsShared, 'Name', 'lstm2')
    reluLayer('name','relu')
    fullyConnectedLayer(numResponses)
    customRegressionLayerObj
    ];

options = trainingOptions('adam', ... 
'MaxEpochs',200,... 
 'MiniBatchSize',216, ... 
'InitialLearnRate',0.01, ...
'LearnRateSchedule','piecewise', ...
 'LearnRateDropPeriod',200, ...
'LearnRateDropFactor',0, ...
'ValidationData',{XValid,YValid}, ...
'Verbose',1, ...
'ValidationFrequency',5, ...
 'Plots','training-progress', ...
'L2Regularization', 0.001); 

%% === Train and Predict ===
net = trainNetwork(XTrain, YTrain, layers, options);

YPredTrain = predict(net, XTrain, 'MiniBatchSize', 1);
YPredTest = predict(net, XTest, 'MiniBatchSize', 1);

% Inverse normalize
YPredTrain = mapminmax('reverse', YPredTrain', PS_out);
YPredTest = mapminmax('reverse', YPredTest', PS_out);
YTrain_real = mapminmax('reverse', cell2mat(YTrain), PS_out);
YTest_real = mapminmax('reverse', cell2mat(YTest), PS_out);

%% === Evaluation ===
train_rmse = sqrt(mean((YPredTrain - YTrain_real).^2));
test_rmse = sqrt(mean((YPredTest - YTest_real).^2));
train_mae = mean(abs(YPredTrain - YTrain_real));
test_mae = mean(abs(YPredTest - YTest_real));

fprintf('Train RMSE: %.4f, MAE: %.4f\n', train_rmse, train_mae);
fprintf('Test  RMSE: %.4f, MAE: %.4f\n', test_rmse, test_mae);

%% === Visualization ===
figure;
subplot(2,1,1);
plot(YTrain_real, 'k-o'); hold on;
plot(YPredTrain, 'r-*');
title('Training Set Prediction');
legend('Actual', 'Predicted');
xlabel('Sample'); ylabel('Discharge');

subplot(2,1,2);
plot(YTest_real, 'k-o'); hold on;
plot(YPredTest, 'b-*');
title('Testing Set Prediction');
legend('Actual', 'Predicted');
xlabel('Sample'); ylabel('Discharge');