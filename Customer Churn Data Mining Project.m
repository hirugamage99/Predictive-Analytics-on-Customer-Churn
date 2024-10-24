%loading dataset
clc;clear;
T=dlmread('C:\Users\Hiroo\Documents\MATLAB\telco.csv', ',','A2..AP1001');

%------------------------------------------------------pre processing
%dropping variables
drop=[12:15,21:25,35:40];
data = T(:,~ismember(1:size(T, 2),drop));

%handling missing values
missing_count=sum(isnan(T),'all');
disp(['Number of missing values: ', num2str(missing_count)]);

%recoding
%education level variable
threshold = 3;
data(:,7) = data(:,7) > threshold;

%divide test and training
rng(123);
churn=data(:,27); %response variable to get train-test indices in the next step
P=cvpartition(churn,'HoldOut',0.2);
train_full = data(P.training, :);
test_full = data(P.test, :);
x_train = train_full(:, 1:26);
y_train = train_full(:, 27);
x_test = test_full(:, 1:26);
y_test = test_full(:,27);

%exploratory data analysis
%Pearson's chi squared test
%correlation coefficient with numerical variables
%num_columns = [2,3,5,6,8,16,17,18,19,20,21];
%correlation_coeffiecient=corr(train_full(:, num_columns));
%correlation coefficient for response with numerical variables
%data2 = data(:,~ismember(1:size(data, 2),num_columns));
%correlation_spearman=corr(data2(:, 1:20),data2(:,21), 'Type', 'Spearman');
%churn_logical = categorical(churn_train);
%pie(churn_logical)
histcounts(y_train)
%-----------------------------------------------------------Balancing

%class frequencies
class_frequencies = histcounts(y_train, 'BinMethod', 'integers');

%class weights
total_samples = numel(y_train);
weights = total_samples ./ (numel(class_frequencies) * class_frequencies);
minority_class_idx = (y_train == 1);
class_weights(minority_class_idx) = weights(1);
class_weights(~minority_class_idx) = weights(2);

categorical_indices = [1,4,7,9,10,11,18:26]; %indices of categorical predictors
categorical_predictors = false(1, size(x_train, 2));  % Initialize as all false
categorical_predictors(categorical_indices) = true;  % Set true for categorical indices

%convert categorical predictors to categorical arrays
x_train(:, categorical_predictors) = categorical(x_train(:, categorical_predictors));

%--------------------------------------- Initial classification tree
%train classification tree with class weights and specifying categorical predictors
tree = fitctree(x_train, y_train, 'ClassNames', unique(y_train), 'Weights', class_weights, 'CategoricalPredictors', categorical_predictors);
view(tree, 'Mode', 'graph');

y_pred_test = predict(tree, x_test); %predictions

%evaluate accuracy on the test set
accuracy_test = sum(y_pred_test == y_test) / numel(y_test);
fprintf('Accuracy on Test Set: %.2f%%\n', accuracy_test * 100);

%---------------------------------------- Optimal Classification tree

%identifying optimal minleafsize
%range of leaf sizes to consider
leafSizes = 1:500;
%arrays to store CV error and leaf size
cvError = zeros(size(leafSizes));
numLeaves = zeros(size(leafSizes)); %zeros vector to collect values

%cross-validation
for i = 1:length(leafSizes)
    % Train decision tree with current leaf size
    tree = fitctree(x_train, y_train, 'CrossVal', 'on', 'MinLeafSize', leafSizes(i), 'ClassNames', unique(y_train), 'Weights', class_weights, 'CategoricalPredictors', categorical_predictors);
    cvError(i) = kfoldLoss(tree);
    numLeaves(i) = sum(tree.Trained{1}.NumNodes == 1);
end

%plot the CV error vs. leaf size
figure;
plot(leafSizes, cvError, 'b');
xlabel('Leaf Size');
ylabel('Cross-Validation Error');
title('Cross-Validation Error vs. Leaf Size');

%leaf size at minimum CV error
minValue = min(cvError);
minIndex = find(cvError == minValue);
minLeafSize = 14;

%classification tree with optimal leaf size 
tree2 = fitctree(x_train, y_train, 'MinLeafSize', minLeafSize, 'ClassNames', unique(y_train), 'Weights', class_weights, 'CategoricalPredictors', categorical_predictors);
view(tree2, 'Mode', 'graph');

y_pred_test = predict(tree2, x_test); %predictions

%evaluate accuracy on the test set
accuracy_test = sum(y_pred_test == y_test) / numel(y_test);
fprintf('Accuracy on Test Set: %.2f%%\n', accuracy_test * 100);

%----------------------------------------- keeping only important variables

%x_train2 and x_test2 are data with only the significant variables identified above
idx_to_keep = [2,3,11,14,16,23,24];
x_train2 = x_train(:, idx_to_keep);
x_test2 = x_test(:, idx_to_keep);

%------------------------------------------------ Random Forest

%finidng optimal number of trees to include in random forest 
%range of numbers of trees to consider
numTreesRange = 50:50:600;
oobErrorVals = zeros(size(numTreesRange)); %zeros vector to collect values

%out of bound errors
for i = 1:length(numTreesRange)
    forest = TreeBagger(numTreesRange(i), x_train2, y_train, 'Method', 'classification', 'OOBPrediction', 'on');    
    err = mean(oobError(forest));
    oobErrorVals(i) = err;
end

%plot OOB error vs. number of trees
figure;
plot(numTreesRange, oobErrorVals, 'b');
xlabel('Number of Trees');
ylabel('Out-of-Bag Error');
title('Out-of-Bag Error vs. Number of Trees');

%find the number of trees with the lowest out-of-bag error
[~, minIndex] = min(oobErrorVals);
optimalNumTrees = numTreesRange(minIndex);

%random forest model with optimal number of trees(=50)
numTrees = optimalNumTrees;
forest = TreeBagger(numTrees, x_train2, y_train, 'Method', 'classification');

y_pred_test = predict(forest, x_test2); %predictions
y_pred_test = cellfun(@str2double, y_pred_test); %convert cell array to double array

%evaluate accuracy
accuracy_test = sum(y_pred_test == y_test) / numel(y_test);
fprintf('Accuracy on Test Set: %.2f%%\n', accuracy_test * 100);

%------------------------------------------------ SVM

%support vector machine with significant variables
svmModel = fitcsvm(x_train2, y_train, 'Standardize', true, 'KernelFunction', 'linear');

y_pred_test = predict(svmModel, x_test2); %predictions
%evaluate accuracy
accuracy_test = sum(y_pred_test == y_test) / numel(y_test);
fprintf('Accuracy on Test Set: %.2f%%\n', accuracy_test * 100);

%------------------------------------------------- KNN

%finding optimal k for knn
numNeighborsRange = 1:20;
cvError = zeros(size(numNeighborsRange)); %zeros vector to collect values
%cross-validation
for i = 1:length(numNeighborsRange)
    knnModel = fitcknn(x_train2, y_train, 'NumNeighbors', numNeighborsRange(i));
    cvError(i) = kfoldLoss(crossval(knnModel));
end

%plot the CV error vs. number of neighbors
figure;
plot(numNeighborsRange, cvError, 'b');
xlabel('Number of Neighbors');
ylabel('Cross-Validation Error');
title('Cross-Validation Error vs. Number of Neighbors');

optimal_k = 5; %elbow point of plot

%implementation of knn
k = optimal_k; %number of nearest neighbors
knnModel = fitcknn(x_train2, y_train, 'NumNeighbors', k);

y_pred_test = predict(knnModel, x_test2); %predictions

%evaluate accuracy on the test set
accuracy_test = sum(y_pred_test == y_test) / numel(y_test);
fprintf('Accuracy on Test Set: %.2f%%\n', accuracy_test * 100);
