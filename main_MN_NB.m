
my_table = readtable('test_sheet_large.csv');
rng(1);

% shuffle table
my_table = my_table(randperm(size(my_table,1)),:);



%% split data 80 - 20 
split_point = round(.20 * size(my_table,1));

% allocate all data untill splitting point as testing data
testing_data = my_table(1:split_point,:);
% remove categorical data from table, algorithm can't process it
testing_labels = categorical(testing_data{:,['labels']});
testing_data(:,['labels']) = [];

% allocate all data after splitting point as testing data
training_data = my_table(split_point+1:end,:);
% remove categorical data from table, algorithm can't process it
training_labels = categorical(training_data{:,['labels']});
training_data(:,['labels']) = [];


%fit model
model = myMN_NB.fit(training_data,training_labels)
% predict labels
prediction = myMN_NB.predict(model, testing_data);

% compute confusion matrix
[confusion_mat, order_class] = confusionmat(testing_labels, prediction)

confusionCH = confusionchart(testing_labels, prediction)

% calculate the overall classification accuracy:
p = sum(diag(confusion_mat)) / sum(confusion_mat(1:1:end))

