function [newFileName,countDiseased,countHealthy,img,result] = BBHA_Predict(tblData,str,num1,num2)
% Predicts if the given parameter, which is the dataset, is
% positive for this disease or negative and returns one value.

%% Importing and Pre-processing Data

countDiseased = 0;
countHealthy = 0;
BH = load('trainedModel.mat');
BH = BH.BH;

dataset = importdata('heart.csv');
if isstruct(dataset)
    data.x = dataset.data(:, 1:end-1);
    data.y = dataset.data(:,end);
else
    data.x = dataset(:, 1:end-1);
    data.y = dataset(:,end);
end
data.rows = size(data.x,1);
data.columns = size(data.x,2);

res = sscanf(tblData,str,[num1,num2]);
data.new = transpose(res);
if isstruct(data.new)
    data.new = data.new.data();
    new_data = data.new;
else
    data.new = data.new();
    new_data = data.new;
end

for j = data.columns : -1:1
    if BH.selection(1,j) == 0
        data.new(:,j) = [];
    end
end
rows = size(data.new,1);
prediction = predict(BH.Mdl,data.new);

for i = 1 : rows
    if prediction(i,1) == 0
        countHealthy = countHealthy + 1;
    else
        countDiseased = countDiseased + 1; 
    end
end

new_data = [new_data, prediction];
[result,~] = vec2str(new_data,[],[],0);
newFileName='Your_Data_Prediction.csv';
figure('visible','off'), histogram(new_data(:,end));
xlabel('Class (0:Healthy 1:Diseased)');
ylabel('Number of Patients');
img = figToImStream('outputType','uint8');

end