function [newFileName,featuresCountOld,featuresCountNew,acc,img,result,selection] = BBHA_Select(tblData,str,num1,num2)
% Returns feature selected data to client along with
% old count of features, new count, the accuracy
% of this given dataset, the byte array of the 
% generated figure, the result data in string and
% the selection (indices) of the best star in the algorithm
% to be used later on in the client-side.

%% Importing and Pre-processing Data

res = sscanf(tblData,str,[num1,num2]);
dataset = transpose(res);
if isstruct(dataset)
    data.x = dataset.data(:, 1:end-1);
    data.y = dataset.data(:,end);
else
    data.x = dataset(:, 1:end-1);
    data.y = dataset(:,end);
end
data.rows = size(data.x,1);
data.columns = size(data.x,2);

%% BBHA

% BBHA Parameters
numStars = 3;
numIter = 15;

% Inititialization of Stars
stars = struct;
BH = struct;
stars.values = (1-0).*rand(numStars,data.columns) + 0;
stars.values = round(stars.values);

% Training Data
Mdl = fitcnb(data.x,data.y);
CVMdl = crossval(Mdl,'KFold',10);
classErr = kfoldLoss(CVMdl);
fprintf('The old accuracy is %.2f%%\n',100 * (1-classErr));

for i = 1:numStars
    count = 0;
    data.new = data.x;
    for j = data.columns : -1:1
        if stars.values(i,j) == 0
            data.new(:,j) = [];
        else
            count = count+1;
        end
    end
    Mdl = fitcnb(data.new,data.y);
    CVMdl = crossval(Mdl,'KFold',10);
    classErr = kfoldLoss(CVMdl);
    fprintf('The new accuracy in iteration %d is %.2f%%\n',i,100 * (1-classErr));
    stars.fitness(i,1) = 1-classErr;
    stars.count(i,1) = count;
end

% Finding maximum fitness value in fitness array
BH.data = [];
BH.fitArr = [];
BH.fitness = stars.fitness(1);
for i = 1:numStars
    if(stars.fitness(i)>=BH.fitness)
        BH.fitness = stars.fitness(i);
        BH.index = i;
        BH.Mdl = Mdl;
        BH.data = data.new;
        for j = 1:data.columns
            BH.selection(1,j) = stars.values(BH.index,j);
        end
    end
end
BH.fitArr = [BH.fitArr,BH.fitness];
BH.count = stars.count(BH.index,:);

% Main Loop
fprintf('************************\n');
iteration = 1;
while (1)
    if iteration >= numIter
        fprintf('************************\n');
        fprintf('\nIterations Completed\n\n');
        break;
    end
    fprintf('Iteration %d :-\n',iteration);
    for a = 1:numStars
        count = 0;
        data.new = data.x;
        for j = data.columns : -1:1
            if stars.values(a,j) == 0
                data.new(:,j) = [];
            else
                count = count + 1;
            end
        end
        stars.count(a,1) = count;
        Mdl = fitcnb(data.new,data.y);
        CVMdl = crossval(Mdl,'KFold',10);
        classErr = kfoldLoss(CVMdl);
        fprintf('Accuracy %d = %.2f%%\n',a,100 * (1-classErr));
        if (1-classErr) > BH.fitness
            BH.index = a;
            BH.fitArr = [BH.fitArr,1-classErr];
            BH.fitness = 1-classErr;
            BH.Mdl = Mdl;
            BH.data = data.new;
            for j = 1:data.columns
                BH.selection(1,j) = stars.values(BH.index,j);
            end
            BH.count = stars.count(a,1);
        elseif (((1-classErr) == (BH.fitness)) && ((stars.count(a,1)) < (BH.count)))
            BH.index = a;
            BH.fitArr = [BH.fitArr,1-classErr];
            BH.fitness = 1-classErr;
            BH.Mdl = Mdl;
            BH.data = data.new;
            for j = 1:data.columns
                BH.selection(1,j) = stars.values(BH.index,j);
            end
            BH.count(a,1) = stars.count(a,1);
        end
        stars.fitness(a,1) = 1-classErr;
        
        % Calculating R
        sum = 0;
        for s = 1 : numStars
            if s ~= BH.index
                sum = sum + stars.fitness(s,1);
            end
        end
        R = BH.fitness/sum;
        
        if sqrt(((BH.index-a)^2)) < R
            stars.values(a,:) = [];
            newStar = (1-0).*rand(1,data.columns) + 0;
            stars.values = [stars.values(1:a-1,:);newStar;stars.values(a:end,:)];
            stars.values(a,:) = round(stars.values(a,:));
        end
    end
    
    for i = 1 : numStars
        for d = 1 : data.columns
            random = (0.2-0.1).*rand + 0.1;
            x = stars.values(BH.index,d)-stars.values(i,d);
            stars.values(i,d) = stars.values(i,d) + random * x;
            if abs(tanh(stars.values(i,d))) > random
                stars.values(i,d) = 1;
            else
                stars.values(i,d) = 0;
            end
        end
    end
    
    fprintf('************************\n');
    iteration = iteration + 1;
end
fprintf('The best accuracy is %.2f%%\t\tIndex : %d\n',100 * BH.fitness,BH.index);

%% Assigning Output Values

featureSelectedData = [BH.data, data.y];
featuresCountOld = data.columns;
featuresCountNew = BH.count;
acc = 100*BH.fitness;

newFileName = 'Feature_Selected_Data.csv';
selection = BH.selection;
[result,~] = vec2str(featureSelectedData,[],[],0);
figure('visible','off'), plot(BH.fitArr,'-*','LineWidth',2);
xlabel('Star');
ylabel('BH Fitness');
img = figToImStream('outputType','uint8');

end