
%data
eegClass = eegNoZ(:,12);
eegData = eegNoZ(:,1:11);

rng(1);

%create data folds
dataLength = length(eegClass);
idx = randperm(dataLength);
folds = 23;
leftOver = mod(dataLength,folds);
grouping = (dataLength - leftOver) / folds;

hold on;

%loop through
for i = 0:folds-1
    %Get training cut of data
    if i == 0
        testing = (idx < (i+1)*grouping);
        testingData = eegData(testing,:);
        testingClass = eegClass(testing,:);
        %training = 
        training = ~(idx < (i+1)*grouping);
        trainingData = eegData(training,:);
        trainingClass = eegClass(training,:);
        
    elseif i == folds
        testing = (idx > (i)*grouping);
        training = ~(idx > (i)*grouping);
        
        testingData = eegData(testing,:);
        testingClass = eegClass(testing,:);
        
        %training = 
        trainingData = eegData(training,:);
        trainingClass = eegClass(training,:);                   
    else
        testing = ((idx < (i+1)*grouping) & (idx > (i)*grouping));
        training = ~((idx < (i+1)*grouping) & (idx > (i)*grouping));
        
        testingData = eegData(testing,:);
        testingClass = eegClass(testing,:);
        
        %training = 
        trainingData = eegData(training,:);
        trainingClass = eegClass(training,:);
        
    end
    
    
        %Train function
        SVMModel = fitcsvm(trainingData,trainingClass,'Standardize',true,'KernelFunction','gaussian',...
    'KernelScale','auto');
        CVSVMModel = crossval(SVMModel);
        eegClassLossOutput = kfoldLoss(CVSVMModel);

        %loop through testing and record Accuracy, FN , FP, TN, TP
        correct = 0;
        errors = 0;
        TN = 0;
        TP = 0;
        FN = 0;
        FP = 0;
        total = 0
        for j = 1:length(testingClass)
        test = testingData(j,:);
        [label,score] = predict(SVMModel,test);
        total = total + 1;
           if label == testingClass(j)
               correct = correct + 1;
               if(label == 0)
                   TN = TN + 1;
                   scatter(test(1),test(2),'b','x')
               else
                   TP = TP + 1;
                   scatter(test(1),test(2),'b','o')
               end
               
           else
               errors = errors + 1;
               if(testingClass(j) == 0)
                   FP = FP + 1;
                   scatter(test(1),test(2),'r','o')
               else
                   FN = FN + 1;
                   scatter(test(1),test(2),'r','x')
               end
           end
        end
        FPm(i+1) = FP
        FNm(i+1) = FN
        TPm(i+1) = TP
        TNm(i+1) = TN
        totalM(i+1) = total
        i = i
        
        Accuracy(i+1) = correct / length(testingClass)
    
end




AccuracyFinal = mean(Accuracy)
TNf = mean(TNm)/mean(totalM)
TPf = mean(TPm)/mean(totalM)
FNf = mean(FNm)/mean(totalM)
FPf = mean(FPm)/mean(totalM)

Acc = (TNf+TPf)/(TPf + TNf + FPf + FNf)
Sens = TPf / (TPf + FNf)
Spec = TNf / (TNf + FPf)
Prec = TPf / (TPf + FPf)

ylabel('Meditation') 
xlabel('Attention')
title('Confusion of SVM Classifier across Meditation and Attention')
hold off;




