###########################################################################################################################
# binary:logistic（二分类）
###########################################################################################################################
#R package installation and loading （R包安装和加载）
rm(list = ls())
library(xgboost)  
library(tidyverse)   ##Data organization and visualization（数据整理和画图）
library(skimr)   ##Viewing the overall situation of data（数据的整体情况）
library(DataExplorer)  ##Data missing values and filling（数据缺失值和填充）
library(caret)   ##Data splitting, model evaluation（数据拆分，模型评估）
library(pROC) ##Plotting the ROC curve（绘制ROC曲线）

#Enter the working directory（进入工作路径）
setwd("/home/mode_construction/")

#Inputdata
raw_data <- read.table("model2_group2_15features.txt",header = T, row.names = 1, sep = "\t")
raw_data$Tissue <- as.factor(raw_data$Tissue)
raw_data$Treatment <- as.factor(raw_data$Treatment)
raw_data$Tissue <- as.numeric(raw_data$Tissue)
raw_data$Treatment <- as.numeric(raw_data$Treatment)
data <- raw_data
data <- data.frame(data)

#Data overview（数据鸟瞰）
skim(data)
plot_missing(data)
#heart <- na.omit(Heart)  ##Directly remove data with NA（直接去除掉有NA的数据）

#Variable type correction（变量类型修正）
data$Group <- factor(data$Group)

#Distribution of the dependent variable（因变量分布情况）
table(data$Group)
    # No    Yes 
# 715126 505148

##############################################################
#Splitting the data（拆分数据）
set.seed(42)
trains <- createDataPartition(
  y=data$Group,
  p=0.7,
  list=F
)
trains2 <- sample(trains,400000)
valids <- setdiff(trains,trains2)

data_train <- data[trains2, ]
data_valid <- data[valids, ]
data_test <- data[-trains, ]

#Distribution of the dependent variable after splitting（拆分后因变量分布）
table(data_train$Group)
    # No    Yes 
# 234479 165521 
table(data_valid$Group)
    # No    Yes 
# 266110 188083
table(data_test$Group)
    # No    Yes 
# 214537 151544

#Data preparation
colnames(data)
dvfunc <- dummyVars(~.,data = data_train[,2:16],fullRank = T)
data_trainX <- predict(dvfunc, newdata = data_train[, 2:16])
data_trainy <- ifelse(data_train$Group == "0", 0,1)

data_validX <- predict(dvfunc, newdata = data_valid[, 2:16])
data_validy <- ifelse(data_valid$Group == "0", 0,1)

data_testX <- predict(dvfunc, newdata = data_test[, 2:16])
data_testy <- ifelse(data_test$Group == "0", 0,1)

dtrain <- xgb.DMatrix(data=data_trainX,
                      label = data_trainy)
dvalid <- xgb.DMatrix(data=data_validX,
                      label = data_validy)
dtest <- xgb.DMatrix(data=data_testX,
                     label = data_testy)

watchlist <- list(train = dtrain,test = dvalid)

#Training the model（训练模型）
fit_xgb_cls <- xgb.train(
  data = dtrain,
  eta =0.3, 
  gamma = 0.001,
  max_depth =2 ,
  subsample = 0.7,
  colsample_bytree = 0.4,
  objective = "binary:logistic",  ##Specify whether the model is binary classification or multivariate, etc.（指定模型是二分类还是多变量等）
  nrounds = 100000,  ##迭代次数---1000
  watchlist = watchlist,
  verbose = 1,
  print_every_n = 100,  ##Displaying the results every 100 iterations（每100次展示结果）
  early_stopping_rounds = 200 
)

fit_xgb_cls

##Importance of features
importance_matrix <- xgb.importance(model = fit_xgb_cls)
print(importance_matrix)


xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = "Gain")

pdf('importance_sort_15features.pdf',width = 8,height = 8)
xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = "Gain")
dev.off()

#############################################
#Prediction（预测）
#Prediction probabilities on the training set（训练集预测概率）
trainpredprob <- predict(fit_xgb_cls,
                         newdata = dtrain)
#ROC curve on the training set.（训练集ROC）
trainroc <- roc(response = data_train$Group,
                predictor = trainpredprob)	

pdf('ROC_15features.pdf',width = 8,height = 8)
plot(trainroc,
     print.auc = TRUE,
     auc.polygon = T,
     grid = T,
     max.auc.polygon = T,
     auc.polygon.col = "skyblue",
     print.thres = T,
     legacy.axes = T,
     bty = "l")
dev.off()

#Youden（约登法则）
bestp <- trainroc$thresholds[
  which.max(trainroc$sensitivities + trainroc$specificities - 1)
]
bestp

#Predicted classification on the training set.（训练集预测分类）
trainpredlab <- as.factor(
  ifelse(trainpredprob > bestp, "1","0")
)

#Confusion matrix on the training set.（训练集混淆分类）
confusionMatrix(data = trainpredlab,    ##Predicted classes（预测类别）
                reference = data_train$Group,   ##Actual class（实际类别）
                positive = "1",
                mode = "everything")

##############################
#Prediction probabilities on the test set.（测试集预测概率）
testpredprob <- predict(fit_xgb_cls,
                        newdata = dtest)
#Predicted classification on the test set.（测试集预测分类）
testpredlab <- as.factor(
  ifelse(testpredprob > bestp, "1", "0")
)
#Confusion matrix on the test set.（测试集混淆矩阵）
confusionMatrix(data = testpredlab,
                reference = data_test$Group,
                positive = "1",
                mode = "everything")

#ROC curve on the test set.（测试集ROC）
testroc <- roc(response = data_test$Group,
               predictor = testpredprob)

pdf('ROC_train_test_15features.pdf',width = 8,height = 8)
plot(trainroc,
     print.auc = TRUE,
     grid = c(0.1,0.2),
     auc.polygon = F,
     max.auc.polygon = T,
     main = "模型效果图示",
     grid.col = c("green","red"))
plot(testroc,
     print.auc = TRUE,
     print.auc.y = 0.4,
     add = T,
     col = "red")
legend("bottomright",
       legend = c("data_train","data_test"),
       col = c(par("fg"),"red"),
       lwd = 2,
       cex = 0.9)
dev.off()

###############################################################################
#Analysis of the features（特征变量的分析）
##SHAP
xgb.plot.shap(data = data_trainX,
              model = fit_xgb_cls,
              top_n = 5)

##top_n = 5（SHAP values of the top 5 features ranked by importance.This step is so slow, so please choose the quantity carefully.）






