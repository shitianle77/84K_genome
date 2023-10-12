###########################################################################################################################
# multi:softprob（多分类）
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
raw_data <- read.table("model1_group4_15features.txt",header = T, row.names = 1, sep = "\t")
raw_data$Tissue <- as.factor(raw_data$Tissue)
raw_data$Treatment <- as.factor(raw_data$Treatment)
raw_data$Tissue <- as.numeric(raw_data$Tissue)
raw_data$Treatment <- as.numeric(raw_data$Treatment)
data <- raw_data
#data <- readxl::read_ecxel()
data <- data.frame(data)

#Data overview（数据鸟瞰）
skim(data)
plot_missing(data)
#heart <- na.omit(Heart)  ##Directly remove data with NA（直接去除掉有NA的数据）

#Variable type correction（变量类型修正）
data$Group <- factor(data$Group)

#Distribution of the dependent variable（因变量分布情况）
table(data$Group)
#0      1      2      3 
#715126 182202 189948 132998

##############################################################
#Splitting the data（拆分数据）
set.seed(42)
trains <- createDataPartition(
  y=data$Group,
  p=0.7,
  list=F
)
trains2 <- sample(trains,450000)
valids <- setdiff(trains,trains2)

data_train <- data[trains2, ]
data_valid <- data[valids, ]
data_test <- data[-trains, ]

#Distribution of the dependent variable after splitting（拆分后因变量分布）
table(data_train$Group)
#0      1      2      3 
#263918  67154  69859  49069
table(data_valid$Group)
#0      1      2      3 
#236671  60388  63105  44030
table(data_test$Group)
#0      1      2      3 
#214537  54660  56984  39899

#Data preparation
data_trainX <- as.matrix(data_train[, 2:16])
data_trainy <- as.numeric(as.character(data_train$Group))

data_validX <- as.matrix(data_valid[, 2:16])
data_validy <- as.numeric(as.character(data_valid$Group))

data_testX <- as.matrix(data_test[, 2:16])
data_testy <- as.numeric(as.character(data_test$Group))

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
  num_class = 4,   ##modify it based on your specific situation. Here, there are four variables (Diff00, Diff0, Diff2, Diff8).
  objective = "multi:softprob",  ##Specify whether the model is binary classification or multivariate, etc.（指定模型是二分类还是多变量等）
  nrounds = 1000000,  ##Number of iterations（迭代次数）
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

pdf('importance_sort_15features_model1.pdf',width = 8,height = 8)
xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = "Gain")
dev.off()

#############################################
#Prediction（预测）
#Prediction probabilities on the training set（训练集预测概率）
trainpredprob <- predict(fit_xgb_cls,
                         newdata = dtrain)
trainpredprob2 <- as.data.frame(
  matrix(trainpredprob, ncol = 4, byrow = T)
)
#The value of ncol should vary according to the number of groups.（ncol值要根据分组数量变化）
colnames(trainpredprob2) <- c("0","1","2","3")
#Predicted classification on the training set.（训练集预测分类）
trainpredprob3 <- trainpredprob2
trainpredprob3$lab <- as.factor(
  apply(trainpredprob2, 1, which.max) -1 
)

#ROC curve on the training set.（训练集ROC）
roc1 <- multiclass.roc(data_train$Group,trainpredprob2[,1])
roc2 <- multiclass.roc(data_train$Group,trainpredprob2[,2])
roc3 <- multiclass.roc(data_train$Group,trainpredprob2[,3])
roc4 <- multiclass.roc(data_train$Group,trainpredprob2[,4])
plot(roc1$rocs[[1]],col = "blue",print.auc = TRUE)
plot.roc(roc1$rocs[[2]],add = TRUE,col = "green",print.auc = TRUE)
plot.roc(roc1$rocs[[3]],add = TRUE,col = "red",print.auc = TRUE)
##Save the images（保存图片）

#All
multiclass.roc(response = data_train$Group,
               predictor = trainpredprob2)

#Confusion matrix on the training set.（训练集混淆矩阵）
confusionMatrix(data = trainpredprob3$lab,  ##Predicted classes（预测类别）
                reference = data_train$Group,   ##Actual class（实际类别）
                mode = "everything")

#Overall results on the training set.（训练集综合结果）
multiClassSummary(
  data.frame(obs = data_train$Group,
             pred = trainpredprob3$lab),
  lev = levels(data_train$Group)
)

##############################
#Prediction probabilities on the test set.（测试集预测概率）
testpredprob <- predict(fit_xgb_cls,
                        newdata = dtest)
testpredprob2 <- as.data.frame(
  matrix(testpredprob, ncol = 4, byrow = T)
)
#The value of ncol should vary according to the number of groups.（ncol值要根据分组数量变化）
colnames(testpredprob2) <- c("0","1","2","3")

#Predicted classification on the test set.（测试集预测分类）
testpredprob3 <- testpredprob2
testpredprob3$lab <- as.factor(
  apply(testpredprob2, 1, which.max) -1 
)

#ROC curve on the test set.（测试集ROC）
roc1 <- multiclass.roc(data_test$Group,testpredprob2[,1])
roc2 <- multiclass.roc(data_test$Group,testpredprob2[,2])
roc3 <- multiclass.roc(data_test$Group,testpredprob2[,3])
roc4 <- multiclass.roc(data_test$Group,testpredprob2[,4])
plot(roc1$rocs[[1]],col = "blue",lty = "dashed",print.auc = TRUE)
plot.roc(roc1$rocs[[2]],add = TRUE,col = "green",lty = "dashed",print.auc = TRUE)
plot.roc(roc1$rocs[[3]],add = TRUE,col = "red",lty = "dashed",print.auc = TRUE)
#plot.roc(roc1$rocs[[4]],add = TRUE,col = "yellow",lty = "dashed")
legend("bottomright",
       legend = c("Diff00-Diff0-test", "Diff00-Diff2-test", "Diff00-Diff8-test"),
       col = c("blue","green","red"),
       lwd = 2.5,
       lty = c("dashed", "dashed", "dashed"))
##Save the images（保存图片）

#All
multiclass.roc(response = data_test$Group,
               predictor = testpredprob2)

#Confusion matrix on the test set.（测试集混淆矩阵）
confusionMatrix(data = testpredprob3$lab,  ##Predicted classes（预测类别）
                reference = data_test$Group,   ##Actual class（实际类别）
                mode = "everything")

#Overall results on the test set.（测试集综合结果）
multiClassSummary(
  data.frame(obs = data_test$Group,
             pred = testpredprob3$lab),
  lev = levels(data_test$Group)
)

###############################################################################
#Analysis of the features（特征变量的分析）
##SHAP
xgb.plot.shap(data = data_trainX,
              model = fit_xgb_cls,
              top_n = 5)

##top_n = 5（SHAP values of the top 5 features ranked by importance.This step is so slow, so please choose the quantity carefully.）








