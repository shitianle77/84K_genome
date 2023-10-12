###########################################################################################################################
# reg:squarederror（回归模型）
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
raw_data <- read.table("model3_regression_15features.txt",header = T, row.names = 1, sep = "\t")
raw_data$Tissue <- as.factor(raw_data$Tissue)
raw_data$Treatment <- as.factor(raw_data$Treatment)
raw_data$Tissue <- as.numeric(raw_data$Tissue)
raw_data$Treatment <- as.numeric(raw_data$Treatment)
data <- raw_data
data <- data.frame(data)

#Data overview（数据鸟瞰）
skim(data)
plot_missing(data)
hist(data$Group,breaks = 100)


##############################################################
#Splitting the data（拆分数据）
set.seed(42)
trains <- createDataPartition( 
  y=data$Group, 
  p=0.85,
  list=F,
  times = 1
)

trains2 <- sample(trains,nrow(data)*0.7)
valids <- setdiff(trains, trains2)
data_train <- data[trains2, ]
data_valid <- data[valids, ]
data_test <- data[-trains, ]

#Distribution of the dependent variable after splitting（拆分后因变量分布）
hist(data_train$Group,breaks = 100)
hist(data_valid$Group,breaks = 100)
hist(data_test$Group,breaks = 100)

#Data preparation
colnames(data)
dvfunc <- dummyVars(~.,data = data_train[,2:16],fullRank = T)
data_trainx <- predict(dvfunc,newdata=data_train[,2:16])
data_trainy <- data_train$Group

data_validx <- predict(dvfunc,newdata=data_valid[,2:16])
data_validy <- data_valid$Group

data_testx <- predict(dvfunc,newdata=data_test[,2:16])
data_testy <- data_test$Group

dtrain <- xgb.DMatrix(data = data_trainx,
                      label = data_trainy)
dvalid <- xgb.DMatrix(data = data_validx,
                      label = data_validy)
dtest <- xgb.DMatrix(data = data_testx,
                      label = data_testy)
watchlist <- list(train = dtrain, test = dvalid)

#Training the model（训练模型）
fit_xgb_reg <- xgb.train(
  data = dtrain,
  eta = 0.3,
  gamma = 0.001,
  max_depth = 2,
  subsample = 0.7,
  colsample_bytree = 0.4,
  objective = "reg:squarederror",
  nrounds = 10000000,
  watchlist = watchlist,
  verbose = 1,
  print_every_n = 100,
  early_stopping_rounds = 200
)

fit_xgb_reg

##Importance of features
importance_matrix <- xgb.importance(model = fit_xgb_reg)
print(importance_matrix)

xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = "Gain")

pdf('importance_sort_15features_tpm.pdf',width = 8,height = 8)
xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = "Gain")
dev.off()

####################################
#Prediction（预测）
#Prediction probabilities on the training set（训练集预测概率）
trainpred <- predict(fit_xgb_reg,
                     newdata = dtrain)
#Training and prediction error metrics（训练及预测误差指标）
defaultSummary(data.frame(obs = data_train$Group,
                          pred = trainpred))

#Visual representation of the predicted results on the training set.（图示训练集预测结果）
pdf('XGBoost_actual_vs_predicted_train.pdf',width = 8,height = 4)
plot(x = data_train$Group,
     y= trainpred,
     xlab = "Actual",
     ylab = "Prediction",
     main = "Train_XGBoost_actual_vs_predicted-train",
     sub = "Training-set")
dev.off()

trainlinmod <- lm(trainpred ~ data_train$Group)
abline(trainlinmod, col = "blue", lwd = 2.5, lty = "solid")
abline(a = 0, b = 1, col = "red", lwd = 2.5, lty = "dashed")
legend("topleft",
       legend = c("Model", "Base"),
       col = c("blue","red"),
       lwd = 2.5,
       lty = c("solid", "dashed"))
##save

##############################
#Prediction probabilities on the test set.（测试集预测概率）
testpred <- predict(fit_xgb_reg,
                     newdata = dtest)
#Training and prediction error metrics（训练及预测误差指标）
defaultSummary(data.frame(obs = data_test$Group,
                          pred = testpred))
      # RMSE   Rsquared        MAE 
# 0.42650046 0.99994183 0.09330611

#Visual representation of the predicted results on the training set.（图示训练集预测结果）
pdf('XGBoost_actual_vs_predicted_test.pdf',width = 8,height = 4)
plot(x = data_test$Group,
     y= testpred,
     xlab = "Actual",
     ylab = "Prediction",
     main = "Train_XGBoost_actual_vs_predicted-test",
     sub = "Test-set")
dev.off()

testlinmod <- lm(testpred ~ data_test$Group)
abline(testlinmod, col = "blue", lwd = 2.5, lty = "solid")
abline(a = 0, b = 1, col = "red", lwd = 2.5, lty = "dashed")
legend("topleft",
       legend = c("Model", "Base"),
       col = c("blue","red"),
       lwd = 2.5,
       lty = c("solid", "dashed"))
##save

####################################
#Displaying the predicted results on the training and test sets together.（训练集和测试集预测结果集中展示）
predresult <- 
  data.frame(obs = c(data_train$Group, data_test$Group),
             pred = c(trainpred, testpred),
             group = c(rep("Train",length(trainpred)),
                       rep("Test",length(testpred))))

png(filename = 'XGBoost_actual_vs_predicted-train_test.png',width = 1600,height = 1000, res = 300 )
ggplot(predresult,
       aes(x = obs, y= pred, fill = group, colour = group)) +
  geom_point(shape = 21, size =1.5,  alpha = 0.3) +
  geom_smooth(method = lm, se= F) +
  geom_abline(intercept = 0, slope = 1) +
  labs(fill = NULL, colour = NULL,
       x = "Actual", y = "Predicted",
       title = "XGBoost_actual_vs_predicted") +
  theme_classic() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5))
dev.off()

pdf('XGBoost_actual_vs_predicted-train_test.pdf',width = 8,height = 4)
ggplot(predresult,
       aes(x = obs, y= pred, fill = group, colour = group)) +
  geom_point(shape = 21, size =3) +
  geom_smooth(method = lm, se=F) +
  geom_abline(intercept = 0, slope = 1) +
  labs(fill = NULL, colour = NULL,
       x = "Actual", y = "Predicted",
       title = "XGBoost_actual_vs_predicted") +
  theme_classic() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5))
dev.off()

	
###############################################################################
#Analysis of the features（特征变量的分析）
##SHAP
xgb.plot.shap(data = data_trainX,
              model = fit_xgb_reg,
              top_n = 5)

##top_n = 5（SHAP values of the top 5 features ranked by importance.This step is so slow, so please choose the quantity carefully.）




