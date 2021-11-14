##########################################################
#
# HarvardX Data Science Capstone Project
# Stefan Kälin
#
##########################################################
# /!\: Running the entire code can take up to 30 min!

# Check and install all required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")
if(!require(naivebayes)) install.packages("naivebayes", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(corrplot)
library(class)
library(doParallel)
library(caTools)
library(Rborist)
library(kernlab)

# Download the raw data file and import it
# Original source is: https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction
dl <- tempfile()
download.file("https://github.com/stkaelin/HarvardX_Data_Science_Capstone_Project/raw/main/data.csv", dl)
raw_data <- read.csv(dl)  # read csv file 

# Clean up of the data
colnames(raw_data) <- append(c("Y"), paste0("X", 2:ncol(raw_data)-1)) # Rename the variables to Y, X1, X2, etc.
raw_data[["Y"]] <- as.factor(raw_data$Y) # Convert Y to a factor
sum(is.na(raw_data))  # Check if there are any NAs in the data set. There are none.

# Split the data
# Training set will be 70% of the raw data, while the test set is 30%
# Inspiration from: https://hrcak.srce.hr/file/375100
set.seed(04071993, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = raw_data$Y, times = 1, p = 0.7, list = FALSE)
training <- raw_data[test_index,]
testing <- raw_data[-test_index,]

rm(dl, raw_data, test_index) # Remove intermediary variables


##########################################################
# Data exploration
##########################################################
mean(as.integer(training$Y))-1
ggplot(training, aes(Y)) +geom_bar()

# Principal component analysis
# https://rafalab.github.io/dsbook/large-datasets.html#pca
princcompanal <- prcomp(training[,2:96])
qplot(2:ncol(training), princcompanal$sdev)


##########################################################
# Analysis
##########################################################

# Naive approach: If one predicts that no company will go bankrupt 
predBase <- factor(array(0, c(length(testing$Y),1)), levels = c(0,1))
confusionMatrix(testing$Y, predBase)
# The accuracy is 0.9677

# The caret libary supports also parallel processing to enhance the speed
# https://topepo.github.io/caret/parallel-processing.html
# Using the doParallel library
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

# Feature Selection: Check which variables are the most important.
# Inspiration from: https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
set.seed(04071993, sample.kind="Rounding")
fitlvq <- train(Y ~ ., data = training, method = "lvq")
# Stop parallel processing:
stopCluster(cl)

importance <- varImp(fitlvq)
print(importance)
plot(importance)

# Select the top 20 most significant variables
modelFormula <- formula(Y ~ X19 + X86 + X40 + X43 + X68 
                          + X23 + X10 + X36 + X1 + X90 
                          + X38 + X37 + X95 + X3 + X69 
                          + X2 + X7 + X8 + X91 + X34)

# Calculate the correlation of these 20 features
varCor <- training %>% select(X19,X86,X40,X43,X68, 
                              X23,X10,X36,X1,X90, 
                              X38,X37,X95,X3,X69, 
                              X2,X7,X8,X91,X34) %>% cor()

# Create a correlation plot
corrplot(varCor)

# Some plots of the most significant features
ggplot(training, aes(Y, X19)) + geom_boxplot()
ggplot(training, aes(Y, X86)) + geom_boxplot()
ggplot(training, aes(X19, X86, color=Y)) + geom_point()

training %>% filter(Y==1) %>% summarize(max(X19)) 
training %>% filter(Y==1) %>% summarize(max(X86)) 

# Fit the different machine learning models and predict the outcomes
# Using the doParallel library
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

# Run the models in parallel
set.seed(04071993, sample.kind="Rounding")
fitrf <- train(modelFormula, data = training, method = "rf")
fitknn <- train(modelFormula, data = training, method = "knn")
fitRborist <- train(modelFormula, data = training, method = "Rborist")
fitLogitBoost <- train(modelFormula, data = training, method = "LogitBoost")
fitnaive_bayes <- train(modelFormula, data = training, method = "naive_bayes")
fitlssvmRadial <- train(modelFormula, data = training, method = "lssvmRadial")

predlvq <- predict(fitlvq, newdata = testing)
predrf <- predict(fitrf, newdata = testing)
predknn <- predict(fitknn, newdata = testing)
predRborist <- predict(fitRborist, newdata = testing)
predLogitBoost <- predict(fitLogitBoost, newdata = testing)
prednaive_bayes <- predict(fitnaive_bayes, newdata = testing)
predlssvmRadial <- predict(fitlssvmRadial, newdata = testing)

# Stop parallel processing:
stopCluster(cl)


##########################################################
# Evaluation
##########################################################

# Compute the confusion matrix for each model
confusionMatrix(testing$Y, predlvq)
confusionMatrix(testing$Y, predrf)
confusionMatrix(testing$Y, predknn)
confusionMatrix(testing$Y, predRborist)
confusionMatrix(testing$Y, predLogitBoost)
confusionMatrix(testing$Y, prednaive_bayes)
confusionMatrix(testing$Y, predlssvmRadial)

# Save the information from the confusion matrix in a data frame
confInfo <- data.frame(matrix(NA, nrow = 8, ncol = 4))
colnames(confInfo) <- c("Method", "Accuracy", "Sensitivity", "Specificity")
confInfo[1:8,1] <- c("lvq", "rf", "knn", "Rborist", "LogitBoost",
                     "naive_bayes","lssvmRadial", "vote")
confInfo[1,2:4] <- c(confusionMatrix(testing$Y, predlvq)$overall[1],
                     confusionMatrix(testing$Y, predlvq)$byClass[1:2])
confInfo[2,2:4] <- c(confusionMatrix(testing$Y, predrf)$overall[1],
                     confusionMatrix(testing$Y, predrf)$byClass[1:2])
confInfo[3,2:4] <- c(confusionMatrix(testing$Y, predknn)$overall[1],
                     confusionMatrix(testing$Y, predknn)$byClass[1:2])
confInfo[4,2:4] <- c(confusionMatrix(testing$Y, predRborist)$overall[1],
                     confusionMatrix(testing$Y, predRborist)$byClass[1:2])
confInfo[5,2:4] <- c(confusionMatrix(testing$Y, predLogitBoost)$overall[1],
                     confusionMatrix(testing$Y, predLogitBoost)$byClass[1:2])
confInfo[6,2:4] <- c(confusionMatrix(testing$Y, prednaive_bayes)$overall[1],
                     confusionMatrix(testing$Y, prednaive_bayes)$byClass[1:2])
confInfo[7,2:4] <- c(confusionMatrix(testing$Y, predlssvmRadial)$overall[1],
                     confusionMatrix(testing$Y, predlssvmRadial)$byClass[1:2])

# Create a voting system
preds <- data.frame(predlvq) %>%
  mutate(predrf, predknn, predRborist, predLogitBoost, prednaive_bayes, predlssvmRadial)

indx <- sapply(preds, is.factor)
preds[indx] <- lapply(preds[indx], function(x) as.numeric(as.character(x)))
preds <- preds %>% mutate(vote=if_else(rowMeans(preds)>=0.5,1,0))

head(preds,10)

confusionMatrix(testing$Y, factor(preds$vote, levels = c(0,1)))

# Save it in the in a data frame
confInfo[8,2:4] <- c(confusionMatrix(testing$Y, factor(preds$vote, levels = c(0,1)))$overall[1],
              confusionMatrix(testing$Y, factor(preds$vote, levels = c(0,1)))$byClass[1:2])

confInfo

# Calculate how many bankruptcies were correctly predicted
predbank <-mutate(preds, Y=as.numeric(testing$Y)-1,
                    hitlvq=if_else(Y==1 & predlvq==1,1,0),
                    hitrf=if_else(Y==1 & predrf==1,1,0),
                    hitlknn=if_else(Y==1 & predknn==1,1,0),
                    hitRborist=if_else(Y==1 & predRborist==1,1,0),
                    hitLogitBoost=if_else(Y==1 & predLogitBoost==1,1,0),
                    hitnaive_bayes=if_else(Y==1 & prednaive_bayes==1,1,0),
                    hitlssvmRadial=if_else(Y==1 & predlssvmRadial==1,1,0),
                    hitvote=if_else(Y==1 & vote==1,1,0)) %>% select(9:17)

colSums(predbank)

# Calculate how many bankruptcies were missed
missedbank <-mutate(preds, Y=as.numeric(testing$Y)-1,
                    missedlvq=if_else(Y==1 & predlvq==0,1,0),
                    missedlrf=if_else(Y==1 & predrf==0,1,0),
                    missedlknn=if_else(Y==1 & predknn==0,1,0),
                    missedRborist=if_else(Y==1 & predRborist==0,1,0),
                    missedLogitBoost=if_else(Y==1 & predLogitBoost==0,1,0),
                    missednaive_bayes=if_else(Y==1 & prednaive_bayes==0,1,0),
                    missedlssvmRadial=if_else(Y==1 & predlssvmRadial==0,1,0),
                    missedvote=if_else(Y==1 & vote==0,1,0)) %>% select(9:17)

colSums(missedbank)

# Calculate the number of false positives (predicted a bankruptcy where there was not)
falsepos <-mutate(preds, Y=as.numeric(testing$Y)-1,
                  falselvq=if_else(Y==0 & predlvq==1,1,0),
                  falselrf=if_else(Y==0 & predrf==1,1,0),
                  falselknn=if_else(Y==0 & predknn==1,1,0),
                  falseRborist=if_else(Y==0 & predRborist==1,1,0),
                  falseLogitBoost=if_else(Y==0 & predLogitBoost==1,1,0),
                  falsenaive_bayes=if_else(Y==0 & prednaive_bayes==1,1,0),
                  falselssvmRadial=if_else(Y==0 & predlssvmRadial==1,1,0),
                  falsevote=if_else(Y==0 & vote==1,1,0)) %>% select(9:17)

colSums(falsepos)

# Calculate the number of predicted survivals
survive <-mutate(preds, Y=as.numeric(testing$Y)-1,
                 survivelvq=if_else(Y==0 & predlvq==0,1,0),
                 survivelrf=if_else(Y==0 & predrf==0,1,0),
                 survivelknn=if_else(Y==0 & predknn==0,1,0),
                 surviveRborist=if_else(Y==0 & predRborist==0,1,0),
                 surviveLogitBoost=if_else(Y==0 & predLogitBoost==0,1,0),
                 survivenaive_bayes=if_else(Y==0 & prednaive_bayes==0,1,0),
                 survivelssvmRadial=if_else(Y==0 & predlssvmRadial==0,1,0),
                 survivevote=if_else(Y==0 & vote==0,1,0)) %>% select(9:17)

colSums(survive)

## End of Code