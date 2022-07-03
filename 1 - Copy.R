# Classification of Cars Using K-Nearest Neighbor and Support Vector Machine
#Loading all the required libraries
install.packages("caret")
install.packages("lattice")
install.packages("ggplot2")
install.packages("MASS")
install.packages("RWeka")
install.packages("e1071")
install.packages("kernlab")
install.packages("plyr")

library(lattice)
library(ggplot2)
library(caret)
library(MASS)
library(e1071)
library(kernlab)
library(plyr)

#reading the csv file
data <- read.csv("car_evaluation.csv", sep = ",", header = F)
#Add the column names
colnames(data) <- c("Buying","Maintenance","Doors","Persons","Luggage_boot","Safety","Class")

#Write the data as a CSV file to directory
write.csv(data, file = "car.csv",row.names = F)

#Split data into Training set and validation set
# create a list of 80% of the total observations for training
validation_index <- createDataPartition(data$Class, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- data[-validation_index,]
# use the remaining 80% of data to training and testing the models
data <- data[validation_index,]

#Display datatype 
sapply(data, class)

#Shows the classes for classification
levels(data$Class)

#% distribution for each class type
dist <- prop.table(table(data$Class)) * 100
cbind(freq=table(data$Class), dist=dist)

# summary of car data
summary(data)

# Plot graph showing distribution of data
plot(data)
plot(data[1:6], col = as.numeric(data$Class))

x <- data[,1:6]
y <- data[7]

#bar graph plot
par(mfrow=c(1,1))
plot(data$Class, ylim = c(0,1000))

#bar graph plot for each variables showing their distribution among class
#Maintenance
ggplot(data, aes(Maintenance, fill = Class)) + geom_bar()
#Buying Price
ggplot(data, aes(Buying, fill = Class)) + geom_bar()
#Number of Doors
ggplot(data, aes(Doors, fill = Class)) + geom_bar()
#Person Capacity
ggplot(data, aes(Persons, fill = Class)) + geom_bar()
# Luggage Boot
ggplot(data, aes(Luggage_boot, fill = Class)) + geom_bar()
#Safety
ggplot(data, aes(Safety, fill = Class)) + geom_bar()


#Number of fold = 10
#The following is used in the training function for computational purposes.

control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#Classification

# kNN (k Nearest Neighbors)
set.seed(7)
fit.knn <- train(Class~., data=data, method="knn", metric=metric, trControl=control)
plot(fit.knn)
# estimate skill of KNN on the data
classify_train <- predict(fit.knn, data)
confusionMatrix(classify_train, data$Class)                    #error
classify_test <- predict(fit.knn, validation)
confusionMatrix(classify_test, validation$Class)              #error
# Create data frame
check <- data.frame(validation$Class)
check$validation.Class<- classify_test
#trying to compare the values of train and test data
table(check$validation.Class)
table(check$Class)

# Predict using the test values
for(i in 1:nrow(validation)) {
  classification <- predict(fit.knn,validation[i,])
  print(classification)
  print(validation[i,])
}


# SVM
set.seed(7)
fit.svm <- train(Class~., data=data, method="svmRadial", metric=metric, trControl=control)
plot(fit.svm)
#estimating the skills of svm on the data
classify_train <- predict(fit.svm, data)
confusionMatrix(classify_train, data$Class)         #error
classify_test <- predict(fit.svm, validation)
confusionMatrix(classify_test, validation$Class)    #error
# Create data frame
check <- data.frame(validation$Class)
check$validation.Class<- classify_test
#trying to compare the values of train and test data
table(check$validation.Class)
table(validation$Class)

# Predict using the test values
for(i in 1:nrow(validation)) {
  classification <- predict(fit.svm,validation[i,])
  print(classification)
  print(validation[i,])
}

#Compares the result of both algorithms
results <- resamples(list(knn=fit.knn, svm=fit.svm))
summary(results)

# model accuracy comparison 
dotplot(results)

#prints all associated parameters
#K nearest neighbor
print(fit.knn)
#Support Vector Machines
print(fit.svm)
