library(caret)

training <- read.csv("../Downloads//pml-training.csv",header = TRUE)
testing <- read.csv("../Downloads//pml-testing.csv",header = TRUE)

training <- training[,colSums(is.na(training)) <= nrow(training)*.9]
training <- training[,colSums(training != "") >= nrow(training)*.9]
training <- subset(training,select = -c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))

inTrain <- createDataPartition(training$classe,p=0.9,list=FALSE)
valid <- training[-inTrain,]
train <- training[inTrain,]

library(doParallel)
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)


tc <- trainControl("repeatedcv", number=3, repeats=5, classProbs=TRUE, savePred=T) 
modFitTC <- train(train$classe ~.,data = train,method="rf",trControl = tc, preProcess = c("center","scale"))

result <- table(valid$classe,predict(modFitTC,valid))

stopCluster(cl)