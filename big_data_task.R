                                   
#################################################### Big Data Task #########################################

###### Libraries #####

if(require("pacman")=="FALSE"){
  install.packages("pacman")
}

pacman::p_load("doParallel", "readr", "dplyr", "plotly", "caret", "mlbench", "corrplot", "C50", "ranger", "ggplot2",
               "BBmisc", "e1071", "randomForest", "caret", "DMwR", "gbm")


#### Create cluster ####
cluster <- makeCluster(5)
registerDoParallel(cluster)

#Reasure amount of cores in use 
getDoParWorkers() 

######### Upload file and set seed #####
set.seed (123)
iphone<-read_csv("iphone_smallmatrix_labeled_8d.csv")
galaxy<-read_csv("galaxy_smallmatrix_labeled_8d.csv")


#################################################################################### Explore correlation iphone ####

##Plot of correlation
corrplot(cor(iphone),order = "hclust")

#Create an upper triangle in order to sort out to highly correlated variables
tmp <- cor(iphone)
tmp[upper.tri(tmp)] <- 0
diag(tmp) <- 0
iPhone<- iphone[, apply(tmp,2,function(x) all(x<=0.8))]
corrplot(cor(iPhone), order ="hclust")


#Delete variables with high correlation
aa <- names(iphone)
ba <- names(iPhone)
setdiff(aa, ba)

DeleteVar <- c("iphone", "htcphone", "samsungcampos", "nokiacampos", "samsungcamneg", "nokiacamneg", "nokiacamunc", "iphonedispos",
               "samsungdispos","sonydispos", "nokiadispos", "iphonedisneg", "samsungdisneg","nokiadisneg", "htcdisneg","samsungdisunc","nokiadisunc",
               "samsungperpos", "nokiaperpos", "htcperpos", "nokiaperneg", "iosperpos", "googleperpos","iosperneg") 

for (i in DeleteVar) {
  iphone[,DeleteVar] <- NULL }

#################################################################################### Explore correlation Galaxy ####

##Plot of correlation
corrplot(cor(galaxy),order = "hclust")

#Create an upper triangle in order to sort out to highly correlated variables
tmpG <- cor(galaxy)
tmpG[upper.tri(tmp)] <- 0
diag(tmpG) <- 0

Galaxy <- galaxy[, apply(tmp,2,function(x) all(x<=0.8))]
corrplot(cor(Galaxy), order ="hclust")

#Delete variables with high correlation
ab <- names(galaxy)
bb <- names(Galaxy)
setdiff(ab, bb)

DeleteVar <- c("iphone", "htcphone", "samsungcampos", "nokiacampos", "samsungcamneg", "nokiacamneg", "nokiacamunc", "iphonedispos",
               "samsungdispos","sonydispos", "nokiadispos", "iphonedisneg", "samsungdisneg","nokiadisneg", "htcdisneg","samsungdisunc","nokiadisunc",
               "samsungperpos", "nokiaperpos", "htcperpos", "nokiaperneg", "iosperpos", "googleperpos","iosperneg") 

for (i in DeleteVar) {
  galaxy[,DeleteVar] <- NULL }

###################################################################################################### IPHONE ####
                                         

#################################### Pre - processing & Feature selection #####

#### Plottimg distrubution of iphone sentiment ####
plot_ly(iphone, x= ~iphonesentiment, type="histogram", name = "Iphone Sentiment", color = "red")

######################################################### Create iPhone sample
set.seed(123)
iphoneSample <- iphone[sample(1:nrow(iphone), 1000, replace=FALSE),]

################################## Investigate variables with highest importance 

control <- trainControl(method="repeatedcv", number=10, repeats=3)

model <- train(iphonesentiment~., data=iphoneSample, method="cforest", preProcess="scale", trControl=control)

print(importance <- varImp(model, scale=FALSE))

plot(importance)

########################################################################### Recursive feature elimination

## Set up rfeControl with Randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

###### RFE and omitting the response variable (attribute 35 iphonesentiment) 
system.time(rfe_iphone_Results <- rfe(iphoneSample[,1:34], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:34), 
                  rfeControl=ctrl))

predictors(rfe_iphone_Results)

#Plot results
plot(rfe_iphone_Results, type=c("g", "o"))

#Create new data set with rfe recommended features
iphoneRFE <- iphone[,predictors(rfe_iphone_Results)]

#Add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphone$iphonesentiment

#Save file
write.csv(iphoneRFE, file ="iphoneR", row.names = F)

iphoneRFE<-read_csv("iphoneR")
iphoneRFE$X1 <- NULL

############################################################################ Recode iphone sentiment

## Recode sentiment to combine factors - then make variable factor

#RFE
iphoneRFE$iphonesentiment    <- recode(iphoneRFE$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4)
iphoneRFE$iphonesentiment <- as.factor(iphoneRFE$iphonesentiment)

#Without RFE
iphone$iphonesentiment    <- recode(iphone$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4)
iphone$iphonesentiment <- as.factor(iphone$iphonesentiment)

######################################################################################### Splitting data ##### 
set.seed(123)
inTrain<- createDataPartition(y = iphoneRFE$iphonesentiment, p = 0.7,list = FALSE)
trainSet  <- iphoneRFE[inTrain,]
testSet   <- iphoneRFE[-inTrain,]

#Balance uneven distributed data - trainSet 2
trainSet2 <- SMOTE(iphonesentiment~.,as.data.frame(trainSet), k=5,perc.under = 200, perc.over = 100)

#Balance uneven data - trainSet 3
DF1<- filter(trainSet, iphonesentiment %in% c(1,2))
DF1 <-upSample(DF1, DF1$iphonesentiment)
DF1$Class<-NULL

DF2<- filter(trainSet, iphonesentiment %in% c(3,4))
DF2 <-upSample(DF2, DF2$iphonesentiment)
DF2$Class<-NULL

trainSet3<- rbind(DF1,DF2)
k<- filter(trainSet3, iphonesentiment %in% c(1,3))
k <- upSample(k, k$iphonesentiment)
k$Class <-NULL

f<- filter(trainSet3, iphonesentiment %in% c(2,4))
f <-upSample(f, f$iphonesentiment)
f$Class <-NULL

trainSet3<- rbind(f, k)

#Plot new distrubution
plot_ly(trainSet,  x= ~iphonesentiment, type="histogram", name = "Iphone Sentiment", color = "red")
plot_ly(trainSet2, x= ~iphonesentiment, type="histogram", name = "Iphone Sentiment", color = "red")
plot_ly(trainSet3, x= ~iphonesentiment, type="histogram", name = "Iphone Sentiment", color = "red")


############################################################# SVM #####

system.time(svm_iphone <- svm(iphonesentiment ~ ., data = trainSet3))

svm_ipred<- predict(svm_iphone, testSet)

print(confusionMatrix(svm_ipred, testSet$iphonesentiment))

##################################################### Random Forest ####
#Algorithm Tune 
set.seed(123)
bestmtry <- tuneRF(trainSet3[1:19], trainSet3$iphonesentiment, stepFactor=1.5, improve= 0.001, ntree=500, Importance = T)
print(bestmtry)

system.time(randomIphone<- randomForest(iphonesentiment ~ ., data=trainSet3, mtry=9, ntree=500,
               importance=TRUE, na.action=na.omit))

randomIphone_pred <- predict(randomIphone, testSet)

print(confusionMatrix(randomIphone_pred, as.factor(testSet$iphonesentiment)))

################################################################################### Large matrix - iPhone ####
set.seed(123)
Large<-read_csv("LargeMatrixCombined.csv")
Large$id <- NULL

#Select vaiables of importance to Galaxy sentiment
largeIphone<- select (Large,"samsunggalaxy", "googleandroid", "htcperneg", "ios", "sonyxperia", "htcdispos", "iphoneperpos","iphonedisunc",  
                    "iphoneperunc", "htcperunc", "htcdisunc", "htccamneg", "htccampos", "iphoneperneg", "iphonecamunc", "iphonecampos",   
                    "iphonecamneg", "htccamunc", "iosperunc")
                      

#Checking that data frames contains the same names 
a <- data.frame(iphoneRFE= 1:19)
b <- data.frame(largeIphone = 1:19)
all(a == b) #TRUE

#Predicted galaxy sentiment 
random_iphone_Large_pred <- predict(randomIphone, largeIphone)

largeIphone$iphonesentiment<-random_iphone_Large_pred

#Plot new sentiment
plot_ly(largeIphone, x= ~iphonesentiment, type="histogram", color = "red")


############################################################################################################ GALAXY #####

### Pre - processing & Feature selection #####

#### Plotting distrubution of iphone sentiment ####
plot_ly(galaxy, x= ~galaxysentiment, type="histogram")

############################################# Create Galaxy sample
set.seed(123)
galaxySample <- galaxy[sample(1:nrow(galaxy), 1000, replace=FALSE),]

############################################ Investigate variables with highest importance 

Gcontrol <- trainControl(method="repeatedcv", number=10, repeats=3)

Gmodel <- train(galaxysentiment~., data=galaxySample, method="cforest", preProcess="scale", trControl=control)

print(G_importance <- varImp(Gmodel, scale=FALSE))

plot(G_importance)

########################################################################### Recursive feature elimination

## Set up rfeControl with randomforest, repeated cross validation and no updates

ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

###### RFE and omitting the response variable (attribute 59 iphonesentiment) 
system.time(rfe_galaxy_results <- rfe(galaxySample[,1:34], 
                                      galaxySample$galaxysentiment, 
                                      sizes=(1:34), 
                                      rfeControl=ctrl))

predictors(rfe_galaxy_results)

# Plot results
plot(rfe_galaxy_results, type=c("g", "o"))

# create new data set with rfe recommended features
galaxyRFE <- galaxy[,predictors(rfe_galaxy_results)]

# add the dependent variable to iphoneRFE
galaxyRFE$galaxysentiment <- galaxy$galaxysentiment

write.csv(galaxyRFE, file = "galaxyR.csv", row.names = F)

galaxyRFE<-read_csv("galaxyR.csv")

############################################################################ Recode galaxy sentiment

## RFE
galaxyRFE$galaxysentiment <- recode(galaxyRFE_RC$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4)
galaxyRFE$galaxysentiment <- as.factor(galaxyRFE$galaxysentiment)

## Without RFE
galaxy$galaxysentiment <- recode(galaxyRFE_RC$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4)
galaxy$galaxysentiment <- as.factor(galaxyRFE$galaxysentiment)

######################################################################################### Splitting data ##### 
inTrainG  <- createDataPartition(y = galaxyRFE$galaxysentiment, p = 0.7,list = FALSE)
training  <- galaxyRFE[inTrainG,]
testing   <- galaxyRFE[-inTrainG,]

############## Balance uneven distributed data - trainSet 2
training2 <- SMOTE(galaxysentiment~.,as.data.frame(training), k=5,perc.under = 200, perc.over = 100)

############## Balance data - trainSet 3
df1 <- filter(training, galaxysentiment %in% c(1,2))
df1 <-upSample(df1, df1$galaxysentiment)
df1$Class <-NULL

df2<- filter(training, galaxysentiment %in% c(3,4))
df2 <-upSample(df2, df2$galaxysentiment)
df2$Class <-NULL

df3<- filter(training, galaxysentiment %in% c(4))
indexG <- sample(1:nrow(df3), 5000)
df3<-df3[indexG, ]

training3<- rbind(df1,df2)

h<- filter(training3, galaxysentiment %in% c(1,3))
h <- upSample(h, h$galaxysentiment)
h$Class <-NULL

j<- filter(training3, galaxysentiment %in% c(2,4))
j <-upSample(j, j$galaxysentiment)
j$Class <-NULL

training3<- rbind(h,j)

#Plot new distrubution
plot_ly(training,  x= ~galaxysentiment, type="histogram")
plot_ly(training2, x= ~galaxysentiment, type="histogram")
plot_ly(training3, x= ~galaxysentiment, type="histogram")


################################################################################### SVM #### 

system.time(svm_galaxy <- svm(galaxysentiment ~ ., data = training3))

svm_pred<- predict(svm_galaxy, testing)
print(confusionMatrix(svm_pred, testing$galaxysentiment))

##################################################################################################### Random Forest
system.time(random_galaxy<- randomForest(galaxysentiment ~ ., data=training3, mtry=9, ntree=500,
                                        importance=TRUE, na.action=na.omit))

#Output
summary(random_galaxy)

#Predict
random_galaxy_pred <- predict(random_galaxy, testing)

#Confusion matrix
print(confusionMatrix(random_galaxy_pred, testing$galaxysentiment))

################################################################################### Test with large matrix - Galaxy ####
set.seed(123)
Large<-read_csv("LargeMatrixCombined.csv")
Large$id <- NULL

#Select vaiables of importance to Galaxy sentiment
largeGalaxy<- select (Large,"samsunggalaxy", "googleandroid", "htcperneg", "ios", "sonyxperia", "htcdispos","iphoneperpos","iphonedisunc",  
                      "htccamneg", "iphoneperunc","htccampos", "htcperunc", "htcdisunc", "iphoneperneg", "iphonecamunc", "iphonecamneg",   
                      "iphonecampos", "htccamunc")
                      
                      
#Checking that data frames contains the same names 
a <- data.frame(galaxyRFE= 1:19)
b <- data.frame(largeGalaxy = 1:19)
all(a == b) #TRUE

#Predicted galaxy sentiment 
random_galaxyLarge_pred <- predict(randomGalaxy, largeGalaxy)

largeGalaxy$galaxysentiment<-random_galaxyLarge_pred

#Plot new sentiment
plot_ly(largeGalaxy, x= ~galaxysentiment, type="histogram")


############################################################################# Visualisations ####

#Small matrices

Small <- as.data.frame(iPhone <-as.factor(iphoneRFE$iphonesentiment), Galaxy<- as.factor(galaxyRFE$galaxysentiment))

plot_ly(data = Small, alpha = 0.6) %>% 
  add_histogram(x = ~Galaxy, name = "Galaxy") %>%
  add_histogram(x = ~iPhone, name = "iPhone") %>%
  layout(
    title = "Sentiments",
    yaxis = list(
      tickfont = list(color = "blue"),
      overlaying = "y",
      side = "left",
      title = "count"
    ), xaxis = list(title = "Sentiments"))


#Predictions

Sentiments <- as.data.frame (as.factor(Galaxy <- largeGalaxy$galaxysentiment), as.factor(iPhone <- largeIphone$iphonesentiment))

plot_ly(data = Sentiments, alpha = 0.6) %>% 
  add_histogram(x = ~Galaxy, name = "Galaxy") %>%
  add_histogram(x = ~iPhone, name = "iPhone") %>%
  layout(
    title = "Predicted sentiments",
    yaxis = list(
      tickfont = list(color = "blue"),
      overlaying = "y",
      side = "left",
      title = "count"
    ),
    xaxis = list(title = "Sentiments")
  )

################################################################################################ 
## Tried PCA to see if performance improved which it did not end up to give a better result
# Below is only the pre-processing and no models

############################################################################################### PCA - iphone ####
IPHONE <-read_csv("iphone_smallmatrix_labeled_8d.csv")

## Split original data
inTrain<- createDataPartition(y = IPHONE$iphonesentiment, p = 0.7,list = FALSE)
trainingP  <- IPHONE[inTrain,]
testingP   <- IPHONE[-inTrain,]

## Excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(trainingP[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

# use predict to apply pca parameters, create training, exclude dependant
train.pca <- predict(preprocessParams, trainingP[,-59])

## Ad the dependent to training
trainingP$iphonesentiment<- recode(trainingP$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4)
train.pca$iphonesentiment <- as.factor(trainingP$iphonesentiment)

#Use predict to apply pca parameters, create testing, exclude dependant
test.pca <- predict(preprocessParams, testingP[,-59])

# add the dependent to training
testingP$iphonesentimente<- recode(testingP$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4)
test.pca$iphonesentiment <- as.factor(testingP$iphonesentiment)
