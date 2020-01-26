#-------------------------------------Set-up environment----------------------------------------#

# Install packages
pacman::p_load(pacman, rstudioapi, caret, doParallel, corrplot, tidyverse)

# Set working directory
path <- getActiveDocumentContext()$path
setwd(dirname(path))

# Set random seed
set.seed(123)

# Set up parallel processing

# Find how many cores are on your machine
detectCores()

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
clusters <-  makeCluster(2)

# Register cluster
registerDoParallel(clusters)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 


#------------------------------------------Read data-------------------------------------------#

# Import data
data_iphone <- read.csv(file = "./data/iphone_smallmatrix_labeled_8d.csv")
data_galaxy <- read.csv(file = "./data/galaxy_smallmatrix_labeled_8d.csv")

#-------------------------------------Initial Exploration--------------------------------------#

# Initial exploration
str(data_iphone)
summary(data_iphone)
head(data_iphone)

str(data_galaxy)
summary(data_galaxy)
head(data_galaxy)

# Make sentiment factor
#data_iphone$iphonesentiment = as.factor(data_iphone$iphonesentiment)
#data_galaxy$iphonesentiment = as.factor(data_iphone$galaxysentiment)

# Check the frequency of and galaxy data
ggplot(data = data_galaxy, aes(x = data_galaxy$samsunggalaxy)) +
  geom_histogram(stat = "count")

# We see that for most intances there are almost no mentions of 
# phones. This leads us to believe that
# these webpages do not really convey much information about the true sentiment of this brand/phone.
# Therefore we will not continue with this data set. It is recommended gather the data again and
# include more webpages that 

# For the same reason we drop the rows that have less than 3 iphone mentions
data_iphone <- data_iphone[data_iphone$iphone > 2,]

# Check the dependant variable
ggplot(data = data_iphone, aes(x = iphonesentiment)) +
  geom_histogram(stat = "count")

# check for missing addresses
sum(is.na(data_iphone))
sum(is.na(data_galaxy))

# check for missing addresses
sum(is.null(data_iphone))
sum(is.null(data_galaxy))

#--------------------------------Preprocessing adn feature selection--------------------------------#

# Drop lines with less than reviews

# Increase max print
options(max.print=1000000)

# Check correlations
corr_galaxy <- cor(data_galaxy)
corr_iphone <- cor(data_iphone)
corrplot(corr_galaxy)

# Create a new data set and remove features highly correlated with the dependant 
# iphoneCOR$featureToRemove <- NULL

# Recode sentiment
data_iphone[data_iphone$iphonesentiment %in% c(4, 5), ]$iphonesentiment <- "positive"
data_iphone[data_iphone$iphonesentiment %in% c(0, 1), ]$iphonesentiment <- "negative"
data_iphone[data_iphone$iphonesentiment %in% c(2, 3), ]$iphonesentiment <- "neutral"

# Convert to factor
data_iphone$iphonesentiment <- as.factor(data_iphone$iphonesentiment)

# Check the results of recoding
ggplot(data = data_iphone, aes(x = iphonesentiment)) +
  geom_histogram(stat = "count")

# Let's check class imbalances
table(data_iphone$iphonesentiment)

# Upsample and downsample
data_iphone <- upSample(x = data_iphone[, -ncol(data_iphone)], y = data_iphone$iphonesentiment)
colnames(data_iphone)[59] <- "iphonesentiment"
table(data_iphone$iphonesentiment)
str(data_iphone)
# iphone data

# NearZeroVar() with saveMetrics = TRUE returns an object containing a table including: frequency ratio, 
# percentage unique, zero variance and near zero variance 
nzvMetrics <- nearZeroVar(data_iphone, saveMetrics = TRUE)
nzvMetrics

# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv <- nearZeroVar(data_iphone, saveMetrics = FALSE) 
nzv


# create a new data set and remove near zero variance features
iphoneNZV <- data_iphone[,-nzv]
str(iphoneNZV)

# Let's sample the data before using RFE
iphoneSample <- data_iphone[sample(1:nrow(data_iphone), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 2,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults

# Save results to RData so we can open these in R markdown
save(rfeResults, file="./saved_objects/rfeResults.RData")

# Plot results
plot(rfeResults, type=c("g", "o"))

# create new data set with rfe recommended features
iphoneRFE <- data_iphone[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- data_iphone$iphonesentiment

# Save results to csv so we can open these in R markdown
write.csv(iphoneRFE, file = "/Users/davidverbiest/google_drive/data_science/ubiqum/m4/t3/data/rfeResults.csv")

# review outcome
str(iphoneRFE)

# Rename iphoneRGE as df for modeling
df <- iphoneRFE

#--------------------------------Modeling--------------------------------#

# Create models and predict on test
models = c("rf", "knn", "svmRadial")
models_fitted <-  list()
models_results <- list()
aggr_confusion_matrix <- list()

# Set-up resampling method
fit_control <-  trainControl(method = "repeatedcv",
                             number = 5,
                             repeats = 3)

for (i in models) {
  
  model <- train(iphonesentiment ~ ., 
                 data = df, 
                 preProc = c("center", "scale"),
                 method = i,
                 trControl = fit_control,
                 metric = "Accuracy")
  
  models_fitted[i] <- list(model)
  
  models_results[i] <- list(model$results)
  
}

# Check model performance and pick best performing model
resamps <- resamples(models_fitted)
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps, layout = c(2, 1))

# Rename iphoneRGE as df for modeling
df <- iphoneRFE

# random forest has te best performance. We train a random forest on all the data
model <- train(iphonesentiment ~ ., 
               data = df, 
               preProc = c("center", "scale"),
               method = "rf",
               tuneLength = 5)

#--------------------------------Predictions--------------------------------#

# Make predictions on the results of the webcrawl
  
# Load webscraped data that will be used to estimate sentiment of the data -->
testing <- read.csv(file = "/Users/davidverbiest/google_drive/data_science/ubiqum/m4/t3/data/join_concat_concat_factors.csv")

# Just like mentioned above we will drop the reviews that have less than 3 mentions of the brand
testing <- testing[testing$iphone > 2,]

# Select same features as in training
testing <- testing[,predictors(rfeResults)]

# Check if features are the same for both training and testing
colnames(testing) == colnames(df[,-ncol(df)])

# Use trained model to make predictions
final_pred <-  predict(model, testing)

# Check sentiment
table_final_pred <- table(final_pred)

# Make a bar chart of the sentiment
final_pred <- as.data.frame(final_pred)

ggplot(final_pred, aes(x = final_pred)) + 
  geom_bar(fill =  "#33A1DE") +
  xlab("Sentiment") +
  ggtitle("Overview iPhone Sentiment")

# Reset settings for parallel computing
stopCluster(clusters)

