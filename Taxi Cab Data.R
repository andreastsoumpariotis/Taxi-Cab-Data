
### STAT 6306 Final Project ###

#Install and load required packages (must be done every single time in AWS)
install.packages("sparklyr")
install.packages("aws.s3")
install.packages("tidyverse")
install.packages("dplyr")
library(sparklyr)
library(aws.s3)
library(tidyverse)
library(dplyr)


#Start a Spark connection
config <- spark_config()                            #dreate a config to tune memory
config[["sparklyr.shell.driver-memory"]] <- "10G"   #set driver memory to 10GB
sc <- spark_connect(master = "yarn",               #connect to the AWS Cluster
                    config = config,
                    spark_home = "/usr/lib/spark")  #this is where AWS puts the Spark Code

# Creativity Part #


# GREEN CAB DATA

start = proc.time()
Taxi_green_tbl = spark_read_csv(sc, path = "s3://stat6306studentfilebucket/Andreas and Webb/Taxi/Green/Green2.csv", infer_schema = TRUE, header = TRUE)
proc.time() - start

sdf_describe(Taxi_green_tbl)
head(Taxi_green_tbl)

partitions <- Taxi_green_tbl %>%
  sdf_random_split(training = 0.1, test = 0.9, seed = 1111) # Trick random split function to sample data so we can plot it using our R sesseion memory

Taxi_green_training <- partitions$training 
Taxi_green_test <- partitions$test

sdf_describe(Taxi_green_training)

#Pick variables we're interested in (apply to both training and test sets)
Taxi_green_training = Taxi_green_training %>% select(passenger_count,trip_distance,fare_amount,tip_amount,total_amount,payment_type)
Taxi_green_test = Taxi_green_test %>% select(passenger_count,trip_distance,fare_amount,tip_amount,total_amount,payment_type)

#Remove all NA values for both training and test
Taxi_green_training = na.omit(Taxi_green_training)
Taxi_green_test = na.omit(Taxi_green_test)

#Correlation between passenger count, trip distance, and total amount
ml_corr(Taxi_green_training, columns = c("passenger_count", "trip_distance", "total_amount"))

#Inspect passenger count, trip distance, total amount, and payment type

summary(Taxi_green_training %>% 
          select(passenger_count) %>%
          collect())

summary(Taxi_green_training %>% 
          select(trip_distance) %>%
          collect())

Taxi1_tbl = Taxi_green_training %>%
  select(payment_type, total_amount) %>%
  collect()
summary(Taxi1_tbl$total_amount)
summary(as.factor(Taxi1_tbl$payment_type))

# Plotting Cleaned Data with dplyr
start = proc.time()
Taxi_green_training %>% 
  select(passenger_count,trip_distance) %>%
  filter(passenger_count > 0 & trip_distance > 0) %>%
  filter(passenger_count < 10 & trip_distance < 100) %>%
  collect() %>%
  ggplot(aes(x = passenger_count, y = trip_distance)) +
  geom_point() + 
  ggtitle("Trip Distance v. Passenger Count") +
  xlab("Passenger Count") +
  ylab("Trip Distance")
proc.time() - start

#Investigate further
summary(Taxi_green_training %>% 
          select(passenger_count,trip_distance) %>%
          filter(passenger_count > 0 & trip_distance > 0) %>%
          filter(passenger_count < 100 & trip_distance < 100) %>%
          collect())

# Linear Regression Model 1
fitlm = Taxi_green_training %>% 
  select(passenger_count,trip_distance) %>%
  filter(passenger_count > 0 & trip_distance > 0) %>%
  filter(passenger_count < 100 & trip_distance < 100) %>%
  ml_linear_regression(trip_distance~passenger_count)

tidy(fitlm) #paremeter estimate table
pred = ml_predict(fitlm,Taxi_green_test)
ml_regression_evaluator(pred, label_col = "trip_distance", metric_name = "rmse")


# Linear Regression Model 2
fitlm2 = Taxi_green_training %>% 
  select(passenger_count,total_amount) %>%
  filter(passenger_count > 0 & total_amount > 0) %>%
  filter(passenger_count < 10 & total_amount < 500) %>%
  ml_linear_regression(total_amount~passenger_count)

tidy(fitlm2) #paremeter estimate table
pred = ml_predict(fitlm2,Taxi_green_test)
ml_regression_evaluator(pred, label_col = "total_amount", metric_name = "rmse")


#Logistic Regression

fitlog <- Taxi_green_training %>% select(payment_type,total_amount) %>% 
  filter(payment_type == 1 | payment_type == 2) %>%
  ml_logistic_regression(payment_type~total_amount)

fitlog
summary(fitlog)
tidy(fitlog)

pred <- ml_predict(fitlog, Taxi_green_test)
ml_binary_classification_evaluator(pred)
ml_multiclass_classification_evaluator(pred, metric_name = "f1")
ml_multiclass_classification_evaluator(pred, metric_name = "accuracy")
ml_multiclass_classification_evaluator(pred, metric_name = "weightedPrecision")
ml_multiclass_classification_evaluator(pred, metric_name = "weightedRecall")


# YELLOW CAB DATA

start = proc.time()
Taxi_yellow_tbl = spark_read_csv(sc, path = "s3://stat6306studentfilebucket/Andreas and Webb/Taxi/Yellow/Yellow.csv", infer_schema = TRUE, header = TRUE)
proc.time() - start

sdf_describe(Taxi_yellow_tbl)
head(Taxi_yellow_tbl)

partitions <- Taxi_yellow_tbl %>%
  sdf_random_split(training = 0.1, test = 0.9, seed = 1111) # Trick random split function to sample data so we can plot it using our R sesseion memory

Taxi_yellow_training <- partitions$training 
Taxi_yellow_test <- partitions$test

#Pick variables we're interested in
Taxi_yellow_training = Taxi_yellow_training %>% select(passenger_count,trip_distance,fare_amount,tip_amount,total_amount,payment_type)
Taxi_yellow_test = Taxi_yellow_test %>% select(passenger_count,trip_distance,fare_amount,tip_amount,total_amount,payment_type)

#Remove NA values
Taxi_yellow_training = na.omit(Taxi_yellow_training)
Taxi_yellow_test = na.omit(Taxi_yellow_test)

#Correlation between a few variables
ml_corr(Taxi_yellow_training, columns = c("passenger_count", "trip_distance", "total_amount"))

#Inspect passenger count, trip distance, total amount, and payment type

summary(Taxi_yellow_training %>% 
          select(passenger_count) %>%
          collect()) #nothing off with passenger count

summary(Taxi_yellow_training %>% 
          select(trip_distance) %>%
          collect()) #outliers in trip distance

Taxi1_tbl = Taxi_yellow_training %>%
  select(payment_type, total_amount) %>%
  collect()

summary(Taxi1_tbl$total_amount) #outliers in total amount
summary(factor(Taxi1_tbl$payment_type)) #four payment type options

# Plotting Cleaned Data with dplyr
start = proc.time()
Taxi_yellow_training %>% 
  select(passenger_count,trip_distance) %>%
  filter(passenger_count > 0 & trip_distance > 0) %>%
  filter(passenger_count < 10 & trip_distance < 100) %>%
  collect() %>%
  ggplot(aes(x = passenger_count, y = trip_distance)) +
  geom_point() + 
  ggtitle("Trip Distance v. Passenger Count") +
  xlab("Passenger Count") +
  ylab("Trip Distance")
proc.time() - start

#Investigate further
summary(Taxi_yellow_training %>% 
          select(passenger_count,trip_distance) %>%
          filter(passenger_count > 0 & trip_distance > 0) %>%
          filter(passenger_count < 100 & trip_distance < 100) %>%
          collect())

# Linear Regression Model 1
fitlm = Taxi_yellow_training %>% 
  select(passenger_count,trip_distance) %>%
  filter(passenger_count > 0 & trip_distance > 0) %>%
  filter(passenger_count < 100 & trip_distance < 100) %>%
  ml_linear_regression(trip_distance~passenger_count)
tidy(fitlm)

# Linear Regression Model 2
fitlm2 = Taxi_yellow_training %>% 
  select(passenger_count,total_amount) %>%
  filter(passenger_count > 0 & total_amount > 0) %>%
  filter(passenger_count < 10 & total_amount < 500) %>%
  ml_linear_regression(total_amount~passenger_count)
tidy(fitlm2)



#Logistic Regression

fitlog <- Taxi_yellow_training %>% select(payment_type,total_amount) %>% 
  filter(payment_type == 1 | payment_type == 2) %>%
  ml_logistic_regression(payment_type~total_amount)

fitlog
summary(fitlog)
tidy(fitlog)

pred <- ml_predict(fitlog, Taxi_yellow_test)
ml_binary_classification_evaluator(pred)
ml_multiclass_classification_evaluator(pred, metric_name = "f1") #f1
ml_multiclass_classification_evaluator(pred, metric_name = "accuracy") #accuracy
ml_multiclass_classification_evaluator(pred, metric_name = "weightedPrecision") # Precision
ml_multiclass_classification_evaluator(pred, metric_name = "weightedRecall") # Recall

# FOR HIRE GROUPS

forhire = spark_read_csv(sc,path = "s3://stat6306studentfilebucket/Andreas and Webb/ForHireTaxi/ForHireVehicle2019.csv", infer_schema = TRUE, header = TRUE, repartition = 100)

sdf_describe(forhire)


partitions <- forhire %>% sdf_random_split(training = 0.999,test = 0.001, seed = 1)
forhirebig <- partitions$training
forhiresmall <- partitions$test

sdf_describe(forhiresmall)

forhiresmall1 <- forhiresmall %>% select(hvfhs_license_num) %>% collect()

forhiresmall2 <- forhiresmall1 %>% mutate(hvfhs_license_num = if_else(hvfhs_license_num == "HV0002","Juno",hvfhs_license_num)) %>%
  mutate(hvfhs_license_num = if_else(hvfhs_license_num == "HV0003","Uber",hvfhs_license_num)) %>%
  mutate(hvfhs_license_num = if_else(hvfhs_license_num == "HV0004","Via",hvfhs_license_num)) %>%
  mutate(hvfhs_license_num = if_else(hvfhs_license_num == "HV0005","Lyft",hvfhs_license_num))

barplot((summary(factor(forhiresmall2$hvfhs_license_num))),main="Histogram of For Hire Rides", 
        xlab="For Hire Ride Service",
        ylab = "Count",
        border="black", 
        col="gray")

forhiresmall3 <- forhiresmall %>% select(SR_Flag) %>% collect()

forhiresmall4 <- forhiresmall3 %>% mutate(SR_Flag = if_else(SR_Flag == 1, "Shared Ride","Non Shared Ride"))

barplot((summary(factor(forhiresmall4$SR_Flag))),
        main = "Percentage of Shared Rides",
        xlab = "Shared Ride or Non Shared (NA)",
        ylab = "Count",
        border = "black",
        col = "gray")



# Prediction Part #

#Yellow Predictions

#Yellow Predictions 2009
yellow2009train <- spark_read_csv(sc, path = "s3://stat6306studentfilebucket/Andreas and Webb/Taxi/Yellow/yellow_2009-03.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
sdf_describe(yellow2009train)

yellow2009test <- spark_read_csv(sc, path = "s3://stat6306project/Yellow Test Student/TaxiTest2009_WO_df.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
sdf_describe(yellow2009test)

yellow2009sub <- yellow2009train %>% select(Tip_Amt,Trip_Distance,Passenger_Count,Payment_Type) %>% filter(Payment_Type == "CREDIT")
yellow2009subtest <- yellow2009test %>% select(Trip_Distance,Passenger_Count,Payment_Type)

fityellow2009 <- yellow2009sub %>% ml_linear_regression(Tip_Amt~Trip_Distance+Passenger_Count)
tidy(fityellow2009)

preds2009 <- ml_predict(fityellow2009,yellow2009subtest)

preds2009subset <- preds2009 %>% select(prediction) %>% collect()
preds2009subset1 <- yellow2009test %>% select(ID) %>% collect()

predictions2009 <- matrix(0,nrow = 32863,ncol = 2)
colnames(predictions2009) <- c("ID","tip_amount")

predictions2009[,1] <- preds2009subset1$ID

predictions2009[,2] <- preds2009subset$prediction

#Yellow Predictions 2011
yellow2011train <- spark_read_csv(sc, path = "s3://stat6306studentfilebucket/Andreas and Webb/Taxi/Yellow/yellow_2011-02.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
sdf_describe(yellow2011train)

yellow2011test <- spark_read_csv(sc, path = "s3://stat6306project/Yellow Test Student/TaxiTest2011_WO_df.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
sdf_describe(yellow2011test)

yellow2011sub <- yellow2011train %>% select(tip_amount,trip_distance,passenger_count,payment_type) %>% filter(payment_type == "CRD")
yellow2011subtest <- yellow2011test %>% select(trip_distance,passenger_count,payment_type)

fityellow2011 <- yellow2011sub %>% ml_linear_regression(tip_amount~trip_distance+passenger_count)
tidy(fityellow2011)

preds2011 <- ml_predict(fityellow2011,yellow2011subtest)

preds2011subset <- preds2011 %>% select(prediction) %>% collect()
preds2011subset1 <- yellow2011test %>% select(ID) %>% collect()

predictions2011 <- matrix(0,nrow = 61535,ncol = 2)
colnames(predictions2011) <- c("ID","tip_amount")

predictions2011[,1] <- preds2011subset1$ID

predictions2011[,2] <- preds2011subset$prediction

#Yellow Predictions 2018
yellow2018train <- spark_read_csv(sc, path = "s3://stat6306studentfilebucket/Andreas and Webb/Taxi/Yellow/yellow_tripdata_2018-03.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
sdf_describe(yellow2018train)

yellow2018test <- spark_read_csv(sc, path = "s3://stat6306project/Yellow Test Student/TaxiTest2018_WO_df.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
sdf_describe(yellow2018test)

yellow2018sub <- yellow2018train %>% select(tip_amount,trip_distance,passenger_count,payment_type) %>% filter(payment_type == 1)
yellow2018subtest <- yellow2018test %>% select(trip_distance,passenger_count,payment_type)

fityellow2018 <- yellow2018sub %>% ml_linear_regression(tip_amount~trip_distance+passenger_count)
tidy(fityellow2018)

preds2018 <- ml_predict(fityellow2018,yellow2018subtest)

preds2018subset <- preds2018 %>% select(prediction) %>% collect()
preds2018subset1 <- yellow2018test %>% select(ID) %>% collect()

predictions2018 <- matrix(0,nrow = 66530,ncol = 2)
colnames(predictions2018) <- c("ID","tip_amount")

predictions2018[,1] <- preds2018subset1$ID

predictions2018[,2] <- preds2018subset$prediction

#Yellow Predictions 2020
yellow2020train <- spark_read_csv(sc, path = "s3://stat6306studentfilebucket/Andreas and Webb/Taxi/Yellow/yellow_tripdata_2020-05.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
sdf_describe(yellow2020train)

yellow2020test <- spark_read_csv(sc, path = "s3://stat6306project/Yellow Test Student/TaxiTest2020_WO_df.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
sdf_describe(yellow2020test)

yellow2020sub <- yellow2020train %>% select(tip_amount,trip_distance,passenger_count,payment_type) %>% filter(payment_type == 1)
yellow2020subtest <- yellow2020test %>% select(trip_distance,passenger_count,payment_type)

fityellow2020 <- yellow2020sub %>% ml_linear_regression(tip_amount~trip_distance+passenger_count)
tidy(fityellow2020)

preds2020 <- ml_predict(fityellow2020,yellow2020subtest)

preds2020subset <- preds2020 %>% select(prediction) %>% collect()
preds2020subset1 <- yellow2020test %>% select(ID) %>% collect()

predictions2020 <- matrix(0,nrow = 70027,ncol = 2)
colnames(predictions2020) <- c("ID","tip_amount")

predictions2020[,1] <- preds2020subset1$ID

predictions2020[,2] <- preds2020subset$prediction


#Put everything together

yellowpredictfinal <- rbind(predictions2009,predictions2011,predictions2018,predictions2020)

Sys.setenv("AWS_ACCESS_KEY_ID" = "AKIAJFPIUJBDHKKBR6SA",
           "AWS_SECRET_ACCESS_KEY" = "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",
           "AWS_DEFAULT_REGION" = "us-east-2")

aws.s3::bucketlist()
aws.s3::get_bucket("ds6306unit13")

s3write_using(yellowpredictfinal, FUN = write.csv,bucket = "stat6306studentfilebucket",object = "Group10YellowTaxiFinalPredictions")


#Green Predictions


#Green Predictions 2015

green2015train <- spark_read_csv(sc, path = "s3://stat6306studentfilebucket/Andreas and Webb/Taxi/Green/Green 2015/green_tripdata_2015-11.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
green2015test <- spark_read_csv(sc, path = "s3://stat6306project/Green Test Student/TaxiTest2015G_WO_df.csv",infer_schema = TRUE, header = TRUE, repartition = 100)

green2015sub <- green2015train %>% select(Trip_distance,Passenger_count,Payment_type) %>% filter(Payment_type == 1 | Payment_type == 2)
green2015subtest <- green2015test %>% select(Trip_distance,Passenger_count)

fitgreen2015 <- green2015sub %>% ml_logistic_regression(Payment_type~Trip_distance+Passenger_count)
tidy(fitgreen2015)

preds2015 <- ml_predict(fitgreen2015,green2015subtest)

preds2015subset <- preds2015 %>% select(prediction) %>% collect()
preds2015subset1 <- green2015test %>% select(ID) %>% collect()

predictions2015 <- matrix(0,nrow = 76323,ncol = 2)
colnames(predictions2015) <- c("ID","tip_amount")

predictions2015[,1] <- preds2015subset1$ID

predictions2015[,2] <- preds2015subset$prediction


#Green Predictions 2017

green2017train <- spark_read_csv(sc, path = "s3://stat6306studentfilebucket/Andreas and Webb/Taxi/Green/Green 2017/green_tripdata_2017-04.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
green2017test <- spark_read_csv(sc, path = "s3://stat6306project/Green Test Student/TaxiTest2017G_WO_df.csv",infer_schema = TRUE, header = TRUE, repartition = 100)

green2017sub <- green2017train %>% select(trip_distance,passenger_count,payment_type) %>% filter(payment_type == 1 | payment_type == 2)
green2017subtest <- green2017test %>% select(trip_distance,passenger_count)

fitgreen2017 <- green2017sub %>% ml_logistic_regression(payment_type~trip_distance+passenger_count)
tidy(fitgreen2017)

preds2017 <- ml_predict(fitgreen2017,green2017subtest)

preds2017subset <- preds2017 %>% select(prediction) %>% collect()
preds2017subset1 <- green2017test %>% select(ID) %>% collect()

predictions2017 <- matrix(0,nrow = 53967,ncol = 2)
colnames(predictions2017) <- c("ID","tip_amount")

predictions2017[,1] <- preds2017subset1$ID

predictions2017[,2] <- preds2017subset$prediction


#Green Predictions 2019

green2019train <- spark_read_csv(sc, path = "s3://stat6306studentfilebucket/Andreas and Webb/Taxi/Green/Green 2019/green_tripdata_2019-06.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
green2019test <- spark_read_csv(sc, path = "s3://stat6306project/Green Test Student/TaxiTest2019G_WO_df.csv",infer_schema = TRUE, header = TRUE, repartition = 100)

green2019sub <- green2019train %>% select(trip_distance,passenger_count,payment_type) %>% filter(payment_type == 1 | payment_type == 2)
green2019subtest <- green2019test %>% select(trip_distance,passenger_count)

fitgreen2019 <- green2019sub %>% ml_logistic_regression(payment_type~trip_distance+passenger_count)
tidy(fitgreen2019)

preds2019 <- ml_predict(fitgreen2019,green2019subtest)

preds2019subset <- preds2019 %>% select(prediction) %>% collect()
preds2019subset1 <- green2019test %>% select(ID) %>% collect()

predictions2019 <- matrix(0,nrow = 23395,ncol = 2)
colnames(predictions2019) <- c("ID","tip_amount")

predictions2019[,1] <- preds2019subset1$ID

predictions2019[,2] <- preds2019subset$prediction


#Green Predictions 2020

green2020train <- spark_read_csv(sc, path = "s3://stat6306studentfilebucket/Andreas and Webb/Taxi/Green/Green 2020/green_tripdata_2020-05.csv",infer_schema = TRUE, header = TRUE, repartition = 100)
green2020test <- spark_read_csv(sc, path = "s3://stat6306project/Green Test Student/TaxiTest2020G_WO_df.csv",infer_schema = TRUE, header = TRUE, repartition = 100)

green2020sub <- green2020train %>% select(trip_distance,passenger_count,payment_type) %>% filter(payment_type == 1 | payment_type == 2)
green2020subtest <- green2020test %>% select(trip_distance,passenger_count)

fitgreen2020 <- green2020sub %>% ml_logistic_regression(payment_type~trip_distance+passenger_count)
tidy(fitgreen2020)

predsgreen2020 <- ml_predict(fitgreen2020,green2020subtest)

predsgreen2020subset <- predsgreen2020 %>% select(prediction) %>% collect()
predsgreen2020subset1 <- green2020test %>% select(ID) %>% collect()

predictionsgreen2020 <- matrix(0,nrow = 15540,ncol = 2)
colnames(predictionsgreen2020) <- c("ID","tip_amount")

predictionsgreen2020[,1] <- predsgreen2020subset1$ID

predictionsgreen2020[,2] <- predsgreen2020subset$prediction


#Put everything together

greenpredictfinal <- rbind(predictions2015,predictions2017,predictions2019,predictionsgreen2020)

Sys.setenv("AWS_ACCESS_KEY_ID" = "AKIAJFPIUJBDHKKBR6SA",
           "AWS_SECRET_ACCESS_KEY" = "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",
           "AWS_DEFAULT_REGION" = "us-east-2")


aws.s3::bucketlist()
aws.s3::get_bucket("ds6306unit13")

s3write_using(greenpredictfinal, FUN = write.csv,bucket = "stat6306studentfilebucket",object = "Group10GreenTaxiFinalPredictions")



