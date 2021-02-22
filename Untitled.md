```R
library(caret)
library(dplyr)
library(xgboost)
library(ggplot2)
```


```R
data_dir=c("C:/Users/sambe/Downloads")
training_file = "pml-training.csv"
test_file = "pml-testing.csv"
```


```R
train <- read.csv(file.path(data_dir, training_file))
test <- read.csv(file.path(data_dir, test_file))
```


```R
head(train)
```


<table>
<thead><tr><th scope=col>X</th><th scope=col>user_name</th><th scope=col>raw_timestamp_part_1</th><th scope=col>raw_timestamp_part_2</th><th scope=col>cvtd_timestamp</th><th scope=col>new_window</th><th scope=col>num_window</th><th scope=col>roll_belt</th><th scope=col>pitch_belt</th><th scope=col>yaw_belt</th><th scope=col>...</th><th scope=col>gyros_forearm_x</th><th scope=col>gyros_forearm_y</th><th scope=col>gyros_forearm_z</th><th scope=col>accel_forearm_x</th><th scope=col>accel_forearm_y</th><th scope=col>accel_forearm_z</th><th scope=col>magnet_forearm_x</th><th scope=col>magnet_forearm_y</th><th scope=col>magnet_forearm_z</th><th scope=col>classe</th></tr></thead>
<tbody>
	<tr><td>1               </td><td>carlitos        </td><td>1323084231      </td><td>788290          </td><td>05/12/2011 11:23</td><td>no              </td><td>11              </td><td>1.41            </td><td>8.07            </td><td>-94.4           </td><td>...             </td><td>0.03            </td><td> 0.00           </td><td>-0.02           </td><td>192             </td><td>203             </td><td>-215            </td><td>-17             </td><td>654             </td><td>476             </td><td>A               </td></tr>
	<tr><td>2               </td><td>carlitos        </td><td>1323084231      </td><td>808298          </td><td>05/12/2011 11:23</td><td>no              </td><td>11              </td><td>1.41            </td><td>8.07            </td><td>-94.4           </td><td>...             </td><td>0.02            </td><td> 0.00           </td><td>-0.02           </td><td>192             </td><td>203             </td><td>-216            </td><td>-18             </td><td>661             </td><td>473             </td><td>A               </td></tr>
	<tr><td>3               </td><td>carlitos        </td><td>1323084231      </td><td>820366          </td><td>05/12/2011 11:23</td><td>no              </td><td>11              </td><td>1.42            </td><td>8.07            </td><td>-94.4           </td><td>...             </td><td>0.03            </td><td>-0.02           </td><td> 0.00           </td><td>196             </td><td>204             </td><td>-213            </td><td>-18             </td><td>658             </td><td>469             </td><td>A               </td></tr>
	<tr><td>4               </td><td>carlitos        </td><td>1323084232      </td><td>120339          </td><td>05/12/2011 11:23</td><td>no              </td><td>12              </td><td>1.48            </td><td>8.05            </td><td>-94.4           </td><td>...             </td><td>0.02            </td><td>-0.02           </td><td> 0.00           </td><td>189             </td><td>206             </td><td>-214            </td><td>-16             </td><td>658             </td><td>469             </td><td>A               </td></tr>
	<tr><td>5               </td><td>carlitos        </td><td>1323084232      </td><td>196328          </td><td>05/12/2011 11:23</td><td>no              </td><td>12              </td><td>1.48            </td><td>8.07            </td><td>-94.4           </td><td>...             </td><td>0.02            </td><td> 0.00           </td><td>-0.02           </td><td>189             </td><td>206             </td><td>-214            </td><td>-17             </td><td>655             </td><td>473             </td><td>A               </td></tr>
	<tr><td>6               </td><td>carlitos        </td><td>1323084232      </td><td>304277          </td><td>05/12/2011 11:23</td><td>no              </td><td>12              </td><td>1.45            </td><td>8.06            </td><td>-94.4           </td><td>...             </td><td>0.02            </td><td>-0.02           </td><td>-0.03           </td><td>193             </td><td>203             </td><td>-215            </td><td> -9             </td><td>660             </td><td>478             </td><td>A               </td></tr>
</tbody>
</table>




```R
dim(train)
```


<ol class=list-inline>
	<li>19622</li>
	<li>160</li>
</ol>




```R
## Percentage of missing data
colMeans(is.na(train))
```


<dl class=dl-horizontal>
	<dt>X</dt>
		<dd>0</dd>
	<dt>user_name</dt>
		<dd>0</dd>
	<dt>raw_timestamp_part_1</dt>
		<dd>0</dd>
	<dt>raw_timestamp_part_2</dt>
		<dd>0</dd>
	<dt>cvtd_timestamp</dt>
		<dd>0</dd>
	<dt>new_window</dt>
		<dd>0</dd>
	<dt>num_window</dt>
		<dd>0</dd>
	<dt>roll_belt</dt>
		<dd>0</dd>
	<dt>pitch_belt</dt>
		<dd>0</dd>
	<dt>yaw_belt</dt>
		<dd>0</dd>
	<dt>total_accel_belt</dt>
		<dd>0</dd>
	<dt>kurtosis_roll_belt</dt>
		<dd>0</dd>
	<dt>kurtosis_picth_belt</dt>
		<dd>0</dd>
	<dt>kurtosis_yaw_belt</dt>
		<dd>0</dd>
	<dt>skewness_roll_belt</dt>
		<dd>0</dd>
	<dt>skewness_roll_belt.1</dt>
		<dd>0</dd>
	<dt>skewness_yaw_belt</dt>
		<dd>0</dd>
	<dt>max_roll_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>max_picth_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>max_yaw_belt</dt>
		<dd>0</dd>
	<dt>min_roll_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>min_pitch_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>min_yaw_belt</dt>
		<dd>0</dd>
	<dt>amplitude_roll_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>amplitude_pitch_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>amplitude_yaw_belt</dt>
		<dd>0</dd>
	<dt>var_total_accel_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_roll_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_roll_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>var_roll_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_pitch_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_pitch_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>var_pitch_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_yaw_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_yaw_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>var_yaw_belt</dt>
		<dd>0.979308938946081</dd>
	<dt>gyros_belt_x</dt>
		<dd>0</dd>
	<dt>gyros_belt_y</dt>
		<dd>0</dd>
	<dt>gyros_belt_z</dt>
		<dd>0</dd>
	<dt>accel_belt_x</dt>
		<dd>0</dd>
	<dt>accel_belt_y</dt>
		<dd>0</dd>
	<dt>accel_belt_z</dt>
		<dd>0</dd>
	<dt>magnet_belt_x</dt>
		<dd>0</dd>
	<dt>magnet_belt_y</dt>
		<dd>0</dd>
	<dt>magnet_belt_z</dt>
		<dd>0</dd>
	<dt>roll_arm</dt>
		<dd>0</dd>
	<dt>pitch_arm</dt>
		<dd>0</dd>
	<dt>yaw_arm</dt>
		<dd>0</dd>
	<dt>total_accel_arm</dt>
		<dd>0</dd>
	<dt>var_accel_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_roll_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_roll_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>var_roll_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_pitch_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_pitch_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>var_pitch_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_yaw_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_yaw_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>var_yaw_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>gyros_arm_x</dt>
		<dd>0</dd>
	<dt>gyros_arm_y</dt>
		<dd>0</dd>
	<dt>gyros_arm_z</dt>
		<dd>0</dd>
	<dt>accel_arm_x</dt>
		<dd>0</dd>
	<dt>accel_arm_y</dt>
		<dd>0</dd>
	<dt>accel_arm_z</dt>
		<dd>0</dd>
	<dt>magnet_arm_x</dt>
		<dd>0</dd>
	<dt>magnet_arm_y</dt>
		<dd>0</dd>
	<dt>magnet_arm_z</dt>
		<dd>0</dd>
	<dt>kurtosis_roll_arm</dt>
		<dd>0</dd>
	<dt>kurtosis_picth_arm</dt>
		<dd>0</dd>
	<dt>kurtosis_yaw_arm</dt>
		<dd>0</dd>
	<dt>skewness_roll_arm</dt>
		<dd>0</dd>
	<dt>skewness_pitch_arm</dt>
		<dd>0</dd>
	<dt>skewness_yaw_arm</dt>
		<dd>0</dd>
	<dt>max_roll_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>max_picth_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>max_yaw_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>min_roll_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>min_pitch_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>min_yaw_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>amplitude_roll_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>amplitude_pitch_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>amplitude_yaw_arm</dt>
		<dd>0.979308938946081</dd>
	<dt>roll_dumbbell</dt>
		<dd>0</dd>
	<dt>pitch_dumbbell</dt>
		<dd>0</dd>
	<dt>yaw_dumbbell</dt>
		<dd>0</dd>
	<dt>kurtosis_roll_dumbbell</dt>
		<dd>0</dd>
	<dt>kurtosis_picth_dumbbell</dt>
		<dd>0</dd>
	<dt>kurtosis_yaw_dumbbell</dt>
		<dd>0</dd>
	<dt>skewness_roll_dumbbell</dt>
		<dd>0</dd>
	<dt>skewness_pitch_dumbbell</dt>
		<dd>0</dd>
	<dt>skewness_yaw_dumbbell</dt>
		<dd>0</dd>
	<dt>max_roll_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>max_picth_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>max_yaw_dumbbell</dt>
		<dd>0</dd>
	<dt>min_roll_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>min_pitch_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>min_yaw_dumbbell</dt>
		<dd>0</dd>
	<dt>amplitude_roll_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>amplitude_pitch_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>amplitude_yaw_dumbbell</dt>
		<dd>0</dd>
	<dt>total_accel_dumbbell</dt>
		<dd>0</dd>
	<dt>var_accel_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_roll_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_roll_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>var_roll_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_pitch_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_pitch_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>var_pitch_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_yaw_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_yaw_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>var_yaw_dumbbell</dt>
		<dd>0.979308938946081</dd>
	<dt>gyros_dumbbell_x</dt>
		<dd>0</dd>
	<dt>gyros_dumbbell_y</dt>
		<dd>0</dd>
	<dt>gyros_dumbbell_z</dt>
		<dd>0</dd>
	<dt>accel_dumbbell_x</dt>
		<dd>0</dd>
	<dt>accel_dumbbell_y</dt>
		<dd>0</dd>
	<dt>accel_dumbbell_z</dt>
		<dd>0</dd>
	<dt>magnet_dumbbell_x</dt>
		<dd>0</dd>
	<dt>magnet_dumbbell_y</dt>
		<dd>0</dd>
	<dt>magnet_dumbbell_z</dt>
		<dd>0</dd>
	<dt>roll_forearm</dt>
		<dd>0</dd>
	<dt>pitch_forearm</dt>
		<dd>0</dd>
	<dt>yaw_forearm</dt>
		<dd>0</dd>
	<dt>kurtosis_roll_forearm</dt>
		<dd>0</dd>
	<dt>kurtosis_picth_forearm</dt>
		<dd>0</dd>
	<dt>kurtosis_yaw_forearm</dt>
		<dd>0</dd>
	<dt>skewness_roll_forearm</dt>
		<dd>0</dd>
	<dt>skewness_pitch_forearm</dt>
		<dd>0</dd>
	<dt>skewness_yaw_forearm</dt>
		<dd>0</dd>
	<dt>max_roll_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>max_picth_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>max_yaw_forearm</dt>
		<dd>0</dd>
	<dt>min_roll_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>min_pitch_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>min_yaw_forearm</dt>
		<dd>0</dd>
	<dt>amplitude_roll_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>amplitude_pitch_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>amplitude_yaw_forearm</dt>
		<dd>0</dd>
	<dt>total_accel_forearm</dt>
		<dd>0</dd>
	<dt>var_accel_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_roll_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_roll_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>var_roll_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_pitch_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_pitch_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>var_pitch_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>avg_yaw_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>stddev_yaw_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>var_yaw_forearm</dt>
		<dd>0.979308938946081</dd>
	<dt>gyros_forearm_x</dt>
		<dd>0</dd>
	<dt>gyros_forearm_y</dt>
		<dd>0</dd>
	<dt>gyros_forearm_z</dt>
		<dd>0</dd>
	<dt>accel_forearm_x</dt>
		<dd>0</dd>
	<dt>accel_forearm_y</dt>
		<dd>0</dd>
	<dt>accel_forearm_z</dt>
		<dd>0</dd>
	<dt>magnet_forearm_x</dt>
		<dd>0</dd>
	<dt>magnet_forearm_y</dt>
		<dd>0</dd>
	<dt>magnet_forearm_z</dt>
		<dd>0</dd>
	<dt>classe</dt>
		<dd>0</dd>
</dl>




```R
trainClasse = train$classe
trainRaw = train[, sapply(train, is.numeric)]
testRaw = test[, sapply(test, is.numeric)]
```


```R
# Remove columns with NA value
trainFilter <- trainRaw[, colSums(is.na(trainRaw)) == 0]
# Attach Classe variable
trainFilter$classe = trainClasse
testFilter <- testRaw[, colSums(is.na(testRaw)) == 0]
```


```R
dim(trainFilter)
dim(testFilter)
```


<ol class=list-inline>
	<li>19622</li>
	<li>57</li>
</ol>




<ol class=list-inline>
	<li>20</li>
	<li>57</li>
</ol>




```R
## remove unwanted columns
unwanted = !grepl("X|timestamp", colnames(trainFilter))
cols = colnames(trainFilter)[unwanted]
trainFilter = trainFilter %>%
select(cols)
```


```R
unwanted = !grepl("X|timestamp", colnames(testFilter))
cols = colnames(testFilter)[unwanted]
testFilter = testFilter %>%
select(cols)
```


```R
dim(trainFilter)
dim(testFilter)
```


<ol class=list-inline>
	<li>19622</li>
	<li>54</li>
</ol>




<ol class=list-inline>
	<li>20</li>
	<li>54</li>
</ol>




```R
## create data partition
set.seed(120)
inTrain <- createDataPartition(trainFilter$classe, p=0.70, list=F)
trainData <- trainFilter[inTrain, ]
validationData <- trainFilter[-inTrain, ]
dim(trainData)
```


<ol class=list-inline>
	<li>13737</li>
	<li>54</li>
</ol>




```R
## use random forest
controlRf <- trainControl(method="cv", 5, allowParallel = TRUE)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf
```


    Random Forest 
    
    13737 samples
       53 predictor
        5 classes: 'A', 'B', 'C', 'D', 'E' 
    
    No pre-processing
    Resampling: Cross-Validated (5 fold) 
    Summary of sample sizes: 10990, 10989, 10990, 10989, 10990 
    Resampling results across tuning parameters:
    
      mtry  Accuracy   Kappa    
       2    0.9929388  0.9910672
      27    0.9967241  0.9958561
      53    0.9937396  0.9920800
    
    Accuracy was used to select the optimal model using the largest value.
    The final value used for the model was mtry = 27.



```R
predict_rf <- predict(modelRf, validationData)
confusionMatrix(validationData$classe, predict_rf)
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1673    0    0    0    1
             B    5 1132    2    0    0
             C    0    1 1025    0    0
             D    0    0    4  959    1
             E    0    0    0    2 1080
    
    Overall Statistics
                                              
                   Accuracy : 0.9973          
                     95% CI : (0.9956, 0.9984)
        No Information Rate : 0.2851          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.9966          
                                              
     Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            0.9970   0.9991   0.9942   0.9979   0.9982
    Specificity            0.9998   0.9985   0.9998   0.9990   0.9996
    Pos Pred Value         0.9994   0.9939   0.9990   0.9948   0.9982
    Neg Pred Value         0.9988   0.9998   0.9988   0.9996   0.9996
    Prevalence             0.2851   0.1925   0.1752   0.1633   0.1839
    Detection Rate         0.2843   0.1924   0.1742   0.1630   0.1835
    Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    Balanced Accuracy      0.9984   0.9988   0.9970   0.9985   0.9989



```R
## use xgboost
controlXGB <- trainControl(method="cv", 5, allowParallel = TRUE)
modelXGB <- train(classe ~ ., data=trainData, method="xgbTree", trControl=controlXGB)
```


```R
modelXGB
```


    eXtreme Gradient Boosting 
    
    13737 samples
       53 predictor
        5 classes: 'A', 'B', 'C', 'D', 'E' 
    
    No pre-processing
    Resampling: Cross-Validated (5 fold) 
    Summary of sample sizes: 10989, 10990, 10991, 10989, 10989 
    Resampling results across tuning parameters:
    
      eta  max_depth  colsample_bytree  subsample  nrounds  Accuracy   Kappa    
      0.3  1          0.6               0.50        50      0.8084013  0.7571550
      0.3  1          0.6               0.50       100      0.8842558  0.8534827
      0.3  1          0.6               0.50       150      0.9144661  0.8917263
      0.3  1          0.6               0.75        50      0.8112409  0.7606834
      0.3  1          0.6               0.75       100      0.8842560  0.8534922
      0.3  1          0.6               0.75       150      0.9131558  0.8900976
      0.3  1          0.6               1.00        50      0.8096393  0.7586782
      0.3  1          0.6               1.00       100      0.8822178  0.8509430
      0.3  1          0.6               1.00       150      0.9142479  0.8914903
      0.3  1          0.8               0.50        50      0.8121861  0.7618746
      0.3  1          0.8               0.50       100      0.8841822  0.8533850
      0.3  1          0.8               0.50       150      0.9173051  0.8953270
      0.3  1          0.8               0.75        50      0.8109489  0.7603557
      0.3  1          0.8               0.75       100      0.8849838  0.8544310
      0.3  1          0.8               0.75       150      0.9133015  0.8902819
      0.3  1          0.8               1.00        50      0.8137884  0.7639473
      0.3  1          0.8               1.00       100      0.8819264  0.8505908
      0.3  1          0.8               1.00       150      0.9146122  0.8919532
      0.3  2          0.6               0.50        50      0.9503541  0.9371878
      0.3  2          0.6               0.50       100      0.9843496  0.9802043
      0.3  2          0.6               0.50       150      0.9945403  0.9930938
      0.3  2          0.6               0.75        50      0.9529749  0.9405175
      0.3  2          0.6               0.75       100      0.9877705  0.9845298
      0.3  2          0.6               0.75       150      0.9954866  0.9942910
      0.3  2          0.6               1.00        50      0.9534105  0.9410670
      0.3  2          0.6               1.00       100      0.9879162  0.9847151
      0.3  2          0.6               1.00       150      0.9951229  0.9938308
      0.3  2          0.8               0.50        50      0.9540662  0.9418920
      0.3  2          0.8               0.50       100      0.9852957  0.9814009
      0.3  2          0.8               0.50       150      0.9946132  0.9931859
      0.3  2          0.8               0.75        50      0.9558135  0.9441084
      0.3  2          0.8               0.75       100      0.9877705  0.9845307
      0.3  2          0.8               0.75       150      0.9959963  0.9949356
      0.3  2          0.8               1.00        50      0.9547216  0.9427300
      0.3  2          0.8               1.00       100      0.9887169  0.9857274
      0.3  2          0.8               1.00       150      0.9954141  0.9941991
      0.3  3          0.6               0.50        50      0.9879888  0.9848070
      0.3  3          0.6               0.50       100      0.9977433  0.9971455
      0.3  3          0.6               0.50       150      0.9983985  0.9979743
      0.3  3          0.6               0.75        50      0.9890084  0.9860967
      0.3  3          0.6               0.75       100      0.9978889  0.9973297
      0.3  3          0.6               0.75       150      0.9986170  0.9982507
      0.3  3          0.6               1.00        50      0.9900273  0.9873854
      0.3  3          0.6               1.00       100      0.9978161  0.9972375
      0.3  3          0.6               1.00       150      0.9990537  0.9988031
      0.3  3          0.8               0.50        50      0.9895172  0.9867396
      0.3  3          0.8               0.50       100      0.9977433  0.9971456
      0.3  3          0.8               0.50       150      0.9984713  0.9980664
      0.3  3          0.8               0.75        50      0.9898814  0.9872009
      0.3  3          0.8               0.75       100      0.9980346  0.9975140
      0.3  3          0.8               0.75       150      0.9988354  0.9985268
      0.3  3          0.8               1.00        50      0.9908279  0.9883975
      0.3  3          0.8               1.00       100      0.9982529  0.9977902
      0.3  3          0.8               1.00       150      0.9986897  0.9983426
      0.4  1          0.6               0.50        50      0.8439997  0.8024649
      0.4  1          0.6               0.50       100      0.9073327  0.8827265
      0.4  1          0.6               0.50       150      0.9330284  0.9152406
      0.4  1          0.6               0.75        50      0.8446544  0.8032675
      0.4  1          0.6               0.75       100      0.9048568  0.8795955
      0.4  1          0.6               0.75       150      0.9323731  0.9144212
      0.4  1          0.6               1.00        50      0.8442914  0.8028655
      0.4  1          0.6               1.00       100      0.9052214  0.8800670
      0.4  1          0.6               1.00       150      0.9314999  0.9133330
      0.4  1          0.8               0.50        50      0.8480763  0.8075692
      0.4  1          0.8               0.50       100      0.9076235  0.8830842
      0.4  1          0.8               0.50       150      0.9363770  0.9194803
      0.4  1          0.8               0.75        50      0.8474215  0.8067541
      0.4  1          0.8               0.75       100      0.9071872  0.8825572
      0.4  1          0.8               0.75       150      0.9353582  0.9181989
      0.4  1          0.8               1.00        50      0.8447275  0.8033395
      0.4  1          0.8               1.00       100      0.9065320  0.8817422
      0.4  1          0.8               1.00       150      0.9321549  0.9141624
      0.4  2          0.6               0.50        50      0.9689169  0.9606739
      0.4  2          0.6               0.50       100      0.9917741  0.9895953
      0.4  2          0.6               0.50       150      0.9967969  0.9959485
      0.4  2          0.6               0.75        50      0.9704461  0.9626154
      0.4  2          0.6               0.75       100      0.9932300  0.9914366
      0.4  2          0.6               0.75       150      0.9975977  0.9969613
      0.4  2          0.6               1.00        50      0.9726291  0.9653734
      0.4  2          0.6               1.00       100      0.9943221  0.9928180
      0.4  2          0.6               1.00       150      0.9977433  0.9971455
      0.4  2          0.8               0.50        50      0.9702992  0.9624306
      0.4  2          0.8               0.50       100      0.9938851  0.9922650
      0.4  2          0.8               0.50       150      0.9973793  0.9966851
      0.4  2          0.8               0.75        50      0.9724104  0.9650970
      0.4  2          0.8               0.75       100      0.9934486  0.9917131
      0.4  2          0.8               0.75       150      0.9974523  0.9967774
      0.4  2          0.8               1.00        50      0.9721927  0.9648254
      0.4  2          0.8               1.00       100      0.9944677  0.9930023
      0.4  2          0.8               1.00       150      0.9981801  0.9976981
      0.4  3          0.6               0.50        50      0.9945406  0.9930941
      0.4  3          0.6               0.50       100      0.9984714  0.9980665
      0.4  3          0.6               0.50       150      0.9988353  0.9985268
      0.4  3          0.6               0.75        50      0.9946859  0.9932782
      0.4  3          0.6               0.75       100      0.9985441  0.9981584
      0.4  3          0.6               0.75       150      0.9987624  0.9984346
      0.4  3          0.6               1.00        50      0.9937396  0.9920811
      0.4  3          0.6               1.00       100      0.9986170  0.9982507
      0.4  3          0.6               1.00       150      0.9989082  0.9986190
      0.4  3          0.8               0.50        50      0.9943221  0.9928181
      0.4  3          0.8               0.50       100      0.9986897  0.9983426
      0.4  3          0.8               0.50       150      0.9986897  0.9983426
      0.4  3          0.8               0.75        50      0.9954867  0.9942911
      0.4  3          0.8               0.75       100      0.9987624  0.9984346
      0.4  3          0.8               0.75       150      0.9988353  0.9985267
      0.4  3          0.8               1.00        50      0.9958507  0.9947517
      0.4  3          0.8               1.00       100      0.9988354  0.9985269
      0.4  3          0.8               1.00       150      0.9989810  0.9987111
    
    Tuning parameter 'gamma' was held constant at a value of 0
    Tuning
     parameter 'min_child_weight' was held constant at a value of 1
    Accuracy was used to select the optimal model using the largest value.
    The final values used for the model were nrounds = 150, max_depth = 3, eta
     = 0.3, gamma = 0, colsample_bytree = 0.6, min_child_weight = 1 and subsample
     = 1.



```R
predict_XGB <- predict(modelXGB, validationData)
confusionMatrix(validationData$classe, predict_XGB)
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1674    0    0    0    0
             B    0 1139    0    0    0
             C    0    0 1026    0    0
             D    0    0    0  964    0
             E    0    0    0    0 1082
    
    Overall Statistics
                                         
                   Accuracy : 1          
                     95% CI : (0.9994, 1)
        No Information Rate : 0.2845     
        P-Value [Acc > NIR] : < 2.2e-16  
                                         
                      Kappa : 1          
                                         
     Mcnemar's Test P-Value : NA         
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Rate         0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000



```R
# collect resamples
model_results <- resamples(list(RF=modelRf, XGB=modelXGB))
# summarize the distributions
summary(model_results)
# boxplots of results
bwplot(model_results)
# dot plots of results
dotplot(model_results)
```


    
    Call:
    summary.resamples(object = model_results)
    
    Models: RF, XGB 
    Number of resamples: 5 
    
    Accuracy 
             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    RF  0.9941755 0.9967237 0.9970888 0.9967241 0.9974527 0.9981798    0
    XGB 0.9981805 0.9989083 0.9992717 0.9990537 0.9992722 0.9996360    0
    
    Kappa 
             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    RF  0.9926320 0.9958553 0.9963176 0.9958561 0.9967776 0.9976977    0
    XGB 0.9976983 0.9986192 0.9990787 0.9988031 0.9990795 0.9995396    0
    



![png](output_18_1.png)



![png](output_18_2.png)



```R
resultRf <- predict(modelRf, testFilter[, -length(names(testFilter))])
resultXGB <- predict(modelXGB, testFilter[, -length(names(testFilter))])
resultRf
resultXGB
confusionMatrix(resultRf, resultXGB)
```


<ol class=list-inline>
	<li>B</li>
	<li>A</li>
	<li>B</li>
	<li>A</li>
	<li>A</li>
	<li>E</li>
	<li>D</li>
	<li>B</li>
	<li>A</li>
	<li>A</li>
	<li>B</li>
	<li>C</li>
	<li>B</li>
	<li>A</li>
	<li>E</li>
	<li>E</li>
	<li>A</li>
	<li>B</li>
	<li>B</li>
	<li>B</li>
</ol>

<details>
	<summary style=display:list-item;cursor:pointer>
		<strong>Levels</strong>:
	</summary>
	<ol class=list-inline>
		<li>'A'</li>
		<li>'B'</li>
		<li>'C'</li>
		<li>'D'</li>
		<li>'E'</li>
	</ol>
</details>



<ol class=list-inline>
	<li>B</li>
	<li>A</li>
	<li>B</li>
	<li>A</li>
	<li>A</li>
	<li>E</li>
	<li>D</li>
	<li>B</li>
	<li>A</li>
	<li>A</li>
	<li>B</li>
	<li>C</li>
	<li>B</li>
	<li>A</li>
	<li>E</li>
	<li>E</li>
	<li>A</li>
	<li>B</li>
	<li>B</li>
	<li>B</li>
</ol>

<details>
	<summary style=display:list-item;cursor:pointer>
		<strong>Levels</strong>:
	</summary>
	<ol class=list-inline>
		<li>'A'</li>
		<li>'B'</li>
		<li>'C'</li>
		<li>'D'</li>
		<li>'E'</li>
	</ol>
</details>



    Confusion Matrix and Statistics
    
              Reference
    Prediction A B C D E
             A 7 0 0 0 0
             B 0 8 0 0 0
             C 0 0 1 0 0
             D 0 0 0 1 0
             E 0 0 0 0 3
    
    Overall Statistics
                                         
                   Accuracy : 1          
                     95% CI : (0.8316, 1)
        No Information Rate : 0.4        
        P-Value [Acc > NIR] : 1.1e-08    
                                         
                      Kappa : 1          
                                         
     Mcnemar's Test P-Value : NA         
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity              1.00      1.0     1.00     1.00     1.00
    Specificity              1.00      1.0     1.00     1.00     1.00
    Pos Pred Value           1.00      1.0     1.00     1.00     1.00
    Neg Pred Value           1.00      1.0     1.00     1.00     1.00
    Prevalence               0.35      0.4     0.05     0.05     0.15
    Detection Rate           0.35      0.4     0.05     0.05     0.15
    Detection Prevalence     0.35      0.4     0.05     0.05     0.15
    Balanced Accuracy        1.00      1.0     1.00     1.00     1.00



```R

```
