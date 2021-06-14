# Time series forecasting for Dow Jones Industrial Average (using Facebook Prophet)
In this post, we try to examine the usefulness of Facebook Prophet in forcasting stock market indices.
```bash
library(xts)
library(quantmod)
library(prophet)
library(ggplot2)
library(forecast)
library(lubridate) 
```

## Read historical data for Dow Jones Industrial Average (adjusted closing prices)
```bash
setwd("F:/My Files/R Studio/DOW S&P500")
dow_data=read.csv('dow_data_v3.csv',header = T,sep = ',')
head(dow_data)
tail(dow_data) 
```
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/1%20head_tail.jpg)

## Standardize the dates and prepare data for conversion to time series xts format, using from year 2008 onward
```bash
dow_data$Date=dmy(dow_data$X) #Standardize the dates using lubridate package

df=data.frame(dow_data$DJI) #Prepare time series format
rownames(df)=dow_data$Date
colnames(df)=c('DJI')

df_xts=as.xts(df) #Convert to xts format
df_xts=df_xts['2008/'] #extract data frm 2008 onward
```
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/2%20time%20series.jpg)

## Plot the chart series
We observe an upward multiplicative (exponential) trend. As prophet library makes time series forecast based on additive regression trend model, we need to do a log transformation to linearize the data using log transformation.
```bash
chartSeries(df_xts,
            theme=chartTheme('white')) 
```
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/3%20dow%20plot.jpg)
         
## Prepare dataframe with variables ds and y assigned for prophet model, perform log transformation
```bash
df=data.frame(df_xts)
df$ds=rownames(df)
ds=df$ds
y=df$DJI
df=data.frame(ds,y)
df$ds=as.Date(df$ds)

y=log(df$y)
df=data.frame(ds,y)
df$ds=as.Date(df$ds)

qplot(ds,y,data=df,
      main='Dow Jones Industrial Average in log scale')
```
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/4%20plot%20log%20scale.jpg)

## Split the data to training and test set
Let's predict and validate for the last 252 trading days

```bash
training_length=length(df$y)-252
test_start=training_length+1

df_training=df[1:training_length,]
df_test=df[test_start:length(df$y),] 
```

## Specify the forecasting model, and make furture prediction using training data
```bash
m=prophet()
m=fit.prophet(m,df_training) 

future=make_future_dataframe(m,periods=252)
forecast=predict(m,future)
```

## Forecast Components
We observe a long term upward trend, as well as seasonality, where the index peaks in the months of May and August and bottoms out in the months of March and November. There is no weekly pattern being observed in the above plots.
```bash
prophet_plot_components(m,forecast) 
```
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/5%20plot%202%20components.jpg)

## Model Performance
We observe an overall linear upward trend, with R-squared at 0.9825. Up to 98.25% of the variation in predicted values can be explained by the variation in actual values in the prediction model. However, this measures the goodness-of-fit and does not provide information on the accuracy of the model.

```bash
pred=forecast$yhat
actuals=df$y
plot(actuals,pred) 
abline(lm(pred~actuals),col='red',lwd=2)
summary(lm(pred~actuals))
```
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/6%20Least%20sq%20plot.jpg)
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/7%20R%20squared_.jpg)

## Plot for forecast and residuals
```bash
#Create dataframe for residuals
residuals_m=df$y-forecast$yhat
df_residuals=data.frame(df$ds,residuals_m)
colnames(df_residuals)=c('ds','residuals') 

#Forecast plot
plot(m,forecast)

#Residual plot
qplot(ds,residuals,data=df_residuals,
      main='Plot of residuals in log scale')+
  geom_vline(xintercept = as.numeric(ymd(df_test[1,1])), 
             color = "blue") 
```
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/8%20plot%20predictions.jpg)
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/9%20plot%20residuals.jpg)

## Model accuracy
We notice that the accuracy metrics of the model deteriorates when we compare the validation set against the training set, suggestive of overfitting. This could be due to the highly volatile nature of the stock index, where the variance of the residuals are not constant, rendering the model which uses the least squares method less accurate. Facebook Prophet is not that useful for predicting stock index in times of high volatility, also, the accuracy for forecasts deteriorates as we project further into the future (medium to long-term).
```bash
#Performance metrics for training set
predicted_train=forecast$yhat[1:(length(forecast$yhat)-252)]
actuals_train=df_training$y
round(accuracy(predicted_train,actuals_train),2)

#Performance metrics for test/validation set
predicted_v=forecast$yhat[(length(forecast$yhat)-252+1):length(forecast$yhat)]
actuals_v=df_test$y
round(accuracy(predicted_v,actuals_v),2) 
```
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/10%20accuracy_training.jpg)
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/11%20Accuracy_test.jpg)

## Accuracy of model in actual scale (test data)
```bash
round(accuracy(exp(predicted_v),exp(actuals_v),2))
```
![](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/12%20Accuracy_test_actual%20scale.jpg)

## Let's rescale back our data and do a plot
```bash
df$dowjones=exp(df$y)
df$predicted=exp(forecast$yhat)

df_residuals$Residuals=df$dowjones-df$predicted

ggplot(df, aes(ds)) + 
  geom_line(aes(y = dowjones, color='dow')) + 
  geom_line(aes(y = predicted, color='predicted'))+
  geom_vline(xintercept = as.numeric(ymd(df_test[1,1])), 
             color = "blue")

qplot(ds,Residuals,data=df_residuals,
      main='Plot of residuals')+
  geom_vline(xintercept = as.numeric(ymd(df_test[1,1])), 
             color = "blue")
```
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/13%20plot%20actual%20scale.jpg)
![Facebook Prophet](https://github.com/Royston-Soh/dow-facebook-prophet/blob/main/pic/14%20plot%20actual%20residuals.jpg)



