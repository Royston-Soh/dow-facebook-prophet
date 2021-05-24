# Time series forecasting For Dow Jones Industrial Average (using Facebook Prophet)

```bash
library(xts)
library(quantmod)
library(prophet)
library(ggplot2)
library(forecast)
```

## Get historical data for Dow Jones Industrial Average (adjusted closing prices)

```bash
getSymbols('DJI',
           from='2008-01-01')

head(DJI)
tail(DJI)
```

### Here’s the code for saving as csv or downloading all the data available
```bash
#Save data as csv
#setwd("F:/My Files/R Studio/DOW S&P500")
#write.csv(dow_data,'dow_data.csv')

#dow_data=get_data('DJI',src = 'yahoo') under qrmtools package for adjusted closing prices
#if we want to load all available data
```

## Plot the chart series
We observe an upward multiplicative (exponential) trend. As prophet library makes time series forecast based on additive regression model, we need to do a log transformation to linearize the data using log transformation.

```bash
chartSeries(DJI,
            theme=chartTheme('white'))
```
         
## Data Preparation: Remove missing values and assign variables ds and y for prophet model
```bash
dow_data=data.frame(DJI)
dow_data$ds=index(DJI)

data=na.omit(dow_data)

ds=data$ds
y=data$DJI.Adjusted
df=data.frame(ds,y)
```

## Log transformation
```bash
y=log(df$y)
df=data.frame(ds,y)
View(df)
```

## Visualization of log transformed data
```bash
qplot(ds,y,data=df,
      main='Dow Jones Industrial Average in log scale')
```

## Split the data to training and test set
Let's predict and validate for the last 252 trading days
```bash
training_length=length(df$y)-252
test_start=training_length+1

df_training=df[1:training_length,]
df_test=df[test_start:length(df$y),]
```

## Specify the forecasting model using training data
```bash
m=prophet()
m=fit.prophet(m,df_training)
```

## Make future prediction for next 252 trading days
```bash
future=make_future_dataframe(m,periods=252)
forecast=predict(m,future)
```

#Forecast components
We observe a long term upward trend, as well as seasonality where the index peaks in the months of January and bottoms out in the months of March and July.
```bash
prophet_plot_components(m,forecast)
```

## Model Performance
We observe an overall linear upward trend, with R-squared at 0.9833. Up to 99.73% of the variation in predicted values can be explained by the variation in actual values in the prediction model. However, this measures the goodness-of-fit and does not provide information on the accuracy of the model.

```bash
pred=forecast$yhat
actuals=df$y
plot(actuals,pred)
abline(lm(pred~actuals),col='red',lwd=2)
summary(lm(pred~actuals))
```

## Residuals Plot
Create dataframe for residuals
```bash
residuals_m=df$y-forecast$yhat
df_residuals=data.frame(df$ds,residuals_m)
colnames(df_residuals)=c('ds','residuals')
```

## Plot for forecast and residuals
```bash
plot(m,forecast)


qplot(ds,residuals,data=df_residuals,
      main='Plot of residuals in log scale')+
  geom_vline(xintercept = (max(ds)-252), 
             color = "blue")
```

## Model accuracy
We notice that the accuracy metrics of the model deteriorates when we compare the validation set against the training set, suggestive of overfitting. This could be due to the highly volatile nature of the stock index, where the variance of the residuals are not constant, rendering the model which uses the least squares method less accurate.
```bash
#Performance metrics for training set
predicted_train=forecast$yhat[1:(length(forecast$yhat)-252)]
actuals_train=df_training$y
round(accuracy(predicted_train,actuals_train),2)

#Performance metrics for test/validation set
### predicted_v=forecast$yhat[(length(forecast$yhat)-252+1):length(forecast$yhat)]
actuals_v=df_test$y
round(accuracy(predicted_v,actuals_v),2)
```                            

## Let's rescale back our data and do a plot
```bash
df$dowjones=exp(df$y)
df$predicted=exp(forecast$yhat)
df_residuals$Residuals=df$dowjones-df$predicted

ggplot(df, aes(ds)) + 
  geom_line(aes(y = dowjones, color='dow')) + 
  geom_line(aes(y = predicted, color='predicted'))+
  geom_vline(xintercept = (max(ds)-252),
             color='blue')

qplot(ds,Residuals,data=df_residuals,
      main='Plot of residuals')+
  geom_vline(xintercept = (max(ds)-252), 
             color = "blue")
```

