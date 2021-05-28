##Time series forecasting For Dow Jones Industrial Average (using Facebook Prophet)


library(xts)
library(quantmod)
library(prophet)
library(ggplot2)
library(forecast)
library(lubridate)


#Read historical data for Dow Jones Industrial Average (adjusted closing prices)
setwd("F:/My Files/R Studio/DOW S&P500")
dow_data=read.csv('dow_data_v3.csv',header = T,sep = ',')
View(dow_data)
str(dow_data)

#Standardize the dates and prepare the data for conversion to time series xts format
dow_data$Date=dmy(dow_data$X)

df=data.frame(dow_data$DJI)
rownames(df)=dow_data$Date
colnames(df)=c('DJI')
View(df)
str(df)

#Let's use the data from year 2008 onward
df_xts=as.xts(df)
df_xts=df_xts['2008/']
View(df_xts)



#Plot the chart series
#We observe an upward multiplicative (exponential) trend
#As prophet library makes time series forecast based on additive regression model, we need to do a log transformation to linearize the data

chartSeries(df_xts,
            theme=chartTheme('white'))


#Prepare dataframe with variables ds and y assigned for prophet model
df=data.frame(df_xts)
df$ds=rownames(df)
ds=df$ds
y=df$DJI
df=data.frame(ds,y)
df$ds=as.Date(df$ds)
View(df)
str(df)


#Log transformation
y=log(df$y)
df=data.frame(ds,y)
df$ds=as.Date(df$ds)
View(df)
str(df)



#Visualization of log transformed data
qplot(ds,y,data=df,
      main='Dow Jones Industrial Average in log scale')




#Split the data to training and test set
#Let's predict and validate for the last 252 trading days
training_length=length(df$y)-252
test_start=training_length+1

df_training=df[1:training_length,]
df_test=df[test_start:length(df$y),]

View(df_test)


#Forecasting model using training data
m=prophet()
m=fit.prophet(m,df_training)

#Prediction eg.next 252 days
future=make_future_dataframe(m,periods=252)
forecast=predict(m,future)
View(forecast)

#Forecast components
prophet_plot_components(m,forecast)


#Model Performance
pred=forecast$yhat
actuals=df$y
plot(actuals,pred)

#Overall pattern is linear, let's draw a linear trend line
abline(lm(pred~actuals),col='red',lwd=2)

#Calculate R sq of the linear line
#up to 99.73% of the variation in predicted values can be explained by the variation in actual values in the prediction model
#This does not provide information on the accuracy of the model
summary(lm(pred~actuals))


#Residuals
residuals_m=df$y-forecast$yhat
View(residuals_m)
df_residuals=data.frame(df$ds,residuals_m)
View(df_residuals)
colnames(df_residuals)=c('ds','residuals')


#Plot forecast and residuals
plot(m,forecast)


qplot(ds,residuals,data=df_residuals,
      main='Plot of residuals in log scale')+
  geom_vline(xintercept = as.numeric(ymd(df_test[1,1])), 
             color = "blue")



#Model accuracy
#Performance metrics for training set
predicted_train=forecast$yhat[1:(length(forecast$yhat)-252)]
actuals_train=df_training$y
round(accuracy(predicted_train,actuals_train),2)

#Performance metrics for test/validation set
predicted_v=forecast$yhat[(length(forecast$yhat)-252+1):length(forecast$yhat)]
actuals_v=df_test$y
round(accuracy(predicted_v,actuals_v),2)
                            
#Rescaled accutacy
round(accuracy(exp(predicted_v),exp(actuals_v),2))

#Let's rescale back our data and do a plot
#Plot forecast and residuals

df$dowjones=exp(df$y)
df$predicted=exp(forecast$yhat)

df_residuals$Residuals=df$dowjones-df$predicted

par(mfrow=c(2,1))
ggplot(df, aes(ds)) + 
  geom_line(aes(y = dowjones, color='dow')) + 
  geom_line(aes(y = predicted, color='predicted'))+
  geom_vline(xintercept = as.numeric(ymd(df_test[1,1])), 
             color = "blue")


qplot(ds,Residuals,data=df_residuals,
      main='Plot of residuals')+
  geom_vline(xintercept = as.numeric(ymd(df_test[1,1])), 
             color = "blue")

par(mfrow=c(1,1))




