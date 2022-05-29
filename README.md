# Convusing
Circa 2018... My first attempt at using Keras, Channel-Wise Networks, and generally an educational experience with Python.  The idea was to use channel-wise convolution newtworks with stock data to make predictions.  Typically Channel-Wise networks are used with image data, whereby the R/G/B/A values are 'separated' into different channels.  The separation here took place with different stock indicator timeframes.  Although the code worked and the models trained, they weren't very good at predicting going forward. But, fun to play with.

The final model design was initially based on Xception but took on a life completely of it's own, producing some pretty neat multi-headed/inter-layer connections.  

![Model -1](/stocks5-5day(index=-1).h5.png)

