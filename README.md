Hello,
This project is concentrated around an AutoEncoder net used for ECG time series data analysis.
AutoEncoder encodes the data (compresses it) using encoder, and then decodes (decompreses) it using decoder.
The main goal is to separate anomalous examples from normal ones.
To do so the net is initialy trained only on normal examples, which it learns to reconstruct (decompres).
After training when receiving anomalous ones it is not able to reconstruct with a low error, that is how we tell we came across an anomaly.
Training is performed in modelTraining.py file, you can find evaluation and visualisation of the results in the ModelEvaluation.ipynb file.
You can also find a script where I preprocess data so it is suitable for the task, just navigate to DataPreparation.ipynb.
Dataset used for the project is available here: https://www.timeseriesclassification.com/description.php?Dataset=ECG5000
I took inspiration from this article: https://www.analyticsvidhya.com/blog/2022/01/complete-guide-to-anomaly-detection-with-autoencoders-using-tensorflow/
Enjoy!
