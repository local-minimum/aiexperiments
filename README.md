# About

Code inspired by http://rubikscode.net/2018/02/19/artificial-neural-networks-series/
Mostly just documenting my experimenting and learning some AI

# Install

Get python 3.6 or 3.7 and pip to match.
Run:

    pip install -r requirements.txt

# Predictive Text

In python3 train:

    import predictive_text
    result = predictive_text.train('myfile.txt')

Then try it out with:

    print(result.speak())

You can save:

    result.save('myfile.h5')

and load:

    result = predictive_text.PredictiveText.load('myfile.h5')
