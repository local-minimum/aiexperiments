# Install

Get python 3.6 or 3.7 and pip to match.
Run:
    pip install -r requirements.txt

# Predictive Text

In python3:

    import predictive_text
    result = predictive_text.train('myfile.txt')
    result.speak()

You can save:

    result.save('myfile.h5')

and load:
    result = predictive_text.PredictiveText.load('myfile.h5')
