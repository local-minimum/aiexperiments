# About

Code inspired by http://rubikscode.net/2018/02/19/artificial-neural-networks-series/
Mostly just documenting my experimenting and learning some AI

# Install

Get python 3.6 or 3.7 and pip to match.
Run:

    pip install -r requirements.txt

# Predictive Text

Couldn't really follow the instructions in the tutorial, I feel like this
model is a bit too prone to over-fitting

In python3 train:
```python
import predictive_text
result = predictive_text.train('myfile.txt')
```
Then try it out with:
```python
print(result.speak())
```
You can save:
```python
result.save('myfile.h5')
````
and load:
```python
result = predictive_text.PredictiveText.load('myfile.h5')
```

# Generate images based on a training set.
```python
import dcgan
X = dcgan.images_to_array('path/to/training/images*.jpg', resize=(64, 64), gray=False)
image_helper = dcgan.ImageHelper('output/images/path', 8, 8, X[0].shape)
gan = dcgan.DCGAN(X[0].shape, image_helper, generator_input_dim=100)
gan.train(20000, X, batch_size=32)
```    
This will train for 20k epochs, it will save a collage of 8 by 8 images every 100 epochs into the path specified in the example.
If all training images already have the same size and the size you want, just omit the `resize` argument.

# Getting an image set from duck duck go

```python
from pathlib import Path
from duckduck.api import Search
from duckduck.callbacks import ImageSaver
with ImageSaver(Path('./image/folder'), overwrite=True) as callback:
        search('search phrase', callback=callback)
```

If you want some quite naive / marginally smart scaling and cropping to get all images of equal size

```python
from imagetools.extract import extract
extract('./image/folder/img*', target=(64, 64), outprefix='cropped')
```
