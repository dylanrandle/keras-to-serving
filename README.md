# keras-to-serving
Simple tool to export serialized Keras models to a Tensorflow Servable

Requirements:
  1. Keras model serialized to h5 with keras.models.save()
  2. Tensorflow 1.1, Keras 2.0, H5Py
  
To use:
  1. ``` python export.py --model_path=/some/path --export_path=/another/path --export_version=some_integer ``
