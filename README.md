# keras-to-serving
Simple tool to export serialized Keras models to a Tensorflow Servable

Requirements:
  1. Keras model serialized to h5 with keras.models.save() [requires H5py]
  2. Tensorflow 1.1.0
  
To use:
  1. ``` python3 export.py --model_path=/some/path --export_path=/another/path --export_version=some_integer ```
