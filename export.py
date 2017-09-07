import argparse
import os

import keras.backend as K
from keras.models import load_model

def load_inference_model(model_path):
	K.set_learning_phase(0)
	return load_model(model_path)

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model_path', type=str, help='path to saved Keras model')
	parser.add_argument('--export_path', type=str, default='./servables/tf_serving_model', help='path to Tensorflow servable')
	parser.add_argument('--export_version', type=int, help='model version (integer)')
	args = parser.parse_args()
	model = load_inference_model(args.model_path)
	export(model, args)

def export(model, args):
	from tensorflow.python.saved_model import builder as saved_model_builder
	from tensorflow.python.saved_model import utils
	from tensorflow.python.saved_model import tag_constants, signature_constants
	from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
	from tensorflow.contrib.session_bundle import exporter

	builder = saved_model_builder.SavedModelBuilder(args.export_path+'/'+args.export_version)
	signature = predict_signature_def(inputs={'inputs': model.input}, outputs={'ouputs': model.output})
	with K.get_session() as sess:
		builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING], signature_def_map={'predict': signature})
		builder.save()

if __name__ == '__main__':
    main()