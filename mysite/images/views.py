from django.shortcuts import render,redirect
import tensorflow as tf
import wikipedia as wiki
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.uploadedfile import SimpleUploadedFile
from .forms import UploadFileForm

def home(request):
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		print(form.is_valid())
		if form.is_valid():
			handle_uploaded_file(request.FILES['image'])
			return redirect('/classify')
	return render(request, 'images/home.html')

def handle_uploaded_file(f):	
	with open('./images/static/images/img/test.jpg','wb+') as destination:
		for chunk in f.chunks():
			destination.write(chunk)

'''
Reminder from s/o on why to use gfile   ---->

The main roles of the tf.gfile module are:
1)To provide an API that is close to Python's file objects, and
2)To provide an implementation based on TensorFlow's C++ FileSystem API.

Regular python file API can be used but use gfile if possibility of using unconventional file system such as google cloud
'''
def classify_image(request):
	context = {}
	# load image
	image_file = './images/static/images/img/test.jpg'
	image = tf.gfile.FastGFile(image_file,'rb').read()

  	# load labels in googley fashion
	labels = [line.rstrip() for line in tf.gfile.GFile('images/retrained_labels.txt')]
  	
  	# load graph
	with tf.gfile.FastGFile('images/retrained_graph.pb', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')
	
	output_layer_name = 'final_result:0'
	input_layer_name = 'DecodeJpeg/contents:0'
	
	with tf.Session() as sess:
		softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
		predictions, = sess.run(softmax_tensor, {input_layer_name: image})
		num_top_predictions = 120
		top_k = predictions.argsort()[-num_top_predictions:][::-1]
		for node_id in top_k:
	  		human_string = labels[node_id]
	  		score = predictions[node_id]

	  		print('%s (score = %.5f)' % (human_string, score))
	print(labels[top_k[0]])
	context_prediction = labels[top_k[0]]
	context_prediction = context_prediction[10:]
	context['prediction'] = context_prediction.upper()
	context['summary'] = wiki.summary(context_prediction, sentences=4)
	print(context['summary'])
	return render(request,'images/classify_image.html',context)

