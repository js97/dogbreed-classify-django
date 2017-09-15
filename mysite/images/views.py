from django.shortcuts import render
import tensorflow as tf
import wikipedia as wiki
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.uploadedfile import SimpleUploadedFile
def home(request):
	if request.method == 'POST':
		form = UploadFileForm(request.FILES)
		if form.is_valid():
			handle_uploaded_file(request.FILES['file'])
			return render(request, 'images/classify.html')

	return render(request, 'images/home.html', {'form':form})

def handle_uploaded_file(f):
	with open('./test.jpg','wb+') as destination:
		for chunk in f.chunks():
			destination.write(chunk)

'''
Reminder from s/o on why to use gfile   ---->

The main roles of the tf.gfile module are:
1)To provide an API that is close to Python's file objects, and
2)To provide an implementation based on TensorFlow's C++ FileSystem API.

Regular python file API can be used but use gfile if possibility of using unconventional file system such as google cloud
'''
def classify_image(request, image):
	'''
	# load image
	image_data = image

  	# load labels in googley fashion
	labels = [line.rstrip() for line in tf.gfile.GFile('mysite/retrained_labels.txt')]
  	# load graph
	with tf.gfile.FastGFile('/mysite/retrained_graph.pb', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
		softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
		predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})
    	
		top_k = predictions.argsort()[-num_top_predictions:][::-1]
		for node_id in top_k:
	  		human_string = labels[node_id]
	  		score = predictions[node_id]
	  		print('%s (score = %.5f)' % (human_string, score))
	'''
	return render(request,'mysite/classify_image.html')
# Create your views here.
