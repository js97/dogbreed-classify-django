from django.conf.urls import url
from . import views

urlpatterns = [
			url(r'^$', views.home, name = 'Home'),
			url(r'^classify$', views.classify_image, name = 'Classify '),	

]