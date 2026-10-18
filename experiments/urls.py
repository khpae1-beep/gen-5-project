from django.urls import path

from . import views

app_name = 'experiments'

urlpatterns = [
    path('', views.experiment_list, name='list'),
]
