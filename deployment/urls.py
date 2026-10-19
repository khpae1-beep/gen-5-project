from django.urls import path

from . import views

app_name = 'deployment'

urlpatterns = [
    path('', views.index, name='index'),
    path('generate/', views.generate, name='generate'),
]
