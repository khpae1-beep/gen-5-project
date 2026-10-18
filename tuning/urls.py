from django.urls import path

from . import views

app_name = 'tuning'

urlpatterns = [
    path('', views.job_list, name='list'),
    path('compare/', views.compare, name='compare'),
    path('<int:pk>/', views.job_detail, name='detail'),
]
