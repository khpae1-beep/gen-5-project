from django.urls import path

from . import views

app_name = 'experiments'

urlpatterns = [
    path('', views.experiment_list, name='list'),
    path('<str:name>/', views.experiment_detail, name='detail'),
    path('<str:name>/compare/', views.compare, name='compare'),
    path('<str:name>/v<int:version>/', views.run_detail, name='run'),
    path('<str:name>/v<int:version>/promote/', views.promote_run, name='promote'),
]
