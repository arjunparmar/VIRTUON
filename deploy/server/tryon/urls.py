from django.urls import path
from . import views

app_name = 'tryon'

urlpatterns = [
    path('', views.TryonView.as_view(), name='home'),
    path('predict', views.TryonPredict.as_view(), name='predict'),
]