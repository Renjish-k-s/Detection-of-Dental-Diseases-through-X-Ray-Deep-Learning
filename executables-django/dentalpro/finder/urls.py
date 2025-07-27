from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.homepage,name='home'),
    path('predict/', views.predict_image, name='predict_image'),

]