"""
URL configuration for voice_pdf_rag project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
"""
from django.contrib import admin
from django.urls import path
from django.views.generic import RedirectView
from rag import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('ask/', views.ask, name='ask'),
    path('summary/', views.summary, name='summary'),
    path('reset/', views.reset, name='reset'),
    path('audio/<path:filename>', views.serve_audio, name='serve_audio'),
    path('upload_pdf/', views.upload_pdf, name='upload_pdf'),
    path('health/', views.health_check, name='health_check'),
    path('favicon.ico', RedirectView.as_view(url='/static/favicon.ico')),
]
