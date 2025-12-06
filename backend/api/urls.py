from django.urls import path
from .views import PredictMalware, UploadApk

urlpatterns = [
    path('predict/', PredictMalware.as_view(), name='predict-malware'),
    path('upload-apk/', UploadApk.as_view(), name='upload-apk'),
]
