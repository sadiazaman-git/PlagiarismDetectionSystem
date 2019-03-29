from django.urls import path, re_path
from plagtrace import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

app_name = "Plagtrace"

urlpatterns = [
    path('', views.index, name="index"),
    path('plagiarism-checker/', views.index, name="index"),
    path('blog', views.blog, name="blog"),
    path('features', views.features, name="features"),
    path('contact_form', views.contact_form, name='contact_form'),
    path('upload_file', views.upload_file, name='upload_file'),
    path('submitted', views.submitted,name='submitted'),
    path('register', views.UserFormView.as_view(),name='register'),
    path('login/', views.login_user, name='login'),
    path('logout/', auth_views.logout, name='logout')



]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



