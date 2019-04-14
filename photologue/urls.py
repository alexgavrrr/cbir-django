from django.urls import path
from django.views.generic import TemplateView

from . import views

app_name = 'photologue'
urlpatterns = [
    path('', TemplateView.as_view(template_name="photologue/home.html"), name='home'),
    path('database/', views.database_list_view, name='database_list'),
    path('database/create/', views.database_create_view, name='database_create'),
    path('database/<slug:slug>/', views.database_detail_view, name='database_detail'),
    path('database/<slug:slug>/edit/', views.database_edit_view, name='database_edit'),

    path('event/create/', views.event_create_view, name='event_create'),
    path('event/<slug:slug>/', views.event_detail_view, name='event_detail'),

]
