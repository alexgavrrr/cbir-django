from django.http import HttpResponse
from django.shortcuts import render

from django.urls import reverse
from django.views.generic import ListView, DetailView

from . import models


class DatabaseListView(ListView):
    model = models.Database
    template_name = 'photologue/database_list.html'
    paginate_by = 20

database_list_view = DatabaseListView.as_view()


class DatabaseDetailView(DetailView):
    model = models.Database
    template_name = 'photologue/database_detail.html'
    queryset = models.Database.objects.all()

database_detail_view = DatabaseDetailView.as_view()


def database_create_view(request):
    method = request.method
    return HttpResponse(f'Create database. Method {method}')


# class PhotoDetailView(DetailView):
#     model = models.Photo
#     template_name = 'photologue/photo_detail.html'
#     queryset = models.Photo.objects.all()
#
# photo_detail_view = PhotoDetailView.as_view()
