import logging
from django.http import HttpResponse
from django.shortcuts import render

from django.urls import reverse
from django.views.generic import ListView, DetailView

from django.http import HttpResponseRedirect
from django.shortcuts import render

from . import forms
from . import models


logger = logging.getLogger('photologue.views')

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
    context = {}
    debug_info = f'Method {method}'
    context['debug_info'] = debug_info
    logger = logging.getLogger('photologue.database.create')

    if method == 'POST':
        form = forms.DatabaseForm(request.POST)
        if form.is_valid():
            database = form.save(commit=False)
            logger.info(f'Database object submitted by user: {database}')
            database.save()
            return HttpResponseRedirect(reverse('photologue:database_list'))
    else:
        form = forms.DatabaseForm()

    context['form'] = form
    return render(request, 'photologue/database_create.html', context)


# class PhotoDetailView(DetailView):
#     model = models.Photo
#     template_name = 'photologue/photo_detail.html'
#     queryset = models.Photo.objects.all()
#
# photo_detail_view = PhotoDetailView.as_view()
