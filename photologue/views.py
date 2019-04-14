import logging
from django.http import HttpResponse
from django.shortcuts import render
from django.template.defaultfilters import slugify


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

    def get_context_data(self, **kwargs):
        context = {}
        logger.info(f'kwargs: {kwargs}')
        events = models.Event.objects.filter(database=kwargs['object'].pk)
        logger.info(f'events: {events}')
        context['events'] = events
        return context

database_detail_view = DatabaseDetailView.as_view()


def database_create_view(request):
    method = request.method
    context = {}
    debug_info = f'Method {method}'
    context['debug_info'] = debug_info
    logger = logging.getLogger('photologue.database.create')

    if method == 'POST':
        form = forms.DatabaseForm(request.POST, request.FILES)
        if form.is_valid():
            database = form.save(commit=False)
            logger.info(f'Database object submitted by user: {database}')

            # if database.description_file:
            #     logger.info(f'There is description file:\n{database.description_file} - {database.description_file.file.name}')
            # else:
            #     logger.info(f'There is NO description file')

            database.save()

            # logger.info(f'After save There is description file:\n'
            #               f'{database.description_file} - {database.description_file.file.name}')

            # TODO: Creare database photos from request.FILES
            logger.info(f"files count: {len(request.FILES.getlist('photos'))}")

            count = 1
            for file_image in request.FILES.getlist('photos'):
                logger.info(f"file_image: {file_image}\n"
                            f"type(file_image): {type(file_image)}\n"
                            f"file_image.name: {file_image.name}\n")
                while True:
                    slug = f'{database.slug}-{count}'
                    if models.DatabasePhoto.objects.filter(slug=slug).exists():
                        count += 1
                        continue
                    break

                database_photo = models.DatabasePhoto(slug=slug,
                                                      database=database,
                                                      image=file_image)
                database_photo.save()

            return HttpResponseRedirect(reverse('photologue:database_list'))
    else:
        form = forms.DatabaseForm()

    context['form'] = form
    return render(request, 'photologue/database_create.html', context)


class EventDetailView(DetailView):
    model = models.Event
    template_name = 'photologue/event_detail.html'
    queryset = models.Event.objects.all()  # TODO: Ensure that it is fine or fix this.

event_detail_view = EventDetailView.as_view()


def event_create_view(request):
    return HttpResponse('Create new event')
