import logging
import os
import shutil
from pathlib import Path

from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import ListView, DetailView

from project.settings import MEDIA_ROOT
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
    context_object_name = 'database'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['events'] = models.Event.objects.filter(database=context['database'].pk)
        return context


database_detail_view = DatabaseDetailView.as_view()


def database_create_view(request):
    context = {}

    if request.method == 'POST':
        form = forms.DatabaseForm(request.POST, request.FILES)
        if form.is_valid():
            database = form.save(commit=False)
            database.save()

            count = 1
            for file_image in request.FILES.getlist('photos'):
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

            return HttpResponseRedirect(reverse('photologue:database_detail', kwargs={'slug': database.slug}))
    else:
        form = forms.DatabaseForm()

    context['form'] = form
    return render(request, 'photologue/database_create.html', context)


def database_edit_view(request, slug):
    context = {}
    database = get_object_or_404(models.Database, slug=slug)
    logger = logging.getLogger('photologue.database.edit')

    if request.method == 'POST':
        form = forms.DatabaseForm(request.POST, request.FILES, instance=database)

        validation_successful = True
        if 'slug' in form.changed_data or 'title' in form.changed_data:
            validation_successful = False
            message = f'slug or title is changed wihich is bad'
            logger.info(message)

            if 'slug' in form.changed_data:
                form.add_error('slug', 'Can not modify slug')
            if 'title' in form.changed_data:
                form.add_error('title', 'Can not modify title')
        # ... next validation steps

        if validation_successful:
            database = form.save(commit=False)
            database.save()

            count = 1
            for file_image in request.FILES.getlist('photos'):
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

            return HttpResponseRedirect(reverse('photologue:database_detail', kwargs={'slug': database.slug}))
    else:
        logger.info('GET')
        form = forms.DatabaseForm(instance=database)

    context['form'] = form
    context['database'] = database
    return render(request, 'photologue/database_edit.html', context)


# class EventDetailView(DetailView):
#     model = models.Event
#     template_name = 'photologue/event_detail.html'
#     context_object_name = 'event'
#
#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         all_photos = models.EventPhoto.objects.filter(event=context['event'])
#         query_photos = all_photos.filter(is_query=True)
#         result_photos = all_photos.filter(is_query=False)
#         context['result_photos'] = result_photos
#         context['query_photos'] = query_photos
#         return context
# event_detail_view = EventDetailView.as_view()

def event_detail_view(request, slug):
    RESULT_PHOTOS_LIMIT = 10
    context = {}
    event = get_object_or_404(models.Event, slug=slug)
    result_photos = event.init_if_needed_and_get_result_photos()
    result_photos_truncated = result_photos[:RESULT_PHOTOS_LIMIT]

    # TODO: HARDCODED. Fix it.
    cbir_database_name = 'buildings'

    context['result_photos'] = result_photos_truncated
    context['query_photos'] = event.get_query_photos()
    return render(request, 'photologue/event_detail.html', context)


def event_create_view(request):
    method = request.method
    database = get_object_or_404(models.Database, slug=request.GET.get('database'))
    context = {}
    context['database'] = database

    if method == 'POST':
        form = forms.EventForm(request.POST, request.FILES)
        if form.is_valid():
            event = form.save(commit=False)
            event.database = database
            event.save()

            # Handling images chosen from existing ones in a database
            query_photos_from_database = form.cleaned_data.get('query_photos_from_database')
            count_event_photo = 1
            for query_photo_from_database in query_photos_from_database:
                while True:
                    event_photo_slug = f'{event.slug}-{count_event_photo}'
                    if models.EventPhoto.objects.filter(slug=event_photo_slug).exists():
                        count_event_photo += 1
                        continue
                    break

                event_photo = models.EventPhoto(slug=event_photo_slug,
                                                event=event,
                                                is_query=True,
                                                # description=...,
                                                # description_file=...,
                                                database_photo=query_photo_from_database, )
                event_photo_path = models.get_storage_path_for_image(
                    event_photo,
                    filename=Path(query_photo_from_database.image.name).name)
                event_photo.image = event_photo_path
                path_to_original_file = os.path.join(MEDIA_ROOT, query_photo_from_database.image.name)
                path_to_new_file = os.path.join(MEDIA_ROOT, event_photo_path)
                shutil.copyfile(path_to_original_file, path_to_new_file)
                event_photo.save()

            # Handling new uploaded images
            for file_image in request.FILES.getlist('query_photos'):
                while True:
                    event_photo_slug = f'{event.slug}-{count_event_photo}'
                    if models.EventPhoto.objects.filter(slug=event_photo_slug).exists():
                        count_event_photo += 1
                        continue
                    break

                database_photo_slug_base = f'{database.slug}-{event_photo_slug}'
                count_database_photo = 1
                while True:
                    database_photo_slug = f'{database_photo_slug_base}-{count_database_photo}'
                    if models.DatabasePhoto.objects.filter(slug=database_photo_slug).exists():
                        count_database_photo += 1
                        continue
                    break

                database_photo = models.DatabasePhoto(slug=database_photo_slug,
                                                      database=database,
                                                      # description=...,
                                                      # description_file=...,
                                                      image=file_image)
                database_photo.save()

                event_photo = models.EventPhoto(slug=event_photo_slug,
                                                event=event,
                                                is_query=True,
                                                # description=...,
                                                # description_file=...,
                                                database_photo=database_photo,
                                                image=file_image, )
                event_photo.save()

            return HttpResponseRedirect(reverse('photologue:event_detail', kwargs={'slug': event.slug}))
    else:
        form = forms.EventForm()

    context['form'] = form
    return render(request, 'photologue/event_create.html', context)
