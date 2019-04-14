import logging
import os
from pathlib import Path
import shutil

from django.core.files import File
from django.core.files.base import ContentFile
from django.http import HttpResponse
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

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

        # logger.info(f'Base context: {context}')
        # events = None
        # context['events'] = events
        # logger.info(f'events: {events}')


database_list_view = DatabaseListView.as_view()


class DatabaseDetailView(DetailView):
    model = models.Database
    template_name = 'photologue/database_detail.html'
    context_object_name = 'database'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        events = models.Event.objects.filter(database=context['database'].pk)
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
            database.save()

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

            return HttpResponseRedirect(reverse('photologue:database_detail', kwargs={'slug': database.slug}))
    else:
        form = forms.DatabaseForm()

    context['form'] = form
    return render(request, 'photologue/database_create.html', context)


def database_edit_view(request, slug):
    method = request.method
    context = {}
    debug_info = f'Method {method}'
    context['debug_info'] = debug_info
    database = get_object_or_404(models.Database, slug=slug)
    context['database_old'] = database
    logger = logging.getLogger('photologue.database.edit')

    if method == 'POST':
        logger.info('POST')
        form = forms.DatabaseForm(request.POST, request.FILES, instance=database)

        logger.info(f'form.errors: {form.errors}')
        for index_error, error in enumerate(form.errors):
            logger.info(f'{index_error} error: {error} type(error): {type(error)}')

        logger.info(f'form.cleaned_data: {form.cleaned_data}')
        logger.info(f'form.has_changed: {form.has_changed()}')
        logger.info(f'form.changed_data: {form.changed_data}')

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
            logger.info('Validation successful')
            logger.info('Updating model instance from form instsance')
            database = form.save(commit=False)
            logger.info(f'database from form: {database}; database.pk: {database.pk}; database.slug: {database.slug}')
            database.save()

            logger.info(f"Images count to upload: {len(request.FILES.getlist('photos'))}")
            count = 1
            for file_image in request.FILES.getlist('photos'):
                # logger.info(f"file_image: {file_image}\n"
                #             f"type(file_image): {type(file_image)}\n"
                #             f"file_image.name: {file_image.name}\n")
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
            logger.info('Validation not successful')
    else:
        logger.info('GET')
        form = forms.DatabaseForm(instance=database)

    context['form'] = form
    context['database'] = database
    return render(request, 'photologue/database_edit.html', context)


class EventDetailView(DetailView):
    model = models.Event
    template_name = 'photologue/event_detail.html'
    context_object_name = 'event'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        all_photos = models.EventPhoto.objects.filter(event=context['event'])
        query_photos = all_photos.filter(is_query=True)
        result_photos = all_photos.filter(is_query=False)
        context['result_photos'] = result_photos
        context['query_photos'] = query_photos
        return context


event_detail_view = EventDetailView.as_view()


# def event_create_view(request):
#     return HttpResponse('Create new event')
def event_create_view(request):
    method = request.method
    context = {}
    debug_info = f'Method {method}'
    context['debug_info'] = debug_info
    logger = logging.getLogger('photologue.event.create')

    logger.info(f'post: {request.POST}')
    logger.info(f'get: {request.GET}')

    database_slug = request.GET.get('database')
    logger.info(f'Database slug: {database_slug}')
    database = get_object_or_404(models.Database, slug=database_slug)
    context['database'] = database

    if method == 'POST':
        form = forms.EventForm(request.POST, request.FILES)
        if form.is_valid():
            event = form.save(commit=False)
            logger.info(f'Event object submitted by user: {event}')
            event.database = database
            event.save()

            # Handling images chosen from existing ones in a database
            query_photos_from_database = form.cleaned_data.get('query_photos_from_database')
            logger.info(f'query_photos_from_database: {query_photos_from_database}')

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
                                                database_photo=query_photo_from_database,)
                event_photo_path = models.get_storage_path_for_image(
                        event_photo,
                        filename=Path(query_photo_from_database.image.name).name)
                logger.info(f'AAAAAAAA. event_photo_path: {event_photo_path}')
                event_photo.image = event_photo_path

                path_to_ready_file = os.path.join(MEDIA_ROOT, query_photo_from_database.image.name)
                logger.info(f'path_to_ready_file: {path_to_ready_file}')
                shutil.copyfile(path_to_ready_file, os.path.join(MEDIA_ROOT, event_photo_path))

                # with open(path_to_ready_file, 'rb') as fin:
                #     event_photo.image.save(event_photo_path, ContentFile('sdfvfvdfsccccc'))

                event_photo.save()

            # Handling new uploaded images
            logger.info(f"files count: {len(request.FILES.getlist('query_photos'))}")
            for file_image in request.FILES.getlist('query_photos'):
                logger.info(f"file_image: {file_image}\n"
                            f"type(file_image): {type(file_image)}\n"
                            f"file_image.name: {file_image.name}\n")

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
                                                image=file_image,)
                event_photo.save()

            return HttpResponseRedirect(reverse('photologue:event_detail', kwargs={'slug': event.slug}))
    else:
        form = forms.EventForm()

    context['form'] = form
    return render(request, 'photologue/event_create.html', context)
