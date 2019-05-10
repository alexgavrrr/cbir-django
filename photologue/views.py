import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from django.contrib import messages
from django.core.files.base import ContentFile
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import ListView, DetailView
from tqdm import tqdm

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
        context['events'] = context['database'].get_events(limit=10)
        return context


database_detail_view = DatabaseDetailView.as_view()


def database_create_view(request):
    context = {}

    if request.method == 'POST':
        form = forms.DatabaseForm(request.POST, request.FILES)
        if form.is_valid():
            database = form.save(commit=False)
            database.save()

            zip_file = form.cleaned_data['zip_file']
            if zip_file:
                # Handling files from zip
                logger.info('Handling files from zip archive')
                zip = zipfile.ZipFile(zip_file)
                count_files_from_zip = 1
                for filename in tqdm(sorted(zip.namelist())):
                    # logger.debug('Reading file "{0}".'.format(filename))

                    if filename.startswith('__') or filename.startswith('.'):
                        logger.debug('Ignoring file "{0}".'.format(filename))
                        continue
                    if os.path.dirname(filename):
                        logger.warning('Ignoring file "{0}" as it is in a subfolder; all images should be in the top '
                                       'folder of the zip.'.format(filename))
                        continue

                    data = zip.read(filename)

                    if not len(data):
                        logger.debug('File "{0}" is empty.'.format(filename))
                        continue

                    # Basic check that we have a valid image.
                    try:
                        file = BytesIO(data)
                        opened = Image.open(file)
                        opened.verify()
                    except Exception:
                        # Pillow doesn't recognize it as an image.
                        # If a "bad" file is found we just skip it.
                        logger.error('Could not process file "{0}" in the .zip archive.'.format(
                            filename))
                        if request:
                            messages.warning(request,
                                             'Could not process file "{0}" in the .zip archive.'
                                             .format(filename),
                                             fail_silently=True)
                        continue

                    while True:
                        slug = f'{database.slug}-fromzip-{count_files_from_zip}'
                        if models.DatabasePhoto.objects.filter(slug=slug).exists():
                            count_files_from_zip += 1
                            continue
                        break
                    database_photo = models.DatabasePhoto(slug=slug,
                                                          database=database)
                    contentfile = ContentFile(data)
                    database_photo.image.save(filename, contentfile)
                    database_photo.name = Path(database_photo.image.name).name
                    database_photo.save()

                zip.close()

                if request:
                    messages.success(request,
                                     'The photos from zip have been added.',
                                     fail_silently=True)

            # Handling files
            logger.info('Handling files')
            count_files = 1
            for file_image in tqdm(request.FILES.getlist('photos')):
                while True:
                    slug = f'{database.slug}-{count_files}'
                    if models.DatabasePhoto.objects.filter(slug=slug).exists():
                        count_files += 1
                        continue
                    break

                database_photo = models.DatabasePhoto(slug=slug,
                                                      database=database,
                                                      image=file_image)
                database_photo.save_when_name_not_inited()

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
                database_photo.save_when_name_not_inited()

            return HttpResponseRedirect(reverse('photologue:database_detail', kwargs={'slug': database.slug}))
    else:
        logger.info('GET')
        form = forms.DatabaseForm(instance=database)

    context['form'] = form
    context['database'] = database
    return render(request, 'photologue/database_edit.html', context)


def event_detail_view(request, slug):
    context = {}
    event = get_object_or_404(models.Event, slug=slug)
    context['event'] = event

    event_life_stage = 'ready'  # ['search', 'basket', 'ready']
    if event_life_stage == 'search':
        return render(request, 'photologue/event_detail_search.html', context)
    elif event_life_stage == 'basket':
        return render(request, 'photologue/event_detail_basket.html', context)
    elif event_life_stage == 'ready':
        RESULT_PHOTOS_LIMIT = 10
        result_photos = event.get_result_photos()
        result_photos_truncated = result_photos[:RESULT_PHOTOS_LIMIT]
        context['result_photos'] = result_photos_truncated
        context['query_photos'] = event.get_query_photos()
        return render(request, 'photologue/event_detail_ready.html', context)
    else:
        raise ValueError(f'event life_stage: {event_life_stage} is wrong')


def event_create_view(request):
    method = request.method
    database = get_object_or_404(models.Database, slug=request.GET.get('database'))
    context = {}
    context['database'] = database

    if method == 'POST':
        form = forms.EventForm(request.POST, request.FILES)
        if form.is_valid():
            event = form.save(commit=False)
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
                                                      image=file_image)
                database_photo.save_when_name_not_inited()

                event_photo = models.EventPhoto(slug=event_photo_slug,
                                                event=event,
                                                is_query=True,
                                                database_photo=database_photo, )
                event_photo.save()

            qe = form.cleaned_data.get('qe')
            sv = form.cleaned_data.get('sv')
            topk = form.cleaned_data.get('topk')
            n_candidates = 100 if 100 >= topk else topk * 2
            max_verified = 20 if 20 > n_candidates else topk
            similarity_threshold = form.cleaned_data.get('similarity_threshold')

            search_params = {
                'sv_enable': sv,
                'qe_enable': qe,
                'topk': topk,
                'n_candidates': n_candidates,
                'max_verified': max_verified,
                'similarity_threshold': similarity_threshold,
            }
            # TODO: Async job
            event.init_if_needed_and_get_result_photos(search_params)
            return HttpResponseRedirect(reverse('photologue:event_detail', kwargs={'slug': event.slug}))
    else:
        form = forms.EventForm(initial={'cbir_index': database.cbir_index_default,
                                        'database': database})

    context['form'] = form
    return render(request, 'photologue/event_create.html', context)


def database_index_info_view(request):
    database_slug = request.GET.get('database')

    if not database_slug:
        return HttpResponse("Database must be provided as get parameter for which to show info")
    else:
        context = {}
        database = get_object_or_404(models.Database, slug=database_slug)
        context['database'] = database
        cbir_indexes = database.cbirindex_set.all()
        context['cbir_indexes'] = cbir_indexes

        return render(request, 'photologue/database_index_info.html', context)


def database_index_detail_view(request, slug):
    database_index = get_object_or_404(models.CBIRIndex, slug=slug)
    context = {}
    context['database_index'] = database_index
    return render(request, 'photologue/database_index_detail_ready.html', context)


def database_index_management_view(request):
    pass


def database_index_create_view(request):
    database_slug = request.GET.get('database')
    database = get_object_or_404(models.Database, slug=database_slug)

    context = {}
    context['database'] = database

    if request.method == 'POST':
        form = forms.CbirIndexForm(request.POST)
        if form.is_valid():
            cbir_index = form.save(commit=False)
            cbir_index.database = database
            des_type = form.cleaned_data.get('des_type')
            algo_params = {'des_type': des_type}

            cbir_index.save()
            # TODO: Make call to build index asynchronous
            cbir_index.build_if_needed(algo_params)

            set_default = form.cleaned_data.get('set_default')
            database_has_default_cbir_index = bool(database.cbir_index_default)
            if set_default or not database_has_default_cbir_index:
                database.cbir_index_default = cbir_index
                database.save()

            return HttpResponseRedirect(reverse('photologue:database_index_detail', kwargs={'slug': cbir_index.slug}))
    else:
        form = forms.CbirIndexForm(initial={'database': database})

    context['form'] = form
    return render(request, 'photologue/database_index_create.html', context)


def database_photo_detail_view(request, slug):
    database_photo = get_object_or_404(models.DatabasePhoto, slug=slug)

    events = [event_photo.event for event_photo in database_photo.eventphoto_set.all()]
    event_id_to_element_id = dict()
    for element_id, event in enumerate(events):
        event_id_to_element_id[event.id] = element_id

    queries_flattened = models.EventPhoto.objects.filter(event__in=events, is_query=True)
    queries = [[]] * len(events)
    for query in queries_flattened:
        queries[event_id_to_element_id[query.event.id]] = query
    context = {}
    context['photo'] = database_photo
    context['events_queries_pairs'] = list(zip(events, queries))

    return render(request, 'photologue/database_photo_detail.html', context)


def event_photo_detail_view(requst, event_slug, pk):
    return HttpResponse('AAA event_photo' + event_slug + str(pk))
