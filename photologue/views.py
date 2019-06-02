import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from django.contrib import messages
from django.core.files.base import ContentFile
from django.db import IntegrityError
from django.http import HttpResponseRedirect, HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import ListView, DetailView
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

from tqdm import tqdm

from . import forms
from . import models


logger = logging.getLogger('photologue.views')

PAGINATE_PHOTOS_BY = 5


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
        context['events'] = context['database'].get_events(limit=None)
        photos = models.DatabasePhoto.objects.filter(database=self.get_object())

        paginator = Paginator(photos, PAGINATE_PHOTOS_BY)
        page = self.request.GET.get('page')
        photos_to_give = None
        try:
            photos_to_give = paginator.page(page)
        except PageNotAnInteger:
            photos_to_give = paginator.page(1)
        except EmptyPage:
            photos_to_give = paginator.page(paginator.num_pages)

        context['photos_to_give'] = photos_to_give
        return context


database_detail_view = DatabaseDetailView.as_view()


def database_create_view(request):
    context = {}

    if request.method == 'POST':
        form = forms.DatabaseForm(request.POST, request.FILES, my_params={'my_mode': 'create'})
        if form.is_valid():
            database = form.save(commit=False)
            database.count = 0
            database.save()
            total_count_files = 0

            zip_file = form.cleaned_data['zip_file']
            if zip_file:
                # Handling files from zip
                logger.info('Handling files from zip archive')
                zip = zipfile.ZipFile(zip_file)
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

                    database_photo = models.DatabasePhoto(database=database)
                    contentfile = ContentFile(data)
                    database_photo.image.save(filename, contentfile)
                    database_photo.name = Path(database_photo.image.name).name
                    database_photo.save()
                    total_count_files += 1

                zip.close()

                if request:
                    messages.success(request,
                                     'The photos from zip have been added.',
                                     fail_silently=True)

            # Handling files
            logger.info('Handling files')
            for file_image in tqdm(request.FILES.getlist('photos')):
                database_photo = models.DatabasePhoto(database=database,
                                                      image=file_image)
                database_photo.save_when_name_not_inited()
                total_count_files += 1

            database.count = total_count_files
            database.save()
            return HttpResponseRedirect(reverse('photologue:database_detail', kwargs={'slug': database.slug}))
    else:
        form = forms.DatabaseForm(my_params={'my_mode': 'create'})

    context['form'] = form
    return render(request, 'photologue/database_create.html', context)


def database_edit_view(request, slug):
    context = {}
    database = get_object_or_404(models.Database, slug=slug)
    logger = logging.getLogger('photologue.database.edit')

    if request.method == 'POST':
        form = forms.DatabaseForm(request.POST, request.FILES, instance=database, my_params={'my_mode': 'edit'})
        if form.is_valid():
            database = form.save(commit=False)

            count_new_photos = 0
            for file_image in request.FILES.getlist('photos'):
                database_photo = models.DatabasePhoto(database=database,
                                                      image=file_image)
                database_photo.save_when_name_not_inited()
                count_new_photos += 1

            database.count = database.count + count_new_photos
            database.save()
            return HttpResponseRedirect(reverse('photologue:database_detail', kwargs={'slug': database.slug}))
    else:
        logger.info('GET')
        form = forms.DatabaseForm(instance=database, my_params={'my_mode': 'edit'})

    context['form'] = form
    context['database'] = database
    return render(request, 'photologue/database_edit.html', context)


def event_detail_by_id_view(request, id):
    event = get_object_or_404(models.Event, id=id)
    return event_detail_view(request, slug=event.slug)


def event_detail_view(request, slug):
    context = {}
    event = get_object_or_404(models.Event, slug=slug)
    context['event'] = event
    event_status = event.status  # ['search', 'basket', 'ready']

    if request.method == 'POST':
        dos = ['add', 'remove', 'commit']
        do = request.POST.get('do')

        if event.status == 'basket':
            if do == 'add':
                database_photos_to_add = None
                if request.POST.get('toggle_all'):
                    found_photos = event.get_found_photos()
                    database_photos_to_add = [found_photo.database_photo for found_photo in found_photos]
                else:
                    database_photos_to_add = [get_object_or_404(models.DatabasePhoto, id=photo_id)
                                              for photo_id
                                              in request.POST.getlist('chosen_photo')]
                for database_photo in database_photos_to_add:
                    event_basket_chosen_photo = models.EventBasketChosenPhoto(
                        description=database_photo.description or str(database_photo),
                        event=event,
                        database_photo=database_photo)
                    try:
                        event_basket_chosen_photo.save()
                    except IntegrityError:
                        logger.warning('Supressed integirty Error')
                return HttpResponseRedirect(reverse('photologue:event_detail', kwargs={'slug': event.slug}))
            elif do == 'commit':
                descriptions = request.POST.getlist('description')
                chosen_photos = models.EventBasketChosenPhoto.objects.filter(event=event)
                print(f'len(descriptions): {len(descriptions)}, len(chosen_photos): {len(chosen_photos)}')
                print(f'AAA descriptions: {descriptions}')
                print(f'AAA chosen_photos: {chosen_photos}')
                while len(descriptions) < len(chosen_photos):
                    descriptions += [chosen_photos[len(descriptions)].description]
                print(f'AAA descriptions: {descriptions}')
                for index_chosen_photo, chosen_photo in enumerate(chosen_photos):
                    event_photo = models.EventPhoto(
                        description=descriptions[index_chosen_photo],
                        is_query=False,
                        event=event,
                        database_photo=chosen_photo.database_photo)
                    event_photo.save()
                event.status = 'ready'
                event.save()
                return HttpResponseRedirect(reverse('photologue:event_detail', kwargs={'slug': event.slug}))
            else:
                return HttpResponseBadRequest(f'Bad do `{do}`. Must be one of {dos}')
        else:
            # event.status != 'basket'
            return HttpResponseBadRequest(f'Bad status: {event.status}. Must be `basket`')
    else:
        # request.method != 'POST'
        if event_status == 'search':
            context['query_photos'] = event.get_query_photos()
            return render(request, 'photologue/event_detail_search.html', context)
        elif event_status == 'basket':
            context['query_photos'] = event.get_query_photos()

            found_photos = event.get_found_photos()
            paginator = Paginator(found_photos, PAGINATE_PHOTOS_BY)
            page = request.GET.get('page', 1)
            found_photos_to_give = None
            try:
                found_photos_to_give = paginator.page(page)
            except PageNotAnInteger:
                found_photos_to_give = paginator.page(1)
            except EmptyPage:
                found_photos_to_give = paginator.page(paginator.num_pages)

            chosen_photos = event.get_chosen_photos()
            paginator = Paginator(chosen_photos, PAGINATE_PHOTOS_BY)
            page = request.GET.get('page', 1)

            # NOTE: We do not paginate chosen_photos because we want all chosen photos to be displayed at once. Otherwise bug.
            # chosen_photos_to_give = None
            # try:
            #     chosen_photos_to_give = paginator.page(page)
            # except PageNotAnInteger:
            #     chosen_photos_to_give = paginator.page(1)
            # except EmptyPage:
            #     chosen_photos_to_give = paginator.page(paginator.num_pages)


            context['found_photos_to_give'] = found_photos_to_give
            context['chosen_photos_to_give'] = chosen_photos
            return render(request, 'photologue/event_detail_basket.html', context)
        elif event_status == 'ready':
            result_photos = event.get_result_photos()
            paginator = Paginator(result_photos, PAGINATE_PHOTOS_BY)
            page = request.GET.get('page', 1)
            result_photos_to_give = None
            try:
                result_photos_to_give = paginator.page(page)
            except PageNotAnInteger:
                result_photos_to_give = paginator.page(1)
            except EmptyPage:
                result_photos_to_give = paginator.page(paginator.num_pages)

            context['result_photos_to_give'] = result_photos_to_give
            context['query_photos'] = event.get_query_photos()
            event_baskets = event.database.get_event_baskets()
            context['event_baskets'] = event_baskets
            return render(request, 'photologue/event_detail_ready.html', context)
        else:
            raise ValueError(f'event status: {event_status} is wrong')


def event_create_view(request):
    method = request.method
    database = get_object_or_404(models.Database, slug=request.GET.get('database'))
    context = {}
    context['database'] = database

    if method == 'POST':
        form = forms.EventForm(request.POST, request.FILES)
        if form.is_valid():
            event = form.save(commit=False)
            query_photo_from_database = form.cleaned_data.get('query_photo_from_database')
            query_photos_uploded = request.FILES.getlist('query_photos')
            if (len(query_photos_uploded) >= 2
                    or (len(query_photos_uploded) == 1 and query_photo_from_database)
                    or (len(query_photos_uploded) == 0 and not query_photo_from_database)):
                logger.warning('Must be one photo')
                form.add_error('query_photos', 'Exactly one photo must be submitted')
                context['form'] = form
                return render(request, 'photologue/event_create.html', context)

            event.status = 'search'
            event.save()

            # Handling new uploaded images
            count_new_photos = 0
            for file_image in query_photos_uploded:
                x = form.cleaned_data.get('x')
                y = form.cleaned_data.get('y')
                w = form.cleaned_data.get('width')
                h = form.cleaned_data.get('height')
                database_photo = models.DatabasePhoto(database=database,
                                                      image=file_image)
                database_photo.save_when_name_not_inited()

                if not (x is None or y is None or w is None or h is None):
                    image = Image.open(database_photo.image)
                    resulting_file_image = image.crop((x, y, w+x, h+y))
                    # resulting_file_image = resulting_file_image.resize((200, 200), Image.ANTIALIAS)
                    resulting_file_image.save(database_photo.image.path)

                event_photo = models.EventPhoto(event=event,
                                                is_query=True,
                                                database_photo=database_photo, )
                event_photo.save()
                count_new_photos += 1

            if query_photo_from_database:
                event_photo = models.EventPhoto(event=event,
                                                is_query=True,
                                                # description=...,
                                                # description_file=...,
                                                database_photo=query_photo_from_database, )
                event_photo.save()

            database.count = database.count + count_new_photos
            database.save()

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
            # event.status = 'search'  # This is already above
            event.init(search_params)
            event.status = 'basket'
            event.save()
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
    context['status'] = 'ready' if database_index.built else 'build'
    context['database_index'] = database_index

    if database_index.built:
        context['difference'] = database_index.database.count - database_index.count_photos_indexed
        return render(request, 'photologue/database_index_detail_ready.html', context)
    else:
        return render(request, 'photologue/database_index_detail_build.html', context)


def database_index_create_view(request):
    context = {}

    if request.method == 'POST':
        form = forms.CbirIndexForm(request.POST)
        if form.is_valid():
            cbir_index = form.save(commit=False)
            database = get_object_or_404(models.Database, id=cbir_index.database.id)
            count_photos_indexed = database.count
            cbir_index.count_photos_indexed = count_photos_indexed
            cbir_index.count_photos_for_training_from_database = count_photos_indexed
            cbir_index.count_photos_for_training_external = 0
            cbir_index.save()

            set_default = form.cleaned_data.get('set_default')
            database_has_default_cbir_index = bool(database.cbir_index_default)
            if set_default or not database_has_default_cbir_index:
                database.cbir_index_default = cbir_index
                database.save()

            des_type = form.cleaned_data.get('des_type')
            algo_params = {'des_type': des_type}
            # TODO: Make call to build index asynchronous
            cbir_index.build_if_needed(algo_params)

            return HttpResponseRedirect(reverse('photologue:database_index_detail', kwargs={'slug': cbir_index.slug}))
    else:

        database_slug = request.GET.get('database')
        try:
            database = models.Database.objects.get(slug=database_slug)
        except models.Database.DoesNotExist:
            database = None
        form = forms.CbirIndexForm(initial={'database': database})

    context['form'] = form
    return render(request, 'photologue/database_index_create.html', context)


def database_photo_detail_view(request, database_slug, pk):
    database = get_object_or_404(models.Database, slug=database_slug)
    database_photo = get_object_or_404(models.DatabasePhoto, database=database, pk=pk)

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
    event = get_object_or_404(models.Event, slug=event_slug)
    event_photo = get_object_or_404(models.EventPhoto, event=event, pk=pk)
    return HttpResponse('AAA event_photo' + '--' + event_slug + '--' + pk)


def event_list_view(request):
    return HttpResponse('EVENT LIST')


def database_index_list_view(request):
    return HttpResponse('DATABASE INDEX LIST')
