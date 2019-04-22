import logging

from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import ListView, DetailView

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


def event_detail_view(request, slug):
    RESULT_PHOTOS_LIMIT = 10
    context = {}
    event = get_object_or_404(models.Event, slug=slug)
    context['event'] = event

    if not event.has_cbir_index():
        is_cbir_index_set = event.set_default_cbir_index_and_return_whether_success()
        if not is_cbir_index_set:
            context['warning_message'] = (f'Photos in the database {event.database} have not been indexed yet. '
                                          f'First, go and create search index')
            logger.info(f'event.database: {event.database}')
            logger.info(f'event.database.slug: {event.database.slug}')
            return render(request, 'photologue/event_detail.html', context)

    result_photos = event.init_if_needed_and_get_result_photos()
    result_photos_truncated = result_photos[:RESULT_PHOTOS_LIMIT]

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
            event.cbir_index = form.cleaned_data.get('cbir_index')
            if event.cbir_index.database != database:
                # TODO: Fine that user's cbir_index is ignored silently? User is not told about it.
                logger.warning(f'User chose incorrect cbir_index connected to database {event.cbir_index.database}. '
                               f'But intends to create an event for {database} database')
                event.cbir_index = database.cbir_index_default

            if not event.cbir_index:
                event.cbir_index = database.cbir_index_default
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
        form = forms.EventForm(initial={'cbir_index': database.cbir_index_default})

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
    return render(request, 'photologue/database_index_detail.html', context)


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

            # TODO: Make call to build index asynchronous
            cbir_index.build_if_needed()

            cbir_index.save()

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
    LIMIT = 10
    photo = get_object_or_404(models.DatabasePhoto, slug=slug)

    photo.eventphoto_set.all()
    events = [event_photo.event for event_photo in photo.eventphoto_set.all()]
    queries = [None] * len(events)

    context = {}
    context['photo'] = photo
    context['events_queries_pairs'] = list(zip(events, queries))

    return render(request, 'photologue/database_photo_detail.html', context)


def event_photo_detail_view(requst, event_slug, pk):
    return HttpResponse('AAA event_photo' + event_slug + str(pk))
