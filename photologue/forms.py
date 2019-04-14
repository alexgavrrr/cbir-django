from django import forms
from django.forms import ModelForm

from .models import Database, Event, DatabasePhoto

FILE_MAX_LENGTH = 100

file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))


class DatabaseForm(ModelForm):
    photos = forms.ImageField(required=False,
                              label='photos',
                              widget=forms.ClearableFileInput(attrs={'multiple': True}))

    class Meta:
        model = Database
        fields = ['date_added', 'title', 'slug', 'description']


class EventForm(ModelForm):
    query_photos = forms.ImageField(required=False,
                              label='query photos',
                              widget=forms.ClearableFileInput(attrs={'multiple': True}))
    query_photos_from_database = forms.ModelMultipleChoiceField(DatabasePhoto.objects.all(),
                                                                required=False,
                                                                label='query photos from database')

    class Meta:
        model = Event
        fields = ['date_added', 'title', 'slug', 'description']
