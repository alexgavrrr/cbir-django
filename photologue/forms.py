from django.db import models
from django import forms
from django.forms import ModelForm

from .models import Database, DatabasePhoto

FILE_MAX_LENGTH = 100

file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

class DatabaseForm(ModelForm):
    lol_field = forms.CharField(required=False,
                                label='Label for lol field',
                                initial='lol-initial')
    photos = forms.ImageField(required=False,
                              label='photos',
                              widget=forms.ClearableFileInput(attrs={'multiple': True}))

    class Meta:
        model = Database
        fields = ['date_added', 'title', 'slug', 'description', 'description_file']
        labels = {'description_file': 'Description file'}
