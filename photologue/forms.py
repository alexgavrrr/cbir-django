from django.db import models
from django import forms
from django.forms import ModelForm

from .models import Database, DatabasePhoto

FILE_MAX_LENGTH = 100



class DatabaseForm(ModelForm):
    lol_field = forms.CharField(required=False,
                                label='Label for lol field',
                                initial='lol-initial')
    photos = forms.ImageField(required=False,
                              label='photos',)

    class Meta:
        model = Database
        fields = ['date_added', 'title', 'slug', 'description', 'description_file']
        labels = {'description_file': 'Description file'}


class ImageForm(forms.ModelForm):
    image = forms.ImageField(label='Image')
    class Meta:
        model = DatabasePhoto
        fields = ('image', )
