from django.db import models
from django import forms
from django.forms import ModelForm

from .models import Database


class DatabaseForm(ModelForm):
    lol_field = forms.CharField(required=True, label='Label for lol field', initial='lol-initial')

    class Meta:
        model = Database
        fields = ['date_added', 'title', 'slug', 'description']
