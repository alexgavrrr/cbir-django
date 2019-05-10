import zipfile
from zipfile import BadZipFile

from django import forms
from django.forms import ModelForm

from .models import (Database,
                     Event,
                     DatabasePhoto,
                     CBIRIndex, )

FILE_MAX_LENGTH = 100

file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))


class DatabaseForm(ModelForm):
    photos = forms.ImageField(required=False,
                              label='photos',
                              widget=forms.ClearableFileInput(attrs={'multiple': True}))
    zip_file = forms.FileField(required=False)

    class Meta:
        model = Database
        fields = ['date_added', 'title', 'slug', 'description']

    def clean_zip_file(self):
        """Open the zip file a first time, to check that it is a valid zip archive.
        We'll open it again in a moment, so we have some duplication, but let's focus
        on keeping the code easier to read!
        """
        zip_file = self.cleaned_data['zip_file']
        if not zip_file:
            return zip_file
        try:
            zip = zipfile.ZipFile(zip_file)
        except BadZipFile as e:
            raise forms.ValidationError(str(e))
        bad_file = zip.testzip()
        if bad_file:
            zip.close()
            raise forms.ValidationError('"%s" in the .zip archive is corrupt.' % bad_file)
        zip.close()  # Close file in all cases.
        return zip_file


class EventForm(ModelForm):
    query_photos = forms.ImageField(required=False,
                                    label='query photos',
                                    widget=forms.ClearableFileInput(attrs={'multiple': True}))
    query_photos_from_database = forms.ModelMultipleChoiceField(DatabasePhoto.objects.all(),
                                                                required=False,
                                                                label='query photos from database')
    qe = forms.BooleanField(required=False, initial=False, label='qe',
                            help_text='Query expansion')
    sv = forms.BooleanField(required=False, initial=False, label='sv',
                            help_text='Rerank by spatial verification')
    topk = forms.IntegerField(required=False, initial=10, label='topk',
                              help_text='Maximum count resulting photos')
    similarity_threshold = forms.IntegerField(required=False, initial=None, label='similarity_threshold',
                                              help_text='Photos with similarity value below than threshold are not returned. '
                                                        'Default is none which means -inf.')

    class Meta:
        model = Event
        fields = ['date_added', 'title', 'slug', 'description', 'database', 'cbir_index']

    def clean(self):
        cleaned_data = super().clean()
        database = cleaned_data.get('database')

        if not database:
            return cleaned_data
        else:
            cleaned_data['cbir_index'] = cleaned_data['cbir_index'] or database.cbir_index_default
            cbir_index = cleaned_data['cbir_index']

            if not cbir_index:
                self.add_error('cbir_index', f'There are no indexes for database {database}. Firstly, create index.')
            elif cbir_index.database != database:
                self.add_error('cbir_index', f'Chosen cbir_index {cbir_index} corresponds to another database')

            return cleaned_data


class CbirIndexForm(ModelForm):
    set_as_default = forms.BooleanField(required=False,
                                        initial=False, )
    des_type = forms.ChoiceField(
        choices=(
            ('HardNetAll', 'HardNetAll'),
            ('l2net', 'l2net'),
            ('sift', 'sift'),
            (None, 'default')
        ),
        required=False
    )

    class Meta:
        model = CBIRIndex
        fields = ['date_added', 'title', 'slug', 'name', 'description', 'database']


class EventDetailBasketForm(ModelForm):
    pass
