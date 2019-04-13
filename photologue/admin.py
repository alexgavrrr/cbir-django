from django.contrib import admin

from . import models

admin.site.register(models.Database)
admin.site.register(models.Event)
