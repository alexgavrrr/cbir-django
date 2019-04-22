from django.contrib import admin

from . import models
from .models import (PhotoEffect, PhotoSize, Watermark)

admin.site.register(models.Database)
admin.site.register(models.Event)
admin.site.register(models.DatabasePhoto)
admin.site.register(models.EventPhoto)
admin.site.register(models.CBIRIndex)


class PhotoEffectAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'color', 'brightness',
                    'contrast', 'sharpness', 'filters', 'admin_sample')
    fieldsets = (
        (None, {
            'fields': ('name', 'description')
        }),
        ('Adjustments', {
            'fields': ('color', 'brightness', 'contrast', 'sharpness')
        }),
        ('Filters', {
            'fields': ('filters',)
        }),
        ('Reflection', {
            'fields': ('reflection_size', 'reflection_strength', 'background_color')
        }),
        ('Transpose', {
            'fields': ('transpose_method',)
        }),
    )


admin.site.register(PhotoEffect, PhotoEffectAdmin)


class PhotoSizeAdmin(admin.ModelAdmin):
    list_display = ('name', 'width', 'height', 'crop', 'pre_cache', 'effect', 'increment_count')
    fieldsets = (
        (None, {
            'fields': ('name', 'width', 'height', 'quality')
        }),
        ('Options', {
            'fields': ('upscale', 'crop', 'pre_cache', 'increment_count')
        }),
        ('Enhancements', {
            'fields': ('effect', 'watermark',)
        }),
    )


admin.site.register(PhotoSize, PhotoSizeAdmin)


class WatermarkAdmin(admin.ModelAdmin):
    list_display = ('name', 'opacity', 'style')


admin.site.register(Watermark, WatermarkAdmin)
