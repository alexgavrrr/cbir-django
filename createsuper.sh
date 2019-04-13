#!/usr/bin/env bash

# python manage.py makemigrations && python manage.py migrate && ./createsuper.sh
echo "from django.contrib.auth.models import User; User.objects.create_superuser('admin', '', 'admin')" | python manage.py shell
