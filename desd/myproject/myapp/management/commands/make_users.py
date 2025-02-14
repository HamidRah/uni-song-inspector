# app/management/commands/create_users.py
from django.core.management.base import BaseCommand
from django.contrib.auth.hashers import make_password
from django.contrib.auth import get_user_model

class Command(BaseCommand):
    def handle(self, *args, **options):
        Profile = get_user_model()
        if not Profile.objects.filter(username="superuser").exists():
            Profile.objects.create_superuser("superuser", password="placeholder")

        if not Profile.objects.filter(username="admin").exists():
            Profile.objects.create_user("admin", password="placeholder")
        
        if not Profile.objects.filter(username="base_user").exists():
            Profile.objects.create_user("base_user", password="placeholder")
        
        if not Profile.objects.filter(username="ml_engineer").exists():
            Profile.objects.create_user("ml_engineer", password="placeholder")
        
        if not Profile.objects.filter(username="accountant").exists():
            Profile.objects.create_user("accountant", password="placeholder")