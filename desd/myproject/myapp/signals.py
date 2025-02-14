from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import Profile, UserTokenCount

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        UserTokenCount.objects.get_or_create(user=instance)
        Profile.objects.get_or_create(user=instance)
    instance.profile.save()