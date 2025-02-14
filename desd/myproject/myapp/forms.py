from django import forms
from django.contrib.auth.forms import UserCreationForm 
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm

from .models import Profile

# class CustomRegistrationForm(UserCreationForm):
#     #UserCreationForm comes with username, password1, password2 by default
#     #only email needs to be added for our custom users
#     email = forms.EmailField()

#     class Meta(UserCreationForm.Meta):
#         model = User
#         fields = ["username", "email", "password1", "password2"]

# class LoginForm(forms.Form):
#     username = forms.CharField()
#     password = forms.CharField(widget=forms.PasswordInput)

class InstrumentDetectionForm(forms.Form):
    audio_file = forms.FileField(
        label='Select an audio file',
        widget=forms.ClearableFileInput(attrs={
            'class': 'block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400',
            'id': 'audio_file',
            'name': 'audio_file',
            'accept': '.mp3,.wav',
            'onchange': 'loadAudioFile(event)'
        })
    )

class LoginAuthenticationForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super(LoginAuthenticationForm, self).__init__(*args, **kwargs)
        for visible in self.visible_fields():
            visible.field.widget.attrs['class'] = 'block w-full px-3 py-2 mt-1 text-gray-700 bg-white border border-gray-300 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500'


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={
        'class': 'block w-full px-4 py-2 mt-2 text-gray-700 bg-white border border-red-300 rounded-md dark:bg-gray-800 dark:text-gray-300 dark:border-red-600 focus:border-red-500 dark:focus:border-red-500 focus:outline-none focus:ring focus:ring-red-500',
        'placeholder': 'Email'
    }))

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
    
    def __init__(self, *args, **kwargs):
        super(UserRegisterForm, self).__init__(*args, **kwargs)
        common_attrs = {
            'class': 'block w-full px-4 py-2 mt-2 text-gray-700 bg-white border border-red-300 rounded-md dark:bg-gray-800 dark:text-gray-300 dark:border-red-600 focus:border-red-500 dark:focus:border-red-500 focus:outline-none focus:ring focus:ring-red-500',
            'placeholder': ''
        }
        self.fields['username'].widget.attrs.update({**common_attrs, 'placeholder': 'Username'})
        self.fields['password1'].widget.attrs.update({**common_attrs, 'placeholder': 'Password'})
        self.fields['password2'].widget.attrs.update({**common_attrs, 'placeholder': 'Repeat Password'})



class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['user_type']
        widgets = {
            'user_type': forms.Select(attrs={
                'class': 'block w-full px-3 py-1.5 text-base font-normal text-gray-700 bg-white bg-clip-padding bg-no-repeat border border-solid border-gray-300 rounded transition ease-in-out m-0 focus:text-gray-700 focus:bg-white focus:border-blue-600 focus:outline-none',
            }),
        }
