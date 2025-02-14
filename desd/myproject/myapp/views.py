from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse
from django.utils import timezone
from django.shortcuts import render, redirect, get_object_or_404
from django.template import RequestContext
import logging
from reportlab.pdfgen import canvas
import json
from datetime import datetime

from .forms import InstrumentDetectionForm
from .models import Log, Action, User, UserTokenCount, Profile
from django.http import JsonResponse
from django.db import connection

# Django Rest Framework imports
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import InstrumentDetectionSerializer
from .audio_preprocessing import preprocess_audio_for_inference
import requests

# Authentication Imports
from django.urls import reverse_lazy
from django.views import generic
from .models import Profile
from .forms import UserRegisterForm, LoginAuthenticationForm
from django.contrib.auth.views import LoginView
from django.views.decorators.csrf import csrf_exempt

from django.contrib.auth.mixins import UserPassesTestMixin
from django.views.generic import TemplateView
import re

logger = logging.getLogger(__name__)

def get_log_data(user, action, status='success', file=None, description=None, feedback=None):
    log_data = {
        'username': user.username,
        'action': action.value.format(username=user.username),
        'status': status,
        'file': file,
        'description': description,
        'feedback': feedback,  # Add the feedback field
    }
    return log_data

def create_log(user, log_data):
    Log.objects.create(user=user, log=log_data, feedback=log_data.get('feedback'))

def handling_music_file(request):
    if request.method == 'POST':
        if 'audio_file' in request.FILES:
            log_data = {
                'action': 'File uploaded',
                'file': request.FILES['audio_file'].name,
            }
            log_data = get_log_data(request.user ,Action.UPLOAD_FILE, 'success', file=request.FILES['audio_file'].name)
            create_log(request.user if request.user.is_authenticated else None, log_data)
            return HttpResponse('File uploaded successfully!',log_data)
    log_data = get_log_data(request.user ,Action.INVALID_FILE, 'error')
    create_log(None, log_data)
    return HttpResponse('File invalid',log_data)

@csrf_exempt
def log_fileupload(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        status = data.get('status')
        file = data.get('file')

        if request.user.is_authenticated:
            log_data = get_log_data(request.user, Action.UPLOAD_FILE, status, file)
            create_log(request.user, log_data)

        return JsonResponse({'message': 'Log created successfully'}, status=201)

    return JsonResponse({'error': 'Invalid request'}, status=400)

def submit_feedback(request):
    if request.method == 'POST' and request.user.is_authenticated:
        prediction = request.POST.get('prediction')
        liked = request.POST.get('feedback') == 'true'
        file_name = request.POST.get('file_name')  # Get the filename from the form data
        
        # Create log data using the get_log_data function
        log_data = get_log_data(
            user=request.user,
            action=Action.FEEDBACK_SUBMITTED,
            status='success',
            file=file_name,  # Use the filename obtained from the form
            description=prediction,
            feedback=liked
        )
        
        # Create the Log entry using the create_log function
        create_log(request.user, log_data)
        
        return redirect('index')
    
    return redirect('index')

def admin_table(request):
    # Execute the query and fetch all rows
    query = """SELECT date, log, user_id, feedback FROM myapp_log ORDER BY date DESC"""
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()

    # Create a list of dictionaries from the query results
    data = []
    for row in rows:
        # Parse the JSON string into a dictionary
        log = json.loads(row[1])
        # Get the user object based on the user_id
        user_id = row[2]
        # Get the feedback value
        feedback = row[3]
        # Create a dictionary with the date, user, JSON fields, and feedback
        date = row[0].strftime('%Y-%m-%d %H:%M:%S')
        entry = {'date': date, 'user': user_id, 'file': log['file'], 'action': log['action'], 'status': log['status'],
                 'description': log['description'], 'feedback': feedback}
        data.append(entry)

    # Return the data as a JSON response
    return JsonResponse({'data': data}, safe=False)

def user_table(request):
    user_id = request.user.id
    # Only display user logs code below
    query = """SELECT date, log, user_id, feedback FROM myapp_log WHERE user_id = {} ORDER BY date DESC""".format(user_id)
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()

    # Create a list of dictionaries from the query results
    data = []
    for row in rows:
        # Parse the JSON string into a dictionary
        log = json.loads(row[1])
        # Get the user object based on the user_id
        user_id = row[2]
        # Get the feedback value
        feedback = row[3]
        # Create a dictionary with the date, user, JSON fields, and feedback
        date = row[0].strftime('%Y-%m-%d %H:%M:%S')
        entry = {'date': date, 'user': user_id, 'file': log['file'], 'action': log['action'], 'status': log['status'],
                 'description': log['description'], 'feedback': feedback}
        data.append(entry)

    # Return the data as a JSON response
    return JsonResponse({'data': data}, safe=False)

def index(request):
    # Initialize default context
    context = {'form': InstrumentDetectionForm(), 'predictions': [], 'file_name': None}

    # Handle authenticated users
    if request.user.is_authenticated:
        token_count = UserTokenCount.objects.get(user=request.user).token_count
        context['token_count'] = token_count

        if request.method == 'POST':
            form = InstrumentDetectionForm(request.POST, request.FILES)
            if form.is_valid() and 'audio_file' in request.FILES:
                uploaded_file = request.FILES['audio_file']
                context['file_name'] = uploaded_file.name

                # Make a request to the InstrumentDetectionView to get the predictions
                view = InstrumentDetectionView().as_view()
                response = view(request)

                # Ensure there's a response and it contains predictions before updating context
                if response and hasattr(response, 'data') and 'predictions' in response.data:
                    context['predictions'] = response.data['predictions']

                    if request.user.is_authenticated:
                        feedback = request.POST.get('feedback')  # Get the user's feedback from the form
                        predictions_string = ', '.join(response.data['predictions'])
                        log_data = get_log_data(request.user, Action.RUN_ALGORITHM, 'success', file=uploaded_file.name, 
                                                description=predictions_string, feedback=feedback)
                        create_log(request.user, log_data)
            else:
                context['form'] = form
    else:
        if request.method == 'POST':
            # Assuming you want to do something specific with the file if it's uploaded
            uploaded_file = request.FILES.get('audio_file')
            if uploaded_file:
                # Process the uploaded file as needed
                pass

    # For GET requests or if form is not valid, render the page with the default or updated context
    if request.user.is_authenticated:
        return render(request, 'index1.html', context)
    else:
        return render(request, 'index2.html')





def users(request):
    if request.user.is_authenticated:
        # Make a request to the admin_table view to get the data
        context = {}
        data_admin = admin_table(request)
        data_user = user_table(request)
        admin_dict = json.loads(data_admin.content)
        user_dict = json.loads(data_user.content)
        token_count = UserTokenCount.objects.get(user=request.user).token_count
        user_profile = request.user.profile
        user = request.user
        all_user_profiles = Profile.objects.all()  # Retrieve all Profile objects

        # Pass the data as a context variable to the template
        # !!! ADMIN DATA ONLY DISPLAYED AND GET IF USER IS ADMIN !!!
        context['admin_data'] = admin_dict['data']
        context['user_data'] = user_dict['data']
        context['token_count'] = token_count
        context['user_profile'] = user_profile
        context['user'] = user
        context['all_user_profiles'] = all_user_profiles  # Add all_user_profiles to the context

        return render(request, 'user_page.html', context)
    return redirect('login')

def handler404(request, *args, **kwargs):
    response = render(request, '404.html', {})
    response.status_code = 404
    return response

def handler500(request, *args, **kwargs):
    response = render(request, '500.html', {})
    response.status_code = 500
    return response

def maintenance(request):
    return render(request, 'maintenance.html')


# Authentication
class RegisterView(generic.CreateView):
    form_class = UserRegisterForm
    success_url = reverse_lazy('index')
    template_name = 'registration/register.html'

    def form_valid(self, form):
        response = super().form_valid(form)
        user = self.object  # Grab the user instance

        # Ensure the user is active; this line might be redundant if you're sure users are active by default
        user.is_active = True
        user.save()

        # Check if the Profile exists, and if not, create it
        if not Profile.objects.filter(user=user).exists():
            Profile.objects.create(user=user, user_type=0)  # Default user type as Basic User

        # Log the user in
        login(self.request, user)

        return response


class CustomLoginView(LoginView):
    authentication_form = LoginAuthenticationForm
    template_name = 'registration/login.html'

    def form_valid(self, form):
        # Create log if user is authenticated
        login(self.request, form.get_user())

        log_data = get_log_data(form.get_user(), Action.LOGIN, 'success')
        create_log(form.get_user(), log_data)

        return super().form_valid(form)


def terms_conditions(request):
    return render(request, 'terms_conditions.html')

def privacy_policy(request):
    return render(request, 'privacy_policy.html')

def pricing(request):
    return render(request, 'pricing.html')

#For testing the receipts ONLY. TODO: delete when working
def generate_pdf(request):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="example.pdf"'

    p = canvas.Canvas(response)
    p.drawString(100, 800, "Hello, this is a PDF!")
    p.showPage()
    p.save()

    return response

# Running the audio file through the model
class InstrumentDetectionView(APIView):
    def post(self, request):
        # Get the user's token count
        user_token_count = UserTokenCount.objects.get(user=request.user)

        # Check if the user has more than one token
        if user_token_count.token_count < 1:
            return Response({'error': 'Insufficient tokens'}, status=status.HTTP_403_FORBIDDEN)

        # Decrease the user's token count by one
        user_token_count.token_count -= 1
        user_token_count.save()

        serializer = InstrumentDetectionSerializer(data=request.data)
        if serializer.is_valid():
            audio_file = serializer.validated_data['audio_file']

            # Save the uploaded file temporarily
            # with open('temp_audio.wav', 'wb') as f:
            #     f.write(audio_file.read())

            # Preprocess the audio file
            preprocessed_data = preprocess_audio_for_inference(audio_file)

            # Prepare data for TensorFlow Serving
            data = json.dumps({"signature_name": "serving_default", "instances": [window.tolist() for window in preprocessed_data]})

            # Send request to TensorFlow Serving
            url = 'http://tensorflow_serving:8501/v1/models/instrument_model/versions/2:predict'
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, data=data, headers=headers)

            # Process the response
            if response.status_code == 200:
                raw_predictions = response.json()['predictions']
                # Convert raw prediction numbers into percentages
                formatted_predictions = self.format_predictions(raw_predictions)
                return Response({"predictions": formatted_predictions}, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Failed to get predictions"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


    def format_predictions(self, predictions):
        instruments = ['Guitar', 'Drum', 'Violin', 'Piano']
        instrument_windows = {instrument: [] for instrument in instruments}

        for window_index, prediction in enumerate(predictions, start=1):
            highest_score_index = prediction.index(max(prediction))
            highest_score_instrument = instruments[highest_score_index]
            instrument_windows[highest_score_instrument].append(window_index)

        formatted_predictions = []
        for instrument, windows in instrument_windows.items():
            if windows:
                window_list = ', '.join(map(str, windows))
                formatted_predictions.append(f"{instrument} - Windows: {window_list}")

        return formatted_predictions
    



class ModelPerformanceView(UserPassesTestMixin, TemplateView):
    template_name = 'model_performance.html'

    def test_func(self):
        return self.request.user.is_authenticated and (self.request.user.is_superuser or self.request.user.profile.user_type == 2)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        metrics_url = 'http://tensorflow_serving:8501/monitoring/prometheus/metrics'
        response = requests.get(metrics_url)

        if response.status_code == 200:
            metrics_data = response.text

            # Extract metrics using regular expressions
            request_count = re.search(r':tensorflow:serving:request_count{model_name="instrument_model",status="OK"} (\d+)', metrics_data)
            request_latency_sum = re.search(r':tensorflow:serving:request_latency_sum{model_name="instrument_model",API="predict",entrypoint="REST"} ([\d\.e\+]+)', metrics_data)
            request_latency_count = re.search(r':tensorflow:serving:request_latency_count{model_name="instrument_model",API="predict",entrypoint="REST"} (\d+)', metrics_data)
            runtime_latency_sum = re.search(r':tensorflow:serving:runtime_latency_sum{model_name="instrument_model",API="Predict",runtime="TF1"} ([\d\.e\+]+)', metrics_data)
            runtime_latency_count = re.search(r':tensorflow:serving:runtime_latency_count{model_name="instrument_model",API="Predict",runtime="TF1"} (\d+)', metrics_data)
            model_load_latency = re.search(r':tensorflow:cc:saved_model:load_latency{model_path="/models/instrument_model/2"} (\d+)', metrics_data)

            # Calculate average latencies in seconds
            avg_request_latency = float(request_latency_sum.group(1)) / float(request_latency_count.group(1)) / 1e6 if request_latency_sum and request_latency_count else None
            avg_runtime_latency = float(runtime_latency_sum.group(1)) / float(runtime_latency_count.group(1)) / 1e6 if runtime_latency_sum and runtime_latency_count else None
            model_load_latency_seconds = float(model_load_latency.group(1)) / 1e6 if model_load_latency else None

            context['metrics'] = {
                'request_count': request_count.group(1) if request_count else None,
                'avg_request_latency': avg_request_latency,
                'avg_runtime_latency': avg_runtime_latency,
                'model_load_latency': model_load_latency_seconds
            }
        else:
            context['metrics'] = None

        return context

def change_user_type(request, user_id):
    if request.method == 'POST':
        user_type = request.POST.get('user_type')
        user_profile = get_object_or_404(Profile, user__id=user_id)  # Get the user profile
        user_profile.user_type = user_type
        user_profile.save()
        return redirect('users')  # Redirect to the users page

def user_has_credits():
    has_credits = False



    return has_credits
