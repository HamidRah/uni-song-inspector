import paypalrestsdk
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.shortcuts import redirect, render
from django.urls import reverse
from .models import Action, UserTokenCount
from .views import get_log_data, create_log
# Create a payment that can be made via the PayPal API
# Create a payment that can be made via the PayPal API
def create_payment(request):
    # Configure PayPal SDK
    paypalrestsdk.configure({
        "mode": settings.PAYPAL_MODE,
        "client_id": settings.PAYPAL_CLIENT_ID,
        "client_secret": settings.PAYPAL_CLIENT_SECRET
    })

    # Create payment object
    payment = paypalrestsdk.Payment({
        "intent": "sale",
        "payer": {
            "payment_method": "paypal",
        },
        "redirect_urls": {
            "return_url": request.build_absolute_uri(reverse('execute_payment')),
            "cancel_url": request.build_absolute_uri(reverse('payment_cancelled')),
        },
        "transactions" : [{
            "item_list" : {
                "items" : [{
                    "name": "Test item",
                    "sku": "test item",
                    "price": "9.99",
                    "currency": "GBP",
                    "quantity": 1,
                }]
            },
            "amount" : {
                "total": "9.99",
                "currency": "GBP"
            },
            "description": "Test payment description"
        }]
    })

    # Successfully communicated with API
    if payment.create():
        print("Payment created successfully!")
        # get url for payment approval
        for link in payment.links:
            if link.rel == "approval_url":
                # turn link into text
                approval_url = str(link.href)
                # send on merry way
                return redirect(approval_url)
    else:
        print(payment.error)


# Execute a successful payment
def execute_payment(request):
    # Get payment id and payer id
    payment_id = request.GET.get('paymentId')
    payer_id = request.GET.get('PayerID')

    #If neither ID, error, restart
    if not payment_id or not payer_id:
        print("no payment")
        #TODO: Change this to a more appropriate path, maybe a generic error page that takes a string:Error to display in a template
        return redirect('handler404')

    # configure API
    paypalrestsdk.configure({
        "mode": settings.PAYPAL_MODE,
        "client_id": settings.PAYPAL_CLIENT_ID,
        "client_secret": settings.PAYPAL_CLIENT_SECRET
    })

    # Check we've got a successful payment
    payment = paypalrestsdk.Payment.find(payment_id)

    # If it we do an the payer IDs match
    if payment.execute({"payer_id": payer_id}):
        print("Payment executed successfully!")

        # Allocate some tokens
        user = request.user
        tokens_purchased = 1

        # increment user_tokens
        # commit changes
        # TODO: Change something here such that the token amount added depends on a detail of the transaction,
        #       i.e. £9.99 payment for one token or £24.99 for
        #
        if request.user.is_authenticated:
            add_tokens(user, tokens_purchased)
            return redirect('success')
        else:
            return redirect('handler404')
    else:
        #TODO: Change this to a more appropriate error message
        print("exiting at the end of execute_payment()")
        return redirect('handler404')


def add_tokens(user, tokens):
    token_count_instance, created = UserTokenCount.objects.get_or_create(user=user)
    token_count_instance.token_count += tokens
    token_count_instance.save()

def payment_cancelled(request):
    return render(request, 'payment_cancelled.html')

def payment_success(request):
    log_data = get_log_data(request.user, Action.PAYMENT_SUCCESSFUL, 'success')
    create_log(request.user, log_data)
    return render(request,'payment_success.html')
