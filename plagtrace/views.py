from django.shortcuts import render, HttpResponseRedirect , HttpResponse, redirect,render_to_response
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.generic import View
from plagtrace.register_form import UserForm
from plagtrace.file_form import DocumentForm
from plagtrace.form import ContactForm
from django.conf import settings
from .models import Document
from .fyp import plag
from plagtrace.display_form import DisplayForm
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pylab
from django.contrib.auth.decorators import login_required
from django.template import RequestContext
import PIL, PIL.Image
import io
from io import *


def handle_uploaded_file(f):
    with open(settings.MEDIA_ROOT,f, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def upload_file(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            content_file = submitted()
            return render(request, 'plagtrace/document_form.html', {'form':form}, {'content_file':content_file})
    else:
        form = DocumentForm()
    return render(request, 'plagtrace/document_form.html', {'form': form})


def index(request):
    return render(request, 'plagtrace/index.html')


def blog(request):
    return render(request, 'plagtrace/blog.html')


def features(request):
    return render(request, 'plagtrace/features.html')


def contact_form(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save(commit=False)
            username = request.POST.get('username', '')
            password = request.POST.get('password', '')
            user = authenticate(username=username, password=password)
            form.save()
            if user is not None:
                if user.is_active:
                    login(request, user)
                    return redirect('plagtrace:contact_form')

        return render(request,'plagtrace/contact.html',{'form': form, 'errorMessage': 'UserName or Password is incorrect'})
    else:
        form = ContactForm()
        return render(request, 'plagtrace/contact.html', {'form': form})

def submitted():
    obj = Document.objects.values_list('file1').latest('id')
    obj = ' '.join(obj)
    obj1 = Document.objects.values_list('file2').latest('id')
    obj1 = ' '.join(obj1)
    plg = plag(obj, obj1)
    return plg


class UserFormView(View):

    form_class = UserForm
    template_name = 'plagtrace/registration_form.html'
    # display blank form

    def get(self, request):
        form = self.form_class(None)
        return render(request, self.template_name, {'form': form})
    # process the data

    def post(self, request):
        form = self.form_class(request.POST)

        if form.is_valid():

            user = form.save(commit=False)
            # cleaned (normalize) data
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user.set_password(password)
            user.save()
            # return User objects if credentials are correct
            user = authenticate(username=username, email=email, password=password)
            if user is not None:

                if user.is_active:
                    login(request, user)
                    return redirect('plagtrace:index')

        return render(request, self.template_name, {'form': form})


def login_user(request):
    context = RequestContext(request)
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return redirect('plagtrace:upload_file')
        else:
            print("invalid login details " + username + " " + password)
            return render_to_response('plagtrace/registration/login.html', {}, context)
    else:
        return render(request, 'plagtrace/registration/login.html',{}, context)


@login_required(login_url='/login/')


def Graph_Loader():
    file_content = submitted()
    context =[]
    for content in file_content:
        context.append(content)
    plg = context[0]
    clean = context[1]
    objects = ('plagiarized', 'Clean')
    width = 0.20
    pylab.ylim([0, 100])
    performance = [plg, clean]
    y_pos = np.arange(len(performance))
    performance = [plag, clean]
    plt.bar(y_pos, performance, align='edge', width=width, alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.yticks(range(0, max(performance)))
    plt.title('Plagiarized vs Clean')
    plt.ylabel('Percentage')
    pylab.grid(True)
    return plt.show()
    '''buffer = io.BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pil_image.save(buffer, "PNG")
    pylab.close()
    return buffer.getvalue() '''





'''
def submitted(request):
    if request.method == 'POST':
        form = DisplayForm(request.POST)
        if form.is_valid():
            form.text("hello")
        return HttpResponseRedirect('submitted')
    else:
        form = DisplayForm()
    return render(request, 'plagtrace/document_form.html', {'form': form})





            for field in request.FILES.keys():
                for form_file in request.FILES.getlist(field):
                    save_uploaded_file_to_media_root(form_file)



def save_uploaded_file_to_media_root(f):
    with open('%s%s' % (settings.MEDIA_ROOT, f.name), 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


'''
