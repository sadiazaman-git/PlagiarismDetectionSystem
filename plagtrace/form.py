from django import forms
from .models import Contact
from django.contrib.auth.models import User
from django.contrib.auth import authenticate


class ContactForm(forms.ModelForm):
    username = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'form-control', 'id': 'form-name', 'type': 'text'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control', 'id': 'materialFormRegisterPasswordEx', 'type': 'password'}))
    subject = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'form-control', 'id': 'form-Subject', 'type': 'text'}))
    description = forms.CharField(widget=forms.Textarea(attrs={
        'class': 'form-control md-textarea', 'id': 'form-text', 'type': 'text', 'rows': '3'}))

    class Meta:
        model = Contact
        fields = ('username', 'password', 'subject', 'description')

    def clean(self):
        try:
            username = User.objects.get(username__iexact=self.cleaned_data['username']).username
        except User.DoesNotExist:
            raise forms.ValidationError("No such username registered")
        password = self.cleaned_data['password']

        self.user = authenticate(username=username, password=password)
        if self.user is None or not self.user.is_active:
            raise forms.ValidationError("username or password is incorrect")
        return self.cleaned_data
