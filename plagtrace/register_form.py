from django.contrib.auth.models import User
from django import forms


class UserForm(forms.ModelForm):
    username = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'form-control', 'id': 'materialFormRegisterNameEx', 'type': 'text'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={
        'class': 'form-control', 'id': 'materialFormRegisterEmailEx', 'type': 'email'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control', 'id': 'materialFormRegisterPasswordEx', 'type': 'password'}))

    class Meta:
        model = User
        fields = ['username','email','password']

    def clean_email(self):
        # Get the email
        email = self.cleaned_data.get('email')

        # Check to see if any users already exist with this email as a username.
        try:
            User.objects.get(email=email)
        except User.DoesNotExist:
            # Unable to find a user, this is fine
            return email

        # A user was found with this as a username, raise an error.
        raise forms.ValidationError('This email address is already in use.')
