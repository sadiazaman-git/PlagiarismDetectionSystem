from django import forms
from .models import Document


class DocumentForm(forms.ModelForm):

    file1 = forms.FileField(widget=forms.FileInput(attrs={'type': 'file', }))
    file2 = forms.FileField(widget=forms.FileInput(attrs={'type': 'file', }))

    class Meta:
        model = Document
        fields = ('file1', 'file2')
