from django import forms


class DisplayForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea(attrs={'class': 'md-textarea form-control',
                                                        'type': 'text', 'id': 'form10', 'rows': '12'}))
