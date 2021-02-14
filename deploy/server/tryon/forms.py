from django import forms
from tryon.models import Tryon

class TryonForm(forms.ModelForm):
    class Meta:
        model = Tryon
        fields = ['pose', 'cloth']