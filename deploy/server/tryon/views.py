from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views.generic import CreateView, ListView
from django.template import Template
from tryon.models import Tryon
from tryon.forms import TryonForm
import os
import sys
sys.path.append(os.path.abspath('model'))
from virtuon import virtuon
from clear import clear
from pairs import pairs

# Create your views here.

class TryonView(CreateView):
    model = Tryon
    template = "index.html"
    success_url = "predict.html"


    def get(self, request):
        clear()
        form = TryonForm()
        ctx = {'form': form}
        return render(request, self.template, ctx)

    def post(self, request):
        form = TryonForm(request.POST, request.FILES or None)
        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template, ctx)
        form.save()
        return render(request, self.template)

class TryonPredict(ListView):
    template = "predict.html"

    def get(self, request):
        pairs()
        virtuon()
        output = ("output/d0.jpg")
        ctx = {"output": output}

        return render(request, self.template, ctx)