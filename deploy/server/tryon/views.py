from django.shortcuts import render, HttpResponse
from django.urls import reverse_lazy
from django.views.generic import CreateView, ListView
from tryon.models import Tryon
from tryon.forms import TryonForm
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath('model'))
from virtuon import virtuon
from clear import clear

# Create your views here.

class TryonView(CreateView):
    model = Tryon
    template = "home.html"


    def get(self, request):
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
        # clear()
        # virtuon()
        base_image = Image.open("model/output/p_rendered/demo_1.jpg")
        response = HttpResponse(content_type="image/png")
        base_image.save(response, "PNG")
        return response
        


    