from django.shortcuts import render, HttpResponse
from django.urls import reverse_lazy
from django.views.generic import CreateView, ListView
from django.template import Template
from tryon.models import Tryon
from tryon.forms import TryonForm
from PIL import Image
import sys
import os
import cv2
import base64
from django.template import RequestContext

sys.path.append(os.path.abspath('model'))
from virtuon import virtuon
from clear import clear
from pairs import pairs

# Create your views here.

class TryonView(CreateView):
    model = Tryon
    template = "home.html"
    success_url = "predict.html"


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
        clear()
        pairs()
        virtuon()
        # # base_image = Image.open("model/output/p_rendered/demo_1.jpg")
        # # response = HttpResponse(content_type="image/png")
        # # base_image.save(response, "PNG")
        output = ("output/d0.jpg")
        # template = Template('{{MEDIA_URL}}:{{output}}')
        ctx = {"output": output}
         # img_path = ('model/output/p_rendered/demo_1.jpg')
        # # img = cv2.imread(img_path)
        # # _, img_encoded = cv2.imencode('.jpg', img)
        # # ctx = {"output": image}
        # # return render(request, self.template, ctx)
        # ctx = RequestContext(request, { 'output': output })

        return render(request, self.template, ctx)