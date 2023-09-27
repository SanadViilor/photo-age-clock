import torch
import streamlit as st
import numpy as np
from torchvision import transforms, models
import mediapipe as mp
from PIL import Image

def margin(box, margin):
    hmargin = margin * box.width
    vmargin = margin * box.height
    hstart = box.xmin - hmargin if (box.xmin - hmargin)>0 else 0
    vstart = box.ymin - vmargin if (box.ymin - vmargin)>0 else 0
    hext = box.xmin + box.width + hmargin if (box.xmin + box.width + hmargin)<1 else 1
    vext = box.ymin + box.height + vmargin if (box.ymin + box.height + vmargin)<1 else 1
    return hstart, vstart, hext, vext

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

st.image('logo_feher.png', width=200)

#age_model = models.resnet50()
#age_model.fc = torch.nn.Linear(in_features=2048, out_features=1)
#age_model.load_state_dict(torch.load('vgg_model.pt', map_location=torch.device('cuda:4')))

## VGG-16 ###
age_model = models.vgg16()
age_model.classifier[3] = torch.nn.Linear(in_features=4096, out_features=1024)
age_model.classifier[6] = torch.nn.Linear(in_features=1024, out_features=100)
age_model.load_state_dict(torch.load('vgg_model.pt', map_location=torch.device('cuda:4')))
#############

age_model.eval()

picture = st.camera_input("")

if picture is not None:
    img = Image.open(picture)
    img = img.convert("RGB")
    np_img = np.array(img)
   
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(np_img)
        if results.detections:
            mp_drawing.draw_detection(np_img, results.detections[0])
            box = results.detections[0].location_data.relative_bounding_box
            hstart, vstart, hext, vext = margin(box, 0.2)
            face_img = img.crop((int(hstart * img.size[0]), 
                                 int(vstart * img.size[1]),
                                 int(hext * img.size[0]), 
                                 int(vext * img.size[1])))

            t = transforms.ToTensor()
            face_img = data_transforms(face_img).detach()
            np_face_img = torch.Tensor.numpy(torch.permute(face_img, (1, 2, 0)))
            #pred = int(age_model(torch.unsqueeze(face_img, 0)))
            ## VGG-16 ###
            pred = int(torch.argmax(age_model(torch.unsqueeze(face_img, 0)), -1))
            #############
            st.header("Age: "+str(pred))
            st.write()
        else:
            np_face_img = img.copy()
            st.write("no face found")
    ##############
    #st.image(np_img, clamp=True, channels='RGB')
    ##############
