import cv2
import numpy as np
from  keras.models import load_model
import pygame
from keras.models import model_from_json
import numpy
def func():
    video=cv2.VideoCapture(0)
    while(True):
        check, frame = video.read()
        start_point = (200, 150)
        end_point = (400, 350)
        color = (90, 90, 90)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img = cv2.rectangle(img, start_point, end_point, color, 1)
        cv2.imshow('a', img)
        img=img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
        img=cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
        ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
        blur = cv2.GaussianBlur(thresh1, (5, 5), 0)
        cv2.imshow('new', blur)
        key=cv2.waitKey(1)
        if key ==ord('q'):
            blur=cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
            blur = numpy.expand_dims(blur, axis=0)
            model = load_model('newmodel.h5')
            pred = model.predict(blur/255)
            cv2.destroyAllWindows()
            return np.argmax(pred)

import random
pygame.init()
textvalue=False
screen=pygame.display.set_mode((600,600))
pygame.display.set_caption('rock paper scissors')
run=True
screen.fill([255, 255, 255])
font=pygame.font.SysFont("monospace",30)
text=font.render("press  to play ",80,(100,100,100))
newtext=font.render("place you hand in square",80,(100,100,100))
text1=0
text2=0
while(run):
    for events in pygame.event.get():
        screen.fill([255, 255, 255])
        if events.type == pygame.QUIT:
            run = False
        if events.type==pygame.KEYDOWN:
            choice=func()
            x=choice
            if choice==0:
                choice="paper"
            if choice==2:
                choice="stone"
            if choice==1:
                choice="scissors"
            textvalue = True
            cmpchoice=random.randint(0, 2)
            u=cmpchoice
            if u == 0:
                cmpchoice = "paper"
            if u == 2:
                cmpchoice = "stone"
            if u == 1:
                cmpchoice = "scissors"
            if cmpchoice==choice:
                text=font.render("tie",100,(100,100,100))
                text1 = font.render("player choice is:" + choice, 40, (100, 100, 100))
                text2 = font.render("computer choice is:" + cmpchoice, 40, (100, 100, 140))
            elif cmpchoice==0 and x==1 or cmpchoice==1 and x==2 or cmpchoice==2 and x==0:
                text = font.render("player choice is:"+choice, 40, (100, 100, 100))
                text1=font.render("computer choice is"+cmpchoice,40, (100, 100, 140))
                text2=font.render("computer wins",40,(100,160))
            else:
                text = font.render("player choice is: " + choice, 40, (100, 100, 100))
                text1 = font.render("computer choice is: " + cmpchoice, 40, (100, 100, 140))
                text2 = font.render("player wins", 40, (100, 160,100))
    screen.blit(text, (20, 50))
    screen.blit(newtext, (20, 100))
    if(textvalue):
        screen.fill((255,255,255))
        screen.blit(text, (20, 50))
        screen.blit(text1,(20,90))
        screen.blit(text2,(20,120))
    pygame.display.update()
pygame.quit()