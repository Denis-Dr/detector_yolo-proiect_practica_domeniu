import time
import cv2
import numpy as np
from PIL import ImageGrab

from detector import Detector

# acuratetea detectorului
__PRECIZIE__ = 0.7
# regiunea de pe ecran unde se face captura
bounding_box = (10, 150, 910, 1050)

#  folosim o blucla while infinita pentru a ne asigura ca primim pe intrare
#  valoarea corecta
while True:
    try:
        dispozitiv = int(input("\nSelectati dispozitivul."
                               "\n0 - camera web; 1 - camptura ecran: "))
    except ValueError:
        print("\nValoare incorecta!")
        # daca se introduce orice in afara de un numar, utilizatorul va primi
        # indicatia sa introduca valoarea corecta
        continue
    else:
        # daca s-a introdus valoarea corecta iesim din bucla
        if dispozitiv in [0, 1]:
            break
        else:
            print("\nValoare incorecta!")

# daca dispozitiv este 0 initializam camera web
if dispozitiv == 0:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, img0 = cap.read()
    if ret is False:
        raise Exception('\nCamera nu este conectata sau este deja in uz!')
# daca dispozitiv este 1 initializam capura de ecran
elif dispozitiv == 1:

    # bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    region = ImageGrab.grab(bbox=bounding_box)
    # convertim imaginea obtinuta in format numpy
    img0 = np.array(region)
    img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)

print("\nSe incarca detectorul . . . ")
# vom folosi prima imagine pentru a initializa detectorul Yolo
yolo = Detector(weights='food_best.pt', source=img0)
# y2 =   Detector(weights='yolov5x6.pt', source=img0)
# y3 =   Detector(weights='yolov5x6.pt', source=img0)
# y4 =   Detector(weights='yolov5x6.pt', source=img0)

# initializare variabile folosite in diverse calcule
minim = 2000
maxim = -1
contor = 0
contor_aux = 0
fps = 0
avg_fps = 0

timeFPS = time.time()

while True:
    timestamp = time.time()

    if dispozitiv == 0:
        ret, img = cap.read()
    if dispozitiv == 1:
        region = ImageGrab.grab(bbox=bounding_box)  # bbox specifies specific region (bbox= x,y,width,height *starts top-left)
        img = np.array(region)  # this is the array obtained from conversion
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # apelam detectorul Yolo cu imaginea curenta
    yolo.detect(source=img, conf_thres=__PRECIZIE__)
    sign, sign_Height = yolo.getResult()

    # y2.detect(source=img, conf_thres=0.70)
    # y3.detect(source=img, conf_thres=0.70)
    # y4.detect(source=img, conf_thres=0.70)

    cv2.putText(img, "Apasa ESC pentru a inchide", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)

    # afisam imaginea procesata
    cv2.imshow("detection", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) == 27:
        print("\nProgram terminat.")
        break

    # ---- calcul FPS ---------------------------------
    contor += 1
    if contor > 10:
        durata = time.time() - timestamp

        if time.time() - timeFPS >= 1.0:
            avg_fps = contor - contor_aux
            contor_aux = contor
            timeFPS = time.time()

        if durata < minim:
            minim = durata
        if durata > maxim:
            maxim = durata
        fps = 1.0 / durata
        print(f'Durata: {durata:2.4f} sec  \tmin: {minim:2.4f} \tmax: {maxim:2.4f} '
              f'\tFPS:{fps:3.2f} \tAvg_FPS: {avg_fps} \tobiect: {sign}')

cap.release()
cv2.destroyAllWindows()
