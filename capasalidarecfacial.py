import cv2 as cv
# acceder a carpetas
import os 
# para reducir la ventana y de esta forma no trabaje tanto la computadora
import imutils as imutils 
dataRuta='D:/Documentos/CURSOS/PYTHON/Proyecto Python RF/Curso/reconocimientofacial1/Data' #hacer comparaciones de las fotos
listaData=os.listdir(dataRuta)
# Se crean las clases para que el programa entienda el entrenamiento
entrenamiento=cv.face.EigenFaceRecognizer_create()
# Clase para decirle a que entramiento va a reconocer
entrenamiento.read('D:/Documentos/CURSOS/PYTHON/Proyecto Python RF/Curso/reconocimientofacial1/EntrenamientoEigenFaceRecognizer.xml')
# se agregan ruidos para que el programa entienda mejor a que persona quiere identificar
ruidos=cv.CascadeClassifier('D:/Documentos/CURSOS/PYTHON/Proyecto Python RF/Curso/reconocimientofacial1/haarcascade_frontalface_default.xml')
camara=cv.VideoCapture(0)
#Preguntamos si nuestra camara esta prendida
while True: 
    respuesta,captura=camara.read()
    if respuesta==False:break
    captura=imutils.resize(captura,width=640)
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    # Deteccion de caras o rostros mediante ruidos
    cara=ruidos.detectMultiScale(grises,1.3,5) 

    # Vamos a enmarcar el rostro.

    #Bluce para hacer un recorrido de cada pixel en la foto "El cuadro de la foto"
    # x & y es Izquierda y derecha. e1 & e2 es arriba y abajo. 
    for(x,y,e1,e2) in cara: 
        # Este apartado sacara fragmentos de nuestro rostro para almacenarlos en carpeta
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        # Tamaño del contorno, ya sea rectangulo o cuadrado
        rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC)
        # Vamos a crear una variable para etiquetar los rostros reconocidos
        resultado=entrenamiento.predict(rostrocapturado)
        # Escribir un texto para la imagen   LINE_AA Hacer que la figura sea cuadra, pero no circulares
        cv.putText(captura, '{}'.format(resultado),(x,y-5),1,1.3,(0,255,0),1,cv.LINE_AA)
        # Etiqutar las fotos
        if resultado[1]<9000:
            cv.putText(captura, '{}'.format(listaData[resultado[0]]), (x,y-20), 2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)
        else:
            cv.putText(captura,"No encontrado", (x,y-20), 2,0.7,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)

    # Comenzamos a obtener los valores sobre la captura que estamos haciendo en el for
    cv.imshow("Resultados", captura)
    # Clave de escape para cerrar la ventana
    if cv.waitKey(1)==ord('s'):
        break
# Cerramos todas las ventanas con la tecla s
camara.release()    
cv.destroyAllWindows()

