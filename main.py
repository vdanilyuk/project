"""import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'

frameWidth = 1920   # Frame Width
frameHeight = 1080   # Frame Height

plateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
minArea = 500

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(cv2.CAP_PROP_FPS, 30)#установка частоты кадров
cap.set(10, 150)
count = 0

# Список разрешенных автомобильных номеров
allowed_plates = ["а033ам774", "CD5678", "EF9101"]

def carplate_extract(image, carplate_haar_cascade):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        carplate_img = image[y+15:y+h-10, x+15:x+w-20]

    return carplate_img


def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized_image

def preprocess_image(img):
    # Применение бинаризации для улучшения контраста
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

    # Применение адаптивной бинаризации
    img_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Устранение шума с помощью медианного фильтра
    img_filtered = cv2.medianBlur(img_adapt, 3)

    return img_filtered


while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Номер Авто", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", imgRoi)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Распознание текста на номерном знаке с помощью pytesseract
                carplate_extract_img = carplate_extract(imgRoi, plateCascade)
                carplate_extract_img = enlarge_img(carplate_extract_img, 150)

                carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_BGR2GRAY)

                text = pytesseract.image_to_string(
                    carplate_extract_img_gray,
                    lang = 'rus+eng',
                    config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                print('Распознанный текст:', text)

                # Проверка распознанного номера на соответствие списку разрешенных номеров
                if text in allowed_plates:
                    cv2.imwrite(f'C:/Users/vladi/Documents/Allowed_Plates/IMAGES{count}.jpg', imgRoi)
                else:
                    cv2.imwrite(f'C:/Users/vladi/Documents/NotAllowed_Plates/IMAGES{count}.jpg', imgRoi)

                cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Скан сохранен", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                cv2.imshow("RESULT", img)
                cv2.waitKey(500)
                count += 1

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

"""
import cv2
import easyocr

reader = easyocr.Reader(['en'])

frameWidth = 1920   # Frame Width
frameHeight = 1080   # Frame Height

plateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
minArea = 500

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(cv2.CAP_PROP_FPS, 30)  # установка частоты кадров
cap.set(10, 150)
count = 0

# Список разрешенных автомобильных номеров
allowed_plates = ["A033AM774", 'A134AA', "A033Am774"]

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Номер Авто", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", imgRoi)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Распознание текста на номерном знаке с помощью easyocr
                result = reader.readtext(imgRoi, detail=0)

                print('Распознанный текст:', result)

                # Применение фильтров для улучшения качества скриншота
                imgRoi = cv2.GaussianBlur(imgRoi, (5, 5), 1)  # Применение размытия
                imgRoi = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)  # Преобразование в оттенки серого
                imgRoi = cv2.Canny(imgRoi, 50, 50)  # Применение фильтра Canny для обнаружения границ

                # Проверка распознанного номера на соответствие списку разрешенных номеров
                if any(item.lower() in allowed_plates for item in result):
                    cv2.imwrite(f'C:/Users/vladi/Documents/Allowed_Plates/IMAGES{count}.jpg', imgRoi)
                else:
                    cv2.imwrite(f'C:/Users/vladi/Documents/NotAllowed_Plates/IMAGES{count}.jpg', imgRoi)

                cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Скан сохранен", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                cv2.imshow("RESULT", img)
                cv2.waitKey(500)
                count += 1


    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

"""import cv2
import easyocr
import csv
import datetime


reader = easyocr.Reader(['en']) # Установка языка для распознавания текста

frameWidth = 1920   # Frame Width
frameHeight = 1080   # Frame Height

plateCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml') # Каскадный классификатор
minArea = 500 # Мин. площадь для обнаружения номерных знаков

cap = cv2.VideoCapture(0) # Видеосъемка
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(cv2.CAP_PROP_FPS, 30)  # Установка частоты кадров
cap.set(10, 150)
count = 0

# Чтение данных из CSV файла
allowed_plates = {}
with open('plates.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        allowed_plates[row[0].lower()] = row[0]

# Определение имени CSV файлов для разрешенных и неразрешенных номеров
allowed_csv_file = 'allowed_plates.csv'
not_allowed_csv_file = 'not_allowed_plates.csv'

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Номер Авто", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", imgRoi)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Распознание текста на номерном знаке с помощью easyocr
                result = reader.readtext(imgRoi, detail=0)

                print('Распознанный текст:', result)

                # Проверка распознанного номера на соответствие списку разрешенных номеров
                recognized_plate = ''.join(result).lower()
                print(f"Распознанный номер: {recognized_plate}")
                if recognized_plate in allowed_plates:
                    print("Номер в списке разрешенных")
                    with open(allowed_csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([recognized_plate, datetime.datetime.now()])
                else:
                    print("Номер НЕ в списке разрешенных")
                    with open(not_allowed_csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([recognized_plate, datetime.datetime.now()])

                cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED) 
                cv2.putText(img, "Скан сохранен", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                cv2.imshow("RESULT", img)
                cv2.waitKey(500) # Ожидание 500 миллесикунд перед продолжением выполнения программы
                count += 1

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""



import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from functools import partial
import cv2
import easyocr
import csv
import datetime
import codecs

# Функция для обработки видео и распознавания номерных знаков
def process_video(allowed_plates):
    reader = easyocr.Reader(['en'])  # Установка языка для распознавания текста
    frameWidth = 1920   # Frame Width
    frameHeight = 1080   # Frame Height

    plateCascade = cv2.CascadeClassifier("C:\\Users\\vladi\\Documents\\haarcascade_russian_plate_number.xml") # Каскадный классификатор
    minArea = 500 # Мин. площадь для обнаружения номерных знаков

    cap = cv2.VideoCapture(0) # Видеосъемка
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Установка частоты кадров
    cap.set(10, 150)
    count = 0

    # Чтение данных из CSV файла
    allowed_plates = {}
    with open('C:\\Users\\vladi\\OneDrive\\Рабочий стол\\проект\\plates.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            allowed_plates[row[0].lower()] = row[0]

    # Определение имени CSV файлов для разрешенных и неразрешенных номеров
    allowed_csv_file = 'allowed_plates.csv'
    not_allowed_csv_file = 'not_allowed_plates.csv'

    while True:
        success, img = cap.read()

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

        for (x, y, w, h) in numberPlates:
            area = w * h
            if area > minArea:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, "Номер Авто", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                imgRoi = img[y:y + h, x:x + w]
                cv2.imshow("ROI", imgRoi)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    # Распознание текста на номерном знаке с помощью easyocr
                    result = reader.readtext(imgRoi, detail=0)

                    print('Распознанный текст:', result)

                    # Проверка распознанного номера на соответствие списку разрешенных номеров
                    recognized_plate = ''.join(result).lower()
                    print(f"Распознанный номер: {recognized_plate}")
                    if recognized_plate in allowed_plates:
                        print("Номер в списке разрешенных")
                        with open(allowed_csv_file, mode='a', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file)
                            writer.writerow([recognized_plate, datetime.datetime.now()])
                    else:
                        print("Номер НЕ в списке разрешенных")
                        with open(not_allowed_csv_file, mode='a', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file)
                            writer.writerow([recognized_plate, datetime.datetime.now()])

                    cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED) 
                    cv2.putText(img, "Скан сохранен", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                    cv2.imshow("RESULT", img)
                    cv2.waitKey(500) # Ожидание 500 миллесикунд перед продолжением выполнения программы
                    count += 1

        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Функция для выбора файла с разрешенными номерами
def select_allowed_plates_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        # Чтение данных из CSV файла и обновление списка разрешенных номерных знаков
        allowed_plates.clear()
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                allowed_plates[row[0].lower()] = row[0]
        messagebox.showinfo("File Selected", "Selected file: " + file_path)

# Создание графического интерфейса
window = tk.Tk()
window.title("Номерные знаки")
window.geometry("400x200")

# Кнопка для выбора файла с разрешенными номерами
select_file_button = tk.Button(window, text="Выбрать файл с разрешенными номерами", command=select_allowed_plates_file)
select_file_button.pack()

# Список разрешенных номерных знаков
allowed_plates = {}

# Кнопка для запуска обработки видео и распознавания номерных знаков
process_video_button = tk.Button(window, text="Обработать видео", command=partial(process_video, allowed_plates))
process_video_button.pack()

window.mainloop()


