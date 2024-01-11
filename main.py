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


