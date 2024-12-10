import cv2
import numpy as np

# Rangli tasvirni RGB formatiga o'zgartirish funksiyasi
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Yuzi aniqlash funksiyasi
def detect_faces(f_cascade, img, scaleFactor=1.1, minNeighbors=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img


# Fayldan rasm yuklash va yuzni aniqlash funksiyasi
def detect_faces_from_image(haar_cascade_path, image_path):
    haar_face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    img = cv2.imread(image_path)
    detected_img = detect_faces(haar_face_cascade, img)

    cv2.imshow('Face Detection - Image', detected_img)

    save_option = input("Natijaviy rasmni saqlashni xohlaysizmi? (ha/yo'q): ").strip().lower()
    if save_option == 'ha':
        output_path = 'result/detected_image.jpg'
        cv2.imwrite(output_path, detected_img)
        print(f"Natijaviy rasm '{output_path}' fayliga saqlandi.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Web kamera orqali yuzni aniqlash funksiyasi
def detect_faces_from_webcam(haar_cascade_path):
    haar_face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_frame = detect_faces(haar_face_cascade, frame)
        cv2.imshow('Face Detection - Webcam (Press Q to Quit)', detected_frame)

        # Natijaviy rasmni saqlash
        if cv2.waitKey(1) & 0xFF == ord('s'):  # 's' tugmasi bosilganda
            output_path = '../result/detected_webcam_frame1.jpg'
            cv2.imwrite(output_path, detected_frame)
            print(f"Natijaviy rasm '{output_path}' fayliga saqlandi.")

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tugmasi bosilganda
            break

    cap.release()
    cv2.destroyAllWindows()


# Asosiy boshqaruv funksiyasi
def main():
    haar_cascade_path = 'data/haarcascade_frontalface_alt.xml'

    # Foydalanuvchi tanlovi
    print("1: Fayldan rasm yuklash")
    print("2: Web kameradan foydalanish")
    choice = input("Tanlovingizni kiriting (1 yoki 2): ")

    if choice == '1':
        image_path = input("Rasmning to'liq yo‘lini kiriting: ")
        detect_faces_from_image(haar_cascade_path, image_path)
    elif choice == '2':
        detect_faces_from_webcam(haar_cascade_path)
    else:
        print("Noto‘g‘ri tanlov. Dasturdan chiqyapmiz.")


if __name__ == "__main__":
    main()
