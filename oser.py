import cv2
import numpy as np
import serial

baudrate = 9600
COMport = '/dev/ttyUSB0'  # Replace with your Arduino's COM port

ser = serial.Serial(COMport, baudrate, timeout=0.1)

cam = cv2.VideoCapture(0)
min_area = 500

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(thresh, 20, 30)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contourss, _ = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    copy_frame = frame.copy()
    cv2.drawContours(copy_frame, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    coin_count = 0
    detected = b'L'  # Default value for detected

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # coin_count = len(contours)
            coin_count += 1
            # coin = cv2.minEnclosingCircle(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(copy_frame, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.circle(frame, tuple(map(int, coin[0])), int(coin[1]), (0, 255, 0), 2)

            detected = b'H'  # Update detected value if a coin is found
            
    cv2.putText(frame, f"Coin Count: {coin_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    ser.write(detected)  # Send the detected value to Arduino

    cv2.imshow("Frame", frame)
    cv2.imshow("Gray", gray)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Edged", edged)
    cv2.imshow("Contours", copy_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()