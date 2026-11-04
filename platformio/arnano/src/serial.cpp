#include <Arduino.h>

void setup() {
    Serial.begin(9600);
    pinMode(LED_BUILTIN, OUTPUT);

    Serial.println("Arduino Nano siap");
}

void loop() {
    if (Serial.available() > 0) {
        char command = Serial.read();

        if (command == '1') {
            digitalWrite(LED_BUILTIN, HIGH);
            Serial.println("LED ON");
        } 
        else if (command == '0') {
            digitalWrite(LED_BUILTIN, LOW);
            Serial.println("LED OFF");
        }
    }
}