/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 https://www.arduino.cc/en/Tutorial/LibraryExamples/Sweep
*/

#include <Servo.h>

Servo myservo1; 
Servo myservo2; 
Servo myservo3; 
Servo myservo4; 
Servo myservo5; 
// twelve servo objects can be created on most boards

// input pin
const int outputPin = 6; 
const int inputPin = 7; // digital pin 7 as inptu
boolean inputPinState = HIGH;
boolean outputPinState = LOW;

// analog declare
const int analogInFinger1 = A0;
const int analogInFinger2 = A1;
const int analogInFinger3 = A2;
const int analogInFinger4 = A3;
const int analogInFinger5 = A4;
const int analogTableDistanceA5 = A5;

int analogInFinger1_value = 0;
int analogInFinger2_value = 0;
int analogInFinger3_value = 0;
int analogInFinger4_value = 0;
int analogInFinger5_value = 0;
int analogTableA5Value = 0;

int min_angle = 0;
int max_angle=170;
int increase_angle_by = 7;
int closing_delay = 40;
int pressure_thresol = 10;
int last_position = min_angle;

int hand_state = 1; // zero means close , 1 means open

void setup() {
  Serial.begin(9600);
  pinMode(inputPin, INPUT);
  //pinMode(outputPin,OUTPUT);
  
  myservo1.attach(8);  // attaches the servo on pin 9 to the servo object
  myservo2.attach(9);
  myservo3.attach(10);
  myservo4.attach(11);
  myservo5.attach(12);

  reset_motor();

}

void reset_motor(){
  //reset motor
  myservo1.write(max_angle);        
  myservo2.write(max_angle);
  myservo3.write(max_angle);
  myservo4.write(max_angle);
  myservo5.write(max_angle);
}

boolean check_pressure_sensor_data(){
  analogInFinger1_value = analogRead(analogInFinger1);
  analogInFinger2_value = analogRead(analogInFinger2);
  analogInFinger3_value = analogRead(analogInFinger3);
  analogInFinger4_value = analogRead(analogInFinger4);
  analogInFinger5_value = analogRead(analogInFinger5);

//  Serial.println("Pressure: ");
//  Serial.print("AF1: ");
//  Serial.print(analogInFinger1_value);
//  Serial.print(", AF2: ");
//  Serial.print(analogInFinger2_value);
//  Serial.print(", AF3: ");
//  Serial.print(analogInFinger3_value);
//  Serial.print(", AF4: ");
//  Serial.print(analogInFinger4_value);
//  Serial.print(", AF5: ");
//  Serial.print(analogInFinger5_value);
  

  if(analogInFinger1_value > pressure_thresol || 
     analogInFinger2_value > pressure_thresol || 
     analogInFinger3_value > pressure_thresol || 
     analogInFinger4_value > pressure_thresol || 
     analogInFinger5_value > pressure_thresol
     ){
//        Serial.println("Pressure: ");
//        Serial.print("AF1: ");
//        Serial.print(analogInFinger1_value);
//        Serial.print(", AF2: ");
//        Serial.print(analogInFinger2_value);
//        Serial.print(", AF3: ");
//        Serial.print(analogInFinger3_value);
//        Serial.print(", AF4: ");
//        Serial.print(analogInFinger4_value);
//        Serial.print(", AF5: ");
//        Serial.print(analogInFinger5_value);
        return false;
     }else{
        return true;
     }
      
}



//void hand_open(){
//    Serial.println("=========Hand Open=======");
//    myservo1.write(max_angle);        
//    myservo2.write(max_angle);
//    myservo3.write(max_angle);
//    myservo4.write(max_angle);
//    myservo5.write(max_angle);
//    
//    hand_state = 1;
//}

void hand_open(){
  //int angle=min_angle;
   int angle = last_position;

  Serial.println("=========Hand Close=======");
  
  while(angle < max_angle){
    //Serial.println("forward moving angle ");
    //Serial.print(angle);

    myservo1.write(angle); 
    myservo2.write(angle);
    myservo3.write(angle);
    myservo4.write(angle);
    myservo5.write(angle);
    
    angle = angle + increase_angle_by;
    delay(closing_delay);
    
  } // while end

  last_position = max_angle; 
  hand_state = 1;

}


void hand_close(){
  int angle=max_angle;


  Serial.println("=========Hand Close=======");
  
  while(angle > min_angle){
    //Serial.println("forward moving angle ");
    //Serial.print(angle);


    if(check_pressure_sensor_data() == false){
       Serial.println("###############false Pressure over thresold");
       break;
    }
    
    myservo1.write(angle); 
    myservo2.write(angle);
    myservo3.write(angle);
    myservo4.write(angle);
    myservo5.write(angle);
    
    angle = angle - increase_angle_by;
    delay(closing_delay);
    
  } // while end

  last_position = angle;
  Serial.println(last_position);
  hand_state = 0;

}

void loop() {

 
//    inputPinState = digitalRead(inputPin);
//    if(inputPinState == HIGH && hand_state == 0){
//        hand_open();
//    }
//    
//    if(inputPinState == LOW && hand_state == 1){
//      hand_close();
//    }


// read from port 0, send to port 1:
  if (Serial.available()) {
    String inStr = Serial.readString();
    
    Serial.println(inStr);
    if(inStr == "1" &&  hand_state == 0){
        hand_open();
    }
    
    if(inStr == "0" && hand_state == 1){
      hand_close();
    }

   
  }
    
    
}
