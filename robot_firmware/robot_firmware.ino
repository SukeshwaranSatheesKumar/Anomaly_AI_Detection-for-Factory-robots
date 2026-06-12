/**
 * Anomaly AI Detection for Factory Robots - Arduino Firmware
 * 
 * Hardware Connections:
 * - Servo motors:
 *   - Base Servo: Signal to Pin 3
 *   - Shoulder Servo: Signal to Pin 5
 *   - Elbow Servo: Signal to Pin 6
 *   - Gripper Servo: Signal to Pin 9
 * - ACS712 Current Sensor:
 *   - OUT Pin: Analog Pin A0
 *   - VCC: 5V
 *   - GND: GND
 * - MPU6050 IMU Sensor:
 *   - SDA: Pin A4 (Uno/Nano) or Pin 20 (Mega)
 *   - SCL: Pin A5 (Uno/Nano) or Pin 21 (Mega)
 *   - VCC: 5V or 3.3V
 *   - GND: GND
 * 
 * This code sweeps the servos through a simulated pick-and-place operation,
 * measures current consumption (ACS712) and accelerations (MPU6050),
 * and streams the telemetry in CSV format over serial:
 * format: base,shoulder,elbow,gripper,current,imu
 */

#include <Wire.h>
#include <Servo.h>

// Servo Definitions
Servo baseServo;
Servo shoulderServo;
Servo elbowServo;
Servo gripperServo;

// Pin Definitions
const int CURRENT_SENSOR_PIN = A0;
const int BASE_SERVO_PIN = 3;
const int SHOULDER_SERVO_PIN = 5;
const int ELBOW_SERVO_PIN = 6;
const int GRIPPER_SERVO_PIN = 9;

// MPU6050 I2C Address
const int MPU_ADDR = 0x68; 

// Variables to track servo target and current angles
int baseAngle = 90;
int shoulderAngle = 90;
int elbowAngle = 90;
int gripperAngle = 45;

// Timing variables
unsigned long lastTelemetryTime = 0;
const unsigned long TELEMETRY_INTERVAL = 50; // 50ms interval (20 Hz)

void setup() {
  // Initialize Serial
  Serial.begin(9600);
  
  // Initialize Servos
  baseServo.attach(BASE_SERVO_PIN);
  shoulderServo.attach(SHOULDER_SERVO_PIN);
  elbowServo.attach(ELBOW_SERVO_PIN);
  gripperServo.attach(GRIPPER_SERVO_PIN);
  
  // Set initial positions
  baseServo.write(baseAngle);
  shoulderServo.write(shoulderAngle);
  elbowServo.write(elbowAngle);
  gripperServo.write(gripperAngle);
  
  // Initialize MPU6050
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0);    // set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);

  delay(500);
}

void loop() {
  // Execute a standard pick-and-place routine in loop
  // We use non-blocking or simple delays. Since it's a loop, we can write a sequence of steps.
  // To keep telemetry flowing while servos move, we increment angles step-by-step.
  
  static int state = 0;
  static int stepCount = 0;
  
  // Robotic routine state machine
  switch (state) {
    case 0: // Rotate base to pickup
      if (baseAngle < 150) { baseAngle++; }
      else { state = 1; }
      break;
    case 1: // Lower shoulder and elbow
      if (shoulderAngle > 45) { shoulderAngle--; }
      if (elbowAngle < 120) { elbowAngle++; }
      if (shoulderAngle <= 45 && elbowAngle >= 120) { state = 2; }
      break;
    case 2: // Close gripper (grip object)
      if (gripperAngle < 90) { gripperAngle++; }
      else { state = 3; delay(300); } // Wait a bit
      break;
    case 3: // Raise arm
      if (shoulderAngle < 90) { shoulderAngle++; }
      if (elbowAngle > 90) { elbowAngle--; }
      if (shoulderAngle >= 90 && elbowAngle <= 90) { state = 4; }
      break;
    case 4: // Rotate base to dropoff
      if (baseAngle > 30) { baseAngle--; }
      else { state = 5; }
      break;
    case 5: // Lower arm for dropoff
      if (shoulderAngle > 50) { shoulderAngle--; }
      if (elbowAngle < 110) { elbowAngle++; }
      if (shoulderAngle <= 50 && elbowAngle >= 110) { state = 6; }
      break;
    case 6: // Open gripper (release object)
      if (gripperAngle > 45) { gripperAngle--; }
      else { state = 7; delay(300); }
      break;
    case 7: // Return to home position
      if (baseAngle < 90) baseAngle++;
      if (baseAngle > 90) baseAngle--;
      if (shoulderAngle < 90) shoulderAngle++;
      if (elbowAngle > 90) elbowAngle--;
      if (elbowAngle < 90) elbowAngle++;
      
      if (baseAngle == 90 && shoulderAngle == 90 && elbowAngle == 90) {
        state = 0; // Restart cycle
        delay(1000);
      }
      break;
  }
  
  // Write to servos
  baseServo.write(baseAngle);
  shoulderServo.write(shoulderAngle);
  elbowServo.write(elbowAngle);
  gripperServo.write(gripperAngle);
  
  // Read sensors and send telemetry at 20Hz (every 50ms)
  unsigned long currentMillis = millis();
  if (currentMillis - lastTelemetryTime >= TELEMETRY_INTERVAL) {
    lastTelemetryTime = currentMillis;
    
    // 1. Read Current Sensor (ACS712)
    // Map analog reading (0-1023) to Current (Amps)
    // For a 5A ACS712, sensitivity is 185mV/A, Vcc/2 (2.5V) is 0 Amps
    int rawCurrent = analogRead(CURRENT_SENSOR_PIN);
    double voltage = (rawCurrent / 1024.0) * 5000.0; // mV
    double currentAmps = (voltage - 2500.0) / 185.0; // Amps (will be positive or negative depending on direction)
    currentAmps = abs(currentAmps); // Get absolute load
    
    // 2. Read IMU (MPU6050) Accelerometer values
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B); // Starting register for Accelerometer Data (ACCEL_XOUT_H)
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6, true); // Read 6 registers (X, Y, Z accel)
    
    int16_t ax = Wire.read() << 8 | Wire.read();
    int16_t ay = Wire.read() << 8 | Wire.read();
    int16_t az = Wire.read() << 8 | Wire.read();
    
    // Compute total acceleration magnitude (vibration metric)
    // Scale factor for default +/-2g range is 16384 LSB/g
    double accelX = ax / 16384.0;
    double accelY = ay / 16384.0;
    double accelZ = az / 16384.0;
    double imuMagnitude = sqrt(accelX * accelX + accelY * accelY + accelZ * accelZ);
    
    // Send CSV line to Serial
    // Format: base,shoulder,elbow,gripper,current,imu
    Serial.print(baseAngle);
    Serial.print(",");
    Serial.print(shoulderAngle);
    Serial.print(",");
    Serial.print(elbowAngle);
    Serial.print(",");
    Serial.print(gripperAngle);
    Serial.print(",");
    Serial.print(currentAmps, 3);
    Serial.print(",");
    Serial.println(imuMagnitude, 3);
  }
  
  delay(15); // Small delay to smooth servo motion
}
