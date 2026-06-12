# Anomaly AI Detection for Factory Robots

An end-to-end Machine Learning and IoT framework designed to detect physical anomalies (collisions, motor stalls, mechanical jams, joint slips, and looseness) in factory robotic arms. 

The system utilizes an **Arduino-based sensor node** to capture real-time telemetry and streams it to an **edge PC** where **Isolation Forest** (classical unsupervised ML) and **LSTM Autoencoder** (deep learning reconstruction) models run real-time inference. It also includes full **ROS2 (Robot Operating System 2)** support to publish joint states and diagnostics.

## 🎥 Project Demonstration

Below is a demonstration of the robotic arm running the telemetry routine and testing the AI anomaly detection system:

![Anomaly AI Detection Demo](demo_video.mp4)

---

## 🛠️ System Architecture

```
+-----------------------------------------------------------+
|                     HARDWARE LAYER                        |
|                                                           |
|  [ACS712 Current]   [MPU6050 IMU]   [4x Servo Motors]     |
|         |                 |                 |             |
|         v                 v                 v             |
|                  [Arduino Uno/Nano/Mega]                  |
+-----------------------------+-----------------------------+
                              | Serial (USB COM @ 9600)
                              v
+-----------------------------------------------------------+
|                     SOFTWARE PIPELINE                     |
|                                                           |
|    [serial_collector.py] ------> [telemetry.csv]          |
|                                         |                 |
|                                         v                 |
|                                  [preprocess.py]          |
|                                  /             \          |
|                                 v               v         |
|                       (Tabular Data)      (Sequence Data) |
|                              |                  |         |
|                              v                  v         |
|                      [train_isolation.py]   [train_lstm_ae.py]
|                              |                  |         |
|                              +--------+---------+         |
|                                       v                   |
|                                 [evaluate.py]             |
|                                       |                   |
|                                       v                   |
|                          [realtime_detector_serial.py]    |
+---------------------------------------+-------------------+
                                        | (Optional)
                                        v
+-----------------------------------------------------------+
|                        ROS2 LAYER                         |
|                                                           |
|    [collector_node.py] -------> /joint_states              |
|                                 /robot/current_draw       |
|                                 /robot/imu_vibration      |
|                                       |                   |
|                                       v                   |
|                         [realtime_detector_ros2.py]       |
|                                       |                   |
|                                       v                   |
|                           /robot/anomaly_status           |
+-----------------------------------------------------------+
```

---

## 🔌 Hardware Setup

### 1. Key Components
* **Microcontroller**: Arduino Uno, Nano, or Mega.
* **Actuators**: 4x Servo Motors (Base, Shoulder, Elbow, Gripper).
* **Current Sensor**: ACS712 (5A module) connected to analog input to measure current draw.
* **IMU (Inertial Measurement Unit)**: MPU6050 (6-axis accel/gyro) connected via I2C to measure arm vibrations.

### 2. Wiring Connections

| Component | Arduino Pin | Description |
|---|---|---|
| **Base Servo** | `D3` | PWM Control Signal |
| **Shoulder Servo** | `D5` | PWM Control Signal |
| **Elbow Servo** | `D6` | PWM Control Signal |
| **Gripper Servo** | `D9` | PWM Control Signal |
| **ACS712 OUT** | `A0` | Analog Voltage Input (Current) |
| **MPU6050 SDA** | `A4` (Uno/Nano) / `20` (Mega) | I2C Data Line |
| **MPU6050 SCL** | `A5` (Uno/Nano) / `21` (Mega) | I2C Clock Line |
| **Power Rails** | `5V` & `GND` | Common VCC and Ground |

> [!WARNING]
> Servo motors can draw significant current when moving under load. Always power your servos using an **external 5V/6V power supply** rather than directly from the Arduino's 5V pin, making sure to connect the external supply's ground to the Arduino's ground.

---

## 💻 IDE Setup (VS Code & Arduino)

> [!NOTE]
> Microsoft's official VS Code "Arduino" extension was deprecated around 2024. To program the Arduino within VS Code, use the modern methods below:

### Option A: PlatformIO IDE Extension (Recommended)
This is the industry standard for embedded software development in VS Code.
1. Open VS Code and go to **Extensions** (`Ctrl+Shift+X`).
2. Search for **PlatformIO IDE** and click install.
3. Open the PlatformIO Home, click **New Project**, select your board (e.g., Arduino Uno) and choose the `Arduino` framework.
4. Copy the firmware code into the `src/main.cpp` file.
5. In `platformio.ini`, add the following to automatically configure parameters:
   ```ini
   [env:uno]
   platform = atmelavr
   board = uno
   framework = arduino
   lib_deps =
       Wire
       Servo
   ```
6. Click the Checkmark icon (Compile) and the Arrow icon (Upload) in the bottom status bar.

### Option B: Official Arduino IDE 2.x (Alternative)
If you prefer a simpler editor, download and use the official Arduino IDE 2.x.
1. Open Arduino IDE 2.x.
2. Copy the code from `robot_firmware/robot_firmware.ino` into a new sketch.
3. Select your Board and Port from the top drop-down menus.
4. Go to **Sketch** -> **Include Library** -> **Manage Libraries...**, verify that the built-in `Servo` library is loaded.
5. Click **Verify** and **Upload**.

---

## 🐍 Software Setup (Python ML)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SukeshwaranSatheesKumar/Anomaly_AI_Detection-for-Factory-robots.git
   cd Anomaly_AI_Detection-for-Factory-robots
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # Activate on Windows:
   .\venv\Scripts\activate
   # Activate on Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ How to Run

### Step 1: Telemetry Data Collection
Connect the Arduino to your PC. Identify its COM port (e.g., `COM5` on Windows or `/dev/ttyUSB0` on Linux). Open `serial_collector.py` and adjust `PORT` if needed, then run:
```bash
python serial_collector.py
```
Let the robot cycle through its movement states. To collect training data representing anomalous states, manually obstruct the arm's motion (generating high current) or shake the sensor link (generating high vibration).

*(Note: If no `telemetry.csv` file exists, the preprocessing script will automatically generate synthetic data so you can test the code immediately).*

### Step 2: Data Preprocessing
Preprocess the telemetry, build features, scale parameters, and create sequence splits:
```bash
python preprocess.py
```

### Step 3: Model Training
Train the anomaly detection models:
* **Option A: Classical ML (Isolation Forest)**
  ```bash
  python train_isolation.py
  ```
* **Option B: Deep Learning (LSTM Autoencoder)**
  ```bash
  python train_lstm_ae.py
  ```

### Step 4: Model Evaluation
Validate model performance on the test sets and generate visualization graphs (`evaluation_plot.png`):
```bash
python evaluate.py
```

### Step 5: Real-Time Inference
Run the detector to listen to the serial port, pre-process telemetry on-the-fly, and alert you of anomalies:
* Run the **Isolation Forest** detector:
  ```bash
  python realtime_detector_serial.py --port COM5 --model if
  ```
* Run the **LSTM Autoencoder** detector:
  ```bash
  python realtime_detector_serial.py --port COM5 --model lstm
  ```

---

## 🤖 ROS2 Integration

If you run your robot inside a ROS2 workspace (Humble/Iron/Jazzy), we have included ready-to-run Python nodes.

1. **Fake/Simulation Mode (No Hardware)**:
   Launch the mock joint and sensor publisher, followed by the detector:
   ```bash
   # Terminal 1: Publish mock robot positions and current spikes
   ros2 run my_robot_package fake_joint_publisher
   
   # Terminal 2: Run inference node using Isolation Forest
   ros2 run my_robot_package realtime_detector_ros2 --ros-args -p model_type:=if
   ```

2. **Hardware Mode (With Connected Robot)**:
   Launch the serial reading collector, followed by the detector:
   ```bash
   # Terminal 1: Stream serial telemetry into ROS2 topics
   ros2 run my_robot_package collector_node --ros-args -p port:=/dev/ttyUSB0
   
   # Terminal 2: Run real-time LSTM detector
   ros2 run my_robot_package realtime_detector_ros2 --ros-args -p model_type:=lstm
   ```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
