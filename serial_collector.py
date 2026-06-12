# serial_collector.py
import serial, time, csv, sys

PORT = 'COM5'   # Windows example, change to /dev/ttyUSB0 on Linux
BAUD = 9600
OUT = 'telemetry.csv'

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("Connected to", PORT)
    with open(OUT, 'w', newline='') as f:
        w = csv.writer(f)
        # header: t,base,shoulder,elbow,gripper,current,imu
        w.writerow(['t','base','shoulder','elbow','gripper','current','imu'])
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            if not line: 
                continue
            parts = line.split(',')
            if len(parts) < 6: 
                print("Bad line:", line)
                continue
            ts = time.time()
            row = [ts] + parts[:6]
            w.writerow(row)
            f.flush()
            print("Wrote:", row)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped")
        sys.exit(0)
