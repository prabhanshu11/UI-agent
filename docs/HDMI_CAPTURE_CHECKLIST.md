# HDMI Capture Card Startup Checklist

**Device:** MacroSilicon MS2109 USB3.0 (534d:2109)
**Symptom:** Black screen / 8KB PNG = no signal

## Quick Fix (90% of cases)

```bash
# 1. Find the device's USB path
lsusb | grep 534d  # Note the Bus/Device
# Map to sysfs path:
for dev in /sys/bus/usb/devices/*/; do
    [ "$(cat $dev/idVendor 2>/dev/null)" = "534d" ] && echo "$(basename $dev): power=$(cat $dev/power/control)"
done

# 2. Disable autosuspend (THE FIX)
echo "on" | sudo tee /sys/bus/usb/devices/BUS-PORT/power/control

# 3. Wait 2s, then capture
sleep 2
ffmpeg -y -f v4l2 -input_format mjpeg -video_size 1920x1080 -i /dev/video0 -frames:v 5 -update 1 /tmp/hdmi_test.png
ls -la /tmp/hdmi_test.png  # >20KB = real content, <20KB = still no signal
```

## Full Diagnostic Steps

### Step 1: USB Detection
```bash
lsusb | grep -i "534d\|macro\|capture"
```
- **Present** → device physically connected, proceed
- **Missing** → check physical cable, try different USB port

### Step 2: USB Autosuspend (MOST COMMON ISSUE)
```bash
cat /sys/bus/usb/devices/BUS-PORT/power/control
```
- **`auto`** → THIS IS THE PROBLEM. Fix: `echo "on" | sudo tee ...`
- **`on`** → autosuspend already disabled, problem is elsewhere

The udev rule in `local-bootstrapping` should prevent this, but may not be applied on all machines.

### Step 3: V4L2 Device
```bash
v4l2-ctl --device=/dev/video0 --get-input
v4l2-ctl --device=/dev/video0 --get-fmt-video
```
- **"Camera 1: ok"** → device is responding
- **Error** → try `/dev/video2`, `/dev/video4` (device index varies)

### Step 4: Actual Capture Test
```bash
ffmpeg -y -f v4l2 -input_format mjpeg -video_size 1920x1080 -i /dev/video0 -frames:v 5 -update 1 /tmp/hdmi_test.png
ls -la /tmp/hdmi_test.png
```
- **>100KB** → Windows desktop with content
- **20-100KB** → Partial signal or simple desktop (wallpaper only)
- **<20KB** → Black/solid screen = no HDMI signal from source

### Step 5: Source Machine
If capture card works but still black:
- Is the Windows laptop lid open / screen on?
- Is Windows outputting to the HDMI port? (Win+P → Duplicate/Extend)
- Is the laptop in sleep/hibernate mode?
- Try moving the ESP32 mouse to wake it: `mouse._send_raw(10, 0)`

## Known Device Paths

| Machine | USB Port | sysfs Path | Video Device |
|---------|----------|------------|--------------|
| Laptop (ThinkPad T14) | USB-A left | 5-4 | /dev/video0 |
| Desktop | TBD | TBD | /dev/video4 |

*Note: sysfs path changes if you plug into a different USB port.*

## Prevention

The udev rule in `local-bootstrapping/system/udev-rules/99-usb-capture.rules` should auto-disable autosuspend. If it's not working:
```bash
# Check if rule is installed
cat /etc/udev/rules.d/99-usb-capture.rules
# Reinstall from local-bootstrapping
sudo cp ~/Programs/local-bootstrapping/system/udev-rules/99-usb-capture.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```
