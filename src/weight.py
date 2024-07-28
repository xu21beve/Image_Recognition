import time
import sys
import RPi.GPIO as GPIO
from hx711 import HX711

leftHx = HX711(17, 27)
rightHx = HX711(23, 22)

def totalWeight():
	return leftHx.rawBytesToWeight(leftHx.readRawBytes()) + rightHx.rawBytesToWeight(rightHx.readRawBytes())

leftHx.setReadingFormat("MSB", "MSB")
rightHx.setReadingFormat("MSB", "MSB")

print("[INFO] Automatically setting the offset.")
leftHx.autosetOffset()

leftOffsetValue = leftHx.getOffset()

rightHx.autosetOffset()
rightOffsetValue = rightHx.getOffset()
print(f"[INFO] Finished automatically setting the left offset. The new value is '{leftOffsetValue}'.")
print(f"[INFO] Finished automatically setting the right offset. The new value is '{rightOffsetValue}'.")


print("[INFO] You can add weight now!")


leftReferenceUnit = (204500 + 180000) / 495.43
rightReferenceUnit = (204500 + 180000) / 495.43

print(f"[INFO] Setting the 'leftReferenceUnit' at {leftReferenceUnit}.")
leftHx.setReferenceUnit(leftReferenceUnit)
print(f"[INFO] Finished setting the 'leftReferenceUnit' at {leftReferenceUnit}.")

print(f"[INFO] Setting the 'rightReferenceUnit' at {rightReferenceUnit}.")
rightHx.setReferenceUnit(rightReferenceUnit)
print(f"[INFO] Finished setting the 'rightReferenceUnit' at {rightReferenceUnit}.")


print("[INFO] Enabling the callback.")
leftHx.enableReadyCallback()
rightHx.enableReadyCallback()
print("[INFO] Finished enabling the callback.")
