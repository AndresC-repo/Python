# GPIO
import RPi.GPIO as GPIO

## Configuration of Pin numbering
GPIO.setmode(GPIO.---)
- BOARD: pin numbers on the PI header of RaspberryPi based hardware will always work. They are printed on boards.
- BCM: Broadcom SOC channel number. Changes between versions.

## Setup a Channel

### INPUT
GPIO.setup(channel, GPIO.IN)

Pull-up/pull-down: When input is not connected value read is undefined -> set pull-resistor -> sets to default value
HW:
- pull-up: resistor from input to 3.3V
- pull-down: resistor from input to 0V
SW: GPIO.setup(GPIO.IN, pull_up_down=GPIO.PULL_UP)

+ Snapshot at anytime: if GPIO.input(channel):
+ Interrupts and edge Detections / Event: Change in state of input
	- wait_for_edge(channel, GPIO.FALLING/RISING/BOTH, timeout=5000)  # blocks execution of program until edge is detected can also be added some waiting time.
	- event_detected(): to be used in loops but wont miss the change in state
	'''
	GPIO.add_event_detected(channel, GPIO.RISING)
	...
	if GPIO.event_detected(channel):
		print("detected")
	'''
+ Threaded Callbacks: runs a second thread for a callback. These functions get inmediate response.
	'''
	def callback_name(channel):
	...

	GPIO.add_event_detect(channel, GPIO.RISING, callback=callback_name)
	
	GPIO.remove_event_detect(channel)
	'''

### OUTPUT
GPIO.setup(channel, GPIO.OUT)
with initial value GPIO.setup(channel, GPIO.OUT, initial=GPIO.HIGH/LOW)

channel_list=[1, 2]
GPIO.setup(channel_list, GPIO.OUT)

### PWM
create an instance ''' p = GPIO.PWM(channel, FREQ)'''
FREQ: 1/temp
DC: Duty cycle (0-100)
Channel needs to be set as output

p.ChangeFrequency(FREQ)
p.ChangeDutyCycle(DC)
p.stop

### CLEAN UP
GPIO.cleanp()
or a single channel GPIO.cleanup(channel)




