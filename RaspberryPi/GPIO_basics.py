import RPi.GPIO as GPIO
import time 

#############################
""" Simple LED On and OFF with time """	
def Led_On_Off(OUT_PIN):
	GPIO.setup(OUT_PIN, GPIO.OUT)
	GPIO.output(OUT_PIN, GPIO.HIGH)
	time.sleep(5)
	GPIO.output(OUT_PIN, GPIO.LOW)
	GPIO.cleanup()

#############################
""" DISTANCE SENSOR """

def Distance_Sensor(OUT_PIN):
	TRIG=4

	GPIO.setup(OUT_PIN, GPIO.OUT)
	GPIO.setup(TRIG, GPIO.IN)

	GPIO.output(TRIG, true)
	time.sleep(0.0001)
	GPIO.output(TRIG, false)
	GPIO.output(OUT_PIN, GPIO.LOW)
	GPIO.cleanup()

###########################
""" PWM Controlled LED """

def PWM_Led(OUT_PIN):
	FREQ=50   # 50Hz
	STEP=5

	GPIO.setup(OUT_PIN, GPIO.OUT)
	p = GPIO.PWM(OUT_PIN, FREQ)
	p.start()
	try:
		while 1:
			for dc in range(0, 101, STEP):
				p.ChangeDutyCycle(dc)
				time.sleep(0.1)
			for dc in range(100, -1, -STEP):
				p.ChangeDutyCycle(dc)
				time.sleep(0.1)
	except KeyBoardInterrupt:
		print(" Ended ")
	p.stop()
	GPIO.cleanup()

#############################

if __name__ == '__main__':
	OUT_PIN=18

	# Configuration
	GPIO.setmode(GPIO.BCM)

	Led_On_Off(OUT_PIN)
	Distance_Sensor(OUT_PIN)
	PWM_Led(OUT_PIN)