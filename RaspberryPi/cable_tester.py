import RPi.GPIO as GPIO
import time 
"""
Pi setup is BOARD
LED OUT is Pin 3
Cable's Heads should be connected as
Cable pins:
"""
CAN_High = [7,11,12]
CAN_Low = [13,15,16]
LV = [18,22,29]
HV = [31,32,33]
LIGHT_OUT = [35,36,37]
GND = [38,40, 26]
FUNCTION_LIST = [CAN_High, CAN_Low, LV, HV, LIGHT_OUT, GND]

BLINK_CT = 3

ERROR_LIST = ["No_error", "Error_High", "ERROR_Low"]
ERROR_NO_ERROR = 0
ERROR_NO_HIGH = 1
ERROR_NO_LOW = 2
ERROR_RANGE = 2  # High and low

FUNCTIONS_TOTAL = 6

### LED ###
PIN_LED = 24

# class error to be expanded with more error descritpion
class error_type:
	def __init__(self):
		# list containing 6 lists, 2 items set to 0
		self.error_list = [[0 for x in range(ERROR_RANGE)] for y in range(FUNCTIONS_TOTAL)] 

	# CAN HIGH
	def set_error_CAN_H(self, error):
		self.error_list[0] = error

	def get_error_CAN_H(self):
		return self.error_list[0]

	# CAN LOW
	def set_error_CAN_L(self, error):
		self.error_list[1] = error

	def get_error_CAN_L(self):
		return self.error_list[1]

	# LV
	def set_error_LV(self, error):
		self.error_list[2] = error

	def get_error_LV(self,):
		return self.error_list[2]

	# HV
	def set_error_HV(self, error):
		self.error_list[3] = error

	def get_error_HV(self):
		return self.error_list[3]
	
	# LIGHT_OUT
	def set_error_LIGHT_OUT(self, error):
		self.error_list[4] = error

	def get_error_LIGHT_OUT(self):
		return self.error_list[4]

	# GND
	def set_error_GND(self, error):
		self.error_list[5] = error

	def get_error_GND(self):
		return self.error_list[5]


	def set_status_all(self, err_list):
		self.error_list = err_list

	# Return status of all
	def return_status_all(self):
		return self.error_list

	def error_in_cable(self):
		return sum(sum(self.return_status_all(),[]))

"""
Test one cable pin functionality combo at a time.
In a three headed cable:
One connectio is output | two are inputs
For BLINK_CT times:
	Output HIGH -> Inputs check if HIGH
	Output LOW -> Inputs check if LOW
"""
def test_connection(cable_list):
	error_high, error_low = 0, 0

	for idx, pin in enumerate(cable_list):
		blink_counter_high = 0
		blink_counter_low = 0

		GPIO.cleanup(cable_list)
		new_list = cable_list[:idx] + cable_list[idx+1:]
		print("cable_list (input)", new_list, " Output ", pin)
		GPIO.setup(pin, GPIO.OUT)
		GPIO.setup(new_list, GPIO.IN)
		time.sleep(0.2)
		GPIO.setup(new_list, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # check

		for blink in range(BLINK_CT):
			# Test High
			time.sleep(0.2)
			GPIO.output(pin, GPIO.HIGH)
			time.sleep(0.2)
			if (GPIO.input(new_list[0]) & (GPIO.input(new_list[1]))):
			    blink_counter_high += 1
			else:
			    pass
			# Test Low
			time.sleep(0.2)
			GPIO.output(pin, GPIO.LOW)
			if (GPIO.input(new_list[0]) & (GPIO.input(new_list[1]))):
			    pass
			else:
			    blink_counter_low += 1
		if (blink_counter_high != BLINK_CT):
			error_high = ERROR_NO_HIGH
		if (blink_counter_low != BLINK_CT):
			error_low = ERROR_NO_LOW
	print("errors", error_high, error_low)
	return [error_high, error_low]

# -------------------------- #
# 		Main				 #
# -------------------------- #

if __name__ == '__main__':

	# Back-up error list
	error_list = [[0 for x in range(ERROR_RANGE)] for y in range(FUNCTIONS_TOTAL)]
	# Create error object
	total_error = error_type()
	# Configuration 
	GPIO.setmode(GPIO.BOARD)
	# Test all (6) different connections in cable
	for idx, connection in enumerate(FUNCTION_LIST):
		error_list[idx] = test_connection(connection)
	# load errors into erro object
	total_error.set_status_all(error_list)
	# --- get status of cable example -----
	error_canH_high, error_canH_low = total_error.get_error_CAN_H()
	print("error_canH_high: ", ERROR_LIST[error_canH_high])
	print("error_canH_low: ", ERROR_LIST[error_canH_low])
	# Turn led ON if no errors are present
	if(total_error.error_in_cable()):
		GPIO.setup(PIN_LED, GPIO.OUT)
		GPIO.output(PIN_LED, GPIO.HIGH)
		print(" ---- LED ON ---- ")
		time.sleep(5)
		GPIO.output(PIN_LED, GPIO.LOW)
		print(" ---- LED OFF ---- ")
	GPIO.cleanup()



