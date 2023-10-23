import math
pi=math.pi

def getpin(EN):
  dic={'REN1': 7, 'REN2': 8, 'LEN1': 12, 'LEN2': 13, 'RPWM1': 5, 'RPWM2': 6, 'LPWM1': 9, 'LPWM2': 10}
  return dic[EN]
  #This function is actually not needed but just in case :")

LL = 2
LM = 11
#LR = A0  ei line e vul ase 'A0' er bodole ekta integer hobe

D_speed = 100 #going speed
T_speed = 70 #turning speed

RXPin, TXPin, GPSBaud = (4, 3, 9600) #Bluetooth & GPS er jonno, arki amader kono kajer na

#------------------------------------------------------------------------------

qmc = [110,-1202, 1256] #Random Compass data as a list 

gps = [20.3799028, 79.6410455] #1st GPS loaction (starting point)

destination = lat_B,lon_B = [23.7505615, 90.3617057] #Destination Coordinates

#------------------------------------------------------------------------------

def getCompassDirection(qmc):  
  x = qmc[0]
  y = qmc[1]
  heading = math.atan(y/x)

  if (heading < 0):
    heading = 360 + heading # making anti clockwise or making positive
    
    if (heading > 338 or heading < 22):
      print("NORTH")

    if (heading > 22 and heading < 68):
      print("NORTH-EAST")

    if (heading > 68 and heading < 113):
      print("EAST")

    if (heading > 113 and heading < 158):
      print("SOUTH-EAST")

    if (heading > 158 and heading < 203):
      print("SOUTH")

    if (heading > 203 and heading < 248):
      print("SOTUH-WEST")

    if (heading > 248 and heading < 293):
      print("WEST")

    if (heading > 293 and heading < 338):
      print("NORTH-WEST")
    
    return heading

def setMovingDirection(cmd):
  # void Forward():

    if cmd=="Forward":
    # digitalWrite(REN1, HIGH)
    # digitalWrite(REN2, HIGH)
    # digitalWrite(LEN1, HIGH)
    # digitalWrite(LEN2, HIGH)

      RPWM1= 0
      RPWM2= D_speed
      LPWM1= 0
      LPWM2= D_speed

  # void Turn_Left():

    if cmd=="Left":
    # digitalWrite(REN1, HIGH)
    # digitalWrite(REN2, HIGH)
    # digitalWrite(LEN1, HIGH)
    # digitalWrite(LEN2, HIGH)

      RPWM1= 0
      RPWM2= T_speed
      LPWM1= 0
      LPWM2= T_speed

  #void Turn_Right():

    if cmd=="Right":
    # digitalWrite(REN1, HIGH)
    # digitalWrite(REN2, HIGH)
    # digitalWrite(LEN1, HIGH)
    # digitalWrite(LEN2, HIGH)

      RPWM1= T_speed
      RPWM2= 0
      LPWM1= T_speed
      LPWM2= 0
    print(cmd)


#setMovingDirection("Forward") #function calling for moving Forward

lat_A = gps[0]
lon_A = gps[1]

# void loop()
while (gps != destination):

  # if (current_gps != gps):  #gps.location.isUpdated()
  #   # Current GPS
  lat_A = gps[0]
  lon_A = gps[1]

  A_B_lat = lat_A - lat_B  #delx= x2-x1
  A_B_lon = lon_A - lon_B  #dely= y2-y1

  # needed formulas....
  
  a =  math.sin(A_B_lat / 2) * math.sin(A_B_lat / 2) + math.sin(A_B_lon / 2) * math.sin(A_B_lon / 2) * math.cos(lat_A) * math.cos(lat_B)
  b = 2 * math.atan(math.sqrt(a)/math.sqrt(1 - a))
  Distance_A_to_B = 6378.137 * b

  print("Dist A to B: ")
  print(Distance_A_to_B)
  print("km ")
  print( 1000 * Distance_A_to_B)
  print("meter ")

  # needed formulas....

  x = (math.sin(lat_B - lat_A) * math.cos(lon_B))
  y = (math.cos(lon_A) * math.sin(lon_B) - math.sin(lon_A) * math.cos(lon_B) * math.cos(lat_B - lat_A))
  z = math.atan(y/x)

  # angel from north
  
  bearing = ((z * 180 / pi + 360))
  bearing = bearing  % 360
  print("bearing :")
  print(bearing)

  heading = getCompassDirection(qmc) #the function will prthe compass direction and geting the heading

  if ((bearing - 10) <= heading and heading <= (bearing + 10)):
    
    setMovingDirection("Forward")
    #Forward()
    # digitalWrite(LL, LOW)
    # digitalWrite(LM, HIGH)
    # analogWrite(LR, 0)

  elif (bearing < heading):

    setMovingDirection("Left")
    #Turn_Left()
    # digitalWrite(LL, LOW)
    # digitalWrite(LM, LOW)
    # analogWrite(LR, 255)
  
  elif (bearing > heading):
    
    setMovingDirection("Right")
    #Turn_Right()
    # digitalWrite(LL, HIGH)
    # digitalWrite(LM, LOW)
    # analogWrite(LR, 0)
