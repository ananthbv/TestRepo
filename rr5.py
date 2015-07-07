# ----------
# Part Five
#
# This time, the sensor measurements from the runaway Traxbot will be VERY 
# noisy (about twice the target's stepsize). You will use this noisy stream
# of measurements to localize and catch the target.
#
# ----------
# YOUR JOB
#
# Complete the next_move function, similar to how you did last time. 
#
# ----------
# GRADING
# 
# Same as part 3 and 4. Again, try to catch the target in as few steps as possible.

from robot import *
from math import *
from matrixn import *
import random
from numpy import *
from copy import *
from scipy import optimize


def next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER = None):
    # This function will be called after each time the target moves. 

    # The OTHER variable is a place for you to store any historical information about
    # the progress of the hunt (or maybe some localization information). Your return format
    # must be as follows in order to be graded properly.
    return turning, distance, OTHER

def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def demo_grading(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we
    will grade your submission."""
    max_distance = 0.97 * target_bot.distance # 0.98 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0
    #For Visualization
    import turtle
    window = turtle.Screen()
    window.bgcolor('white')
    chaser_robot = turtle.Turtle()
    chaser_robot.shape('arrow')
    chaser_robot.color('blue')
    chaser_robot.resizemode('user')
    chaser_robot.shapesize(0.3, 0.3, 0.3)
    broken_robot = turtle.Turtle()
    broken_robot.shape('turtle')
    broken_robot.color('green')
    broken_robot.resizemode('user')
    broken_robot.shapesize(0.3, 0.3, 0.3)
    size_multiplier = 15.0 #change size of animation
    chaser_robot.hideturtle()
    chaser_robot.penup()
    chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
    chaser_robot.showturtle()
    broken_robot.hideturtle()
    broken_robot.penup()
    broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
    broken_robot.showturtle()
    measuredbroken_robot = turtle.Turtle()
    measuredbroken_robot.shape('circle')
    measuredbroken_robot.color('red')
    measuredbroken_robot.penup()
    measuredbroken_robot.resizemode('user')
    measuredbroken_robot.shapesize(0.1, 0.1, 0.1)
    broken_robot.pendown()
    chaser_robot.pendown()
    #End of Visualization
    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:
        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print "You got it right! It took you ", ctr, " steps to catch the target."
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)

        # Don't try to move faster than allowed!
        if distance > max_distance:
            distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()
        #Visualize it
        measuredbroken_robot.setheading(target_bot.heading*180/pi)
        measuredbroken_robot.goto(target_measurement[0]*size_multiplier, target_measurement[1]*size_multiplier-100)
        measuredbroken_robot.stamp()
        broken_robot.setheading(target_bot.heading*180/pi)
        broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
        chaser_robot.setheading(hunter_bot.heading*180/pi)
        chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
        #End of visualization
        ctr += 1
        if ctr >= 1000:
            print "It took too many steps to catch the target."
    return caught


def circular_regression(measurements):

    if len(measurements) >= 3:
        xp = []
        yp = []
        other = measurements
        for e in other:
            xp.append(e[0])
            yp.append(e[1])

        x = r_[xp]
        y = r_[yp]

        # coordinates of the barycenter
        x_m = mean(x)
        y_m = mean(y)
        #print x_m, y_m


        method_2 = "leastsq"

        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return sqrt((x-xc)**2 + (y-yc)**2)

        def f_2(c):
            """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        Ri_2 = calc_R(xc_2, yc_2)
        R_2 = Ri_2.mean()

        #print "center, radius", center_2, R_2
        return center_2, R_2
    return [0.0, 0.0], 0.0


    #if OTHER is None:
    #    OTHER = []
    #else:
    #    OTHER.append([target_measurement[0], target_measurement[1]])

    #if len(OTHER) >= 3:
    #    circular_regression(OTHER)


def demo_grading2(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we 
    will grade your submission."""
    max_distance = 0.97 * target_bot.distance # 0.97 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0

    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:

        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print "You got it right! It took you ", ctr, " steps to catch the target."
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)
        
        # Don't try to move faster than allowed!
        if distance > max_distance:
            distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()

        ctr += 1            
        if ctr >= 1000:
            print "It took too many steps to catch the target."
    return caught



def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi

def get_heading(hunter_position, target_position):
    """Returns the angle, in radians, between the target and hunter positions"""
    hunter_x, hunter_y = hunter_position
    target_x, target_y = target_position
    heading = atan2(target_y - hunter_y, target_x - hunter_x)
    heading = angle_trunc(heading)
    return heading


def estimate_curr_pos(measurement2, db_a, estimates, OTHER = None):
    """Estimate the next (x, y) position of the wandering Traxbot
    based on noisy (x, y) measurements."""

    measurement = list(measurement2)
    #print measurement[0], measurement[1]
    #print OTHER

    if not OTHER:
        OTHER = [[0.0, 0.0], 0.0]

    previous = [OTHER[0], OTHER[1]]
    current_angle = atan2(measurement[1] - previous[1], measurement[0] - previous[0])
    previous_angle = OTHER[2]
    d = sqrt((measurement[0] - previous[0]) ** 2 + (measurement[1] - previous[1]) ** 2)
    db_a.append(d)
    db_a2 = r_[db_a]
    db_m = mean(db_a2)
    #d = db_m
    print "dbm = ", db_m


    estimate_angle = current_angle + current_angle - previous_angle
    print "previous = " + str(previous_angle), "current = " + str(current_angle), "estimate = " + str(estimate_angle)
    d = db_m
    xy_estimate = [measurement[0] + d * cos(estimate_angle), measurement[1] + d * sin(estimate_angle)]
    OTHER = [measurement, current_angle]

    # You must return xy_estimate (x, y), and OTHER (even if it is None)
    # in this order for grading purposes.
    return xy_estimate, estimate_angle, db_a, OTHER


def ekf(x, P, measurements, estimates, center, radius, db_a):

    #for n in range(len(measurements)):
    # measurement update

    xy_estimate, estimate_angle, db_a, temp = estimate_curr_pos(measurements[-1], db_a, estimates, estimates[-1])
    estimate_angle = temp[-1]

    x = matrixn([[xy_estimate[0]], [xy_estimate[1]], [estimate_angle]])
    alphap = estimate_angle
    F = matrixn([[-1.0 * radius * sin(alphap),  1.0, 0.0],
                [0.0, radius * cos(alphap), 1.0],
                [0.,                          0.0, 1.0]])

    P = F * P * F.transpose() #+ Q
    C = matrixn([[1., 0., 0.], [0., 1., 0.]])
    z = matrixn([measurements[-1]])
    y = z.transpose() - (H * x)
    S = (H * P * H.transpose()) + R

    K = P * H.transpose() * S.inverse()

    x = x + (K * y)
    P = (I - K * H) * P

    temp = x.value
    d = distance_between(xy_estimate, [temp[0][0], temp[1][0]])
    xe = temp[0][0] + d * cos(alphap + temp[2][0])
    ye = temp[1][0] + d * sin(alphap+ temp[2][0])
    ae = (temp[2][0] + alphap) % 2.0 * pi
    xe = temp[0][0]
    ye = temp[1][0]
    ae = estimate_angle #temp[2][0] % 2.0 * pi
    # prediction
    #x = F * x + u
    #P = F * P * F.transpose()
    #print "--------", xe, ye, ae
    return xe, ye, ae, db_a



def naive_next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER):
    """This strategy always tries to steer the hunter directly towards where the target last
    said it was and then moves forwards at full speed. This strategy also keeps track of all 
    the target measurements, hunter positions, and hunter headings over time, but it doesn't 
    do anything with that information."""
    center = [0.0, 0.0]
    radius = 0.0

    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = [target_measurement]
        hunter_positions = [hunter_position]
        hunter_headings = [hunter_heading]
        estimates = [[0.0, 0.0, 0.0]]
        db_a = [0.0]
        pending_steps = 0
        OTHER = (measurements, hunter_positions, hunter_headings, estimates, db_a, pending_steps) # now I can keep track of history
    else: # not the first time, update my history
        OTHER[0].append(target_measurement)
        OTHER[1].append(hunter_position)
        OTHER[2].append(hunter_heading)
        measurements, hunter_positions, hunter_headings, estimates, db_a, pending_steps = OTHER # now I can always refer to these variables

    #print estimates
    print "target = ", target.x, target.y, target.heading
    center, radius = circular_regression(measurements)
    print "measurement center, radius: ", center, radius
    print "estimates center, radius: ", circular_regression(estimates)

    x_estimate, y_estimate, angle_estimate, db_a = ekf(x, P, measurements, estimates, center, radius, db_a)

    #print OTHER[3]

    OTHER[3].append([x_estimate, y_estimate, angle_estimate])

    turnangle = []
    allds = 0.0
    for i in range(len(OTHER[3])):
        if i == 0:
            turnangle.append(abs(OTHER[3][i][-1]))
        else:
            turnangle.append(abs(OTHER[3][i][-1] - OTHER[3][i - 1][-1]))
        #print turnangle
        if (len(OTHER[3]) > 1) & i + 1 <= len(OTHER[3]):
            allds += distance_between([OTHER[3][i][0], OTHER[3][i][1]], [OTHER[3][(i + 1) % len(OTHER[3])][0], \
                                                                         OTHER[3][(i + 1) % len(OTHER[3])][1]])

    turnangle = r_[turnangle]
    allds = r_[allds]

    print "avg turn angle, avg dist between 2 points = ", mean(turnangle), (allds/len(OTHER[3]))[0]

    #print "estimates = ", estimates
    #print "measurements = ", measurements
    num = len(estimates)
    xavg = 0.0
    yavg = 0.0
    rev_estimates = copy(estimates)
    rev_estimates.reverse()
    #print rev_estimates
    #xy_estimates = rev_estimates[-3 ::-1*30] + rev_estimates[-3::30]
    xy_estimates = measurements[num - 1::-1*30] + measurements[num - 1 + 30::30] + estimates[num - 1::-1*30] + estimates[num - 1 + 30::30]

    print "len estimate = ", len(xy_estimates)
    """print "---------------", num - 1, estimates[num - 1::-30]
    print "---------------", num - 1, estimates[num - 1 + 30::30]
    print "---------------", num + 3, estimates[(num - 1 +  3) ::-1*30]
    print "---------------", num + 3, estimates[(num -1 + 3 + 30)::30]"""
    #print len(measurements), measurements
    #print xy_estimates
    for i in range(len(xy_estimates)):
        xavg += xy_estimates[i][0]
        yavg += xy_estimates[i][1]

    xavg /= len(xy_estimates)
    yavg /= len(xy_estimates)
    #print x, y

    #print x, y

    print "estimate by avg = ", xavg, yavg, distance_between([xavg, yavg], center)

    x_estimate = xavg
    y_estimate = yavg

    c, r = circular_regression(xy_estimates)
    print "estimate by regression = ", c

    #x_estimate = new_x_estimate
    #y_estimate = new_y_estimate
    #print "next estimate = ", (x_estimate + new_x_estimate)/2.0 , (new_y_estimate + y_estimate)/2.0, angle_estimate

    #heading_to_target = get_heading(hunter_position, target_measurement)
    print "hunter position = " + str(hunter_position), "estimate position = " + str([x_estimate, y_estimate, angle_estimate] )
    heading_to_target = get_heading(hunter_position, [x_estimate, y_estimate])
    heading_difference = heading_to_target - hunter_heading
    turning =  heading_difference # turn towards the target
    distance = distance_between(hunter_position, [x_estimate, y_estimate])
    #if distance > max_distance:
    #    distance = max_distance
    #distance = max_distance # full speed ahead!
    print "--------------------------------------------------"
    return turning, distance, OTHER




target = robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)
measurement_noise = 2.0 * target.distance                        # VERY NOISY!!
#print measurement_noise
target.set_noise(0.0, 0.0, measurement_noise)

hunter = robot(-10.0, -10.0, 0.0)

#measurement_noise = 0.05
x = matrixn([[0.], [0.], [0.0]])                                  # initial state (location and velocity)
P = matrixn([[measurement_noise, 0., 0.], [0., measurement_noise, 0.], [0., 0., measurement_noise]])  # initial uncertainty
u = matrixn([[0.], [0.], [0.]])                                   # external motion
F = matrixn([[1., 0., 0.], [0, 1., 0.], [0, 0., 0.]])             # next state function
H = matrixn([[1., 0., 0.], [0., 1., 0.]])                         # measurement function
R = matrixn([[measurement_noise, 0.],
             [0., measurement_noise]])                                 # measurement uncertainty
I = matrixn([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])           # identity matrix
Q = matrixn([[1.00001 * (measurement_noise ** 2), 0., 0.],
             [0., 1.00001 * (measurement_noise ** 2), 0.],
             [0., 0., 1.00001 * (measurement_noise ** 2)]])

print demo_grading2(hunter, target, naive_next_move)

