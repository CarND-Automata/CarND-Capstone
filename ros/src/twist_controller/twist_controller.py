from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class TwistController(object):
    def __init__(self, parameters):
        # TODO: Implement
        self.parameters = parameters

        self.throttle_controller = PID(kp=5, ki=0.5, kd=0.5, mn=parameters['decel_limit'], mx=parameters['accel_limit'])
        self.throttle_lpf = LowPassFilter(tau=3, ts=1)

        self.steering_controller = YawController(parameters['wheel_base'], parameters['steer_ratio'], parameters['min_speed'],
                                            parameters['max_lat_accel'], parameters['max_steer_angle'])
        self.steering_lpf = LowPassFilter(tau=3, ts=1)


    def control(self, twist_cmd, current_twist, dt):
        # TODO: Change the arg, kwarg list to suit your needs
        target_velocity = twist_cmd.twist.linear.x
        target_yaw_rate = twist_cmd.twist.angular.z

        current_velocity = current_twist.twist.linear.x

        velocity_error = target_velocity - current_velocity

        throttle = self.throttle_controller.step(velocity_error, dt)
        # throttle = self.throttle_lpf.filt(throttle)

        steer = self.steering_controller.get_steering(target_velocity, target_yaw_rate, current_velocity)
        steer = self.steering_lpf.filt(steer)



        if throttle > 0.0:
            brake = 0.0
        else:
            brake = -throttle
            throttle = 0.0
        #
        #     if deceleration < self.cp.brake_deadband:
        #         deceleration = 0.0
        #
        #     brake = deceleration * (self.cp.vehicle_mass + self.cp.fuel_capacity * GAS_DENSITY) * self.cp.wheel_radius

        # Return throttle, brake, steer
        return throttle, brake, steer


    def reset(self):
        self.throttle_controller.reset()
