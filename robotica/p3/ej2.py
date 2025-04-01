from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from tkinter import simpledialog
import re


class RoboboController:
    def __init__(self, robobo, sim, default_speed=20, default_time=4):
        self.robobo = robobo
        self.sim = sim
        self.default_speed = default_speed
        self.default_time = default_time

        # Dictionary of movement strategies - lambda functions that return wheel speeds
        self.movement_strategies = {
            "forward": (lambda speed: (speed, speed)),
            "back": (lambda speed: (-speed, -speed)),
            "left": (lambda speed: (-speed, speed)),
            "right": (lambda speed: (speed, -speed)),
        }

        # Adding compound movements
        self.compound_movements = {
            "forward-left": self._move_forward_left,
            "forward-right": self._move_forward_right,
            "back-left": self._move_back_left,
            "back-right": self._move_back_right,
        }

    def _move_forward_left(self, speed, time=None):
        """Execute a forward-left compound movement"""
        self.robobo.moveWheelsByTime(
            speed // 2, speed, self.default_time if time is None else time
        )
        return True

    def _move_forward_right(self, speed, time=None):
        """Execute a forward-right compound movement"""
        self.robobo.moveWheelsByTime(
            speed, speed // 2, self.default_time if time is None else time
        )
        return True

    def _move_back_left(self, speed, time=None):
        """Execute a back-left compound movement"""
        self.robobo.moveWheelsByTime(
            -speed, -speed // 2, self.default_time if time is None else time
        )
        return True

    def _move_back_right(self, speed, time=None):
        """Execute a back-right compound movement"""
        self.robobo.moveWheelsByTime(
            -speed // 2, -speed, self.default_time if time is None else time
        )
        return True

    def parse_command(self, user_response):
        """
        Parse the user command and move the robot
        """
        if user_response is None or user_response.strip() == "":
            print("No command entered")
            return None

        # Standardize input
        user_response = user_response.lower().strip()

        if user_response == "stop" or user_response == "quit":
            print("Stopping motors")
            self.robobo.stopMotors()
            if user_response == "quit":
                self.sim.disconnect()
                return False
            return True

        # Check for compound movements first
        for compound_move, move_function in self.compound_movements.items():
            if compound_move in user_response:
                # Extract speed and time parameters
                speeds = re.findall(r"speed (\d+)", user_response)
                current_speed = int(speeds[0]) if speeds else self.default_speed

                time_matches = re.findall(r"\btime (\d+(?:\.\d+)?)\b", user_response)
                move_time = float(time_matches[0]) if time_matches else None

                print(
                    f"Executing compound movement: {compound_move} with speed: {current_speed}"
                    + (f" for {move_time} seconds" if move_time else "")
                )

                return move_function(current_speed, move_time)

        # Get directions and speeds from text
        directions = re.findall(r"\b(forward|back|left|right)\b", user_response)

        # If no direction found, report error
        if not directions:
            print("No valid direction found in command")
            return True

        # Improved speed parsing
        speeds = re.findall(r"speed (\d+)", user_response)
        current_speed = int(speeds[0]) if speeds else self.default_speed

        # Improved time parsing
        time_matches = re.findall(r"\btime (\d+(?:\.\d+)?)\b", user_response)
        move_time = float(time_matches[0]) if time_matches else self.default_time

        # Execute movements
        for direction in directions:
            if direction in self.movement_strategies:
                # Get wheel speeds (speeds can be different)
                left, right = self.movement_strategies[direction](current_speed)

                # Move the robot
                print(
                    f"Moving {direction} at speed {current_speed} for {move_time} seconds"
                )
                self.robobo.moveWheelsByTime(left, right, move_time)

        return True

    def get_text_command(self):
        """
        Get command from user via dialog box
        """
        return simpledialog.askstring("Input", "Enter command:")


def main(ip="localhost"):
    # Initialize simulator and robot
    sim = RoboboSim(ip)
    sim.connect()
    sim.resetSimulation()

    robobo = Robobo(ip)
    robobo.connect()

    controller = RoboboController(robobo, sim)

    print("Robobo Text Controller")
    print("======================")
    print("Enter text commands to control Robobo.")
    print("Available commands: forward, back, left, right, stop, quit")
    print("You can also use compound commands: forward-left, forward-right, etc.")
    print("You can specify speed: 'forward speed 30'")
    print("You can specify time: 'forward time 3.5'")

    try:
        while True:
            user_response = controller.get_text_command()

            if user_response is None or user_response.lower() == "exit":
                print("Exiting program")
                break

            result = controller.parse_command(user_response)

            if result is False:
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        # Clean up
        robobo.stopMotors()
        sim.disconnect()
        print("Disconnected from Robobo and simulator")


if __name__ == "__main__":
    main()
