# Examples Go-to-Point File
# Christopher Banks
# 10/14/18
# This file provides an example of how to send commands to the Robotarium for moving a number of quadcopters from one pose
# to another.
from utilities.Robotarium import RobotariumEnvironment

TIMEOUT = False

if __name__ == "__main__":
    # creates robotarium object, indictate if user would like to save data
    robotarium = RobotariumEnvironment(save_data=True)

    # robotarium object sets a random number of agents to be created

    # sets the number of agents
    # robotarium.number_of_agents = 5

    #iterate until time limit is reached, the max time our quadcopters can run is 5 minutes so experiments will be
    #limited to that time

    #Iteration method is arbitrary, we will provide a function to check if experiment will timeout however

    # instantiates Robotarium and initializes quadcopters
    robotarium.build()

    while TIMEOUT is False:
        # retrieve quadcopter poses (numpy array, n x m x 3) where n is the quadcopter index
        x = robotarium.get_quadcopter_poses()

        # Insert code here

        # Set desired pose
        robotarium.set_desired_pose(x_desired)


        # send pose commands to robotarium
        robotarium.update_pose()

        #call timeout function to check if algorithm will run to completion
        TIMEOUT = robotarium.check_timeout()

        # also users can self limit runtime of program
        if robotarium.run_time() > 60:
            TIMEOUT = True






