import torch
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class WheelFootCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 16
        num_actions = 6

    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'#trimesh
        measure_heights = True
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.
        dynamic_friction = 1.
        restitution = 0.5
        # trimesh only:
        terrain_proportions = [0.0, 0.5, 0.5, 0.5, 0.0, 0.0]
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain_proportions = [0.5]
        # measured_points_x = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]  # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, 0.2, 0.3, 0.4, 0.5]

        """Simulated TOF sensor"""
        measured_points_x = [0.1]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [0.0001]

    class commands:
        curriculum = False
        basic_max_curriculum = 2.5
        advanced_max_curriculum = 1.5
        curriculum_threshold = 0.7
        num_commands = 3  # default: lin_vel_x, height, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 20  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-5, 5]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            height = [0.24, 0.35]
            heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        #rot = [0., 1., 0., 0]
        default_joint_angles = {
            'laj' : 0.,
            'llj' : -1.,
            'lwj' : 0.,
            'raj' : 0.,
            'rlj' : 1.,
            'rwj' : 0.,
        }
        # default_joint_angles = {
        #     'la_joint' : 0,
        #     'll_joint' : 0.,
        #     'lf_joint' : 0,
        #     'ra_joint' : 0.,
        #     'rl_joint' : 0.,
        #     'rf_joint' : 0,
        # }

    class control( LeggedRobotCfg.control ):

        control_type = 'T+P'
        stiffness = {   
                        'laj': 12.8 ,
                        'llj': 14.8 ,
                        'lwj': 8.8 ,
                        'raj': 12.8 ,
                        'rlj': 14.8 ,
                        'rwj': 8.8
                     }
        damping = {
                    'laj': 0.8 ,
                    'llj': 0.8 ,
                    'lwj': 0.8 ,
                    'raj': 0.8 ,
                    'rlj': 0.8 ,
                    'rwj': 0.8
                   }

        # stiffness = {
        #     'la_joint': 14.8,
        #     'll_joint': 16.8,
        #     'lf_joint': 8.8,
        #     'ra_joint': 14.8,
        #     'rl_joint': 16.8,
        #     'rf_joint': 8.8
        # }
        # damping = {
        #     'la_joint': 1.,
        #     'll_joint': 1.25,
        #     'lf_joint': 0.8,
        #     'ra_joint': 1.,
        #     'rl_joint': 1.25,
        #     'rf_joint': 0.8
        # }

        action_scale = 1
        decimation = 10

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/wfs/urdf/wfs.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/standuptest/urdf/standuptest.urdf'
        name = "wheel_foot"
        base_name = "base"
        penalize_contacts_on = ["base", "la", "ra", "ll", "rl"]
        init_orientation = [0, 0., 0.]
        # penalize_contacts_on = ["base", "la", "ra"]
        # terminate_after_contacts_on = ["base", "la", "ra", "ll", "rl"]
        # terminate_after_contacts_on = ["base", "la", "ra"]
        terminate_after_contacts_on = ["base"]
        """TODO:"""
        self_collisions = 1

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = False
        added_mass_range = [-0.5, 0.5]
        randomize_inertia = False
        randomize_inertia_range = [0.1, 0.3]
        randomize_base_com = True
        rand_com_vec = [0.05, 0.05, 0.05]
        randomize_base_pose = False
        randomize_base_pose_range = 0.01
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.9, 1.1]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [0.5, 1.5]
        randomize_action_delay = True
        delay_ms_range = [0, 50]

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            base_height = 10.
            feet_air_time = -1.
            # base_height_enhance = -10.
            lin_vel_z = -1.0
            base_height_acc = -5e-8
            ang_vel_xy = -0.05
            dof_vel = -1e-5
            dof_acc = -2e-7
            # torques = -0.0001
            # torque_limits = -0.01
            dof_pos_limits = -10
            action_rate = -0.5
            action_smooth = -0.5
            orientation = -100.
            collision = -50.
            lr_diff = -6.
            
            
            wheel_vel = -1e-7
        clip_single_rewards = 1.0
        
        soft_dof_pos_limit = (
            0.97 # percentage of urdf limits, values above this limit are penalized
        )
        max_contact_force = 100.
        only_positive_rewards = False
        soft_dof_vel_limit = 1.
        base_height_target = 0.3
        soft_torque_limit = 0.95
        tracking_sigma = 0.25

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            base_acc = 1. 
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 1.0
            height_measurements = 10.0
            torque = 0.1
            scale_actions = 1.

        scale_actions = 1.
        clip_observations = 100.0
        clip_actions = 1.0


    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values

        class noise_scales:
            dof_pos = 0.1
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            height_measurements = 0.1

    class sim:
        dt =  0.001
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 20
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 4
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)



class WheelFootCfgPPO( LeggedRobotCfgPPO ):
    class policy: 
        init_noise_std = 1.
        # actor_hidden_dims = [128, 256, 256, 64]
        # critic_hidden_dims = [128, 256, 256, 64]
        # actor_hidden_dims = [256, 128, 64]
        # critic_hidden_dims = [256, 128, 64]
        # actor_hidden_dims = [256, 256, 256]
        # critic_hidden_dims = [256, 256, 256]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.9
        desired_kl = 0.005
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        # policy_class_name = 'ActorCriticRecurrent'
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 48 # per iteration
        max_iterations = 20000 # number of policy updates
        run_name = ''
        experiment_name = 'wheel_foot'
        # save_interval = 5000
