# -*- encoding: utf-8 -*-
# @Time    : 2024-10-10
# @File    : uncertain_seir_vector.py
import math
import os
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

config = {
 "font.family": "serif",
 "font.serif": ["Times New Roman"],
 "font.size": 16,
 "axes.unicode_minus": False,
 "mathtext.fontset": "stix",
}
plt.rcParams.update(config)


action_to_u0 = torch.tensor([
    0, 0.05, 0.20, 0.5, 1.0
    # 0, 0.2, 0.5, 1.0
], dtype=torch.float32)
action_to_u1 = torch.tensor([
    0, 0.2, 1.0
    # 0, 1.0
], dtype=torch.float32)


class EpidemicModel:
    S : int = 0
    E_undetected : int = 1
    E_detected : int = 2
    I_undetected : int = 3
    I_detected : int = 4
    I_reported : int = 5
    QE : int = 6
    QI : int = 7
    R : int = 8

    E = [E_undetected, E_detected, QE]
    I = [I_undetected, I_detected, I_reported, QI]
    can_isolated = [E_detected, I_reported, I_detected]

    def __init__(self, args, env_count=20, is_evaluation=False):
        args_dict = vars(args)
        self.city = args.city  # city name
        self.env_count = int(env_count)  # number of parallel environments
        self.is_evaluation = is_evaluation
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.device_name) # cuda or cpu
        # self.action_to_u = action_to_u.to(self.device)
        self.action_to_u0 = action_to_u0.to(self.device)
        self.action_to_u1 = action_to_u1.to(self.device)
        self.action_dim = self.action_to_u0.size(0) * self.action_to_u1.size(0)

        data_dir = '../data/' if 'env_data_dir' not in args_dict else args.env_data_dir
        data_dir += self.city  # select data directory based on city name

        self.action_max = max(self.action_to_u0.shape[0], self.action_to_u1.shape[0])
        if args.simulate_scale == 'district':
            self.OD = torch.from_numpy(np.load(data_dir + '/flow.npy')).float().to(self.device)
            self.POP = torch.from_numpy(np.load(data_dir + '/population.npy')).float().to(self.device)
        elif args.simulate_scale == 'community':
            data_dir += f'/community_{args.zone_num}'
            self.OD = torch.from_numpy(np.load(data_dir + '/flow.npy')).float().to(self.device)
            self.POP = torch.from_numpy(np.load(data_dir + '/population.npy')).float().to(self.device)
        self.OD = self.OD / self.OD.sum(dim=-1, keepdim=True)
        self.ZONE_NUM = self.POP.shape[0]

        # Add dimension 0 for batch operations, corresponding to env_count
        self.OD = self.OD.unsqueeze(0).expand(env_count, -1, -1)
        self.original_OD = self.OD.clone()  # store original flow matrix
        self.contagious_OD = self.OD.clone()  # flow matrix for infected individuals
        self.POP = self.POP.unsqueeze(0).expand(env_count, -1)
        self.TOTAL_POP = self.POP.sum(dim=-1)

        # Epidemic parameters (infected mobility ratio, beta, latent period, recovery period)
        self.Pm = torch.tensor(0.8 if "ODE_Pm" not in args_dict else args.ODE_Pm, device=self.device)
        self.beta = 0.8 if "ODE_beta" not in args_dict else args.ODE_beta
        self.R0 = 'high' if "R0" not in args_dict else args.R0
        self.betas = torch.full((self.ZONE_NUM, ), self.beta, device=self.device)
        self.betas = self.betas.unsqueeze(0).expand(env_count, -1)
        self.beta_E_rate = torch.tensor(0.5, device=self.device)
        self.original_betas = self.betas.clone()  # store original beta values
        self.sigma = torch.tensor(1 / 3 if "ODE_sigma" not in args_dict else args.ODE_sigma, device=self.device)
        self.gamma = torch.tensor(1 / 7 if "ODE_gamma" not in args_dict else args.ODE_gamma, device=self.device)
        self.gamma_q = torch.tensor(1 / 7 if "ODE_gamma_q" not in args_dict else args.ODE_gamma, device=self.device)
        # Detection rate for E and I compartments
        self.detect_E_rate = torch.tensor(0.60 if "ODE_detect_E_rate" not in args_dict else args.ODE_detect_E_rate, device=self.device)
        self.detect_I_rate = torch.tensor(0.95 if "ODE_detect_I_rate" not in args_dict else args.ODE_detect_I_rate, device=self.device)
        # Detection efficiency function exponent parameter
        self.detection_efficiency_exp_param = torch.tensor(0.6 if "detection_efficiency_exp_param" not in args_dict else args.detection_efficiency_exp_param, device=self.device)
        # Reporting rate for I compartment
        self.report_I_rate0 = 1.0 if "ODE_report_I_rate" not in args_dict else args.ODE_report_I_rate

        # Daily imported cases
        self.daily_imported_cases = 0 if "daily_imported_cases" not in args_dict else args.daily_imported_cases
        # Whether to use delayed quarantine
        self.use_delay_quara = False if "use_delay_quara" not in args_dict else args.use_delay_quara

        # Medical capacity and ylim for different cities (for plotting)
        city_capacity = {'sz': 4e6, 'tokyo': 2.2e6, 'nyc': 2e6, 'sh': 5.5e6}
        self.capacity = torch.tensor(city_capacity[self.city], device=self.device)
        # self.capacity = self.capacity
        self.capacity = 0.5 * self.capacity

        self.ylim = self.capacity.item() * 4

        # Simulation period
        self.period = 120 if "ODE_period" not in args_dict else args.ODE_period

        # Observation window size (days)
        self.WINDOW_SIZE = 7 if "WINDOW_SIZE" not in args_dict else args.WINDOW_SIZE
            
        # Observation dimension per zone per step
        self.local_obs_dim = 2 if "local_obs_dim" not in args_dict else args.local_obs_dim
        self.state_contain_detected = False if "state_contain_detected" not in args_dict else args.state_contain_detected
        self.state_contain_Q = False if "state_contain_Q" not in args_dict else args.state_contain_Q
        self.state_contain_R = False if "state_contain_R" not in args_dict else args.state_contain_R
        self.state_contain_detect_rate = False if "state_contain_detect_rate" not in args_dict else args.state_contain_detect_rate
        assert 2 + 2 * self.state_contain_detected + self.state_contain_Q + self.state_contain_R + self.state_contain_detect_rate == self.local_obs_dim, "local_obs_dim error"

        # Whether to use reward shaping
        self.use_reward_shaping = False if "use_reward_shaping" not in args_dict else args.use_reward_shaping

        # Reward function weights
        self.reward_weights = [1, 1, 1] if "reward_weights" not in args_dict else args.reward_weights
        self.cost_ratio_t2q = 1/8 if "cost_ratio_test_to_quarantine" not in args_dict else args.cost_ratio_test_to_quarantine
        self.local_reward_weight = 1.0 if "local_reward_weight" not in args_dict else args.local_reward_weight
        # Whether to disable responsibility distribution in reward function
        self.disable_response_distribution = False if "disable_response_distribution" not in args_dict else args.disable_response_distribution

        """ Uncertainty 1: Incomplete observation """
        # Incomplete observation based on reporting rate
        # Whether to apply incomplete observation to new I cases
        self.use_obs_imperfect = False if "use_obs_imperfect" not in args_dict else args.use_obs_imperfect
        self.obs_imperfect_up = 1.0 if "obs_imperfect_up" not in args_dict else args.obs_imperfect_up
        self.obs_imperfect_down = 0.1 if "obs_imperfect_down" not in args_dict else args.obs_imperfect_down
        if self.obs_imperfect_down > self.obs_imperfect_up:
            raise ValueError("obs_imperfect_down > obs_imperfect_up", self.obs_imperfect_down, self.obs_imperfect_up)
        self.mask_rate_down = None if "mask_rate_down" not in args_dict else args.mask_rate_down
        self.mask_rate_up = None if "mask_rate_up" not in args_dict else args.mask_rate_up
        self.mask_duration_down = None if "mask_duration_down" not in args_dict else args.mask_duration_down
        self.mask_duration_up = None if "mask_duration_up" not in args_dict else args.mask_duration_up

        """ Uncertainty 2: Sudden abnormal events """
        # Abnormal spatiotemporal behavior (not enabled by default, triggered externally)
        self.gather_to_some_region = False if "gather_to_some_region" not in args_dict else args.gather_to_some_region  # I gathering to certain regions

        """ Uncertainty 3: Transmission parameter variation """
        # Beta variation causes state transition uncertainty
        self.use_beta_change = False if "use_beta_change" not in args_dict else args.use_beta_change  # beta variation
        self.beta_matrix = None if "beta_matrix" not in args_dict else args.beta_matrix  # beta variation matrix
        self.original_beta_matrix = self.beta_matrix.clone() if self.beta_matrix is not None else None  # store original beta matrix
        self.state_contain_beta = False if "state_contain_beta" not in args_dict else args.state_contain_beta  # whether to include beta in state
        if self.state_contain_beta: raise NotImplementedError
        
        """ Uncertainty 4: Action execution uncertainty """
        # Uncertainty in action execution effects
        self.use_action_uncertainty = False if "use_action_uncertainty" not in args_dict else args.use_action_uncertainty  # action execution uncertainty

    def seed(self, seed=3047):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def reset(self, rand_idxs=None):
        """
        Reset the simulation environment to initial state.

        This function:
        - Resets the current day to 1.
        - Initializes action array as zero vector for all zones.
        - Clears historical cost data (reward, sdo, fdo, ado).
        - Sets simulation state including population initialization.
        - Resets simulation result arrays.
        - Computes and returns initial observation.

        Returns:
            Observation array representing current environment state.
        """
        self.day = 1
        # Initialize action array
        self.actions = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # Initialize historical cost data, each with shape (env_count, period)
        self.history_cost = {
            "reward": torch.zeros((self.env_count, self.period, self.ZONE_NUM), device=self.device),
            "test_num": torch.zeros((self.env_count, self.period, self.ZONE_NUM), device=self.device),
            "quara_num": torch.zeros((self.env_count, self.period, self.ZONE_NUM), device=self.device),
            "quara_days": torch.zeros((self.env_count, self.period, self.ZONE_NUM), device=self.device),
            "local_reward": torch.zeros((self.env_count, self.period, self.ZONE_NUM), device=self.device),
            "global_reward": torch.zeros((self.env_count, self.period), device=self.device),
            'local_infe_cost': torch.zeros((self.env_count, self.period), device=self.device),
            "local_test_cost": torch.zeros((self.env_count, self.period), device=self.device),
            "local_quara_cost": torch.zeros((self.env_count, self.period), device=self.device),
            "global_infe_cost": torch.zeros((self.env_count, self.period), device=self.device),
            "global_test_cost": torch.zeros((self.env_count, self.period), device=self.device),
            "global_quara_cost": torch.zeros((self.env_count, self.period), device=self.device),
        }

        # Initialize simulation state, expanded to (env_count, ZONE_NUM, 9) 
        # Compartments: S, E_undetected, E_detected, I_undetected, I_detected, I_reported, QE, QI, R
        self.simState = torch.zeros((self.env_count, self.ZONE_NUM, 9), device=self.device).float()
        self.simState[:, :, 0] = self.POP

        # Set initial infection seeds
        self.set_init_seed(rand_idxs=rand_idxs)

        # Initialize simulation result array: (env_count, period+1, ZONE_NUM, 9)
        self.simRes = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, 9), device=self.device)
        self.simRes[:, 0, :, :] = self.simState

        # Initialize daily new event arrays
        # Information to rebuild 2: daily new E
        self.daily_new_E = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_new_E[:, 0, :] = self.simState[:, :, self.E_undetected]
        self.daily_new_I = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_new_Q = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_new_report = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # Record daily testing and quarantine
        self.daily_test_num = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_quara_num = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # Record delayed quarantine
        self.daily_delay_quara_num = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # Record new E caused by infected in each zone (for reward calculation per zone)
        self.daily_new_E_self_cause = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # Record daily recoveries from known I and quarantine compartments (helps track recovery trends)
        self.daily_known2R = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)
        self.daily_detect_E = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)  # detected new E
        self.daily_detect_I = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)  # detected new I
        self.daily_estimate_new_EI = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM), device=self.device)

        # Randomize detection rates for each zone in each environment
        if self.use_obs_imperfect:
            self.obs_imperfect()

        # Observation missing rate tracking
        self.obs_missing_total_days = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)  # days with missing observation
        self.has_been_observed = torch.zeros((self.env_count, self.ZONE_NUM), dtype=torch.bool, device=self.device)  # whether observed

        # Record historical observations (reduce redundant computation)
        self.history_local_obs = torch.zeros((self.env_count, self.period + 1, self.ZONE_NUM, self.local_obs_dim), device=self.device)

        # Record rebuilt information (optional, empty by default)
        self.rebuild_states = []

        # Build initial observation window
        obs = self._get_obs()  # shape: (env_count, zone_num, window_size, local_obs_dim * 2)

        # Record penalty count
        if self.use_reward_shaping:
            self.penalty_times = 0

        return obs

    def set_init_seed(self, init_infection=100, rand_idxs=None):
        """
        Set initial infection seeds.

        This function:
        - Randomly selects initial infection locations.
        - Adds initial infected population to initial state.
        """
        if rand_idxs is None:
            # Method 1: Randomization
            # random.seed(3074)
            rand_list = [random.randint(0, self.ZONE_NUM - 1) for _ in range(init_infection)]
            rand_idxs = torch.tensor(rand_list, device=self.device)
            # Replicate indices for each batch
            rand_idxs = rand_idxs.repeat(self.env_count, 1)  # shape: (env_count, init_infection)

            # # Method 2: Select top 100 most populated zones
            # rand_idxs = torch.argsort(self.POP, dim=-1, descending=True)[:, :init_infection]
            # rand_idxs = rand_idxs.to(self.device)



        # Initialize counter for each batch
        counts = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)

        # Use scatter_add_ for parallel accumulation, update infected count per environment
        counts = counts.scatter_add_(1, rand_idxs, torch.ones_like(rand_idxs, dtype=torch.float))

        # Update corresponding columns in simState
        self.simState[:, :, self.E_undetected] += counts
        self.simState[:, :, self.S] -= counts

    def _imported_cases(self, case_num):
        """Import external cases"""
        # Generate normally distributed case_num_new, ensure within 0-2*case_num
        while True:
            sampled_value = random.gauss(case_num, case_num/2)
            if 0 <= sampled_value <= 2*case_num:
                case_num_new = round(sampled_value)
                break

        if case_num_new == 0:
            return

        case_num = case_num_new
        rand_list = [random.randint(0, self.ZONE_NUM - 1) for _ in range(case_num)]
        rand_idxs = torch.tensor(rand_list, device=self.device)
        # Replicate indices for each batch
        rand_idxs = rand_idxs.repeat(self.env_count, 1)  # shape: (env_count, init_infection)

        # Initialize counter for each batch
        counts = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)

        # Use scatter_add_ for parallel accumulation, update infected count per environment
        counts = counts.scatter_add_(1, rand_idxs, torch.ones_like(rand_idxs, dtype=torch.float))

        # Update corresponding columns in simState
        self.simState[:, :, self.E_undetected] += counts
        self.simState[:, :, self.S] -= counts


    def obs_imperfect(self):
        """Incomplete observation: construct detection rate and detection delay per zone, should be called on each reset"""
        # 1. Detection rate uniformly distributed in [down, up] for each zone
        up = self.obs_imperfect_up
        down = self.obs_imperfect_down
        self.I_real_report_rate = torch.rand((self.env_count, self.ZONE_NUM), device=self.device) * (up - down) + down  # shape: (env_count, zone_num)



    def _delay_day_distribution(self, day_cnt=5, mean=1, std=1.5):
        """Generate a delay distribution of length day_cnt following normal distribution"""
        days = torch.arange(day_cnt + 1, dtype=torch.float32, device=self.device)
        cdf_values = 0.5 * (1 + torch.erf((days - 0.5 - mean) / (std * math.sqrt(2))))
        probabilities = torch.diff(cdf_values)
        probabilities /= probabilities.sum()
        return probabilities


    def _add_noise(self, x, std = 0.05):
        """Add Gaussian noise"""
        noise = 1 + torch.randn_like(x, device=self.device) * std
        noise = torch.clamp(noise, min=0, max=1)
        x = x * noise
        return x

    def _process_less_than_one(self, x, threshold=0.01):
        if threshold != 0:
            # # Create uniform random values with same shape as x
            # random_values = torch.rand_like(x)
            #
            # # For elements less than threshold, probabilistically set to 0 or 1
            # mask = x < threshold
            # x[mask] = (x[mask] > random_values[mask]).float()

            # Directly remove values below threshold
            x[x < threshold] = 0
        return x

    def adjust_contagious_OD(self, regions = [0], increase_ratio = [0.5]):
        """Simulate crowd gathering
        This function adjusts contagious_OD
        :param regions: zones to adjust
        :param increase_ratio: ratio to increase
        """
        # Ensure regions and increase_ratio have same length
        assert len(regions) == len(increase_ratio), "regions and increase_ratio must have equal length"

        self.contagious_OD = self.OD.clone()
        # Iterate through each region and its corresponding increase ratio
        for region, ratio in zip(regions, increase_ratio):
            # Increase inflow ratio for specified region
            self.contagious_OD[:, :, region] *= (1 + ratio)

        # Re-normalize to keep outflow ratios sum to 1
        # Note: this is simpler but actual increase will be slightly lower than specified
        sum_along_dim2 = self.contagious_OD.sum(dim=2, keepdim=True)
        self.contagious_OD /= sum_along_dim2

        # Print increase ratio
        for region in regions:
            print(f"Region {region}: from {self.OD[:, :, region].sum(dim=1).mean(dim=0)} to {self.contagious_OD[:, :, region].sum(dim=1).mean(dim=0)}")


    def reset_contagious_OD(self):
        """Reset contagious_OD to original OD matrix"""
        self.contagious_OD = self.OD.clone()

    def adjust_OD(self, regions = [0], increase_ratio = [0.5], beta_increase_ratio = [0.5]):
        """Simulate crowd gathering (considering overall flow matrix and elevated beta risk in gathering zones)
        This function adjusts OD matrix
        
        Args:
            regions: zones to adjust
            increase_ratio: inflow ratio to increase
            beta_increase_ratio: beta ratio to increase
        """
        # 1. Ensure all lists have same length
        assert len(regions) == len(increase_ratio) == len(beta_increase_ratio), "regions, increase_ratio and beta_increase_ratio must have equal length"

        # 2. Adjust OD
        self.OD = self.original_OD.clone()
        # 2.1. Iterate through each region and its corresponding increase ratio
        for region, ratio in zip(regions, increase_ratio):
            # Increase inflow ratio for specified region
            self.OD[:, :, region] *= (1 + ratio)

        # 2.2. Re-normalize to keep outflow ratios sum to 1
        # Note: actual increase will be slightly lower than specified
        sum_along_dim2 = self.OD.sum(dim=2, keepdim=True)
        self.OD /= sum_along_dim2
        self.contagious_OD = self.OD.clone()

        # 3. Adjust beta
        if self.use_beta_change and self.beta_matrix is not None:
            # Need to adjust beta_matrix
            self.beta_matrix = self.original_beta_matrix.clone()
            for region, ratio in zip(regions, beta_increase_ratio):
                self.beta_matrix[:, :, region] *= (1 + ratio)
        else:
            # Need to adjust betas
            self.betas = self.original_betas.clone()
            for region, ratio in zip(regions, beta_increase_ratio):
                self.betas[region] *= (1 + ratio)

        # Print increase ratio
        for region in regions:
            print(
                f"Region {region}: from {self.original_OD[:, :, region].sum(dim=1).mean(dim=0)} to {self.OD[:, :, region].sum(dim=1).mean(dim=0)}")

    def reset_OD(self):
        """Reset OD to original matrix, beta to original betas or beta_matrix"""
        self.OD = self.original_OD.clone()
        self.contagious_OD = self.OD.clone()
        if self.use_beta_change and self.beta_matrix is not None:
            self.beta_matrix = self.original_beta_matrix.clone()
        else:
            self.betas = self.original_betas.clone()


    def dynamic_beta_change(self, type = 'none', std = 0.1):
        """Dynamically adjust beta matrix to simulate beta variation

        Adjust beta_matrix based on parameters (shape: env_count, period, zone_num)

        Args:
            type: 'none' = no adjustment, 'spatial' = spatial variation, 
                  'temporal' = temporal variation, 'spatiotemporal' = both
            std: standard deviation for adjustment
        """
        if type not in ['none', 'spatial', 'temporal', 'spatiotemporal']:
            raise ValueError('Invalid beta change type: ', type)
        if type == 'none':
            self.use_beta_change = False
            return
        else:
            self.use_beta_change = True

        # Generate random beta matrix for each environment based on type and std
        if type == 'spatial':
            beta_spatial = torch.full((self.ZONE_NUM, ), self.beta, device=self.device)
            self.beta_matrix = torch.zeros((self.env_count, self.period, self.ZONE_NUM)).to(self.device)
            # Generate noise for all batches
            noise = torch.randn(self.env_count, self.ZONE_NUM, device=self.device) * std
            beta_spatial_noise = beta_spatial * (1 + noise)  # shape: (env_count, zone_num)
            self.beta_matrix += beta_spatial_noise.unsqueeze(1).expand(-1, self.period, -1)
        elif type == 'temporal':
            beta_temporal = torch.full((self.period, ), self.beta, device=self.device)
            self.beta_matrix = torch.zeros((self.env_count, self.period, self.ZONE_NUM)).to(self.device)
            noise = torch.randn(self.env_count, self.period, device=self.device) * std
            beta_temporal_noise = beta_temporal * (1 + noise)  # shape: (env_count, period)
            self.beta_matrix += beta_temporal_noise.unsqueeze(2).expand(-1, -1, self.ZONE_NUM)
        elif type == 'spatiotemporal':
            beta_spatialtemporal = torch.full((self.period, self.ZONE_NUM), self.beta, device=self.device)
            self.beta_matrix = torch.zeros((self.env_count, self.period, self.ZONE_NUM)).to(self.device)
            noise = torch.randn(self.env_count, self.period, self.ZONE_NUM, device=self.device) * std
            beta_spatialtemporal_noise = beta_spatialtemporal * (1 + noise)  # shape: (env_count, period, zone_num)
            self.beta_matrix += beta_spatialtemporal_noise
        self.original_beta_matrix = self.beta_matrix.clone()

    def _uncertain_action(self, u0, u1, std = 0.1):
        """Action uncertainty: affects system state transition"""
        actual_u0 = u0 * (1 + std * torch.randn_like(u0))
        actual_u1 = u1 * (1 + std * torch.randn_like(u1))
        actual_u0 = torch.clamp(actual_u0, min=0, max=1)
        actual_u1 = torch.clamp(actual_u1, min=0, max=1)
        return actual_u0, actual_u1


    def step(self, action = None):
        """
        Execute one step based on action.

        This function:
        - Increments day by 1
        - Updates environment state based on action
        - Computes reward
        - Records history info, returns episode history when reaching self.period days

        Returns:
            New state, reward, done flag, episode history info
        """
        self.day += 1
        # Import external cases (select people from S to E, total population unchanged)
        self._imported_cases(self.daily_imported_cases)

        # Record action
        if action is None:
            action = torch.zeros((self.env_count, self.ZONE_NUM), device=self.device)
        # Ensure action is a tensor of shape (env_count, ZONE_NUM)
        assert action.shape == (self.env_count, self.ZONE_NUM), "Action shape must be (env_count, ZONE_NUM, 2)"
        self.actions[:, self.day - 2, :] = action  # shape: (env_count, ZONE_NUM)

        # If beta is dynamic, update beta based on day
        if self.use_beta_change:
            self.betas = self.beta_matrix[:, self.day - 2, :]

        # Map action to u0, u1
        p_test, p_quara = self._action_to_u(action)  # (env_count, ZONE_NUM)

        """Whether to add uncertainty to action"""
        if self.use_action_uncertainty:
            # Add uncertainty to action before execution
            actual_p_test = self._add_noise(p_test, std=0.1)
            actual_p_quara = self._add_noise(p_quara, std=0.1)
            # actual_p_test, actual_p_quara = self._uncertain_action(p_test, p_quara)
        else:
            # Simple action perturbation, helpful for training
            actual_p_test = self._add_noise(p_test, std=0.05)
            # actual_p_test = p_test
            actual_p_quara = p_quara

        """Incomplete observation means adding perturbation to reporting rate"""
        if self.use_obs_imperfect:
            actual_p_report = self.report_I_rate0 * torch.ones(self.env_count, self.ZONE_NUM, device=self.device)

            # TODO: whether to add this randomness - if added, even known zones will have unknown reporting rate
            # I_real_report_rate: uniformly distributed in 0.1-1.0, simulating incomplete observation effect
            actual_p_report = self._add_noise(self.report_I_rate0 * self.I_real_report_rate, std=0.1)

            mask = self._random_mask()
            actual_p_report[mask] = 0

            self.obs_missing_total_days += (mask & ~self.has_been_observed).to(torch.int32)
            self.has_been_observed |= (~mask)
        else:
            actual_p_report = self.report_I_rate0

        """3 steps: transmission dynamics; active reporting; testing-quarantine"""
        # 1. State transition caused by transmission
        # Compartment model step
        contagious_infects = (self.beta_E_rate * self.simState[:, :, [self.E_undetected, self.E_detected]].sum(dim=-1)
                    + self.Pm * self.simState[:, :, [self.I_undetected, self.I_detected, self.I_reported]].sum(dim=-1))  # (env_count, ZONE_NUM)
        # Calculate transmission impact to each zone using contagious_OD transpose
        contagious_OD = self.OD * contagious_infects.unsqueeze(-1)  # (env_count, ZONE_NUM, ZONE_NUM)
        contagious_toJ = contagious_OD.sum(dim=1)  # (env_count, ZONE_NUM)
        contagious_OD_ratio = contagious_OD / (contagious_toJ.unsqueeze(1) + 1e-12)  # column normalize, ratio of infected from i to j
        # contagious_toJ = torch.bmm(self.OD.transpose(2, 1), contagious_infects.unsqueeze(-1)).squeeze(-1)  # (env_count, ZONE_NUM)
        # If outflow infected < 1, probability becomes 0/1
        # contagious_toJ = self._process_less_than_one(contagious_toJ)
        contagious_toJ = torch.clamp(contagious_toJ, min=0.0)

        all_toJ = torch.bmm(self.OD.transpose(2, 1), self.POP.unsqueeze(-1)).squeeze(-1)  # (env_count, ZONE_NUM)

        contagious_ratio_toJ = contagious_toJ / all_toJ  # (env_count, ZONE_NUM)
        contagious_ratio_toJ = torch.clamp(contagious_ratio_toJ, min=0.0)

        # Calculate contagious_ratioToJ with betas
        modified_ratio = contagious_ratio_toJ * self.betas  # (env_count, ZONE_NUM)
        lam = torch.bmm(self.OD, modified_ratio.unsqueeze(-1)).squeeze(-1)  # (env_count, ZONE_NUM)
        lam = torch.clamp(lam, min=0.0)

        # Calculate state transition caused by transmission
        S2Eun = self.simState[:, :, self.S] * lam
        Eun2Iun = self.sigma * self.simState[:, :, self.E_undetected]
        Ede2Ide = self.sigma * self.simState[:, :, self.E_detected]
        Iun2R = self.gamma * self.simState[:, :, self.I_undetected]
        Ide2R = self.gamma * self.simState[:, :, self.I_detected]
        Ire2R = self.gamma * self.simState[:, :, self.I_reported]
        QE2QI = self.sigma * self.simState[:, :, self.QE]
        QI2R = self.gamma_q * self.simState[:, :, self.QI]

        dS = -S2Eun
        dE_undetected = S2Eun - Eun2Iun
        dE_detected = - Ede2Ide
        dI_undetected = Eun2Iun - Iun2R
        dI_detected = Ede2Ide - Ide2R
        dI_reported = - Ire2R
        dQE = - QE2QI
        dQI = QE2QI - QI2R
        dR = Iun2R + Ide2R + Ire2R + QI2R

        # Update state
        dState = torch.stack([dS, dE_undetected, dE_detected, dI_undetected, dI_detected, dI_reported, dQE, dQI, dR],
            dim=2)  # (env_count, ZONE_NUM, 9)
        self.simState += dState

        # Calculate new E then update state (for infection cost in reward function)
        # Return new E to each zone
        S_toJ = torch.bmm(self.OD.transpose(2, 1), self.simState[:, :, self.S].unsqueeze(-1)).squeeze(-1)  # (env_count, ZONE_NUM)
        new_E_inJ = S_toJ * modified_ratio  # (env_count, ZONE_NUM) new_E_inJ.sum() == S2Eun.sum(), new E remaining in zone j
        new_E_backI = (new_E_inJ.unsqueeze(1) * contagious_OD_ratio).sum(dim=-1)  # (env_count, ZONE_NUM)
        self.daily_new_E_self_cause[:, self.day - 1, :] = new_E_backI

        # 2. State transition caused by active reporting
        Iun2Ire = actual_p_report * self.simState[:, :, self.I_undetected]
        self.simState[:, :, self.I_undetected] += - Iun2Ire
        self.simState[:, :, self.I_reported] += Iun2Ire
        self.simState[self.simState < 0] = 0
        assert (self.simState[:, :, :] >= 0).all(), "all state must be non-negative"
        self.daily_new_report[:, self.day - 1, :] = Iun2Ire

        # 3.1. State transition - Testing
        # E and I to be tested under current testing rate
        test_valid_num = actual_p_test * self.POP / self._cal_detection_scale()
        # test_num = torch.floor(test_num)
        test_num_rate = test_valid_num / (self.simState[:, :, [self.E_undetected, self.I_undetected]].sum(dim=-1) + 1e-8)
        test_num_rate = torch.clamp(test_num_rate, min=0, max=1)
        # test_num_rate = actual_p_test
        new_detect_E = test_num_rate * self.simState[:, :, self.E_undetected] * self.detect_E_rate  # (env_count, ZONE_NUM)
        new_detect_I = test_num_rate * self.simState[:, :, self.I_undetected] * self.detect_I_rate
        self.simState[:, :, self.E_undetected] -= new_detect_E
        self.simState[:, :, self.I_undetected] -= new_detect_I
        self.simState[:, :, self.E_detected] += new_detect_E
        self.simState[:, :, self.I_detected] += new_detect_I
        self.simState[self.simState < 0] = 0
        if not (self.simState[:, :, :] >= 0).all():
            print("all state must be non-negative")
        assert (self.simState[:, :, :] >= 0).all(), "all state must be non-negative"
        self.daily_detect_E[:, self.day - 1, :] = new_detect_E
        self.daily_detect_I[:, self.day - 1, :] = new_detect_I

        # 3.2. State transition - Quarantine
        # Total quarantine / to be quarantined
        # TODO: Confirm today's planned quarantine number, distribute to next 1-2 days
        quara_plan_num = torch.min(actual_p_quara * self.POP, self.simState[:, :, self.can_isolated].sum(dim=-1))
        self.__delay_quara(quara_plan_num)
        # Use today's available quarantine capacity to determine quarantine rate
        quara_num_rate = self.daily_delay_quara_num[:, self.day - 1, :] / (self.simState[:, :, self.can_isolated].sum(dim=-1) + 1e-12)
        # quara_num_rate = quara_plan_num / (self.simState[:, :, self.can_isolated].sum(dim=-1) + 1e-12)
        quara_num_rate = torch.clamp(quara_num_rate, min=0, max=1)
        # quara_num_rate = actual_p_quara
        quarantine_E1 = quara_num_rate * self.simState[:, :, self.E_detected]  # (env_count, ZONE_NUM)
        quarantine_I1 = quara_num_rate * self.simState[:, :, self.I_detected]
        quarantine_I2 = quara_num_rate * self.simState[:, :, self.I_reported]
        self.simState[:, :, self.E_detected] -= quarantine_E1
        self.simState[:, :, self.I_detected] -= quarantine_I1
        self.simState[:, :, self.I_reported] -= quarantine_I2
        self.simState[:, :, self.QE] += quarantine_E1
        self.simState[:, :, self.QI] += quarantine_I1 + quarantine_I2
        self.simState[self.simState < 0] = 0
        assert (self.simState[:, :, :] >= 0).all(), "all state must be non-negative"

        # Record new E and I
        self.daily_new_E[:, self.day - 1, :] = S2Eun  # (env_count, zone_num)
        self.daily_new_I[:, self.day - 1, :] = Eun2Iun + Ede2Ide + QE2QI  # (env_count, zone_num)
        self.daily_known2R[:, self.day - 1, :] = Ire2R + Ide2R + QI2R  # (env_count, zone_num)
        self.daily_new_Q[:, self.day - 1, :] = quarantine_E1 + quarantine_I1 + quarantine_I2

        # Correct small values in state
        self.simState = self._process_less_than_one(self.simState)  # When count < 1, probabilistic transition to 0/1
        self.simState[:, :, 0] = self.POP - self.simState[:, :, 1:].sum(dim=2)  # Correct total population
        self.simState[self.simState < 0] = 0
        assert (self.simState[:, :, :] >= 0).all(), f"all state must be non-negative in {self.day}"
        self.simRes[:, self.day - 1, :, :] = self.simState.clone()

        # if self.day in range(10, 60, 10):
        #     self.__cal_pearsonr()

        # Construct state/obs
        obs = self._get_obs()  # shape: (env_count, window_size, zone_num, local_obs_dim * 2)
            
        # Calculate reward (based on actual action effect)
        reward = self._reward_func(actual_p_test, actual_p_quara)  # shape: (env_count, zone_num)

        info = {}
        done = False
        if self.day > self.period:
            # Calculate metrics
            ep_r = self.history_cost['reward'].sum(dim=1).mean(dim=-1)  # (env_count,)
            total_infections = self.daily_new_I.sum(dim=[1,2])  # (env_count)
            total_test_num = self.history_cost['test_num'].sum(dim=[1,2])  # (env_count,)
            total_quara_num = self.history_cost['quara_num'].sum(dim=[1,2])  # (env_count,)
            total_quara_days = self.history_cost['quara_days'].sum(dim=[1,2])  # (env_count,)
            total_control_cost = total_quara_num + total_test_num * self.cost_ratio_t2q / (1/self.sigma + 1/self.gamma)
            score = (torch.exp((20 * 500 * total_infections) / self.TOTAL_POP)
                     + torch.exp(1 * 500 * total_control_cost / self.TOTAL_POP))  # (env_count)
            score_avg = (torch.exp((20 * 500 * total_infections.mean()) / self.TOTAL_POP[0])
                     + torch.exp(1 * 500 * total_control_cost.mean() / self.TOTAL_POP[0]))  # (1)
            # rw = self.reward_weights.copy()
            # rw = [w / 100 for w in rw]
            # score = rw[0] * total_infections + rw[1] * total_test_num + rw[2] * total_quara_days
            # score = (torch.exp((rw[0] * 10 * total_infections) / self.TOTAL_POP)
            #          + torch.exp((rw[2] * 10 * total_control_cost) / self.TOTAL_POP))  # (env_count)
            # Add avg zone score metric
            zone_infections  = self.daily_new_I.sum(dim=1)  # (env_count, zone_num)
            zone_test_num = self.history_cost['test_num'].sum(dim=1)    # (env_count, zone_num)
            zone_quara_num = self.history_cost['quara_num'].sum(dim=1)  # (env_count, zone_num)
            zone_control_cost = zone_quara_num + zone_test_num * self.cost_ratio_t2q / (1/self.sigma + 1/self.gamma)
            avg_zone_score = (torch.exp((20 * 500 * zone_infections) / self.POP)
                     + torch.exp(1 * 500 * zone_control_cost / self.POP)).mean(dim=-1)  # (env_count)
            print(f"({'Eval' if self.is_evaluation else 'Train'} Step End)\tTime: {datetime.now():%H:%M:%S}",
                  "\t reward:", f"{ep_r.mean().item():.4f}",
                  "\t total_infections:", f"{total_infections.mean().item():.4f}",
                  "\t total_test_num:", f"{total_test_num.mean().item():.4f}",
                  "\t total_quarantine_num:", f"{total_quara_num.mean().item():.4f}"
                  "\t total_quarantine_days:", f"{total_quara_days.mean().item():.4f}",
                  "\t total_control_cost:", f"{total_control_cost.mean().item():.4f}",
                  "\t score:", f"{score.mean().item():.4f}",
                  "\t avg_zone_score:", f"{avg_zone_score.mean().item():.4f}")
            # print('******\n', 'local_infe_cost:', self.history_cost['local_infe_cost'].sum(dim=1).mean(),
            #       'local_test_cost:', self.history_cost['local_test_cost'].sum(dim=1).mean(),
            #       'local_quara_cost:', self.history_cost['local_quara_cost'].sum(dim=1).mean())
            # print('global_infe_cost:', self.history_cost['global_infe_cost'].sum(dim=1).mean(),
            #       'global_test_cost:', self.history_cost['global_test_cost'].sum(dim=1).mean(),
            #       'global_quara_cost:', self.history_cost['global_quara_cost'].sum(dim=1).mean())
            if self.use_reward_shaping:
                print("penalty_times:", self.penalty_times)

            info = {
                'ep_r': ep_r,
                'total_infections': total_infections,
                'total_test_num': total_test_num,
                'total_quara_num': total_quara_num,
                'total_control_cost': total_control_cost,
                'total_quarantine_days': total_quara_days,
                'score': score,
                'score_avg': score_avg,
                'avg_zone_score': avg_zone_score,
            }

            done = True

        # In theory done should be batched, but for this env all start and end together
        return obs, reward, done, info

    def _cal_detection_scale(self):
        """Calculate number of people to test for detecting one infected based on infection rate"""
        EI_rate = self.simState[:, :, [self.E_undetected, self.I_undetected]].sum(dim=-1) / self.POP  # (env_count, zone_num)
        scale = torch.zeros_like(EI_rate)
        mask = EI_rate != 0
        scale[mask] = 1 / torch.pow(EI_rate[mask], self.detection_efficiency_exp_param)
        scale[~mask] = 1e8  # When infection rate is 0, fill with very large value
        return scale

        # f_min = 1.0
        # f_max = 1000.0
        #
        # return f_min + (f_max - f_min) * (1 - EI_rate)


    def _random_mask(self):
        # # Method 1: Mask specified zones
        # mask = torch.zeros((self.env_count, self.ZONE_NUM), dtype=torch.bool, device=self.device)
        # mask[:, 0:int(self.ZONE_NUM * 0.15)] = True
        # return mask

        # Method 2: Random mask
        mask_rate_down = 0
        mask_rate_up = 0.3
        mask_duration_down = 14
        mask_duration_up = 28
        if self.R0 == 'low':
            mask_rate_down = 0.3
            mask_rate_up = 0.5
            mask_duration_down = 28
            mask_duration_up = 42
        mask_rate_down = self.mask_rate_down if self.mask_rate_down is not None else mask_rate_down
        mask_rate_up = self.mask_rate_up if self.mask_rate_up is not None else mask_rate_up
        mask_duration_down = self.mask_duration_down if self.mask_duration_down is not None else mask_duration_down
        mask_duration_up = self.mask_duration_up if self.mask_duration_up is not None else mask_duration_up

        rate = torch.rand(self.env_count, device=self.device) * (mask_rate_up - mask_rate_down) + mask_rate_down  # random number in range
        mask = torch.rand(self.env_count, self.ZONE_NUM, device=self.device) < rate.unsqueeze(-1).expand(-1, self.ZONE_NUM)
        # # Zones with large diagonal elements have greater impact when masked
        # mask[:, 563] = False
        cds = torch.randint(mask_duration_down, mask_duration_up, (self.env_count, 1), device=self.device).squeeze(-1)
        if not hasattr(self, 'old_zone_mask'):
            self.old_zone_mask = mask
            self.count_downs = cds
        else:
            # Reset countdown for environments at 0
            cd_mask = self.count_downs <= 0
            self.count_downs[cd_mask] = cds[cd_mask]
            cd_mask = cd_mask.unsqueeze(-1).expand(-1, self.ZONE_NUM)
            self.old_zone_mask[cd_mask] = mask[cd_mask]
        self.count_downs -= 1
        return self.old_zone_mask

        # # Method 3: Read mask from file
        # if not hasattr(self, 'mask') or self.mask is None:
        #     raise NotImplementedError("mask not set")
        # return self.mask

    def set_mask(self, mask):
        self.mask = mask

    def _action_to_u(self, action):
        """Map action (discrete value) to two control measures

        Args:
            action: (env_count, ZONE_NUM)
                Both control measures: larger value = stricter control
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        # action = action.to(torch.long)
        u1_cnt = self.action_to_u1.shape[0]
        u_p_test = self.action_to_u0[(action / u1_cnt).to(torch.long)] * 0.1 # (env_count, ZONE_NUM)
        u_p_quara = self.action_to_u1[(action % u1_cnt).to(torch.long)] * 0.01 # (env_count, ZONE_NUM)
        return u_p_test, u_p_quara

    def _get_history_padding(self, datas, day, window_size):
        """
        Get padded history window data.

        When day >= window_size, extract most recent window_size days of data.
        When day < window_size, pad zeros at the front of time dimension.

        Args:
            datas: input observation data, shape (env_count, total_days, ZONE_NUM)
            day: current day (time step), starting from 1
            window_size: number of days to look back

        Returns:
            Padded window data, shape (env_count, window_size, ZONE_NUM)
        """
        if day >= window_size:
            # Get most recent window_size days of data
            window = datas[:, (day - window_size):day, :]
        else:
            # Calculate days to pad
            padding = window_size - day
            # Get existing days of data
            current_data = datas[:, :day, :]  # shape: (env_count, day, ZONE_NUM)
            # Pad zeros at the front of time dimension
            pad = [0] * 2 * len(current_data.shape)
            pad[-4] = padding
            pad = tuple(pad)
            window = torch.nn.functional.pad(current_data, pad, "constant", 0)  # shape: (env_count, window_size, ZONE_NUM)
    
        return window.clone()

    def _get_obs(self):
        """Get observation

        Returns:
            shape: (env_count, ZONE_NUM, window_size, local_obs_dim * 2)
        """
        """Observation 1: Observation and estimation of existing EI, estimation of new E"""
        # curr known EI
        curr_known_EI = self.simState[:, :, [self.E_detected, self.I_reported, self.I_detected]].sum(dim=-1)
        estimate_Eun = self.daily_detect_E[:, self.day - 1, :] / self.detect_E_rate * (1 - self.detect_E_rate)  # Eun compartment, accurate when test_num_rate reaches 1
        estimate_Iun =  self.daily_detect_I[:, self.day - 1, :] / self.detect_I_rate * (1 - self.detect_I_rate)  # Iun compartment
        # new E
        # estimate_new_E = self.daily_detect_E[:, self.day - 1, :] + self.daily_detect_I[:, self.day - 1, :]    # Use newly detected E to estimate new E
        estimate_new_E = self.daily_detect_E[:, self.day - 1, :] + self.daily_detect_I[:, self.day - 1, :] + self.daily_new_report[:, self.day - 1, :]    # Use newly observed EI to estimate new E
        # estimate_new_E = self.daily_detect_E[:, self.day - 1, :]    # Use newly detected E to estimate new E

        curr_obs = [curr_known_EI + estimate_Eun + estimate_Iun, estimate_new_E]  # [existing, Eun]

        # curr_known_EI += self.simState[:, :, [self.QE, self.QI]].sum(dim=-1)

        # """Observation 2: Observation and estimation of existing I, estimation of new I"""
        # # curr known I
        # curr_known_I = self.simState[:, :, [self.I_reported, self.I_detected]].sum(dim=-1)
        # estimate_Iun = self.daily_detect_I[:, self.day - 1, :] / self.detect_I_rate * (1 - self.detect_I_rate) # estimate current Iun
        # # new I estimation
        if self.state_contain_detected:
            I_detected = self.simState[:, :, self.I_detected]
            E_detected = self.simState[:, :, self.E_detected]
            curr_obs.append(I_detected)
            curr_obs.append(E_detected)
        if self.state_contain_Q:
            Q = self.simState[:, :, [self.QE, self.QI]].sum(dim=-1)
            curr_obs.append(Q)
        if self.state_contain_R:
            total_knownR = self.daily_known2R.sum(dim=1)  # shape: (env_count, zone_num)
            curr_obs.append(total_knownR)
        # # Normalize
        curr_obs = torch.stack(curr_obs, dim=-1)
        curr_obs = curr_obs / self.POP.unsqueeze(-1)
        if self.state_contain_detect_rate:
            action = self.actions[:, self.day - 2, :]
            p_test, _ = self._action_to_u(action)  # (env_count, ZONE_NUM)
            curr_obs = torch.cat([curr_obs, p_test.unsqueeze(-1)], dim=-1)
            # # curr action
            # action = self.actions[:, self.day - 2, :].unsqueeze(-1).clone()  # shape: (env_count, zone_num)
            # curr_obs = torch.cat([curr_obs, action], dim=-1)

        self.history_local_obs[:, self.day - 1, :, :] = curr_obs  # shape: (env_count, zone_num, local_obs_dim)

        local_obs = self._get_history_padding(self.history_local_obs, self.day, self.WINDOW_SIZE) # shape: (env_count, WINDOW_SIZE, ZONE_NUM, local_obs_dim)
        global_obs = local_obs.mean(dim=2) # shape: (env_count, WINDOW_SIZE, local_obs_dim)
        global_obs_expanded = global_obs.unsqueeze(2).repeat(1, 1, local_obs.size(2), 1)
        combined_obs = torch.cat([local_obs, global_obs_expanded], dim=-1) # shape: (env_count, WINDOW_SIZE, ZONE_NUM, local_obs_dim * 2)

        return combined_obs

    def _reward_func(self, actual_p_test, actual_p_quara):
        """Calculate reward for each zone

        Note: Each zone has its own temporal order, but spatial order only appears in global reward

        Returns:
            shape: (env_count, zone_num)
        """

        # 1. Infection cost (E, I, new)
        # # a. Based on existing I per zone
        # infections = state[:, :, [self.I_undetected, self.I_detected, self.I_reported]].sum(dim=-1) # (env_count, zone_num)
        # local_infe_cost = (20 * infections) ** 2 / self.POP # (env_count, zone_num)
        # global_infe_cost = (20 * infections.sum(dim=1)) ** 2 / self.TOTAL_POP # (env_count,)

        # # Based on new I per zone
        # new_infections = self.daily_new_E[:, self.day - 1, :] + self.daily_new_I[:, self.day - 1, :] # (env_count, zone_num)
        # avg_infections = self.daily_new_I.sum(dim=1) / (self.day - 1)  # (env_count, zone_num)
        # # new_infections = avg_infections
        # local_infe_cost = (20 * new_infections) ** 2 / self.POP  # (env_count, zone_num)
        # global_infe_cost = (20 * new_infections.sum(dim=1)) ** 2 / self.TOTAL_POP  # (env_count,)

        if not self.disable_response_distribution:
            # Based on new E caused by each zone
            new_E_self_cause = self.daily_new_E_self_cause[:, self.day - 1, :]  # (env_count, zone_num)
            local_infe_cost = new_E_self_cause / self.POP  # immediate reward
            global_infe_cost = (new_E_self_cause.sum(dim=1)) / self.TOTAL_POP  # global reward
        else:
            # Based on new E in each zone itself
            new_E_self_region = self.daily_new_E[:, self.day - 1, :]
            local_infe_cost = new_E_self_region / self.POP
            global_infe_cost = (new_E_self_region.sum(dim=1)) / self.TOTAL_POP

        # 2. Testing cost actual_p_test
        test_num = actual_p_test * self.POP  # (env_count, zone_num)
        self.daily_test_num[:, self.day - 1, :] = test_num
        local_test_cost = test_num / self.POP  # (env_count, zone_num)
        global_test_cost = test_num.sum(dim=1) / self.TOTAL_POP  # (env_count,)

        # 3. Quarantine cost Q
        quara_num = self.daily_new_Q[:, self.day - 1, :]  # shape: (env_count, zone_num)
        self.daily_quara_num[:, self.day - 1, :] = quara_num
        # quara_num takes max of quara_num and actual_p_quara
        if not self.is_evaluation:
            quara_num = torch.max(quara_num, actual_p_quara * 20)
            # quara_num = torch.max(quara_num, actual_p_quara * 200)
        local_quara_cost = quara_num / self.POP  # (env_count, zone_num)
        global_quara_cost = quara_num.sum(dim=-1) / self.TOTAL_POP  # (env_count,)

        rw = self.reward_weights.copy()

        # 1.global reward
        global_reward = - (rw[0] * global_infe_cost + rw[1] * global_test_cost + rw[2] * global_quara_cost) # shape: (env_count,)
        global_reward_expaned = global_reward.unsqueeze(1).repeat(1, self.ZONE_NUM) # shape: (env_count, zone_num)

        # 2.local reward
        local_reward = -(rw[0] * local_infe_cost + rw[1] * local_test_cost + rw[2] * local_quara_cost) # shape: (env_count, zone_num)
        reward = self.local_reward_weight * local_reward + (1 - self.local_reward_weight) * global_reward_expaned # shape: (env_count, zone_num)
        # if self.day in [10,20,30,40,50,60,70,80,90,100,110]:
        #     print(f'day:{self.day}-------------------')
        #     print('local_infe_cost:', local_infe_cost[:,:5].mean(dim=0))
        #     print('local_test_cost:', local_test_cost[:,:5].mean(dim=0))
        #     print('local_quara_cost:', local_quara_cost[:,:5].mean(dim=0))
        #     print('reward:', reward[:,:5].mean(dim=0))


        if self.use_reward_shaping:
            # If there are reported cases but no testing in that zone, apply penalty (report should be from previous day)
            state = self.simRes[:, self.day - 2, :, :]
            infections = state[:, :, [self.E_detected, self.I_reported, self.I_detected]].sum(dim=-1)  # (env_count, zone_num)
            infections_large = infections > 0.1
            action_small = actual_p_test == 0  # (env_count, zone_num)
            zone_mask = infections_large & action_small
            # If no infections but strong control measures applied
            infections_small = infections < 0.0005
            action_large = actual_p_test >= 0.005
            # zone_mask = (infections_small & action_large)
            zone_mask = (infections_small & action_large) | zone_mask
            if zone_mask.any():
                self.penalty_times += zone_mask.sum().item()
            # Apply penalty, e.g. add a large negative value
            penalty_factor = 2  # penalty factor
            penalty = penalty_factor * (zone_mask.float())
            reward -= penalty

        # Record reward, ensuring independent recording for each batch
        self.history_cost['reward'][:, self.day - 2, :] = reward.clone()
        self.history_cost['test_num'][:, self.day - 2, :] = test_num
        self.history_cost['quara_num'][:, self.day - 2, :] = quara_num
        self.history_cost['quara_days'][:, self.day - 2, :] = self.simState[:, :, [self.QE,self.QI]].sum(dim=2)
        self.history_cost['local_reward'][:, self.day - 2, :] = local_reward
        self.history_cost['global_reward'][:, self.day - 2] = global_reward
        self.history_cost['local_infe_cost'][:, self.day - 2] = local_infe_cost.mean(dim=-1)
        self.history_cost['local_test_cost'][:, self.day - 2] = local_test_cost.mean(dim=-1)
        self.history_cost['local_quara_cost'][:, self.day - 2] = local_quara_cost.mean(dim=-1)
        self.history_cost['global_infe_cost'][:, self.day - 2] = global_infe_cost.mean(dim=-1)
        self.history_cost['global_test_cost'][:, self.day - 2] = global_test_cost.mean(dim=-1)
        self.history_cost['global_quara_cost'][:, self.day - 2] = global_quara_cost.mean(dim=-1)


        return reward

    def __delay_quara(self, quara_plan_num):
        """Distribute today's planned quarantine to next 1-2 days
        Args:
            quara_plan_num: (env_count, zone_num)
        """
        if self.use_delay_quara:
            # Distribute to today, tomorrow, day after
            for i, r in enumerate([0.3, 0.4, 0.3]):
                if self.day - 1 + i > self.period:
                    continue
                self.daily_delay_quara_num[:, self.day - 1 + i, :] += r * quara_plan_num
        else:
            self.daily_delay_quara_num[:, self.day - 1, :] += quara_plan_num

    def __cal_pearsonr(self):
        mask = torch.rand(self.ZONE_NUM) < 0.3  # True for ratio below threshold
        curr_EI = self.simState[0, :, [self.E_undetected, self.E_detected,
                                       self.I_undetected, self.I_detected, self.I_reported]].sum(dim=-1)
        # curr_EI = curr_EI / self.POP[0]
        know_EI = torch.zeros_like(curr_EI)
        know_EI[~mask] = curr_EI[~mask]

        OD = self.OD[0]
        OD_t = OD.transpose(0, 1)

        pre = torch.mm(OD_t, know_EI.unsqueeze(1)) / torch.mm(OD_t, self.POP[0].unsqueeze(1))
        pre = torch.mm(OD, pre)[:, 0] * self.POP[0]

        # Numerical correction
        pre = pre * (curr_EI[~mask].sum() / pre[~mask].sum() + 1e-12)

        y = curr_EI[mask]
        x = pre[mask]

        # y = (curr_EI / self.POP[0])[mask]
        # x = (pre[:, 0] / self.POP[0])[mask]

        from scipy.stats import pearsonr
        r, p_value = pearsonr(x.cpu().numpy(), y.cpu().numpy())
        print(f"Pearson Correlation Coefficient: {r:.4f}")
        print(f"P-value: {p_value:.4e}")

        # import pandas as pd
        # data = pd.DataFrame({'x': x.numpy(), 'y': y.numpy()})
        # data.to_csv('output.csv', index=False)

    def extract_observe(self):
        obs = self.history_local_obs[:, :, :, :2] * self.POP.unsqueeze(1).unsqueeze(-1)
        return obs
    def extract_daily_quara_num(self):
        daily_quara_num = self.daily_quara_num.clone()
        return daily_quara_num

    def record_rebuild_state(self, rebuild_state: torch.Tensor):
        assert rebuild_state.shape == (self.env_count, self.ZONE_NUM, 2), "rebuild_state shape error. expected (env_count, zone_num, 2)"
        self.rebuild_states.append(rebuild_state)
    def extract_rebuild_state(self):
        if len(self.rebuild_states) == self.period:
            print("env.rebuild_state length: ", len(self.rebuild_states))
        if len(self.rebuild_states) == 0:
            return torch.zeros(1,1,1,0)
        r_s = torch.stack(self.rebuild_states, dim=1)
        return r_s

    def get_observation_missing_rate(self):
        """
        Get observation missing rate since last reset.

        Returns:
            Missing rate for each environment, shape (env_count,)
        """
        total_zone_days = self.ZONE_NUM * (self.day - 1)
        missing_rate = self.obs_missing_total_days.sum(dim=-1) / total_zone_days
        return missing_rate

    def render_all_rooms(self):
        # Plot all compartments
        S = self.simRes[:, :, :, self.S].sum(dim=2).mean(dim=0) # shape: (period+1)
        E_undetected = self.simRes[:, :, :, self.E_undetected].sum(dim=2).mean(dim=0)
        E_detected = self.simRes[:, :, :, self.E_detected].sum(dim=2).mean(dim=0)
        I_undetected = self.simRes[:, :, :, self.I_undetected].sum(dim=2).mean(dim=0)
        I_detected = self.simRes[:, :, :, self.I_detected].sum(dim=2).mean(dim=0)
        I_reported = self.simRes[:, :, :, self.I_reported].sum(dim=2).mean(dim=0)
        QE = self.simRes[:, :, :, self.QE].sum(dim=2).mean(dim=0)
        QI = self.simRes[:, :, :, self.QI].sum(dim=2).mean(dim=0)
        R = self.simRes[:, :, :, self.R].sum(dim=2).mean(dim=0)

        # Cumulative infections
        daily_new_I = self.daily_new_I.sum(dim=2).mean(dim=0)
        daily_new_I_cum = daily_new_I.cumsum(dim=0)  # shape: (period+1)

        # Daily new cases


        plt.figure(dpi=120, figsize=(6, 5))
        plt.grid(linestyle='-.', axis='both')

        plt.plot(S.cpu(), label='S (avg)', linewidth=1)
        plt.plot(E_undetected.cpu(), label='E_undetected (avg)', linewidth=1)
        plt.plot(E_detected.cpu(), label='E_detected (avg)', linewidth=1)
        plt.plot(I_undetected.cpu(), label='I_undetected (avg)', linewidth=1)
        plt.plot(I_detected.cpu(), label='I_detected (avg)', linewidth=1)
        plt.plot(I_reported.cpu(), label='I_reported (avg)', linewidth=1)
        plt.plot(QE.cpu(), label='QE (avg)', linewidth=1)
        plt.plot(QI.cpu(), label='QI (avg)', linewidth=1)
        plt.plot(R.cpu(), label='R (avg)', linewidth=1)

        plt.plot(daily_new_I_cum.cpu(), label='daily_new_I_cum (avg)', linewidth=2, color='red', linestyle='--')
        plt.plot(daily_new_I.cpu(), label='daily_new_I (avg)', linewidth=2, color='red')

        plt.legend(fontsize=10, loc='upper right')
        plt.xlim(0, self.period)
        plt.show()

    def render(self, title='', fig_dir=None) :
        plt.figure(dpi=120, figsize=(7, 5))
        plt.grid(linestyle='-.', axis='both')

        # Calculate average of infection sum across all environments
        # infections = self.simRes[:, :, :, [self.I_undetected, self.I_detected, self.I_reported]].sum(dim=[2,3]).mean(dim=0)  # (period+1,)
        # plt.plot(infections.cpu(), label='Infections (avg)', color='orange', linewidth=2)
        # exposeds = self.simRes[:, :, :, [self.E_undetected, self.E_detected]].sum(dim=[2,3]).mean(dim=0)
        # plt.plot(exposeds.cpu(), label='Exposeds (avg)', color='green', linewidth=2)

        # Observed EI
        estimate_total_EI = (self.history_local_obs[:, :, :, 0] * self.POP.unsqueeze(1)).sum(dim=2).mean(dim=0)
        plt.plot(estimate_total_EI.cpu(), label='Obs Total EI (avg)', color='red', linewidth=2, linestyle='--', alpha=0.5)
        # Observed new EI
        estimate_new_E = (self.history_local_obs[:, :, :, 1] * self.POP.unsqueeze(1)).sum(dim=2).mean(dim=0)
        # plt.plot(estimate_new_E.cpu(), label='Obs New E (avg)', color='green', linewidth=2, linestyle='--', alpha=0.5)

        # Total EI
        total_EI = self.simRes[:, :, :, [self.E_undetected, self.E_detected,
                                         self.I_undetected, self.I_detected, self.I_reported]].sum(dim=[2,3]).mean(dim=0)
        plt.plot(total_EI.cpu(), label='Actual Total EI (avg)', color='red', linewidth=2, alpha=0.5)
        # Actual new EI
        actual_new_E = self.simRes[:, :, :, self.E_undetected].sum(dim=2).mean(dim=0)
        # plt.plot(actual_new_E.cpu(), label='Actual New E (avg)', color='green', linewidth=2, alpha=0.5)


        # # E_undetected variation (observe external input)
        # E_undetected = self.simRes[:, :, :, [self.E_undetected]].sum(dim=[2,3]).mean(dim=0)
        # plt.plot(E_undetected.cpu(), label='E_undetected (avg)', color='green', linewidth=2)

        plt.axhline(ls='-.', color='grey')
        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0)

        plt.legend(fontsize=12, loc='upper left')
        plt.twinx()

        # Plot total testing and total infections
        reward = self.history_cost['reward'].mean(dim=0)
        test_num = self.history_cost['test_num'].mean(dim=0)
        quara_days = self.history_cost['quara_days'].mean(dim=0)
        local_reward = self.history_cost['local_reward'].mean(dim=0)
        global_reward = self.history_cost['global_reward'].mean(dim=0)

        # plt.plot(test_num.cpu(), label='Test_num (avg)', color='green', linewidth=1)
        # plt.plot(quara_num.cpu(), label='Quara_num (avg)', color='red', linewidth=1)

        # Plot two types of actions
        action = self.actions
        u0, u1 = self._action_to_u(action)
        u0, u1 = u0.mean(dim=[0,2]), u1.mean(dim=[0,2])  # (period+1)
        plt.plot(u0.cpu(), label='Test rate (avg)', linewidth=1, linestyle='--', color='green', alpha=0.5)
        plt.plot(u1.cpu(), label='Quara rate (avg)', linewidth=1, linestyle='--', color='green')
        # plt.ylim(0, 0.1)
        plt.ylim(0, )

        plt.legend(fontsize=12, loc='upper right')

        plt.title("Daily Current Infection (avg) (" + self.city + ")" if title == '' else title)

        # If fig_dir is None, display the figure
        if fig_dir is None:
            plt.show()
        else:
            # Ensure output directory exists
            os.makedirs(fig_dir, exist_ok=True)

            # Generate file path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(fig_dir, f"{timestamp}.png")

            # Save figure
            plt.savefig(file_path, bbox_inches='tight')
            print(f"Figure saved to {file_path}")

        # Close figure to free memory
        plt.close()



    def render_one_env(self, title='', env_idx=0) :
        plt.figure(dpi=120, figsize=(6, 5))
        plt.grid(linestyle='-.', axis='both')

        # Method needs to be rewritten

        plt.show()

    def render_one_region(self, title='', env_idx=0, region_idx=0) :
        plt.figure(dpi=120, figsize=(6, 5))
        plt.grid(linestyle='-.', axis='both')

        # Observed EI
        estimate_total_EI = (self.history_local_obs[env_idx, :, region_idx, 0] * self.POP[env_idx, region_idx])
        plt.plot(estimate_total_EI.cpu(), label='estimate_total_EI', color='red', linewidth=2, linestyle='--',
                 alpha=0.5)
        # Observed new EI
        estimate_new_E = (self.history_local_obs[env_idx, :, region_idx, 1] * self.POP[env_idx, region_idx])
        plt.plot(estimate_new_E.cpu(), label='estimate_new_E', color='green', linewidth=2, linestyle='--',
                 alpha=0.5)

        # Total EI
        total_EI = self.simRes[env_idx, :, region_idx, [self.E_undetected, self.E_detected,
                                         self.I_undetected, self.I_detected, self.I_reported]].sum(dim=-1)
        plt.plot(total_EI.cpu(), label='Total EI', color='red', linewidth=2, alpha=0.5)
        # Actual new EI
        # actual_new_E = self.daily_new_E[env_idx, :, region_idx]
        actual_new_E = self.simRes[env_idx, :, region_idx, self.E_undetected]
        plt.plot(actual_new_E.cpu(), label='Actual New E', color='green', linewidth=2, alpha=0.5)

        plt.axhline(ls='-.', color='grey')
        plt.xlim(0, self.period)
        plt.xticks(range(0, self.period + 1, 20))
        plt.ylim(0, )

        plt.legend(fontsize=8, loc='upper left')
        plt.twinx()

        # Plot two types of actions
        action = self.actions[env_idx, :, region_idx]  # [period+1]
        u0, u1 = self._action_to_u(action)
        plt.plot(u0.cpu(), label='Action_0 (avg)', linewidth=1, linestyle='--')
        plt.plot(u1.cpu(), label='Action_1 (avg)', linewidth=1, linestyle='--')
        # plt.ylim(0, action_to_u.size(0))

        plt.legend(fontsize=8, loc='upper right')

        plt.title(f"Daily Current Infection (env idx={env_idx}, region idx={region_idx}) (" + self.city + ")" if title == '' else title)
        plt.show()

    def close(self):
        print('close Environment')
        pass

def measure_run_time():
    # 1. Test runtime
    from config import args
    args.simulate_scale = 'community'
    args.zone_num = 643

    # args.device_name = 'cpu'
    env = EpidemicModel(args, env_count=2)
    actions = np.ones((env.env_count, 120, env.ZONE_NUM))
    actions = torch.from_numpy(actions).float().to(env.device)

    random.seed(3074)
    rand_list = [random.randint(0, env.ZONE_NUM - 1) for _ in range(100)]
    rand_idxs = torch.tensor(rand_list, device=env.device)
    # Replicate indices for each batch
    rand_idxs = rand_idxs.repeat(env.env_count, 1)  # shape: (env_count, init_infection)

    import time

    start_time = time.time()
    for i in range(5):
        env.reset(rand_idxs)
        ep_s = 0
        while True:

            s_, r, done, info = env.step(action=actions[:, ep_s, :] * (i % 3))
            ep_s += 1

            if done:
                (ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado,
                 ep_infections_rate, ep_affected_rate, ep_health_cost, ep_economy_cost, ep_order_cost) = info.values()

                # Peak day of total infections needs to be calculated per batch
                peak_days = torch.argmax(env.simRes[:, :, :, 2].sum(dim=2), dim=1).tolist()  # (env_count,)

                for batch_index in range(env.env_count):
                    print(
                        f"Batch {batch_index}, level {i} || reward: {ep_r[batch_index]:.4f}\t"
                        f"overload: {ep_overload[batch_index]:.4f}\t intensity: {ep_intensity[batch_index]:.2f}\t"
                        f"tdo: {ep_tdo[batch_index]:.4f}\t sdo: {ep_sdo[batch_index]:.4f}\t"
                        f"fdo: {ep_fdo[batch_index]:.2f}\t ado: {ep_ado[batch_index]:.2f}\t"
                        f"peak_day: {peak_days[batch_index]}"
                    )

                break

    print("--- Time elapsed: %s seconds ---" % (time.time() - start_time))

def test_consistency():
    # 2. Test consistency (should be very consistent)
    from config import args
    # args.use_obs_imperfect = True
    # args.obs_imperfect_down = 0.5
    args.device_name = 'cpu'
    env = EpidemicModel(args, env_count=10)
    actions = np.ones((env.env_count, 120, env.ZONE_NUM)) * 2
    np.random.seed(3047)
    action1 = np.random.randint(0, env.action_max, env.ZONE_NUM)
    np.random.seed(305)
    action2 = np.random.randint(0, env.action_max, env.ZONE_NUM)
    actions[:, 5:10, 0:10] = action1[0:10]
    actions[:, 10:15, 10:20] = action2[10:20]
    actions = torch.from_numpy(actions).float().to(env.device)

    random.seed(3074)
    rand_list = [random.randint(0, env.ZONE_NUM - 1) for _ in range(100)]
    rand_idxs = torch.tensor(rand_list, device=env.device)
    # Replicate indices for each batch
    rand_idxs = rand_idxs.repeat(env.env_count, 1)  # shape: (env_count, init_infection)

    import time
    start_time = time.time()
    for i in range(1):
        env.reset(rand_idxs)
        ep_s = 0
        ep_r = torch.zeros((env.env_count,), device=env.device)
        while True:

            s_, r, done, info = env.step(action=actions[:, ep_s, :])
            env.reset_contagious_OD()
            ep_s += 1
            ep_r += r



            if done:
                (ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado,
                 ep_affected_rate, ep_health_cost, ep_economy_cost, ep_order_cost) = info.values()
                # env.render()
                env.render_one_region()
                # Peak day of total infections needs to be calculated per batch
                daily_new_I = env.daily_new_I.cpu().numpy()
                daily_curr_I = env.simRes[:, :, :, 2].sum(dim=2)
                peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

                for batch_index in range(env.env_count):
                    total_new_I = sum(daily_new_I[batch_index])
                    print(f"Total new infections: {total_new_I}, Peak current infections: {daily_curr_I[batch_index, peak_days[batch_index]]:.8f}")
                    print(
                        f"Batch {batch_index}, level {i} || reward: {ep_r[batch_index]:.4f}\t"
                        f"overload: {ep_overload[batch_index]:.4f}\t intensity: {ep_intensity[batch_index]:.2f}\t"
                        f"tdo: {ep_tdo[batch_index]:.4f}\t sdo: {ep_sdo[batch_index]:.4f}\t"
                        f"fdo: {ep_fdo[batch_index]:.2f}\t ado: {ep_ado[batch_index]:.2f}\t"
                        f"peak_day: {peak_days[batch_index]}"
                    )

                break

    print("--- Time elapsed: %s seconds ---" % (time.time() - start_time))


def test_control_timely():
    from config import args

    # torch.manual_seed(3047)
    args.simulate_scale = 'community'
    args.zone_num = 643
    # args.ODE_zero_threshold = 0.05

    # Incomplete observation
    # args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.5

    # Gathering behavior
    # args.gather_to_some_region = True
    regions = [0]
    increase_ratio = [10]
    beta_increase_ratio = [3]

    # Dynamic beta
    # args.use_beta_change = True
    args.env_beta_change_rule = 'spatial'

    # Action uncertainty
    # args.use_action_uncertainty = True

    # args.device_name = 'cuda:0'
    args.device_name = 'cpu'
    device = torch.device(args.device_name)
    env = EpidemicModel(args, env_count=2)
    env.seed(3047)

    rand_list = [0 for _ in range(100)]
    rand_idxs = torch.tensor(rand_list, device=env.device)
    # Replicate indices for each batch
    rand_idxs = rand_idxs.repeat(env.env_count, 1)  # shape: (env_count, init_infection)

    env.reset(rand_idxs=rand_idxs)

    # Dynamic beta
    if env.use_beta_change:
        env.dynamic_beta_change(type=args.env_beta_change_rule)

    while True:

        if env.day == 45 and env.gather_to_some_region:
            env.adjust_OD(regions=regions, increase_ratio=increase_ratio, beta_increase_ratio=beta_increase_ratio)

        action = torch.ones(env.env_count, env.ZONE_NUM).to(device)
        action *= 0
        if env.day > 10:
            action *= 2
        action[:, 0] = 4
        s_, r, done, info = env.step(action=action)

        if env.day == 50:
            env.reset_OD()

        if done:
            (ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado,
             ep_infections_rate, ep_affected_rate, ep_health_cost, ep_economy_cost, ep_order_cost) = info.values()
            env.render()
            env.render_one_region()
            # Peak day of total infections needs to be calculated per batch
            daily_new_I = env.daily_new_I.cpu().numpy()
            daily_curr_I = env.simRes[:, :, :, 2].sum(dim=2)
            peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

            for env_idx in range(env.env_count):
                total_new_I = sum(daily_new_I[env_idx])
                print(f"Total new infections: {total_new_I}, Peak current infections: {daily_curr_I[env_idx, peak_days[env_idx]]:.8f}")
                print(
                    f"Batch {env_idx}, level {0} || reward: {ep_r[env_idx]:.4f}\t"
                    f"overload: {ep_overload[env_idx]:.4f}\t intensity: {ep_intensity[env_idx]:.2f}\t"
                    f"tdo: {ep_tdo[env_idx]:.4f}\t sdo: {ep_sdo[env_idx]:.4f}\t"
                    f"fdo: {ep_fdo[env_idx]:.2f}\t ado: {ep_ado[env_idx]:.2f}\t"
                    f"peak_day: {peak_days[env_idx]}"
                )

            break

def calculate_R0_from_params(env):
    # Contribution from exposed
    e_contribution = env.beta * env.beta_E_rate * (1 / env.sigma)
    # Contribution from infected
    i_contribution = env.beta * env.Pm * (1 / env.gamma)
    # Total R0
    R0 = e_contribution + i_contribution
    return R0
def exponential_growth(t, r, C0):
    """Exponential growth model: N(t) = C0 * exp(r*t)"""
    return C0 * np.exp(r * t)

def calculate_R0_from_simulation(env, start=1, end=30):
    from scipy.optimize import curve_fit
    # 1. Extract simulation data (daily new infections)
    daily_new_I = env.daily_new_I.mean(dim=0).sum(dim=-1).cpu().numpy()  # aggregate all environments and zones
    # Take exponential growth phase (first 30 days, assuming sufficient susceptible population)
    t = np.arange(start, end)
    new_cases = daily_new_I[start:end]  # skip day 0

    # 2. Fit exponential growth model to get growth rate r
    try:
        popt, _ = curve_fit(exponential_growth, t, new_cases, p0=(0.1, 1))
        r = popt[0]  # growth rate
    except:
        return np.nan

    # 3. Calculate generation interval T_g (average of latent + infectious period)
    T_g = (1/env.sigma.item() + 1/env.gamma.item()) / 2  # unit: days

    # 4. Calculate R0
    R0 = 1 + r * T_g + (r * T_g) ** 2
    return R0
def robust_R0_calculation(env):
    R0_list = []
    for start in range(1, 16):
        R0 = calculate_R0_from_simulation(env, start=start, end=30)
        print(f"start: {start}, R0: {R0:.4f}")
        R0_list.append(R0)
    print(f"R0: {min(R0_list)} - {max(R0_list)}")
def cal_R0():
    from config import args

    args.simulate_scale = 'community'
    args.zone_num = 654

    # Low R0 experiment
    args.ODE_beta = 0.4
    args.R0 = 'low'

    args.state_contain_detected = True
    args.state_contain_Q = False
    args.state_contain_R = False
    args.state_contain_detect_rate = False
    args.local_obs_dim = 2 + 2 * args.state_contain_detected + args.state_contain_Q + args.state_contain_R + args.state_contain_detect_rate

    # args.device_name = 'cpu'
    device = torch.device(args.device_name)

    env = EpidemicModel(args, env_count=20)
    env.seed(3047)
    env.reset()

    while True:
        action = torch.zeros(env.env_count, env.ZONE_NUM).to(device)

        s_, r, done, info = env.step(action=action)

        if done:
            env.render()
            env.render_all_rooms()
            # env.render_one_region()
            # Peak day of total infections needs to be calculated per batch
            daily_new_I = env.daily_new_I.sum(dim=-1).cpu().numpy()
            daily_curr_I = env.simRes[:, :, :, [1, 2, 3, 4, 5]].sum(dim=(2, 3))
            peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

            for env_idx in range(env.env_count):
                total_new_I = sum(daily_new_I[env_idx])
                print(f"Total new infections: {total_new_I}, Peak current infections: {daily_curr_I[env_idx, peak_days[env_idx]]:.8f}")

            break

    # Calculate R0
    R0_param = calculate_R0_from_params(env)  # based on parameters
    R0_simulation = calculate_R0_from_simulation(env)  # based on simulation data
    robust_R0_calculation(env)
    print(f"R0 from parameters: {R0_param:.2f}")
    print(f"R0 from simulation: {R0_simulation:.2f}")

def natural_transmission():
    from config import args
    args.simulate_scale = 'community'
    args.zone_num = 654

    # args.device_name = 'cpu'
    device = torch.device(args.device_name)

    env = EpidemicModel(args, env_count=1)
    env.seed(3047)
    env.reset()

    while True:
        action = torch.zeros(env.env_count, env.ZONE_NUM).to(device)

        s_, r, done, info = env.step(action=action)

        if done:
            env.render()
            env.render_all_rooms()
            # Peak day of total infections needs to be calculated per batch
            daily_new_I = env.daily_new_I.sum(dim=-1).cpu().numpy()
            daily_curr_I = env.simRes[:, :, :, [1, 2, 3, 4, 5]].sum(dim=(2, 3))
            peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

            for env_idx in range(env.env_count):
                total_new_I = sum(daily_new_I[env_idx])
                print(f"Total new infections: {total_new_I}, Peak current infections: {daily_curr_I[env_idx, peak_days[env_idx]]:.8f}")

            break

    # Export env.simRes to ./draw/simulator_visualization/simRes.npy
    np.save('../draw/simulator_visualization/simRes.npy', env.simRes.cpu().numpy())

def output_imperfect_observation(output_dir="../draw/natural_transmission_observation_levels/data"):
    from config import args
    args.simulate_scale = 'community'
    args.zone_num = 654

    # args.ODE_beta = 0.4
    # args.R0 = 'low'

    # args.device_name = 'cpu'
    device = torch.device(args.device_name)

    env = EpidemicModel(args, env_count=100)
    env.seed(3047)
    env.reset()

    while True:
        action = torch.zeros(env.env_count, env.ZONE_NUM).to(device)
        s_, r, done, info = env.step(action=action)

        if done:
            env.render()
            env.render_all_rooms()
            # Output true state and perfect observation data
            simRes = env.simRes.cpu().numpy()
            perfect_obs = env.extract_observe().cpu().numpy()
            daily_new_E = env.daily_new_E.cpu().numpy()
            # np.save(f"{output_dir}/simRes_low.npy", simRes)
            np.save(f"{output_dir}/simRes1.npy", simRes)
            np.save(f"{output_dir}/perfect_obs.npy", perfect_obs)
            np.save(f"{output_dir}/daily_new_E.npy", daily_new_E)
            break

    # Partial observation
    args.use_obs_imperfect = True
    args.obs_imperfect_down = 0.1
    args.obs_imperfect_up = 1
    env = EpidemicModel(args, env_count=100)
    env.seed(3047)
    env.reset()

    while True:
        action = torch.zeros(env.env_count, env.ZONE_NUM).to(device)
        s_, r, done, info = env.step(action=action)

        if done:
            env.render()
            env.render_all_rooms()
            # Output true state and partial observation data
            simRes = env.simRes.cpu().numpy()
            partial_obs = env.extract_observe().cpu().numpy()
            np.save(f"{output_dir}/simRes2.npy", simRes)
            np.save(f"{output_dir}/partial_obs.npy", partial_obs)
            print("missing rate:", env.get_observation_missing_rate().mean())
            break


if __name__ == '__main__':
    # measure_run_time()
    # exit(0)
    # test_control_timely()
    # exit(0)
    # cal_R0()
    # exit(0)
    # natural_transmission()
    # exit(0)
    output_imperfect_observation()
    exit(0)
    from config import args


    args.simulate_scale = 'community'
    args.zone_num = 654

    args.state_contain_detected = True
    args.state_contain_Q = False
    args.state_contain_R = False
    args.state_contain_detect_rate = False
    args.local_obs_dim = 2 + 2 * args.state_contain_detected + args.state_contain_Q + args.state_contain_R + args.state_contain_detect_rate

    # Incomplete observation
    # args.use_obs_imperfect = True
    args.obs_imperfect_down = 1.0

    # Gathering behavior
    # args.gather_to_some_region = True
    regions = [0]
    increase_ratio = [10]
    beta_increase_ratio = [3]

    # Dynamic beta
    # args.use_beta_change = True
    args.env_beta_change_rule = 'spatial'

    # Action uncertainty
    # args.use_action_uncertainty = True


    # args.device_name = 'cuda:0'
    args.device_name = 'cpu'
    device = torch.device(args.device_name)

    args.ODE_detect_E_rate = 0
    # args.ODE_detect_I_rate = 1.0

    env = EpidemicModel(args, env_count=2)
    env.seed(3047)
    env.reset()

    # Dynamic beta
    if env.use_beta_change:
        env.dynamic_beta_change(type=args.env_beta_change_rule)


    while True:

        if env.day == 45 and env.gather_to_some_region:
            env.adjust_OD(regions=regions, increase_ratio=increase_ratio, beta_increase_ratio=beta_increase_ratio)

        action=torch.ones(env.env_count, env.ZONE_NUM).to(device)

        action = action * 4

        s_, r, done, info = env.step(action=action)

        if env.day == 50:
            env.reset_OD()

        if done:
            # (ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado,
            #  ep_infections_rate, ep_affected_rate, ep_health_cost, ep_economy_cost, ep_order_cost) = info.values()
            #
            env.render()
            env.render_all_rooms()
            # env.render_one_region()
            # Peak day of total infections needs to be calculated per batch
            daily_new_I = env.daily_new_I.sum(dim=-1).cpu().numpy()
            daily_curr_I = env.simRes[:, :, :, [1,2,3,4,5]].sum(dim=(2,3))
            peak_days = torch.argmax(daily_curr_I, dim=1).tolist()  # (env_count,)

            for env_idx in range(env.env_count):
                total_new_I = sum(daily_new_I[env_idx])
                print(f"Total new infections: {total_new_I}, Peak current infections: {daily_curr_I[env_idx, peak_days[env_idx]]:.8f}")

            break