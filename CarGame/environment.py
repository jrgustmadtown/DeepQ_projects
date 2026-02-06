from config import GRID_SIZE, ACTIONS, AGENT_A_START, CRASH_REWARD


class CarGameEnvironment:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.rows = grid_size
        self.cols = grid_size
        self.actions = ACTIONS
        self.start_a = AGENT_A_START
        self.start_b = (grid_size - 1, grid_size - 1)
        self.square_rewards = self._compute_square_rewards()

    def _compute_square_rewards(self):
        rewards = {}
        corners = [
            (0, 0),
            (0, self.grid_size - 1),
            (self.grid_size - 1, 0),
            (self.grid_size - 1, self.grid_size - 1)
        ]
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                min_dist = min(abs(row - cr) + abs(col - cc) for cr, cc in corners)
                rewards[(row, col)] = 2 ** min_dist
        return rewards

    def get_square_reward(self, position):
        return self.square_rewards.get(position, 0)

    def get_available_actions(self, pos):
        row, col = pos
        actions = []
        if row > 0:
            actions.append('up')
        if row < self.grid_size - 1:
            actions.append('down')
        if col > 0:
            actions.append('left')
        if col < self.grid_size - 1:
            actions.append('right')
        return actions

    def is_valid_position(self, pos):
        row, col = pos
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def get_next_position(self, pos, action):
        row, col = pos
        if action == 'up':
            next_pos = (row - 1, col)
        elif action == 'down':
            next_pos = (row + 1, col)
        elif action == 'left':
            next_pos = (row, col - 1)
        elif action == 'right':
            next_pos = (row, col + 1)
        else:
            next_pos = (row, col)
        if not self.is_valid_position(next_pos):
            next_pos = pos
        return next_pos

    def reset(self):
        return (self.start_a, self.start_b)

    def step(self, state, joint_action):
        pos_a, pos_b = state
        action_a, action_b = joint_action
        next_pos_a = self.get_next_position(pos_a, action_a)
        next_pos_b = self.get_next_position(pos_b, action_b)
        next_state = (next_pos_a, next_pos_b)
        if next_pos_a == next_pos_b:
            reward_a = -CRASH_REWARD
            reward_b = -CRASH_REWARD
            done = False
        else:
            reward_a_square = self.get_square_reward(next_pos_a)
            reward_b_square = self.get_square_reward(next_pos_b)
            reward_a = reward_a_square
            reward_b = reward_b_square
            done = False
        return next_state, (reward_a, reward_b), done

    def get_all_states(self):
        states = []
        for a_row in range(self.grid_size):
            for a_col in range(self.grid_size):
                for b_row in range(self.grid_size):
                    for b_col in range(self.grid_size):
                        pos_a = (a_row, a_col)
                        pos_b = (b_row, b_col)
                        if pos_a != pos_b:
                            states.append((pos_a, pos_b))
        return states

    def get_all_joint_actions(self):
        joint_actions = []
        for a_action in self.actions:
            for b_action in self.actions:
                joint_actions.append((a_action, b_action))
        return joint_actions
