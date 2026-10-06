import random
import numpy as np
from sklearn.linear_model import SGDRegressor

# 0 = open path, 1 = wall, 2 = danger zone.
# 10x10 grid, checked with a BFS so the goal is always reachable.
GRID = [
    [0, 0, 2, 0, 1, 2, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 0, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 2, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 2, 2, 2, 0, 0],
    [0, 2, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
]
START = (0, 0)
GOAL = (9, 9)

# Row and column changes for each action.
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

ROWS = len(GRID)
COLUMNS = len(GRID[0])
NUMBER_OF_ACTIONS = len(ACTIONS)
NUMBER_OF_FEATURES = ROWS * COLUMNS * NUMBER_OF_ACTIONS

# reward system, one value per situation
REWARD_NORMAL_MOVE = -1      # moving to a normal open cell
REWARD_INVALID_MOVE = -5     # trying to move outside the grid
REWARD_WALL = -8             # trying to move into a wall
REWARD_DANGER_ZONE = -15     # stepping into a danger zone (still allowed)
REWARD_GOAL = 50             # reaching the target


def step(state, action):
    """Return next state, reward, and termination flag."""
    row = state[0] + ACTIONS[action][0]
    column = state[1] + ACTIONS[action][1]

    # trying to leave the grid: stay in place, get penalized
    if not (0 <= row < ROWS and 0 <= column < COLUMNS):
        return state, REWARD_INVALID_MOVE, False

    # trying to walk into a wall: stay in place, get penalized
    if GRID[row][column] == 1:
        return state, REWARD_WALL, False

    next_state = (row, column)

    # reaching the goal ends the episode
    if next_state == GOAL:
        return next_state, REWARD_GOAL, True

    # stepping into a danger zone: allowed, but costly
    if GRID[row][column] == 2:
        return next_state, REWARD_DANGER_ZONE, False

    # normal open cell
    return next_state, REWARD_NORMAL_MOVE, False


def encode(state, action):
    """Encode one state-action pair as a one-hot vector."""
    features = np.zeros(NUMBER_OF_FEATURES, dtype=float)
    state_index = state[0] * COLUMNS + state[1]
    feature_index = state_index * NUMBER_OF_ACTIONS + action
    features[feature_index] = 1.0
    return features


def predict_q_values(model, state):
    """Predict the Q-value of each available action."""
    features = np.array([
        encode(state, action)
        for action in range(NUMBER_OF_ACTIONS)
    ])
    return model.predict(features)


def train(episodes=1000):
    """Learn Q-values, then evaluate the policy."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    rng = random.Random(42)
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    max_steps_per_episode = 300  # bigger grid, so more steps are allowed

    # incremental linear model for Q-values
    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        learning_rate="constant",
        eta0=0.1,
        random_state=42,
    )

    # initialize before calling predict()
    model.partial_fit(
        np.zeros((1, NUMBER_OF_FEATURES)),
        np.array([0.0]),
    )
    successes = 0
    rewards = []

    for _ in range(episodes):
        state = START
        total = 0
        for _ in range(max_steps_per_episode):
            if rng.random() < epsilon:
                action = rng.randrange(NUMBER_OF_ACTIONS)
            else:
                q_values = predict_q_values(model, state)
                best_actions = np.flatnonzero(
                    q_values == q_values.max()
                ).tolist()
                action = rng.choice(best_actions)

            next_state, reward, terminated = step(state, action)

            # terminal states have no future reward
            if terminated:
                target = float(reward)
            else:
                next_q_values = predict_q_values(model, next_state)
                target = reward + gamma * float(next_q_values.max())

            # learn from one observed transition
            features = encode(state, action).reshape(1, -1)
            model.partial_fit(features, np.array([target]))

            state = next_state
            total += reward
            if terminated:
                successes += 1
                break

        # one total reward recorded per episode
        rewards.append(total)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # evaluate without updating the model (no exploration)
    state = START
    path = [state]
    steps = []
    max_eval_steps = 300
    for number in range(1, max_eval_steps + 1):
        q_values = predict_q_values(model, state)
        action = int(np.argmax(q_values))
        next_state, reward, terminated = step(state, action)

        cell_type = "Goal" if next_state == GOAL else (
            "Danger Zone" if GRID[next_state[0]][next_state[1]] == 2 else "Path"
        )

        steps.append({
            "number": number,
            "state": state,
            "action": ACTION_NAMES[action],
            "next_state": next_state,
            "cell_type": cell_type,
            "reward": reward,
        })

        path.append(next_state)
        state = next_state
        if terminated:
            break

    reached_goal = state == GOAL
    total_eval_reward = sum(s["reward"] for s in steps)

    # build a display table from model predictions, for every open/start cell
    q_table = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            position = (row, column)
            if GRID[row][column] != 1 and position != GOAL:
                q_table.append({
                    "state": position,
                    "action_values": predict_q_values(
                        model, position
                    ).tolist(),
                })

    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": round(successes / episodes * 100, 1),
        "final_average": round(
            sum(rewards[-100:]) / len(rewards[-100:]), 2
        ),
        "final_epsilon": round(epsilon, 3),
        "reached_goal": reached_goal,
        "path": path,
        "steps": steps,
        "total_eval_reward": total_eval_reward,
        "q_table": q_table,
    }