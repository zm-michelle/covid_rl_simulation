import numpy as np 
import pickle
import torch
import torch.nn as nn
from torch.distributions import Categorical
from CovidSim import gameEnvironment
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from datetime import datetime 
from tqdm.auto import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import os
import json
import time
def get_named_action_counts(flat_actions, idx_action_counts):
        action_names ={}

        if isinstance(idx_action_counts, torch.Tensor):
            action_counts = idx_action_counts.tolist()
        else:
            action_counts = list(idx_action_counts.values())

 
        assert len(action_counts) == len(flat_actions)
        for action_desc, count in zip(flat_actions, action_counts):
            if action_desc==('*', {}):
                continue
            if count > 0:   
                action_names[str(action_desc)] = count
             
 
        return action_names

def plot_actions(flat_actions, idx_action_counts, save_path='action_dist.png'):
        action_names = get_named_action_counts(flat_actions, idx_action_counts)

        if not action_names:   
            print("No actions to plot")
            return {}
        labels = [str(action) for action in action_names.keys()]
        counts = list(action_names.values())

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(labels)), counts, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Actions', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title('Action Distribution', fontsize=14, fontweight='bold')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Action plot saved to {save_path}")
        return action_names

class Agent:
    def __init__(self, env, hidden_size=512, gamma =0.99, temperature=1.0, device=torch.device('cpu')):
        super().__init__()

        self.env = env
        self.gamma =gamma
        self.hiden_size = hidden_size
        self.n_actions = len(env.flat_actions)
        self.temperature = temperature 
        self.observation_space = len(env.get_observation())
        self.device = device

        self.in_size = self.observation_space
        self.out_size = self.n_actions  
        self.observation_space 

        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.out_size)
        ).double().to(device)

        self.critic = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        ).double().to(device)
    

    def train_epoch(self, verbose=False):
        actions_recorded = torch.zeros(self.out_size).to(self.device)
        rewards = []
        critic_vals = []
        action_logprob_vals = []

        obs = torch.tensor(self.env.reset(), dtype=torch.float64).to(self.device)
        done = False
 
        while not done:
            action_logits = self.actor(obs)
            valid_actions = self.env.get_valid_actions()

            if verbose:
                print(f"Turn: {self.env.turn_actions + 1}/{self.env.turns_per_day} of day {self.env.sim.day}")

            #-------masking action------------

            mask = torch.full_like(action_logits, float('-inf'))
            mask[valid_actions] = 0.0

            masked_logits = action_logits + mask 
            masked_logits /= self.temperature

            if verbose == True:
                print("Masked Logits: ",masked_logits)

            dist = Categorical(logits=masked_logits)
            

            action = dist.sample()
            action_log_prob = dist.log_prob(action)

            actions_recorded[action] += 1

            pred = torch.squeeze(self.critic(obs).view(-1))
            action_logprob_vals.append(action_log_prob)
            critic_vals.append(pred)

            if verbose == True:
                print(f"Taking action: {self.env.flat_actions[action.item()]}")
            try:
                next_obs, reward, done, _ = self.env.step(action.item())

            except Exception as e:
                print(f"ERROR during step: {e}")
                import traceback
                traceback.print_exc()
                raise
            

            obs = torch.tensor(next_obs, dtype=torch.float64).to(self.device)
            rewards.append(torch.tensor(reward).double().to(self.device))
 

        if verbose == True:
            print(f"Total reward: {sum(rewards)}")
        total_reward = sum(rewards)
        for t_i in range(len(rewards)):
            G = 0
            for t in range(t_i, len(rewards)):
                G += rewards[t] * (self.gamma ** (t - t_i))
            rewards[t_i] = G

        def f(inp):
            return torch.stack(tuple(inp), 0)

        rewards = f(rewards)
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-15)


        return rewards, f(critic_vals), f(action_logprob_vals), total_reward, actions_recorded
    
    @staticmethod
    def compute_loss(action_p_vals, G, V, critic_loss=nn.SmoothL1Loss()):
        assert len(action_p_vals) == len(G) == len(V)
        advantage = G - V.detach()
        return -(torch.sum(action_p_vals * advantage)), critic_loss(G, V)
    
    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


    def test(self, verbose=False, plot_results='test_results.png'):
        actions_recorded = torch.zeros(self.out_size).to(self.device)
        rewards = []
        
        obs = torch.tensor(self.env.reset(), dtype=torch.float64).to(self.device)
        done = False

        infections = []
        new_infections =[]
        immune_people = []
        removed_people = []
        action_names = []
        while not done:
            action_logits = self.actor(obs)
            valid_actions = self.env.get_valid_actions()

            if verbose:
                print(f"Turn: {self.env.turn_actions + 1}/{self.env.turns_per_day} of day {self.env.sim.day}")

            #-------masking action------------
            mask = torch.full_like(action_logits, float('-inf'))
            mask[valid_actions] = 0.0

            masked_logits = action_logits + mask 

            if verbose == True:
                print("Masked Logits: ", masked_logits)
 
            action = torch.argmax(masked_logits)
            actions_recorded[action] += 1
            action_name =self.get_action_name(action)
            
            obs, reward, done, info = self.env.step(action.item())
            obs = torch.tensor(obs, dtype=torch.float64).to(self.device)
            rewards.append(reward)

            final_state = self.env.state.iloc[-1]

            #print(f"Turn: {self.env.turn_actions + 1}/{self.env.turns_per_day} of day {self.env.sim.day}")
            if self.env.turn_actions ==2:
                infections.append(int(final_state['infected']))
                new_infections.append(int(final_state['new_infections']))
                immune_people.append(int(final_state['immune']))
                removed_people.append(int(final_state['removed']))
                action_names.append(str(action_name))  

        if plot_results is not None:
     
            plt.figure(figsize=(10, 6))
            plt.plot(infections, label='Total Infections', linewidth=2)
            plt.plot(new_infections, label='New Infections', linewidth=2)  
            plt.plot(removed_people, label='Removed People', linewidth=2)

            plt.xlabel('Days', fontsize=12)
            plt.ylabel('Number of People', fontsize=12)
            plt.title('Game Over Time', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_results)
            plt.close()

            plot_actions(self.env.flat_actions, actions_recorded.cpu(), save_path=plot_results[:-4]+"_actions.png")

            with open(f'{plot_results[:-3]}_actions.csv', 'w') as f:
                for word in action_names:
                    f.write(word + '\n')
        results = {
            'rewards': [float(r) for r in rewards],   
            'total_reward': float(sum(rewards)),   
            'actions_recorded': actions_recorded.cpu().tolist(),  
            'infections': infections,
            'new_infections': new_infections,
            'immune_people': immune_people,
            'removed_people': removed_people,
            'named_actions': action_names
        }
        if plot_results is not None:
            with open(f'{plot_results[:-4]}.json', 'w') as f:
                json.dump(results, f, indent=4)

        return results
    def get_action_name(self, action_idx):
        return self.env.flat_actions[action_idx]
    
     
    
            



def Empty():
    print("empty")



def train(lr_actor=1e-4, lr_critic=1e-6, epochs=3000, model_dir='models',verbose=True,device=torch.device('cpu'),temperature=1.0):
    NETWORK_FILE ='Community.gml'
     
    r = []  

    #-------------- track results---------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'covid_sim_{timestamp}')
    writer = SummaryWriter(log_dir)

    env = gameEnvironment(rl_agent=Empty)
    agent = Agent(env, device=device, temperature=temperature)
    #avg_r = 0   
    actor_optim = optim.Adam(agent.actor.parameters(), lr=lr_actor)
    critic_optim = optim.Adam(agent.critic.parameters(), lr=lr_critic)

    if verbose == True:
        print("Start Training Loop")
    #------- training loop----------
    for i in tqdm(range(epochs), desc="Training"):
        critic_optim.zero_grad()
        actor_optim.zero_grad()

        if verbose == True:
            print("Running epoch...")
        rewards, critic_vals, action_lp_vals, total_reward, actions_recorded = agent.train_epoch( verbose=verbose)
        r.append(total_reward)

        actor_loss, critic_loss = agent.compute_loss(action_p_vals=action_lp_vals, G=rewards, V=critic_vals)

        actor_loss.backward()
        critic_loss.backward()

        actor_grad_norm = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=float('inf'))
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=float('inf'))

        actor_optim.step()
        critic_optim.step()

        final_state = env.state.iloc[-1]
        writer.add_scalar('Network/susceptible', final_state['susceptible'], i)
        writer.add_scalar('Network/infected', final_state['infected'], i)
        writer.add_scalar('Network/removed', final_state['removed'], i)
        writer.add_scalar('Network/immune', final_state['immune'], i)
        writer.add_scalar('Network/money', final_state['liquid'], i)

        writer.add_scalar('Reward/Total', total_reward, i)
        
        writer.add_scalar('GradNorm/Critic', actor_grad_norm, i)
        writer.add_scalar('GradNorm/Actor', critic_grad_norm, i)

        writer.add_scalar('Loss/Actor', actor_loss.item(), i)
        writer.add_scalar('Loss/Critic', critic_loss.item(), i)

        if i % 100 == 0:
            torch.save({
                'epoch': i,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': actor_optim.state_dict(),
                'critic_optimizer_state_dict': critic_optim.state_dict(),
                'reward': total_reward,
            }, os.path.join(model_dir, f'checkpoint_epoch_{i}.pt'))
    torch.save({
        'epoch': epochs,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
    }, os.path.join(model_dir, 'final_model.pt'))
def test(model_path,  device= torch.device('cpu'), plot_results=None):
    r = []  
    env = gameEnvironment(rl_agent=Empty)
    agent = Agent(env, device=device)
    agent.load_model(model_path)
    results = agent.test(plot_results=None )
    return results


def n_tests(n, model_path, tests_folder, device=None):
    """
    Run n tests and aggregate results
    
    Args:
        n: Number of tests to run
        model_path: Path to the trained model
        tests_folder: Folder to save test results
        device: Torch device (CPU or CUDA)
    
    Returns:
        Dictionary with aggregated results
    """
    if device is None:
        device = torch.device('cpu')
    
    # Initialize lists to store results
    rewards = []
    reward_sum = []
    actions_recorded = []
    infections = []
    new_infections = []
    immune_people = []
    removed_people = []
    all_named_actions = []
    
    # Create tests folder
    os.makedirs(tests_folder, exist_ok=True)
    
    print(f"Running {n} tests...")
    
    for i in tqdm(range(n), desc="Testing"):
        # Run individual test
        test_plot_path = os.path.join(tests_folder, f'test_{i}.png')
        results = test(model_path, device=device, plot_results=test_plot_path)
        
        # Append results
        rewards.append(results['rewards'])
        reward_sum.append(results['total_reward'])
        actions_recorded.append(results['actions_recorded'])
        infections.append(results['infections'])
        new_infections.append(results['new_infections'])
        immune_people.append(results['immune_people'])
        removed_people.append(results['removed_people'])
        all_named_actions.append(results['named_actions'])
    
    # Compute statistics
    reward_sum_array = np.array(reward_sum)
    
    aggregated_results = {
        'n_tests': n,
        'rewards': rewards,
        'reward_sum': reward_sum,
        'reward_mean': float(np.mean(reward_sum_array)),
        'reward_std': float(np.std(reward_sum_array)),
        'reward_min': float(np.min(reward_sum_array)),
        'reward_max': float(np.max(reward_sum_array)),
        'actions_recorded': actions_recorded,
        'infections': infections,
        'new_infections': new_infections,
        'immune_people': immune_people,
        'removed_people': removed_people,
        'named_actions': all_named_actions
    }
    
    # Save aggregated results
    with open(os.path.join(tests_folder, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated_results, f, indent=4)
    
    # Create summary plots
    create_summary_plots(aggregated_results, tests_folder)
    
    # Save CSV with all named actions
    with open(os.path.join(tests_folder, 'all_actions.csv'), 'w') as f:
        f.write('Test_ID,Action\n')
        for i, actions in enumerate(all_named_actions):
            for action in actions:
                f.write(f'{i},{action}\n')
    
    print(f"\nTest Summary:")
    print(f"  Mean Reward: {aggregated_results['reward_mean']:.2f}")
    print(f"  Std Reward: {aggregated_results['reward_std']:.2f}")
    print(f"  Min Reward: {aggregated_results['reward_min']:.2f}")
    print(f"  Max Reward: {aggregated_results['reward_max']:.2f}")
    
    return aggregated_results

def create_summary_plots(results, save_dir):
    """Create summary plots from aggregated test results"""
    
    # Plot 1: Mean ± SEM trajectories for infections, new_infections, and removed_people
    plt.figure(figsize=(12, 7))
    
    # Prepare data for each metric
    metrics = {
        'Total Infections': results['infections'],
        'New Infections': results['new_infections'],
        'Removed People': results['removed_people']
    }
    
    colors = {'Total Infections': 'blue', 'New Infections': 'orange', 'Removed People': 'green'}
    
    for metric_name, metric_data in metrics.items():
        # Find max length and pad arrays
        max_len = max(len(run) for run in metric_data)
        padded_data = np.array([run + [np.nan] * (max_len - len(run)) for run in metric_data])
        
        # Calculate mean and SEM
        mean_values = np.nanmean(padded_data, axis=0)
        sem_values = np.nanstd(padded_data, axis=0) / np.sqrt(np.sum(~np.isnan(padded_data), axis=0))
        
        days = np.arange(len(mean_values))
        
        # Plot mean line
        plt.plot(days, mean_values, label=metric_name, color=colors[metric_name], linewidth=2)
        
        # Plot shaded SEM region
        plt.fill_between(days, 
                         mean_values - sem_values, 
                         mean_values + sem_values, 
                         alpha=0.3, 
                         color=colors[metric_name])
    
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Mean ± SEM Across Runs', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_mean_sem.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Action heatmap showing action types over games
    # Parse action names to extract action types
    all_action_sequences = results['named_actions']
    
    # Extract action types (first element of tuple)
    action_types_per_run = []
    for run_actions in all_action_sequences:
        action_types = []
        for action_str in run_actions:
            try:
                # Parse the string representation of the tuple
                # Extract the action type (first element before comma)
                action_type = action_str.split("'")[1] if "'" in action_str else 'unknown'
                action_types.append(action_type)
            except:
                action_types.append('unknown')
        action_types_per_run.append(action_types)
    
    # Get unique action types and assign colors
    all_action_types = set()
    for run in action_types_per_run:
        all_action_types.update(run)
    
    unique_actions = sorted(list(all_action_types))
    action_to_idx = {action: idx for idx, action in enumerate(unique_actions)}
    
    # Create color map
    n_actions = len(unique_actions)
    cmap = plt.cm.get_cmap('tab10' if n_actions <= 10 else 'tab20')
    action_colors = {action: cmap(i / n_actions) for i, action in enumerate(unique_actions)}
    
    # Create the heatmap data
    max_days = max(len(run) for run in action_types_per_run)
    n_runs = len(action_types_per_run)
    
    # Create matrix for heatmap
    action_matrix = np.full((max_days, n_runs), np.nan)
    
    for run_idx, action_sequence in enumerate(action_types_per_run):
        for day_idx, action_type in enumerate(action_sequence):
            action_matrix[day_idx, run_idx] = action_to_idx[action_type]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create custom colormap
    from matplotlib.colors import ListedColormap
    colors_list = [action_colors[action] for action in unique_actions]
    custom_cmap = ListedColormap(colors_list)
    
    # Plot heatmap
    im = ax.imshow(action_matrix.T, aspect='auto', cmap=custom_cmap, 
                   interpolation='nearest', vmin=0, vmax=n_actions-1)
    
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Game / Epic', fontsize=12)
    ax.set_title('Testing Gameplay - Action Types', fontsize=14, fontweight='bold')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=action_colors[action], label=action) 
                      for action in unique_actions]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
             loc='upper left', fontsize=8, ncol=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'action_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a simpler action distribution plot
    plt.figure(figsize=(12, 6))
    action_counts = Counter()
    for run in action_types_per_run:
        action_counts.update(run)
    
    actions_sorted = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    action_names = [a[0] for a in actions_sorted]
    counts = [a[1] for a in actions_sorted]
    
    bars = plt.bar(range(len(action_names)), counts, 
                   color=[action_colors[a] for a in action_names],
                   edgecolor='black', alpha=0.7)
    
    plt.xlabel('Action Type', fontsize=12, fontweight='bold')
    plt.ylabel('Total Count Across All Runs', fontsize=12, fontweight='bold')
    plt.title('Action Type Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(len(action_names)), action_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'action_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
 
if __name__ == "__main__":
    #train(device=torch.device('cpu'),temperature=1.5,  epochs=3000)
    #test('models/final_model.pt', device=torch.device('cpu'))
    n =200
    model_path = 'best_dec8/final_model.pt'
    tests_folder = 'tests_dec8'

    n_tests(n, model_path, tests_folder, device=torch.device('cpu'))
    
    