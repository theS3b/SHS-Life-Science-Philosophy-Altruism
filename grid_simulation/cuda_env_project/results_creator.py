# simulation_runner.py

import os
# enable the new allocator tunable to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from grid_simulation import get_rgb_map
from cuda_square_simulation import SquareSimulation
from agent import PolicyNet, select_action, clever_single_action

# Disable interactive plotting
plt.ioff()

# Initialize policy network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_shape = (3, 5, 5)
policy_net = PolicyNet(obs_shape).to(device)
policy_net.load_state_dict(torch.load('policy_net.pth', map_location=device))
policy_net.eval()


def get_results_imgs(simulation,
                     iterations=50,
                     type1='random',
                     type2='random',
                     type3='random',
                     save_dir=None,
                     title=None):
    """
    Run the simulation for a fixed number of iterations,
    collect population fitness sums, and save two plots:
    1) final RGB distribution map
    2) time-series of fitness sums
    """

    pop1_size, pop2_size, pop3_size = [], [], []

    # Run entire loop without building any autograd graph
    with torch.inference_mode():
        for _ in range(iterations):
            if type1 == 'Reinforcement Learning Agent':
                _ = simulation.configurable_actions_grid(
                    type1=None, type2=type2, type3=type3
                )
                observations, obs_indices = simulation.get_deep_observations(0)
                action = select_action(policy_net, observations)
                action_grid = clever_single_action(
                    simulation, action, obs_indices, type2, type3
                )
            else:
                action_grid = simulation.configurable_actions_grid(
                    type1=type1, type2=type2, type3=type3
                )

            simulation.step(action_grid)

            # compute mean sum of each channel
            s1 = simulation.grid[:, 0].sum(dim=(1, 2)).mean().cpu().item()
            s2 = simulation.grid[:, 1].sum(dim=(1, 2)).mean().cpu().item()
            s3 = simulation.grid[:, 2].sum(dim=(1, 2)).mean().cpu().item()
            pop1_size.append(s1)
            pop2_size.append(s2)
            pop3_size.append(s3)

    # Final stats
    final1, final2, final3 = pop1_size[-1], pop2_size[-1], pop3_size[-1]
    total = final1 + final2 + final3
    safe_total = max(total, 1e-6)

    result_str = (
        "\nFinal population fitness:\n"
        f"Population 1 ({type1}): {final1}\n"
        f"Population 2 ({type2}): {final2}\n"
        f"Population 3 ({type3}): {final3}\n\n"
        f"Total fitness: {total}\n\n"
        "Final population %'s fitness:\n"
        f"Population 1 ({type1}): {final1 / safe_total * 100:.2f}%\n"
        f"Population 2 ({type2}): {final2 / safe_total * 100:.2f}%\n"
        f"Population 3 ({type3}): {final3 / safe_total * 100:.2f}%\n"
    )
    print(result_str)

    # --- Plot 1: Final distribution map ---
    fig, ax = plt.subplots()
    rgb_map = get_rgb_map(simulation, plt)
    ax.imshow(rgb_map)
    ax.axis('off')
    ax.legend(
        handles=[
            Patch(facecolor='red',   label=type1.capitalize()),
            Patch(facecolor='blue',  label=type2.capitalize()),
            Patch(facecolor='green', label=type3.capitalize()),
        ],
        loc='upper right',
        bbox_to_anchor=(1.5, 1.0)
    )
    fig.suptitle('Exemple de Distribution Finale des Populations')
    if save_dir:
        out = os.path.join(save_dir, f'{title}_final_distr.pdf')
        fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 2: Time-series of fitness sums ---
    fig, ax = plt.subplots()
    x = np.arange(iterations)
    ax.plot(x, pop1_size, linestyle='--', marker='o', markevery=10, label=type1.capitalize())
    ax.plot(x, pop2_size,           marker='x', markevery=7,  label=type2.capitalize())
    ax.plot(x, pop3_size,           marker='s', markevery=13, label=type3.capitalize())
    ax.set_xlabel('Itérations')
    ax.set_ylabel('Somme des Valeurs Sélectives')
    ax.set_title('Somme des Valeurs Sélectives par Population')
    ax.legend()
    if save_dir:
        out = os.path.join(save_dir, f'{title}_time_plot.pdf')
        fig.savefig(out, bbox_inches='tight')
    plt.close(fig)

    return result_str


if __name__ == '__main__':
    # Simulation settings
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    SAVE_DIR = os.path.join(parent_dir, 'simulation_imgs')
    os.makedirs(SAVE_DIR, exist_ok=True)

    iterations = 1000
    nb_batches, rows, cols = 2000, 20, 20
    cuda_intensive_batch = 500
    populations = {
        "red":   {"p": 0.2, "mean_v": 1.0, "std_v": 0.1},
        "blue":  {"p": 0.2, "mean_v": 1.0, "std_v": 0.1},
        "green": {"p": 0.2, "mean_v": 1.0, "std_v": 0.1},
    }

    simulation_parameters = [
        ('bonus_0_0',  0.00, 0.00),
        ('bonus_0_04', 0.00, 0.04),
        ('bonus_04_0', 0.04, 0.00),
        ('bonus_04_04',0.04, 0.04),
        ('bonus_08_0', 0.08, 0.00),
        ('bonus_08_04',0.08, 0.04),
        ('bonus_0_08', 0.00, 0.08),
        ('bonus_04_08',0.04, 0.08),
        ('bonus_08_08',0.08, 0.08),
    ]

    scenarios = [
        ('attackVsRandom',        'attacking', 'random', 'random'),
        ('givingVsRandom',        'giving',    'random', 'random'),
        ('completeVsRandom',      'complete',  'random', 'random'),
        ('completeVsGivingVsAttacking', 'complete','giving','attacking'),
        ('RLAgentVsRandom',       'Reinforcement Learning Agent', 'random', 'random'),
        ('RLAgentVsAttacker',     'Reinforcement Learning Agent', 'attacking', 'attacking'),
        ('RLAgentVsGiving',       'Reinforcement Learning Agent', 'giving',    'giving'),
    ]

    all_results = []

    for name, donation, growth in simulation_parameters:
        title = f"don_{donation}_growth_{growth}"
        print(f"\n=== Parameter set: {name} (don={donation}, growth={growth}) ===")

        for folder, t1, t2, t3 in scenarios:
            print(f"Running scenario: {folder}")
            sim = SquareSimulation(
                nb_batch=nb_batches if folder != 'RLAgentVsRandom' and folder != 'RLAgentVsAttacker' and folder != 'RLAgentVsGiving' else cuda_intensive_batch,
                rows=rows, cols=cols,
                populations=populations, device=device,
                FITNESS_DONATION_BONUS=donation,
                FITNESS_GROWTH_VALUE=growth
            )
            sim.reset()

            scenario_dir = os.path.join(SAVE_DIR, folder)
            os.makedirs(scenario_dir, exist_ok=True)

            res = get_results_imgs(
                sim,
                iterations=iterations,
                type1=t1, type2=t2, type3=t3,
                save_dir=scenario_dir,
                title=title
            )
            all_results.append(f"Parameters: {name}, Scenario: {folder}\n{res}\n")

            # free GPU memory and Python references
            del sim
            gc.collect()
            torch.cuda.empty_cache()

    # Write summary file
    summary_path = os.path.join(SAVE_DIR, "results_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("\n".join(all_results))

    print(f"\nAll simulations complete. Summary written to {summary_path}")
