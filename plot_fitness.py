"""
Fitness Curve Plotter for CMA-ES Optimization
Plots fitness (energy) vs generation from JSON data exported by cdf_thick_panel_2.py

Usage:
    python plot_fitness.py              # Opens file dialog to select JSON
    python plot_fitness.py input.json  # Uses specified JSON file
    python plot_fitness.py input.json -o output.png
"""

import json
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Try to import tkinter for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Configure Arial font
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12


def load_data(json_path: str) -> dict:
    """Load optimization data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_fitness_curve(data: dict, output_path: str = None, show: bool = True):
    """
    Plot fitness curve with mean, std, and min from CMA-ES optimization.

    Parameters:
        data: Dictionary containing 'gen', 'avg', 'std', 'min' keys
        output_path: Path to save the figure (optional)
        show: Whether to display the plot
    """
    # Scientific color palette
    COLOR_MEAN = '#1f77b4'      # Professional blue
    COLOR_BEST = '#d62728'       # Distinct red for best
    COLOR_STD = '#7fb3d5'       # Lighter blue for std region

    # Extract data
    generations = np.array(data['gen'])
    avg_fitness = np.array(data['avg'])
    std_fitness = np.array(data['std'])
    min_fitness = np.array(data['min'])

    # Create figure with proper size
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    # Plot std as shaded region first (behind lines)
    ax.fill_between(
        generations,
        avg_fitness - std_fitness,
        avg_fitness + std_fitness,
        alpha=0.25,
        color=COLOR_STD,
        linewidth=0,
        label='Mean ± Std'
    )

    # Plot mean fitness curve
    ax.plot(generations, avg_fitness, '-', linewidth=2.5, color=COLOR_MEAN, label='Mean Fitness')

    # Plot min fitness curve
    ax.plot(generations, min_fitness, '-', linewidth=2.5, color=COLOR_BEST, label='Best Fitness (Min)')

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Set labels with larger font
    ax.set_xlabel('Generation', fontsize=22, fontname='Arial', fontweight='bold')
    ax.set_ylabel('Residual Energy (Fitness)', fontsize=22, fontname='Arial', fontweight='bold')

    # Configure 4-sided box (spines on all sides)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # Add grid (major ticks only, simplified)
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6, color='gray', which='major')

    # Add legend
    ax.legend(loc='upper right', fontsize=17, framealpha=0.95, edgecolor='black')

    # Set tick parameters (major only) with padding
    ax.tick_params(axis='both', which='major', labelsize=18, direction='in', length=8, width=1.5, pad=8)

    # Ensure x-axis covers all generations
    if len(generations) > 0:
        ax.set_xlim([generations.min(), generations.max()])

    # Tight layout
    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Figure saved to: {output_path}")

    # Show plot
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def select_json_file():
    """Open a file dialog to select JSON file."""
    if not TKINTER_AVAILABLE:
        print("Error: tkinter is not available. Please specify input file via command line.")
        return None

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    file_path = filedialog.askopenfilename(
        title='Select Fitness Data JSON File',
        filetypes=[
            ('JSON Files', '*.json'),
            ('All Files', '*.*')
        ],
        initialdir=os.getcwd()
    )

    root.destroy()
    return file_path if file_path else None


def main():
    parser = argparse.ArgumentParser(
        description='Plot fitness curve from CMA-ES optimization JSON data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python plot_fitness.py              # Opens file dialog to select JSON
    python plot_fitness.py data.json
    python plot_fitness.py data.json -o fitness_curve.png
    python plot_fitness.py /path/to/data.json -o output.pdf
        '''
    )

    parser.add_argument('input', type=str, nargs='?', default=None,
                        help='Input JSON file path (optional, opens file dialog if not provided)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path (optional, if not provided, plot is displayed)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the plot (only save if -o is specified)')

    args = parser.parse_args()

    # If no input file provided via command line, open file dialog
    if args.input is None:
        print("No input file specified, opening file dialog...")
        args.input = select_json_file()
        if args.input is None:
            print("No file selected. Exiting.")
            return 0
    else:
        args.input = os.path.abspath(args.input)

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Load data
    print(f"Loading data from: {args.input}")
    data = load_data(args.input)

    # Validate required keys
    required_keys = ['gen', 'avg', 'std', 'min']
    for key in required_keys:
        if key not in data:
            print(f"Error: Missing required key '{key}' in JSON data")
            return 1

    # Print data summary
    print(f"Data loaded successfully:")
    print(f"  - Generations: {len(data['gen'])}")
    print(f"  - Population size: {data.get('num', 'N/A')}")
    avg_init = f"{data['avg'][0]:.4f}" if data['avg'] else 'N/A'
    avg_final = f"{data['avg'][-1]:.4f}" if data['avg'] else 'N/A'
    min_best = f"{min(data['min']):.4f}" if data['min'] else 'N/A'
    print(f"  - Initial fitness (avg): {avg_init}")
    print(f"  - Final fitness (avg): {avg_final}")
    print(f"  - Best fitness (min): {min_best}")

    # Generate output path if not provided
    if args.output is None and args.no_show:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_fitness_curve.png"

    # Plot
    show = not args.no_show
    plot_fitness_curve(data, output_path=args.output, show=show)

    return 0


if __name__ == '__main__':
    exit(main())
