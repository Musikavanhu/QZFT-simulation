# Quantum Zeta Field Theory (QZFT) Simulator

A PyTorch-based simulator for exploring the behavior of a quantum-like scalar field on the complex plane, with special focus on the Riemann zeta function's zeros and critical line.

## Overview

This application simulates a quantum-like scalar field ϕ(s) over the complex plane s = σ + it, where:
- σ ∈ [0, 1] (Real part)
- t ∈ [0, 50] (Imaginary part)

The field dynamics are governed by:
- Potential function: V(s) = |ζ(s)|⁻², where ζ(s) is the Riemann zeta function
- Collapse penalty: C(s) = α * |Re(s) - 0.5|²

The field exhibits stability near zeta zeros on the critical line (Re(s) = 0.5) and instability elsewhere.

## Features

- Discrete simulation of the complex domain from Re(s) = 0.4 to 0.6, Im(s) = 0 to 50
- Calculation of ζ(s) using the mpmath library
- Visualization of the 2D field potential heatmap
- Highlighting of ζ(s) ≈ 0 points on the critical line
- Web dashboard for interactive exploration
- Data export as CSV for further analysis

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the simulator with default parameters:

```bash
python QZFT_RiemannCollapseSim.py
```

Customize simulation parameters:

```bash
python QZFT_RiemannCollapseSim.py --re_min 0.3 --re_max 0.7 --im_min 0 --im_max 30 --step 0.05 --alpha 2.0
```

Options:
- `--re_min`: Minimum value for Re(s) (default: 0.4)
- `--re_max`: Maximum value for Re(s) (default: 0.6)
- `--im_min`: Minimum value for Im(s) (default: 0)
- `--im_max`: Maximum value for Im(s) (default: 50)
- `--step`: Step size for discretization (default: 0.1)
- `--alpha`: Weight for collapse penalty function (default: 1.0)
- `--device`: Device to use (cpu or cuda)
- `--save_plot`: Path to save plot (default: qzft_plot.png)
- `--save_data`: Save data files (NPZ and CSV)

### Web Dashboard

Start the web server:

```bash
python QZFT_web_dashboard.py
```

Then open your browser and navigate to:

```
http://localhost:5000
```

The web interface allows you to:
- Adjust simulation parameters
- View visualizations in real-time
- Identify zeta zeros
- Download simulation data as CSV

## Technical Details

The simulation discretizes the complex domain and calculates:
1. |ζ(s)| - The absolute value of the Riemann zeta function
2. V(s) = |ζ(s)|⁻² - The potential function (high near zeros)
3. C(s) = α * |Re(s) - 0.5|² - The collapse penalty (high away from critical line)
4. Total potential - Combined field behavior

PyTorch is used for tensor operations, allowing for GPU acceleration if available.

## Visualization Examples

The application generates four visualization panels:
1. |ζ(s)| - Shows where the zeta function approaches zero (dark regions)
2. V(s) - Potential function (bright spots indicate high potential near zeros)
3. C(s) - Collapse penalty (increases with distance from critical line)
4. Total potential - Combined field behavior

## License

MIT 