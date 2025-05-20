#!/usr/bin/env python3
"""
Quantum Zeta Field Theory (QZFT) Riemann Collapse Simulator
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mpmath
import argparse
from tqdm import tqdm

class QZFTRiemannCollapseSim:
    def __init__(self, re_min=0.4, re_max=0.6, im_min=0, im_max=50, 
                 step_size=0.01, alpha=1.0, device=None):
        """
        Initialize the QZFT Riemann Collapse Simulator
        
        Parameters:
        -----------
        re_min, re_max: float
            Range of real part of complex plane (σ)
        im_min, im_max: float
            Range of imaginary part of complex plane (t)
        step_size: float
            Step size for discretization
        alpha: float
            Weight for collapse penalty function C(s) = α * |Re(s) - 0.5|²
        device: str
            PyTorch device ('cpu' or 'cuda')
        """
        self.re_min = re_min
        self.re_max = re_max
        self.im_min = im_min
        self.im_max = im_max
        self.step_size = step_size
        self.alpha = alpha
        
        # Set precision for mpmath
        mpmath.mp.dps = 25
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Create grid for complex plane
        self.create_grid()
        
    def create_grid(self):
        """Create a grid of complex numbers in the s-plane"""
        re_range = np.arange(self.re_min, self.re_max + self.step_size, self.step_size)
        im_range = np.arange(self.im_min, self.im_max + self.step_size, self.step_size)
        
        self.sigma_grid, self.t_grid = np.meshgrid(re_range, im_range)
        self.shape = self.sigma_grid.shape
        
        # Create complex grid
        self.s_grid = self.sigma_grid + 1j * self.t_grid
        
        # Initialize tensors to store results
        self.zeta_abs = np.zeros(self.shape)
        self.potential_V = np.zeros(self.shape)
        self.collapse_C = np.zeros(self.shape)
        self.total_potential = np.zeros(self.shape)
        
    def calculate_zeta(self):
        """Calculate the Riemann zeta function for the entire grid"""
        print("Calculating zeta function values...")
        for i in tqdm(range(self.shape[0])):
            for j in range(self.shape[1]):
                s = self.s_grid[i, j]
                # Calculate zeta using mpmath
                zeta_s = mpmath.zeta(complex(s.real, s.imag))
                self.zeta_abs[i, j] = abs(zeta_s)
        
        # Replace extremely small values to avoid division by zero
        self.zeta_abs = np.maximum(self.zeta_abs, 1e-15)
        
    def calculate_potentials(self):
        """Calculate the potential functions"""
        print("Calculating potential functions...")
        
        # V(s) = |ζ(s)|⁻²
        self.potential_V = 1.0 / (self.zeta_abs ** 2)
        
        # C(s) = α * |Re(s) - 0.5|²
        self.collapse_C = self.alpha * np.abs(self.sigma_grid - 0.5) ** 2
        
        # Total potential is the sum
        self.total_potential = self.potential_V + self.collapse_C
        
        # Convert to PyTorch tensors
        self.potential_V_tensor = torch.from_numpy(self.potential_V).float().to(self.device)
        self.collapse_C_tensor = torch.from_numpy(self.collapse_C).float().to(self.device)
        self.total_potential_tensor = torch.from_numpy(self.total_potential).float().to(self.device)
        
    def find_zeta_zeros(self, threshold=0.1):
        """Find approximate zeros of zeta function"""
        zeros_idx = np.where(self.zeta_abs < threshold)
        zeros = []
        
        for i, j in zip(zeros_idx[0], zeros_idx[1]):
            zeros.append((self.s_grid[i, j], self.zeta_abs[i, j]))
            
        return zeros
        
    def run_simulation(self):
        """Run the full simulation"""
        self.calculate_zeta()
        self.calculate_potentials()
        
    def plot_results(self, save_path=None):
        """Visualize the results"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        # Plot 1: |ζ(s)|
        im = axes[0, 0].imshow(self.zeta_abs, extent=[self.re_min, self.re_max, self.im_min, self.im_max], 
                              aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title(r'$|\zeta(s)|$')
        axes[0, 0].set_xlabel(r'Re(s) $\sigma$')
        axes[0, 0].set_ylabel(r'Im(s) $t$')
        plt.colorbar(im, ax=axes[0, 0])
        
        # Plot 2: V(s) = |ζ(s)|⁻² (using log scale)
        im = axes[0, 1].imshow(self.potential_V, extent=[self.re_min, self.re_max, self.im_min, self.im_max], 
                              aspect='auto', origin='lower', cmap='plasma', norm=LogNorm())
        axes[0, 1].set_title(r'Potential $V(s) = |\zeta(s)|^{-2}$')
        axes[0, 1].set_xlabel(r'Re(s) $\sigma$')
        axes[0, 1].set_ylabel(r'Im(s) $t$')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Plot 3: C(s) = α * |Re(s) - 0.5|²
        im = axes[1, 0].imshow(self.collapse_C, extent=[self.re_min, self.re_max, self.im_min, self.im_max], 
                              aspect='auto', origin='lower', cmap='inferno')
        axes[1, 0].set_title(r'Collapse Penalty $C(s) = \alpha |Re(s) - 0.5|^2$')
        axes[1, 0].set_xlabel(r'Re(s) $\sigma$')
        axes[1, 0].set_ylabel(r'Im(s) $t$')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Total Potential (using log scale for better visualization)
        im = axes[1, 1].imshow(self.total_potential, extent=[self.re_min, self.re_max, self.im_min, self.im_max], 
                              aspect='auto', origin='lower', cmap='magma', norm=LogNorm())
        axes[1, 1].set_title(r'Total Potential $V(s) + C(s)$')
        axes[1, 1].set_xlabel(r'Re(s) $\sigma$')
        axes[1, 1].set_ylabel(r'Im(s) $t$')
        plt.colorbar(im, ax=axes[1, 1])
        
        # Highlight critical line
        for ax in axes.flat:
            ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.5, label='Critical Line')
            
        # Highlight zeta zeros
        zeros = self.find_zeta_zeros()
        if zeros:
            for s, zeta_val in zeros:
                for ax in axes.flat:
                    ax.plot(s.real, s.imag, 'o', color='red', markersize=5)
                    
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
            
        plt.show()
        
    def save_data(self, filename='qzft_data.npz'):
        """Save simulation data to file"""
        np.savez(filename, 
                sigma_grid=self.sigma_grid,
                t_grid=self.t_grid,
                zeta_abs=self.zeta_abs,
                potential_V=self.potential_V,
                collapse_C=self.collapse_C,
                total_potential=self.total_potential)
        print(f"Data saved to {filename}")
        
    def save_csv(self, filename='qzft_data.csv'):
        """Save data as CSV for further analysis"""
        data = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                data.append([
                    self.sigma_grid[i, j],
                    self.t_grid[i, j],
                    self.zeta_abs[i, j],
                    self.potential_V[i, j],
                    self.collapse_C[i, j],
                    self.total_potential[i, j]
                ])
        
        np.savetxt(filename, data, delimiter=',', 
                  header='sigma,t,zeta_abs,potential_V,collapse_C,total_potential',
                  comments='')
        print(f"CSV data saved to {filename}")


def main():
    """Main function to run the simulation from command line"""
    parser = argparse.ArgumentParser(description='Quantum Zeta Field Theory Riemann Collapse Simulator')
    parser.add_argument('--re_min', type=float, default=0.4, help='Minimum value for Re(s)')
    parser.add_argument('--re_max', type=float, default=0.6, help='Maximum value for Re(s)')
    parser.add_argument('--im_min', type=float, default=0, help='Minimum value for Im(s)')
    parser.add_argument('--im_max', type=float, default=50, help='Maximum value for Im(s)')
    parser.add_argument('--step', type=float, default=0.1, help='Step size for discretization')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for collapse penalty function')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu or cuda)')
    parser.add_argument('--save_plot', type=str, default='qzft_plot.png', help='Path to save plot')
    parser.add_argument('--save_data', action='store_true', help='Save data files')
    
    args = parser.parse_args()
    
    # Initialize and run simulation
    sim = QZFTRiemannCollapseSim(
        re_min=args.re_min,
        re_max=args.re_max,
        im_min=args.im_min,
        im_max=args.im_max,
        step_size=args.step,
        alpha=args.alpha,
        device=args.device
    )
    
    sim.run_simulation()
    sim.plot_results(save_path=args.save_plot)
    
    if args.save_data:
        sim.save_data()
        sim.save_csv()


if __name__ == '__main__':
    main() 