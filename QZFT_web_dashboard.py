#!/usr/bin/env python3
"""
QZFT Web Dashboard - Flask interface for the Quantum Zeta Field Theory simulator
"""

import os
import base64
import io
from flask import Flask, render_template, request, send_file, jsonify
import matplotlib.pyplot as plt
import numpy as np
from QZFT_RiemannCollapseSim import QZFTRiemannCollapseSim

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Run the QZFT simulation with parameters from the form"""
    # Get parameters from the form
    re_min = float(request.form.get('re_min', 0.4))
    re_max = float(request.form.get('re_max', 0.6))
    im_min = float(request.form.get('im_min', 0))
    im_max = float(request.form.get('im_max', 50))
    step_size = float(request.form.get('step_size', 0.1))
    alpha = float(request.form.get('alpha', 1.0))
    
    # Initialize simulator
    simulator = QZFTRiemannCollapseSim(
        re_min=re_min,
        re_max=re_max,
        im_min=im_min,
        im_max=im_max,
        step_size=step_size,
        alpha=alpha,
        device='cpu'  # Use CPU for web service
    )
    
    # Run simulation
    simulator.run_simulation()
    
    # Generate plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: |ζ(s)|
    im = axes[0, 0].imshow(simulator.zeta_abs, 
                          extent=[re_min, re_max, im_min, im_max], 
                          aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title(r'$|\zeta(s)|$')
    axes[0, 0].set_xlabel(r'Re(s) $\sigma$')
    axes[0, 0].set_ylabel(r'Im(s) $t$')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Plot 2: V(s) = |ζ(s)|⁻²
    from matplotlib.colors import LogNorm
    im = axes[0, 1].imshow(simulator.potential_V, 
                          extent=[re_min, re_max, im_min, im_max], 
                          aspect='auto', origin='lower', cmap='plasma', 
                          norm=LogNorm())
    axes[0, 1].set_title(r'Potential $V(s) = |\zeta(s)|^{-2}$')
    axes[0, 1].set_xlabel(r'Re(s) $\sigma$')
    axes[0, 1].set_ylabel(r'Im(s) $t$')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Plot 3: C(s) = α * |Re(s) - 0.5|²
    im = axes[1, 0].imshow(simulator.collapse_C, 
                          extent=[re_min, re_max, im_min, im_max], 
                          aspect='auto', origin='lower', cmap='inferno')
    axes[1, 0].set_title(r'Collapse Penalty $C(s) = \alpha |Re(s) - 0.5|^2$')
    axes[1, 0].set_xlabel(r'Re(s) $\sigma$')
    axes[1, 0].set_ylabel(r'Im(s) $t$')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 4: Total Potential
    im = axes[1, 1].imshow(simulator.total_potential, 
                          extent=[re_min, re_max, im_min, im_max], 
                          aspect='auto', origin='lower', cmap='magma', 
                          norm=LogNorm())
    axes[1, 1].set_title(r'Total Potential $V(s) + C(s)$')
    axes[1, 1].set_xlabel(r'Re(s) $\sigma$')
    axes[1, 1].set_ylabel(r'Im(s) $t$')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Highlight critical line
    for ax in axes.flat:
        ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.5, label='Critical Line')
    
    # Highlight zeta zeros
    zeros = simulator.find_zeta_zeros()
    if zeros:
        for s, zeta_val in zeros:
            for ax in axes.flat:
                ax.plot(s.real, s.imag, 'o', color='red', markersize=5)
    
    plt.tight_layout()
    
    # Save plot to a buffer
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    plt.close(fig)
    
    # Encode the image as base64
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    
    # Create CSV data for download
    csv_data = []
    for i in range(simulator.shape[0]):
        for j in range(simulator.shape[1]):
            csv_data.append([
                simulator.sigma_grid[i, j],
                simulator.t_grid[i, j],
                simulator.zeta_abs[i, j],
                simulator.potential_V[i, j],
                simulator.collapse_C[i, j],
                simulator.total_potential[i, j]
            ])
    
    csv_buffer = io.StringIO()
    np.savetxt(csv_buffer, csv_data, delimiter=',', 
              header='sigma,t,zeta_abs,potential_V,collapse_C,total_potential',
              comments='')
    csv_string = csv_buffer.getvalue()
    
    # Get information about zeros
    zeros_info = []
    if zeros:
        for s, zeta_val in zeros:
            zeros_info.append({
                'real': float(s.real),
                'imag': float(s.imag),
                'zeta_abs': float(zeta_val)
            })
    
    return jsonify({
        'plot_image': img_base64,
        'zeros': zeros_info,
        'csv_data': csv_string,
        'parameters': {
            're_min': re_min,
            're_max': re_max,
            'im_min': im_min,
            'im_max': im_max,
            'step_size': step_size,
            'alpha': alpha
        }
    })

@app.route('/download_csv', methods=['POST'])
def download_csv():
    """Download the simulation data as a CSV file"""
    csv_data = request.form.get('csv_data', '')
    
    # Create a file-like object from the CSV string
    csv_buffer = io.StringIO(csv_data)
    
    # Create a response with the CSV file
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='qzft_data.csv'
    )

# Create templates directory and HTML template
@app.before_first_request
def create_templates():
    """Create templates directory and HTML files if they don't exist"""
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Zeta Field Theory Explorer</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <style>
        body {
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .title {
            margin-bottom: 30px;
            color: #343a40;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .plot-img {
            width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .zero-point {
            margin-bottom: 5px;
        }
        .parameter-label {
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title text-center">Quantum Zeta Field Theory Explorer</h1>
        
        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0">Simulation Parameters</h4>
                    </div>
                    <div class="card-body">
                        <form id="simulation-form">
                            <div class="mb-3">
                                <label for="re_min" class="form-label">Re(s) Min:</label>
                                <input type="number" step="0.1" class="form-control" id="re_min" name="re_min" value="0.4">
                            </div>
                            <div class="mb-3">
                                <label for="re_max" class="form-label">Re(s) Max:</label>
                                <input type="number" step="0.1" class="form-control" id="re_max" name="re_max" value="0.6">
                            </div>
                            <div class="mb-3">
                                <label for="im_min" class="form-label">Im(s) Min:</label>
                                <input type="number" step="1" class="form-control" id="im_min" name="im_min" value="0">
                            </div>
                            <div class="mb-3">
                                <label for="im_max" class="form-label">Im(s) Max:</label>
                                <input type="number" step="1" class="form-control" id="im_max" name="im_max" value="50">
                            </div>
                            <div class="mb-3">
                                <label for="step_size" class="form-label">Step Size:</label>
                                <input type="number" step="0.01" class="form-control" id="step_size" name="step_size" value="0.1">
                            </div>
                            <div class="mb-3">
                                <label for="alpha" class="form-label">Collapse Penalty Weight (α):</label>
                                <input type="number" step="0.1" class="form-control" id="alpha" name="alpha" value="1.0">
                            </div>
                            <button type="submit" class="btn btn-primary w-100">Run Simulation</button>
                        </form>
                        
                        <div class="loading mt-3">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p class="mt-2">Running simulation, please wait...</p>
                        </div>
                        
                        <form id="download-form" class="mt-3" style="display: none;">
                            <input type="hidden" id="csv_data" name="csv_data">
                            <button type="submit" class="btn btn-success w-100">Download CSV Data</button>
                        </form>
                    </div>
                </div>
                
                <div class="card" id="zeros-card" style="display: none;">
                    <div class="card-header bg-danger text-white">
                        <h4 class="mb-0">Detected Zeta Zeros</h4>
                    </div>
                    <div class="card-body">
                        <div id="zeros-list"></div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header bg-dark text-white">
                        <h4 class="mb-0">Simulation Results</h4>
                    </div>
                    <div class="card-body text-center">
                        <p class="text-muted" id="initial-message">Run a simulation to see results</p>
                        <img id="plot-image" class="plot-img" style="display: none;" alt="QZFT Simulation Plot">
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#simulation-form').on('submit', function(e) {
                e.preventDefault();
                
                // Show loading spinner
                $('.loading').show();
                $('#plot-image').hide();
                $('#initial-message').hide();
                $('#download-form').hide();
                $('#zeros-card').hide();
                
                // Collect form data
                const formData = new FormData(this);
                
                // Submit AJAX request
                $.ajax({
                    url: '/run_simulation',
                    type: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: function(response) {
                        // Hide loading spinner
                        $('.loading').hide();
                        
                        // Update plot image
                        $('#plot-image').attr('src', 'data:image/png;base64,' + response.plot_image);
                        $('#plot-image').show();
                        
                        // Update CSV data for download
                        $('#csv_data').val(response.csv_data);
                        $('#download-form').show();
                        
                        // Update zeros list
                        if (response.zeros && response.zeros.length > 0) {
                            let zerosHtml = '';
                            response.zeros.forEach(function(zero, index) {
                                zerosHtml += `
                                    <div class="zero-point">
                                        <div><span class="parameter-label">Zero ${index+1}:</span> s = ${zero.real.toFixed(5)} + ${zero.imag.toFixed(5)}i</div>
                                        <div><span class="parameter-label">|ζ(s)|:</span> ${zero.zeta_abs.toExponential(5)}</div>
                                    </div>
                                    <hr>
                                `;
                            });
                            $('#zeros-list').html(zerosHtml);
                            $('#zeros-card').show();
                        }
                    },
                    error: function() {
                        $('.loading').hide();
                        alert('Error running simulation. Please try again with different parameters.');
                    }
                });
            });
            
            $('#download-form').on('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                
                // Create a temporary form to submit the data
                const tempForm = document.createElement('form');
                tempForm.method = 'POST';
                tempForm.action = '/download_csv';
                
                const input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'csv_data';
                input.value = $('#csv_data').val();
                
                tempForm.appendChild(input);
                document.body.appendChild(tempForm);
                tempForm.submit();
                document.body.removeChild(tempForm);
            });
        });
    </script>
</body>
</html>
    """
    
    # Save the template
    with open('templates/index.html', 'w') as f:
        f.write(index_html)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 