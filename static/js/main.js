// ExoFinder Pro - Main JavaScript
// Global variables
let currentSection = 'home';

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('ExoFinder Pro initialized');
    initializeApp();
});

// Initialize the application
function initializeApp() {
    // Set up navigation
    setupNavigation();
    
    // Set up form handlers
    setupFormHandlers();
    
    // Show home section by default
    showSection('home');
}

// Set up navigation event listeners
function setupNavigation() {
    // Navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const section = this.getAttribute('onclick') ? 
                this.getAttribute('onclick').match(/'([^']+)'/)[1] : 
                this.getAttribute('href').substring(1);
            showSection(section);
        });
    });
    
    // Dashboard cards (make them clickable)
    const dashboardCards = document.querySelectorAll('.dashboard-card');
    dashboardCards.forEach((card, index) => {
        card.addEventListener('click', function() {
            const sections = ['home', 'classify', 'data', 'lightcurve', 'habitable'];
            if (index < sections.length) {
                showSection(sections[index]);
            }
        });
    });
}

// Set up form handlers
function setupFormHandlers() {
    // Prediction form
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePrediction);
    }
    
    // Batch prediction form
    const batchPredictionForm = document.getElementById('batchPredictionForm');
    if (batchPredictionForm) {
        batchPredictionForm.addEventListener('submit', handleBatchPrediction);
    }
    
    // Light curve form
    const lightCurveForm = document.getElementById('lightCurveForm');
    if (lightCurveForm) {
        lightCurveForm.addEventListener('submit', handleLightCurveAnalysis);
    }
    
    // Generate light curve form
    const generateLightCurveForm = document.getElementById('generateLightCurveForm');
    if (generateLightCurveForm) {
        generateLightCurveForm.addEventListener('submit', handleLightCurveGeneration);
    }
    
    // Habitability form
    const habitableForm = document.getElementById('habitableForm');
    if (habitableForm) {
        habitableForm.addEventListener('submit', handleHabitabilityCalculation);
    }
}

// Show specific section
function showSection(sectionId) {
    // Update global variable
    currentSection = sectionId;
    
    // Hide all sections
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.classList.remove('active');
    });
    
    // Show requested section
    const sectionToShow = document.getElementById(`${sectionId}-section`);
    if (sectionToShow) {
        sectionToShow.classList.add('active');
    }
    
    // Update navigation active state
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.classList.remove('active');
        const linkSection = link.getAttribute('onclick') ? 
            link.getAttribute('onclick').match(/'([^']+)'/)[1] : 
            link.getAttribute('href').substring(1);
        if (linkSection === sectionId) {
            link.classList.add('active');
        }
    });
    
    // Scroll to top smoothly
    window.scrollTo({ top: 0, behavior: 'smooth' });
    
    console.log(`Switched to section: ${sectionId}`);
}

// Handle single prediction
async function handlePrediction(e) {
    e.preventDefault();
    console.log('Handling single prediction');
    
    const form = e.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);
    
    // Convert string values to numbers
    for (const key in data) {
        data[key] = parseFloat(data[key]);
    }
    
    console.log('Prediction data:', data);
    
    // Show loading state
    const resultDiv = document.getElementById('predictionResult');
    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Analyzing...</span>
                </div>
                <p class="mt-2">Analyzing exoplanet characteristics...</p>
            </div>
        `;
        resultDiv.style.display = 'block';
    }
    
    try {
        // In a real implementation, this would call the backend API
        // For now, we'll simulate a response
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Simulate API response
        const simulatedResult = {
            status: 'success',
            prediction: ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'][Math.floor(Math.random() * 3)],
            confidence: 'High'
        };
        
        displayPredictionResult(simulatedResult);
        
    } catch (error) {
        console.error('Prediction error:', error);
        displayError('Error: ' + error.message);
    }
}

// Handle batch prediction
async function handleBatchPrediction(e) {
    e.preventDefault();
    console.log('Handling batch prediction');
    
    const form = e.target;
    const formData = new FormData(form);
    
    // Show loading state
    const resultDiv = document.getElementById('predictionResult');
    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Processing...</span>
                </div>
                <p class="mt-2">Processing batch analysis...</p>
            </div>
        `;
        resultDiv.style.display = 'block';
    }
    
    try {
        // In a real implementation, this would call the backend API
        // For now, we'll simulate a response
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Simulate API response
        const simulatedResult = {
            status: 'success',
            predictions: [
                { koi_period: 15.2, predicted_disposition: 'CONFIRMED' },
                { koi_period: 5.1, predicted_disposition: 'CANDIDATE' },
                { koi_period: 3.2, predicted_disposition: 'FALSE POSITIVE' },
                { koi_period: 8.7, predicted_disposition: 'CONFIRMED' },
                { koi_period: 12.4, predicted_disposition: 'CANDIDATE' }
            ]
        };
        
        displayBatchPredictionResult(simulatedResult);
        
    } catch (error) {
        console.error('Batch prediction error:', error);
        displayError('Error: ' + error.message);
    }
}

// Handle light curve analysis
async function handleLightCurveAnalysis(e) {
    e.preventDefault();
    console.log('Handling light curve analysis');
    
    const form = e.target;
    const formData = new FormData(form);
    const timeStr = formData.get('time');
    const fluxStr = formData.get('flux');
    
    // Parse the comma-separated values
    const time = timeStr.split(',').map(str => parseFloat(str.trim())).filter(n => !isNaN(n));
    const flux = fluxStr.split(',').map(str => parseFloat(str.trim())).filter(n => !isNaN(n));
    
    if (time.length === 0 || flux.length === 0 || time.length !== flux.length) {
        displayError('Invalid time or flux data. Please ensure both arrays have the same length.');
        return;
    }
    
    const data = {
        time: time,
        flux: flux
    };
    
    console.log('Light curve data:', data);
    
    // Show loading state
    const resultDiv = document.getElementById('lightCurveResult');
    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Analyzing...</span>
                </div>
                <p class="mt-2">Analyzing light curve features...</p>
            </div>
        `;
        resultDiv.style.display = 'block';
    }
    
    try {
        // In a real implementation, this would call the backend API
        // For now, we'll simulate a response
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Simulate API response
        const simulatedResult = {
            status: 'success',
            features: {
                transit_depth: 0.015,
                transit_duration: 2.3,
                periodicity: 0.95,
                shape_analysis: 0.87,
                statistical_significance: 0.92
            }
        };
        
        displayLightCurveAnalysisResult(simulatedResult);
        
    } catch (error) {
        console.error('Light curve analysis error:', error);
        displayError('Error: ' + error.message);
    }
}

// Handle light curve generation
async function handleLightCurveGeneration(e) {
    e.preventDefault();
    console.log('Handling light curve generation');
    
    const form = e.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);
    
    // Convert string values to numbers
    for (const key in data) {
        data[key] = parseFloat(data[key]);
    }
    
    console.log('Generation parameters:', data);
    
    // Show loading state
    const resultDiv = document.getElementById('lightCurveResult');
    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-success" role="status">
                    <span class="visually-hidden">Generating...</span>
                </div>
                <p class="mt-2">Generating synthetic light curve...</p>
            </div>
        `;
        resultDiv.style.display = 'block';
    }
    
    try {
        // In a real implementation, this would call the backend API
        // For now, we'll simulate a response
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Simulate API response
        const simulatedResult = {
            status: 'success',
            time: Array.from({length: 100}, (_, i) => i * 0.5),
            flux: Array.from({length: 100}, (_, i) => 1.0 + 0.01 * Math.sin(i * 0.1) - 0.02 * Math.random())
        };
        
        displayGeneratedLightCurve(simulatedResult);
        
    } catch (error) {
        console.error('Light curve generation error:', error);
        displayError('Error: ' + error.message);
    }
}

// Handle habitability calculation
async function handleHabitabilityCalculation(e) {
    e.preventDefault();
    console.log('Handling habitability calculation');
    
    const form = e.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);
    
    // Convert string values to numbers
    for (const key in data) {
        data[key] = parseFloat(data[key]);
    }
    
    console.log('Habitability parameters:', data);
    
    // Show loading state
    const resultDiv = document.getElementById('habitableResult');
    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-success" role="status">
                    <span class="visually-hidden">Calculating...</span>
                </div>
                <p class="mt-2">Calculating habitability probability...</p>
            </div>
        `;
        resultDiv.style.display = 'block';
    } else {
        // Try the light curve result div as fallback
        const fallbackDiv = document.getElementById('lightCurveResult');
        if (fallbackDiv) {
            fallbackDiv.innerHTML = `
                <div class="text-center">
                    <div class="spinner-border text-success" role="status">
                        <span class="visually-hidden">Calculating...</span>
                    </div>
                    <p class="mt-2">Calculating habitability probability...</p>
                </div>
            `;
            fallbackDiv.style.display = 'block';
        }
    }
    
    try {
        // In a real implementation, this would call the backend API
        // For now, we'll simulate a response
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Simulate API response
        const simulatedResult = {
            status: 'success',
            habitability_probability: Math.random() * 0.8 + 0.1, // 10-90%
            details: {
                distance_score: Math.random(),
                temperature_score: Math.random(),
                size_score: Math.random(),
                inner_hz: 0.95,
                outer_hz: 1.4
            }
        };
        
        displayHabitabilityResult(simulatedResult);
        
    } catch (error) {
        console.error('Habitability calculation error:', error);
        displayError('Error: ' + error.message);
    }
}

// Display prediction result
function displayPredictionResult(result) {
    const resultDiv = document.getElementById('predictionResult');
    if (!resultDiv) return;
    
    if (result.status === 'success') {
        let badgeClass = '';
        let icon = '';
        let bgColor = '';
        
        switch(result.prediction.toLowerCase()) {
            case 'confirmed':
                badgeClass = 'bg-success';
                icon = '<i class="fas fa-check-circle"></i>';
                bgColor = 'rgba(39, 174, 96, 0.2)';
                break;
            case 'candidate':
                badgeClass = 'bg-warning text-dark';
                icon = '<i class="fas fa-search"></i>';
                bgColor = 'rgba(243, 156, 18, 0.2)';
                break;
            case 'false positive':
                badgeClass = 'bg-danger';
                icon = '<i class="fas fa-times-circle"></i>';
                bgColor = 'rgba(231, 76, 60, 0.2)';
                break;
            default:
                badgeClass = 'bg-secondary';
                icon = '<i class="fas fa-question-circle"></i>';
                bgColor = 'rgba(149, 165, 166, 0.2)';
        }
        
        resultDiv.innerHTML = `
            <div class="alert alert-success" style="background: ${bgColor}; border: 1px solid rgba(255,255,255,0.3);">
                <div class="d-flex align-items-center">
                    <div class="me-3" style="font-size: 2em;">${icon}</div>
                    <div>
                        <h4>Classification Result</h4>
                        <div class="d-flex align-items-center gap-3">
                            <span class="badge ${badgeClass} fs-5 px-4 py-2">
                                ${result.prediction.toUpperCase()}
                            </span>
                            ${result.confidence && result.confidence !== 'N/A' ? 
                              `<span class="text-muted">Confidence: ${(parseFloat(result.confidence) * 100).toFixed(1)}%</span>` : 
                              '<span class="text-muted">Confidence: High</span>'}
                        </div>
                    </div>
                </div>
            </div>
        `;
    } else {
        resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${result.message}</div>`;
    }
    
    resultDiv.style.display = 'block';
}

// Display batch prediction result
function displayBatchPredictionResult(result) {
    const resultDiv = document.getElementById('predictionResult');
    if (!resultDiv) return;
    
    if (result.status === 'success') {
        // Count predictions by class
        const classCounts = {};
        result.predictions.forEach(pred => {
            const disposition = pred.predicted_disposition || pred.koi_disposition;
            if (disposition) {
                classCounts[disposition] = (classCounts[disposition] || 0) + 1;
            }
        });
        
        resultDiv.innerHTML = `
            <div class="alert alert-success">
                <h4><i class="fas fa-file-alt me-2"></i>Batch Analysis Results</h4>
                <p>Processed ${result.predictions.length} exoplanet candidates</p>
                
                <div class="row mt-3">
                    <div class="col-md-4">
                        <div class="card bg-success text-white">
                            <div class="card-body text-center">
                                <h3>${classCounts['CONFIRMED'] || 0}</h3>
                                <p>Confirmed Planets</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-warning text-dark">
                            <div class="card-body text-center">
                                <h3>${classCounts['CANDIDATE'] || 0}</h3>
                                <p>Candidates</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-danger text-white">
                            <div class="card-body text-center">
                                <h3>${classCounts['FALSE POSITIVE'] || 0}</h3>
                                <p>False Positives</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-3">
                    <button class="btn btn-primary" onclick="downloadBatchResults()">
                        <i class="fas fa-download me-2"></i>Download Results (CSV)
                    </button>
                </div>
            </div>
        `;
    } else {
        resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${result.message}</div>`;
    }
    
    resultDiv.style.display = 'block';
}

// Display light curve analysis result
function displayLightCurveAnalysisResult(result) {
    const resultDiv = document.getElementById('lightCurveResult');
    if (!resultDiv) return;
    
    if (result.status === 'success') {
        let featureHtml = '';
        let count = 0;
        
        for (const [key, value] of Object.entries(result.features)) {
            if (count >= 6) break; // Limit to first 6 features
            
            featureHtml += `
                <div class="col-md-6 mb-3">
                    <div class="card bg-light text-dark">
                        <div class="card-body">
                            <h6 class="mb-1">${key.replace(/_/g, ' ')}</h6>
                            <p class="mb-0">${Number(value).toFixed(6)}</p>
                        </div>
                    </div>
                </div>
            `;
            
            count++;
        }
        
        resultDiv.innerHTML = `
            <div class="alert alert-success">
                <h4><i class="fas fa-wave-square me-2"></i>Light Curve Analysis Results</h4>
                <p>Extracted ${Object.keys(result.features).length} features from the light curve</p>
                
                <div class="row mt-3">
                    ${featureHtml}
                </div>
                
                <div class="mt-3">
                    <button class="btn btn-primary" onclick="plotLightCurve()">
                        <i class="fas fa-chart-line me-2"></i>Plot Light Curve
                    </button>
                </div>
            </div>
        `;
    } else {
        resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${result.message}</div>`;
    }
    
    resultDiv.style.display = 'block';
}

// Display generated light curve
function displayGeneratedLightCurve(result) {
    const resultDiv = document.getElementById('lightCurveResult');
    if (!resultDiv) return;
    
    if (result.status === 'success') {
        resultDiv.innerHTML = `
            <div class="alert alert-success">
                <h4><i class="fas fa-plus me-2"></i>Synthetic Light Curve Generated</h4>
                <p>Created light curve with ${result.time.length} data points</p>
                <p><strong>Parameters:</strong> Period=5.0 days, Duration=0.3 days, Depth=0.02</p>
                
                <div class="mt-3">
                    <button class="btn btn-primary" onclick="plotGeneratedLightCurve(${JSON.stringify(result.time)}, ${JSON.stringify(result.flux)})">
                        <i class="fas fa-chart-line me-2"></i>Plot Light Curve
                    </button>
                </div>
            </div>
        `;
    } else {
        resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${result.message}</div>`;
    }
    
    resultDiv.style.display = 'block';
}

// Display habitability result
function displayHabitabilityResult(result) {
    const resultDiv = document.getElementById('habitableResult') || document.getElementById('lightCurveResult');
    if (!resultDiv) return;
    
    if (result.status === 'success') {
        const percentage = (result.habitability_probability * 100).toFixed(2);
        const progressBarWidth = Math.min(percentage, 100);
        
        let alertClass = 'alert-success';
        let bgColor = 'rgba(39, 174, 96, 0.2)';
        
        if (percentage < 30) {
            alertClass = 'alert-warning';
            bgColor = 'rgba(243, 156, 18, 0.2)';
        } else if (percentage > 70) {
            alertClass = 'alert-success';
            bgColor = 'rgba(39, 174, 96, 0.2)';
        }
        
        resultDiv.innerHTML = `
            <div class="alert ${alertClass}" style="background: ${bgColor}; border: 1px solid rgba(255,255,255,0.3);">
                <h4><i class="fas fa-seedling me-2"></i>Habitability Analysis</h4>
                <p><strong>Habitability Probability:</strong> ${percentage}%</p>
                
                <div class="progress mt-3" style="height: 25px;">
                    <div class="progress-bar ${percentage < 30 ? 'bg-warning' : 'bg-success'}" 
                         style="width: ${progressBarWidth}%;">
                        ${percentage}%
                    </div>
                </div>
                
                <div class="row mt-3">
                    <div class="col-md-6">
                        <p><strong>Distance Score:</strong> ${(result.details.distance_score * 100).toFixed(1)}%</p>
                        <p><strong>Temperature Score:</strong> ${(result.details.temperature_score * 100).toFixed(1)}%</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Size Score:</strong> ${(result.details.size_score * 100).toFixed(1)}%</p>
                        <p><strong>Habitable Zone:</strong> ${result.details.inner_hz.toFixed(2)} - ${result.details.outer_hz.toFixed(2)} AU</p>
                    </div>
                </div>
            </div>
        `;
    } else {
        resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${result.message}</div>`;
    }
    
    resultDiv.style.display = 'block';
}

// Display error message
function displayError(message) {
    const resultDivs = [
        document.getElementById('predictionResult'),
        document.getElementById('lightCurveResult'),
        document.getElementById('habitableResult')
    ];
    
    resultDivs.forEach(div => {
        if (div) {
            div.innerHTML = `<div class="alert alert-danger">Error: ${message}</div>`;
            div.style.display = 'block';
        }
    });
}

// Download batch results as CSV
function downloadBatchResults() {
    // In a real implementation, this would download the actual results
    alert('In a real implementation, this would download the batch results as a CSV file.');
}

// Plot light curve (placeholder)
function plotLightCurve() {
    // Create a modal to display the plot
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'lightCurveModal';
    modal.tabIndex = -1;
    modal.innerHTML = `
        <div class="modal-dialog modal-xl">
            <div class="modal-content" style="background: rgba(0,0,0,0.9); color: white;">
                <div class="modal-header">
                    <h5 class="modal-title">Light Curve Analysis</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <canvas id="lightCurveChart" width="800" height="400"></canvas>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    
    // Show the modal
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
    
    // Wait for modal to be shown, then create the chart
    modal.addEventListener('shown.bs.modal', function() {
        const ctx = document.getElementById('lightCurveChart').getContext('2d');
        
        // Generate sample data for visualization (this would be replaced with real data in a full implementation)
        const time = Array.from({length: 100}, (_, i) => i * 0.5);
        const flux = Array.from({length: 100}, (_, i) => {
            const t = time[i];
            // Create a transit-like event
            const transitDepth = 0.02;
            const transitDuration = 0.5;
            const transitPhase = (t % 5.0) - 2.5; // Transit every 5 days, centered at 0
            const transit = Math.abs(transitPhase) < transitDuration/2 ? -transitDepth : 0;
            return 1.0 + transit + 0.001 * Math.random(); // Add some noise
        });
        
        // Create the chart
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: time,
                datasets: [{
                    label: 'Normalized Flux',
                    data: flux,
                    borderColor: 'rgba(52, 152, 219, 1)',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (days)',
                            color: 'white'
                        },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)'
                        },
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Normalized Flux',
                            color: 'white'
                        },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)'
                        },
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: 'rgba(255,255,255,0.2)',
                        borderWidth: 1
                    }
                }
            }
        });
    });
    
    // Remove modal from DOM when hidden
    modal.addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(modal);
    });
}

// Plot generated light curve (with actual data)
function plotGeneratedLightCurve(time, flux) {
    // Create a modal to display the plot
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'lightCurveModal';
    modal.tabIndex = -1;
    modal.innerHTML = `
        <div class="modal-dialog modal-xl">
            <div class="modal-content" style="background: rgba(0,0,0,0.9); color: white;">
                <div class="modal-header">
                    <h5 class="modal-title">Generated Light Curve</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <canvas id="lightCurveChart" width="800" height="400"></canvas>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    
    // Show the modal
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
    
    // Wait for modal to be shown, then create the chart
    modal.addEventListener('shown.bs.modal', function() {
        const ctx = document.getElementById('lightCurveChart').getContext('2d');
        
        // Create the chart
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: time,
                datasets: [{
                    label: 'Normalized Flux',
                    data: flux,
                    borderColor: 'rgba(46, 204, 113, 1)',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (days)',
                            color: 'white'
                        },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)'
                        },
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Normalized Flux',
                            color: 'white'
                        },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)'
                        },
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: 'rgba(255,255,255,0.2)',
                        borderWidth: 1
                    }
                }
            }
        });
    });
    
    // Remove modal from DOM when hidden
    modal.addEventListener('hidden.bs.modal', function() {
        document.body.removeChild(modal);
    });
}

// Update dashboard metrics
function updateDashboard() {
    // This would normally fetch live data from the backend
    // For now, we'll just simulate updates
    
    const accuracyDisplay = document.getElementById('accuracy-display');
    if (accuracyDisplay) {
        // Simulate slight variations in accuracy
        const current = parseFloat(accuracyDisplay.textContent);
        const variation = (Math.random() * 0.2) - 0.1; // -0.1 to +0.1 variation
        const newValue = Math.max(70, Math.min(80, current + variation));
        accuracyDisplay.textContent = newValue.toFixed(1) + '%';
    }
}

// Periodically update dashboard metrics
setInterval(updateDashboard, 30000); // Update every 30 seconds

// Global functions for inline onclick handlers
window.showSection = showSection;
window.downloadBatchResults = downloadBatchResults;
window.plotLightCurve = plotLightCurve;
window.plotGeneratedLightCurve = plotGeneratedLightCurve;

console.log('ExoFinder Pro JavaScript loaded successfully');