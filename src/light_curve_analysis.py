import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import io
import base64

def generate_synthetic_light_curve(period=10.0, duration=0.2, depth=0.01, snr=10.0, t_max=50.0, noise_level=0.001):
    """
    Generate a synthetic light curve with a transit signal
    
    Parameters:
    - period: orbital period in days
    - duration: transit duration in days
    - depth: transit depth (fraction of light blocked)
    - snr: signal-to-noise ratio
    - t_max: total observation time in days
    - noise_level: level of noise to add
    """
    # Create time array
    t = np.linspace(0, t_max, int(t_max * 24 * 2))  # 30-minute cadence
    
    # Initialize flux as 1.0 (normalized)
    flux = np.ones_like(t)
    
    # Add multiple transits based on period
    transit_times = np.arange(0, t_max, period)
    
    for transit_time in transit_times:
        # Create a transit shape (trapezoidal model for simplicity)
        in_transit = (t >= transit_time - duration/2) & (t <= transit_time + duration/2)
        
        if np.any(in_transit):
            # Create trapezoidal transit shape
            local_t = t[in_transit] - transit_time
            transit_shape = np.ones_like(local_t)
            
            # Simple trapezoidal shape
            ingress_egress = duration * 0.2  # ingress/egress duration
            
            # Ingress
            ingress_mask = (local_t >= -duration/2) & (local_t < -duration/2 + ingress_egress)
            transit_shape[ingress_mask] = 1 - depth + depth * (local_t[ingress_mask] + duration/2) / ingress_egress
            
            # Egress
            egress_mask = (local_t > duration/2 - ingress_egress) & (local_t <= duration/2)
            transit_shape[egress_mask] = 1 - depth + depth * (duration/2 - local_t[egress_mask]) / ingress_egress
            
            # Full transit (flat bottom)
            full_transit_mask = (local_t >= -duration/2 + ingress_egress) & (local_t <= duration/2 - ingress_egress)
            transit_shape[full_transit_mask] = 1 - depth
            
            flux[in_transit] = transit_shape
    
    # Add noise
    noise = np.random.normal(0, noise_level, size=len(t))
    flux += noise
    
    return t, flux

def detect_transits_in_light_curve(time, flux, period_range=(1, 30), n_periods=1000):
    """
    Detect potential transits using Box Least Squares (BLS) method
    """
    from astropy.timeseries import BoxLeastSquares
    
    # Create a BLS periodogram
    model = BoxLeastSquares(time, flux)
    
    # Define the period range to search
    periods = np.linspace(period_range[0], period_range[1], n_periods)
    
    # Calculate the BLS periodogram
    results = model.autopower(periods, duration=0.1)  # Fixed duration for initial search
    
    # Find the best period
    best_period_idx = np.argmax(results.power)
    best_period = results.period[best_period_idx]
    
    # Get the best fit
    best_fit = model.model(time, best_period, 0.1, results.transit_time[best_period_idx])
    
    return {
        'periods': results.period,
        'power': results.power,
        'best_period': best_period,
        'best_power': results.power[best_period_idx],
        'best_fit': best_fit,
        'transit_time': results.transit_time[best_period_idx]
    }

def plot_light_curve(time, flux, title="Exoplanet Light Curve", save_path=None):
    """
    Create a visualization of the light curve
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, flux, 'b-', alpha=0.7)
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized Flux')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        # Convert plot to base64 string for web display
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        return img_str

def analyze_light_curve_features(time, flux):
    """
    Extract key features from the light curve for classification
    """
    features = {}
    
    # Convert to numpy arrays if they aren't already
    time = np.asarray(time)
    flux = np.asarray(flux)
    
    # Basic statistics
    features['mean_flux'] = float(np.mean(flux)) if len(flux) > 0 else 0.0
    features['std_flux'] = float(np.std(flux)) if len(flux) > 0 else 0.0
    features['median_flux'] = float(np.median(flux)) if len(flux) > 0 else 0.0
    features['min_flux'] = float(np.min(flux)) if len(flux) > 0 else 0.0
    features['max_flux'] = float(np.max(flux)) if len(flux) > 0 else 0.0
    features['flux_range'] = float(features['max_flux'] - features['min_flux']) if len(flux) > 0 else 0.0
    
    # Variability metrics
    features['variance'] = float(np.var(flux)) if len(flux) > 0 else 0.0
    
    # Handle skewness and kurtosis with error handling
    try:
        features['skewness'] = float(pd.Series(flux).skew()) if len(flux) > 0 else 0.0
    except:
        features['skewness'] = 0.0
    
    try:
        features['kurtosis'] = float(pd.Series(flux).kurtosis()) if len(flux) > 0 else 3.0
    except:
        features['kurtosis'] = 3.0
    
    # Periodicity analysis using periodogram
    try:
        if len(time) > 10:  # Need sufficient data points
            # Calculate sampling interval
            dt = np.median(np.diff(time)) if len(time) > 1 else 1.0
            
            # Simple periodogram (would be better with Lomb-Scargle for irregular sampling)
            freqs = np.fft.fftfreq(len(time), dt)
            fft_power = np.abs(np.fft.fft(flux))**2
            positive_freqs = freqs > 0
            
            if np.sum(positive_freqs) > 0:
                dominant_freq_idx = np.argmax(fft_power[positive_freqs])
                features['dominant_period'] = float(1.0 / freqs[positive_freqs][dominant_freq_idx]) if freqs[positive_freqs][dominant_freq_idx] != 0 else 0.0
            else:
                features['dominant_period'] = 0.0
        else:
            features['dominant_period'] = 0.0
    except Exception as e:
        features['dominant_period'] = 0.0  # Fallback if FFT fails
    
    # Transit-specific metrics
    try:
        # Calculate rolling statistics to detect possible transits
        if len(flux) > 50:  # Need sufficient data points
            flux_series = pd.Series(flux)
            rolling_window = min(50, len(flux) // 2)  # Adjust window size based on data length
            rolling_min = flux_series.rolling(window=rolling_window, center=True).min()
            min_diff = flux_series - rolling_min
            
            # Find significant dips (potential transits)
            significant_dips = min_diff[min_diff > 0]
            if len(significant_dips) > 0:
                features['max_flux_drop'] = float(significant_dips.max())
                features['avg_flux_drop'] = float(significant_dips.mean())
            else:
                features['max_flux_drop'] = 0.0
                features['avg_flux_drop'] = 0.0
        else:
            features['max_flux_drop'] = 0.0
            features['avg_flux_drop'] = 0.0
    except Exception as e:
        features['max_flux_drop'] = 0.0
        features['avg_flux_drop'] = 0.0
    
    # Additional transit metrics
    try:
        # Find local minima (potential transit bottoms)
        from scipy.signal import find_peaks
        
        # Invert the flux to find minima as peaks
        inverted_flux = -flux
        peaks, _ = find_peaks(inverted_flux, height=-np.mean(flux) + 0.1*np.std(flux))
        
        if len(peaks) > 0:
            # Calculate transit depths
            transit_depths = np.mean(flux) - flux[peaks]
            features['mean_transit_depth'] = float(np.mean(transit_depths)) if len(transit_depths) > 0 else 0.0
            features['max_transit_depth'] = float(np.max(transit_depths)) if len(transit_depths) > 0 else 0.0
            
            # Calculate transit durations
            if len(peaks) > 1:
                transit_intervals = np.diff(time[peaks])
                features['mean_transit_interval'] = float(np.mean(transit_intervals)) if len(transit_intervals) > 0 else 0.0
            else:
                features['mean_transit_interval'] = 0.0
        else:
            features['mean_transit_depth'] = 0.0
            features['max_transit_depth'] = 0.0
            features['mean_transit_interval'] = 0.0
    except Exception as e:
        features['mean_transit_depth'] = 0.0
        features['max_transit_depth'] = 0.0
        features['mean_transit_interval'] = 0.0
    
    # Ensure all features are valid numbers
    for key in list(features.keys()):
        try:
            # Convert to float and check for valid values
            val = float(features[key])
            if np.isnan(val) or np.isinf(val):
                features[key] = 0.0
            else:
                features[key] = val
        except (ValueError, TypeError):
            features[key] = 0.0
    
    # Add aliases for features expected by the frontend
    features['transit_depth'] = features.get('mean_transit_depth', 0.0)
    features['transit_duration'] = features.get('mean_transit_interval', 0.0)
    features['periodicity'] = features.get('dominant_period', 0.0)
    
    return features

if __name__ == "__main__":
    # Test the light curve functions
    print("Testing light curve generation...")
    
    # Generate a synthetic light curve
    t, f = generate_synthetic_light_curve(period=5.0, duration=0.3, depth=0.02, snr=10.0)
    print(f"Generated light curve with {len(t)} points")
    
    # Extract features
    features = analyze_light_curve_features(t, f)
    print("Extracted features:")
    for key, value in list(features.items())[:10]:  # Show first 10 features
        print(f"  {key}: {value:.4f}")
    
    # Plot the light curve
    plot_path = plot_light_curve(t, f, "Test Exoplanet Light Curve", "test_light_curve.png")
    print(f"Light curve plot saved to: {plot_path}")
    
    # Detect transits
    print("Detecting transits...")
    try:
        detection_results = detect_transits_in_light_curve(t, f)
        print(f"Best detected period: {detection_results['best_period']:.4f} days")
        print(f"Best power: {detection_results['best_power']:.4f}")
    except ImportError:
        print("Astropy not available, skipping transit detection")