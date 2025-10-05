import numpy as np

def calculate_habitable_zone(stellar_teff, stellar_radius):
    """
    Calculate the inner and outer boundaries of the habitable zone using the 
    conservative and optimistic models based on stellar parameters.
    
    Parameters:
    - stellar_teff: Stellar effective temperature in Kelvin
    - stellar_radius: Stellar radius in solar radii
    
    Returns:
    - Inner and outer boundaries in AU for both conservative and optimistic HZ
    """
    # Convert stellar radius to solar luminosity (approximate)
    # L/Lsun = (R/Rsun)^2 * (T/Tsun)^4
    tsun = 5778.0  # Solar temperature in K
    lsun = (stellar_radius ** 2) * ((stellar_teff / tsun) ** 4)
    
    # Conservative habitable zone (water loss and maximum greenhouse limits)
    # Based on work by Kopparapu et al. 2013
    if stellar_teff < 3700:
        # For cool stars, use different approximations
        inner_edge = 0.35 * np.sqrt(lsun)
        outer_edge = 1.15 * np.sqrt(lsun)
    else:
        # Calculate Seff coefficients (for Sun, Seff values are approximated)
        # For conservative HZ
        # Inner (runaway greenhouse)
        a1, b1, c1, d1 = 1.7763, 3.8095e-4, -1.9396e-8, 5.5974e-12
        # Outer (maximum greenhouse)
        a2, b2, c2, d2 = 0.3210, 5.5470e-5, -4.3345e-9, -5.5556e-13
        
        # Calculate Seff (effective stellar flux)
        s_eff_sun1 = a1 + b1*stellar_teff + c1*(stellar_teff**2) + d1*(stellar_teff**3)
        s_eff_sun2 = a2 + b2*stellar_teff + c2*(stellar_teff**2) + d2*(stellar_teff**3)
        
        # Calculate distances
        inner_edge = np.sqrt(lsun / s_eff_sun1)
        outer_edge = np.sqrt(lsun / s_eff_sun2)
    
    return inner_edge, outer_edge

def calculate_planet_equilibrium_temperature(stellar_teff, stellar_radius, orbital_distance_au, albedo=0.3):
    """
    Calculate the equilibrium temperature of a planet
    
    Parameters:
    - stellar_teff: Stellar effective temperature in Kelvin
    - stellar_radius: Stellar radius in solar radii
    - orbital_distance_au: Orbital distance in AU
    - albedo: Planet albedo (default 0.3 for Earth-like)
    
    Returns:
    - Equilibrium temperature in Kelvin
    """
    # Stellar luminosity relative to Sun
    tsun = 5778.0  # Solar temperature in K
    lsun = (stellar_radius ** 2) * ((stellar_teff / tsun) ** 4)
    
    # Calculate equilibrium temperature
    # Teq = Tsun * sqrt(Rstar/(2*a)) * (1 - A)^0.25
    # where A is albedo
    
    # Convert stellar radius to AU to match orbital distance units
    stellar_radius_au = stellar_radius * 0.00465  # 1 solar radius ~ 0.00465 AU
    
    teq = stellar_teff * np.sqrt(stellar_radius_au / (2 * orbital_distance_au)) * ((1 - albedo) ** 0.25)
    return teq

def calculate_habitability_probability(planet_params):
    """
    Calculate a habitability probability based on multiple factors
    
    Parameters:
    - planet_params: Dictionary containing planet parameters
    
    Returns:
    - Habitability probability (0-1 scale)
    """
    
    # Extract parameters
    orbital_distance_au = planet_params.get('orbital_distance_au', 1.0)
    planet_temp = planet_params.get('equilibrium_temp', 288)
    planet_radius = planet_params.get('radius_earth', 1.0)
    stellar_teff = planet_params.get('stellar_teff', 5778)
    stellar_radius = planet_params.get('stellar_radius', 1.0)
    
    # Calculate habitable zone boundaries
    inner_hz, outer_hz = calculate_habitable_zone(stellar_teff, stellar_radius)
    
    # Calculate individual habitability scores
    # 1. Distance from habitable zone
    if inner_hz <= orbital_distance_au <= outer_hz:
        dist_score = 1.0
    else:
        # Calculate normalized distance from habitable zone
        if orbital_distance_au < inner_hz:
            dist_score = 1.0 - min(1.0, (inner_hz - orbital_distance_au) / inner_hz)
        else:
            dist_score = 1.0 - min(1.0, (orbital_distance_au - outer_hz) / outer_hz)
        
        # Clamp to 0-1 range and reduce the score significantly outside HZ
        dist_score = max(0.0, dist_score)
    
    # 2. Temperature habitability (liquid water range: ~273-373 K)
    temp_score = 0.0
    if 273 <= planet_temp <= 373:
        temp_score = 1.0
    elif 253 <= planet_temp <= 393:  # Broader range allows for some tolerance
        # Use a sigmoid-like function for gradual decrease
        if planet_temp < 273:
            temp_score = (planet_temp - 253) / 20
        else:  # planet_temp > 373
            temp_score = (393 - planet_temp) / 20
        temp_score = max(0.0, min(1.0, temp_score))
    
    # 3. Size habitability (Earth-like: 0.8 - 1.5 Earth radii)
    size_score = 0.0
    if 0.8 <= planet_radius <= 1.5:
        size_score = 1.0
    elif 0.5 <= planet_radius <= 2.0:
        # Gradual decrease outside Earth-like range
        if planet_radius < 0.8:
            size_score = (planet_radius - 0.5) / 0.3  # From 0.5 to 0.8
        else:  # planet_radius > 1.5
            size_score = (2.0 - planet_radius) / 0.5  # From 1.5 to 2.0
        size_score = max(0.0, min(1.0, size_score))
    
    # Weighted average of all scores
    habitability = (dist_score * 0.5 + temp_score * 0.3 + size_score * 0.2)
    
    return habitability, {
        'distance_score': dist_score,
        'temperature_score': temp_score,
        'size_score': size_score,
        'inner_hz': inner_hz,
        'outer_hz': outer_hz
    }

# Example usage
if __name__ == "__main__":
    # Test the habitability calculations
    
    # Example: Earth-like planet around Sun-like star
    planet_params = {
        'orbital_distance_au': 1.0,  # 1 AU = Earth's distance
        'equilibrium_temp': 255,     # Earth's equilibrium temp without greenhouse
        'radius_earth': 1.0,         # Earth radius
        'stellar_teff': 5778,        # Sun's temperature in K
        'stellar_radius': 1.0        # Sun's radius in solar radii
    }
    
    prob, details = calculate_habitability_probability(planet_params)
    
    print("Habitability Analysis Results:")
    print(f"  Habitability Probability: {prob:.3f}")
    print(f"  Distance Score: {details['distance_score']:.3f}")
    print(f"  Temperature Score: {details['temperature_score']:.3f}")
    print(f"  Size Score: {details['size_score']:.3f}")
    print(f"  Habitable Zone: {details['inner_hz']:.3f} - {details['outer_hz']:.3f} AU")
    
    # Example: Proxima Centauri b (approximation)
    proxima_b_params = {
        'orbital_distance_au': 0.05,  # ~0.05 AU
        'equilibrium_temp': 234,      # Estimated
        'radius_earth': 1.27,         # Estimated
        'stellar_teff': 3042,         # Proxima Centauri temperature
        'stellar_radius': 0.154       # Proxima Centauri radius in solar radii
    }
    
    prob2, details2 = calculate_habitability_probability(proxima_b_params)
    
    print("\nProxima Centauri b Analysis:")
    print(f"  Habitability Probability: {prob2:.3f}")
    print(f"  Distance Score: {details2['distance_score']:.3f}")
    print(f"  Temperature Score: {details2['temperature_score']:.3f}")
    print(f"  Habitable Zone: {details2['inner_hz']:.3f} - {details2['outer_hz']:.3f} AU")