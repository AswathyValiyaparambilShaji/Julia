using DSP


# Function to calculate Coriolis frequency f based on latitude in degrees
function coriolis_frequency(latitude_deg)
    Omega = 7.292e-5  # Earth's angular velocity in rad/s
    latitude_rad = deg2rad(latitude_deg)  # Convert degrees to radians
    f = 2 * Omega * sin(latitude_rad)  # Coriolis parameter in rad/s
    return f  # Return absolute value to work in both hemispheres
end





