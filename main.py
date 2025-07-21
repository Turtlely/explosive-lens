import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as mticker
from scipy.ndimage import laplace

# --- Simulation Parameters ---
grid_size = 20         # Physical size of the domain [-grid_size/2, grid_size/2]
resolution = 300       # Grid resolution (resolution x resolution)
dt = 0.1               # Time step (ms)
n_lenses = 5           # Number of explosive lenses
v1, v2 = 2.0, 1.0      # Wave speeds for different media (cm/ms)
lens_distance = 4.0    # Distance from center to each lens (radius) (cm)
td = 3.5               # Time for wavefront to reach center (ms)
sigma = 0.25            # Gaussian width of detonation (standard deviation)

# --- Grid Setup ---
x = np.linspace(-grid_size/2, grid_size/2, resolution)
y = np.linspace(-grid_size/2, grid_size/2, resolution)
xv, yv = np.meshgrid(x, y)

# --- Lens Geometry Functions ---
def initial_state(n_points, domain_size, num_lenses, radius, std_dev):
    x = np.linspace(-domain_size/2, domain_size/2, n_points)
    y = np.linspace(-domain_size/2, domain_size/2, n_points)
    xv, yv = np.meshgrid(x, y)

    field = np.zeros_like(xv)
    angles = np.linspace(0, 2 * np.pi, num_lenses, endpoint=False) + np.pi

    for alpha in angles:
        cx = radius * np.cos(alpha)
        cy = radius * np.sin(alpha)
        gauss = np.exp(-((xv - cx)**2 + (yv - cy)**2) / (2 * std_dev**2))
        field += gauss

    return field

def media_check(x, y, num_lenses):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    alphas = np.linspace(0, 2 * np.pi, num_lenses, endpoint=False)

    inside = np.zeros_like(r, dtype=bool)
    for alpha in alphas:
        expr = (np.sqrt(r**2 + 2 * r * lens_distance * np.cos(theta + alpha) + lens_distance**2) / v1 + r / v2) < td
        inside |= expr

    return inside

def lens_boundary(cutoff_radius, alpha=0, num_points=500):
    """
    Compute the boundary of a single explosive lens.

    Parameters:
    cutoff_radius : float
        Only plot boundary points where r > cutoff_radius (to exclude degenerate parts).
    alpha : float, optional
        Angular offset of the lens in radians (default is 0).
    num_points : int, optional
        Number of points to compute along the boundary (default is 500).

    Returns:
    (x, y) : tuple of ndarrays
        Cartesian coordinates of the lens boundary points.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Discriminant calculation based on the boundary condition equation
    term1 = lens_distance * v2**2 * np.cos(theta + alpha) + v1**2 * v2 * td
    discriminant = term1**2 - (v2**2 - v1**2) * (v2**2 * lens_distance**2 - v1**2 * v2**2 * td**2)

    # Initialize r with NaN and compute only where discriminant is valid (>= 0)
    r = np.full_like(theta, np.nan)
    valid = discriminant >= 0
    r[valid] = (-term1[valid] + np.sqrt(discriminant[valid])) / (v2**2 - v1**2)

    # Apply cutoff condition to exclude points too close to center
    r[r <= cutoff_radius] = np.nan

    # Convert polar coordinates to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y

def damping_mask(shape, border_width, damping_factor):
    """
    Create a damping mask with values in [damping_factor, 1] 
    where damping_factor applies at the border and 1 in the center.

    Parameters:
    shape : tuple
        Shape of the simulation grid (ny, nx).
    border_width : int
        Width of the damping border in grid points.
    damping_factor : float
        Minimum multiplier at the boundary (e.g., 0.95 or lower).
    
    Returns:
    mask : ndarray
        Damping multiplier array.
    """
    ny, nx = shape
    mask = np.ones((ny, nx))

    for i in range(border_width):
        coeff = damping_factor + (1 - damping_factor) * (i / border_width)
        mask[i, :] *= coeff
        mask[-(i+1), :] *= coeff
        mask[:, i] *= coeff
        mask[:, -(i+1)] *= coeff

    return mask

# --- Initialize Wave Fields ---
u = initial_state(resolution, grid_size, n_lenses, lens_distance, sigma)
v = np.zeros_like(u)

# --- Generate Media Velocity Map ---
inside_mask = media_check(xv, yv, n_lenses)
c = np.where(inside_mask, v2, v1)

# --- Animation Setup ---
fig, ax = plt.subplots()
im = ax.imshow(u, extent=[-grid_size/2, grid_size/2, -grid_size/2, grid_size/2], origin='lower',
               cmap='seismic', vmin=-0.25, vmax=0.25)
plt.colorbar(im, ax=ax, label='Overpressure')

# --- Plot lens boundaries on top of the heatmap ---
r_cutoff = (-(lens_distance * v2**2 * np.cos((n_lenses-1)*np.pi/n_lenses) + v1**2 * v2 * td) + np.sqrt((lens_distance * v2**2 * np.cos((n_lenses-1)*np.pi/n_lenses) + v1**2 * v2 * td)**2 - (v2**2 - v1**2) * (v2**2 * lens_distance**2 - v1**2 * v2**2 * td**2))) / (v2**2 - v1**2)

for alpha in np.linspace(0, 2 * np.pi, n_lenses, endpoint=False):

    x_b, y_b = lens_boundary(r_cutoff, alpha=alpha)
    ax.plot(x_b, y_b, color='black', linewidth=1)

damping = damping_mask(u.shape, border_width=10, damping_factor=0.95)

# Simulation Loop
def animate(frame):
    global u, v
    lap_u = laplace(u, mode='constant')
    a = lap_u * c**2
    v += a * dt
    u += v * dt
    v *= damping
    u *= damping

    im.set_data(u)
    ax.set_title(f'Explosive Lens Simulation | Time = {frame * dt:.2f}ms')
    return [im]

anim = FuncAnimation(fig, animate, frames=600, interval=1, blit=True)

# Set axis labels
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")

# Parameters for polar grid
r_max = grid_size / 2
num_circles = 5
num_radials = 12

# Radii for concentric circles
radii = np.linspace(0, r_max, num_circles + 1)[1:]  # skip radius=0

# Draw concentric circles
theta_grid = np.linspace(0, 2 * np.pi, 360)
for r in radii:
    x_circle = r * np.cos(theta_grid)
    y_circle = r * np.sin(theta_grid)
    ax.plot(x_circle, y_circle, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

# Draw radial lines
angles = np.linspace(0, 2 * np.pi, num_radials, endpoint=False)
for angle in angles:
    x_line = [0, r_max * np.cos(angle)]
    y_line = [0, r_max * np.sin(angle)]
    ax.plot(x_line, y_line, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

    # Label each radial line with its angle in degrees
    deg = np.degrees(angle)

# Set equal aspect ratio for correct circle shapes
ax.set_aspect('equal', 'box')

# Tighten axis limits to focus on the spherical wavefront
ax.set_xlim(-lens_distance * 1.5, lens_distance * 1.5)
ax.set_ylim(-lens_distance * 1.5, lens_distance * 1.5)

# Save or show
anim.save('detonation.gif', writer='ffmpeg', fps=120)
#plt.show()

