import numpy as np
import matplotlib.pyplot as plt
import cv2

def render_stitched_frame(rgb_img, waypoints=None, waypoint_index=2):
    rgb_img_resized = cv2.resize(rgb_img, (320, 240))
    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=100)
    ax.set_title("Predicted Waypoints")
    ax.grid(True)
    if waypoints is not None:
        for i in range(waypoints.shape[0]):
            ax.plot(-waypoints[i, :, 1], waypoints[i, :, 0], color='cyan', alpha=0.3)
        ax.plot(-waypoints[0, :, 1], waypoints[0, :, 0], color='blue', linewidth=2, label='Trajectory')
        selected_wp = waypoints[0, waypoint_index]
        ax.scatter(-selected_wp[1], selected_wp[0], color='red', s=40, label='Target Waypoint', zorder=5)
    ax.scatter(0, 0, color='black', marker='^', s=80, label='Robot', zorder=10)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlabel("Left/Right (m)")
    ax.set_ylabel("Forward (m)")
    
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    if plot_img.shape[0] != rgb_img_resized.shape[0]:
        plot_img = cv2.resize(plot_img, (int(plot_img.shape[1] * rgb_img_resized.shape[0] / plot_img.shape[0]), rgb_img_resized.shape[0]))
    return np.hstack((rgb_img_resized, plot_img))

# Test with mock data
rgb = np.zeros((120, 160, 3), dtype=np.uint8)
# Random shape (8, 8, 2)
wp = np.random.randn(8, 8, 2) * 0.5 + np.array([1.0, 0.0]) # Bias forward
out = render_stitched_frame(rgb, wp)
print("Output shape:", out.shape)
