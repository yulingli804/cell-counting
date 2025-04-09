import os
import numpy as np
import cv2
from skimage import io, exposure, measure, draw
from cellpose import models
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import glob
import re
from matplotlib.widgets import Slider



def select_folder():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.attributes('-topmost', True)  # Make sure it appears on top
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title='Select Folder Containing Images')
    root.destroy()  # Properly destroy the window

    return folder_selected



def find_channel_images(folder, channel_suffix="CH2", extensions=("tif", "tiff", "jpg", "jpeg")):
    matches = []
    pattern = re.compile(rf"{re.escape(channel_suffix)}\.(?:{'|'.join(extensions)})$", re.IGNORECASE)
    for root, _, files in os.walk(folder):
        for file in files:
            if pattern.search(file):
                matches.append(os.path.join(root, file))
    return sorted(matches)



def preprocess_image(path, low_in=2, high_in=180, low_out=90, high_out=140, gamma=0.69):
    image = io.imread(path)
    if image.ndim == 3:
        image_gray = image[:, :, 0]  # use RFP or specific channel
    else:
        image_gray = image

    # Adaptive histogram equalization
    image_eq = exposure.equalize_adapthist(image_gray, clip_limit=0.03)

    # Intensity scaling
    image_adj = exposure.rescale_intensity(
        image_eq, 
        in_range=(low_in / 255, high_in / 255),
        out_range=(low_out / 255, high_out / 255)
    )

    # Gamma correction
    image_gamma = exposure.adjust_gamma(image_adj, gamma)

    return image_gamma, image

def run_cellpose(image, diameter=30, gpu=True):
    model = models.Cellpose(model_type='cyto2', gpu=gpu)
    masks, _, _, _ = model.eval(image, diameter=diameter, flow_threshold=1.0, cellprob_threshold=0.0)
    return masks

def extract_regions(masks, image_gray, area_min=450, area_max=10010):
    props = measure.regionprops(masks, intensity_image=image_gray)

    results = []
    for p in props:
        if area_min <= p.area <= area_max:
            results.append({
                'area': p.area,
                'mean_intensity': p.mean_intensity,
                'centroid': p.centroid,
                'bbox': p.bbox,
                'mask': p.filled_image
            })
    return results

def visualize_results(original_img, results, title=''):
    plt.figure(figsize=(10, 10))
    plt.imshow(original_img, cmap='gray')
    for r in results:
        y, x = r['centroid']
        plt.plot(x, y, 'ro', markersize=2)
        minr, minc, maxr, maxc = r['bbox']
        plt.gca().add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='yellow', linewidth=0.5))
    plt.title(title)
    plt.axis('off')
    plt.show()


def interactive_correction(overlay_img, cell_results):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.3)
    ax.imshow(overlay_img)
    
    if cell_results:
        pts = np.array([r['centroid'][::-1] for r in cell_results])  # (x, y) from (row, col)
        ax.scatter(pts[:, 0], pts[:, 1], s=30, c='lime', marker='o', label='Detected Cells')
    else:
        pts = np.empty((0, 2))
    
    deletion_candidates = []
    new_points = []
    poly_line = None
    
    ax.set_title("Delete points: Click on points to mark for deletion, then click 'Finish Deleting Points'")
    
    # Create buttons for deletion and addition
    ax_del = plt.axes([0.7, 0.1, 0.25, 0.075])
    btn_del = Button(ax_del, "Finish Deleting Points")
    ax_add = plt.axes([0.7, 0.2, 0.25, 0.075])
    btn_add = Button(ax_add, "Finish Adding Points")
    ax_add.set_visible(False)
    
    def on_click_delete(event):
        nonlocal pts, deletion_candidates
        if event.inaxes != ax:
            return
        if pts.size == 0:
            return
        x_click, y_click = event.xdata, event.ydata
        distances = np.hypot(pts[:, 0] - x_click, pts[:, 1] - y_click)
        if distances.min() < 10:  # if close enough to an existing point
            idx = int(np.argmin(distances))
            if idx not in deletion_candidates:
                deletion_candidates.append(idx)
                ax.plot(pts[idx, 0], pts[idx, 1], 'rx', markersize=12)
                fig.canvas.draw()
    
    cid_delete = fig.canvas.mpl_connect('button_press_event', on_click_delete)
    
    def on_click_add(event):
        nonlocal new_points, poly_line
        if event.inaxes != ax:
            return
        new_points.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'bo', markersize=8)
        if len(new_points) > 1:
            xs, ys = zip(*new_points)
            if poly_line is not None:
                poly_line.remove()
            poly_line = ax.plot(xs, ys, 'b-', linewidth=1)[0]
        fig.canvas.draw()
    
    def finish_deletion(event):
        nonlocal cell_results, pts, deletion_candidates
        if deletion_candidates:
            cell_results = [r for i, r in enumerate(cell_results) if i not in deletion_candidates]
        if cell_results:
            pts = np.array([r['centroid'][::-1] for r in cell_results])
        else:
            pts = np.empty((0, 2))
        ax.cla()
        ax.imshow(overlay_img)
        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=30, c='lime', marker='o', label='Detected Cells')
        ax.set_title("Add points: Click to add new points, then click 'Finish Adding Points'")
        fig.canvas.draw()
        fig.canvas.mpl_disconnect(cid_delete)
        ax_del.set_visible(False)
        ax_add.set_visible(True)
        cid_add = fig.canvas.mpl_connect('button_press_event', on_click_add)
        finish_deletion.cid_add = cid_add  # store cid_add for later removal
    
    btn_del.on_clicked(finish_deletion)
    
    def finish_addition(event):
        nonlocal new_points, cell_results, poly_line
        if new_points:
            xs, ys = zip(*new_points)
            # Close the polygon by adding the first point at the end
            xs = list(xs) + [new_points[0][0]]
            ys = list(ys) + [new_points[0][1]]
            ax.plot(xs, ys, 'b-', linewidth=2)
            # Calculate the centroid of the added shape
            centroid_x = np.mean([p[0] for p in new_points])
            centroid_y = np.mean([p[1] for p in new_points])
            new_cell = {
                'area': None,
                'mean_intensity': None,
                'centroid': (centroid_y, centroid_x),
                'bbox': None,
                'mask': None,
                'manual': True
            }
            cell_results.append(new_cell)
            fig.canvas.draw()
        fig.canvas.mpl_disconnect(finish_deletion.cid_add)
        print(f"Final number of cells: {len(cell_results)}")
        plt.close(fig)
    
    btn_add.on_clicked(finish_addition)
    
    plt.show()
    return cell_results

def analyze_image(path, signal_name='RFP', exposure_time=1.0):
    print(f"Processing {os.path.basename(path)}...")
    image_proc, image_orig = preprocess_image(path)
    masks = run_cellpose(image_proc)
    results = extract_regions(masks, image_proc)

    # Normalize by exposure time
    for r in results:
        r['norm_intensity'] = r['mean_intensity'] / exposure_time

    visualize_results(image_orig, results, title=f"{signal_name} | {len(results)} cells")
    return results


if __name__ == '__main__':
    folder = select_folder()

    print("All .tif files:")
    for f in glob.glob(os.path.join(folder, '**', '*.tif'), recursive=True):
        print(f)


    if not folder:
        print("No folder selected. Exiting.")
        exit()

    image_paths = find_channel_images(folder, channel_suffix="CH2")

    print(f"Found {len(image_paths)} images.")

    for path in image_paths:
        analyze_image(path, signal_name='RFP', exposure_time=1.0)