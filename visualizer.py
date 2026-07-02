import cv2
import numpy as np
import imageio
import torch
import matplotlib
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pdb

class GraphToVideo:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 12))  # Increased figure size
        self.axes = {}
        self.lines = {}
        self.images = {}
        self.animation = None
        self._anim = None  # Prevent garbage collection

    def timestep_check(self, data_dict):
        min_timestep = min(value.shape[0] for value in data_dict.values())
        for key, value in data_dict.items():
            if value.shape[0] != min_timestep:
                print(f"Warning: Time step size for '{key}' is different. Resizing to {min_timestep} timesteps.")
        return min_timestep

    def resize_timesteps(self, data_dict, min_timestep):
        for key in data_dict:
            data_dict[key] = data_dict[key][:min_timestep]
        return data_dict

    def get_axes(self, data_dict):
        num_videos = sum(1 for value in data_dict.values() if (value.ndim == 4 or (value.ndim == 3 and value.shape[-1] != 1)))
        num_graphs = len(data_dict) - num_videos

        if num_videos == 0:
            gs = GridSpec(num_graphs, 1, figure=self.fig, hspace=0.5, wspace=0.3)  # Only graphs
        else:
            gs = GridSpec(num_graphs + 1, num_videos, figure=self.fig, hspace=0.5, wspace=0.3)  # Graphs and videos

        video_index = 0
        graph_index = 0
        
        for title, value in data_dict.items():
            if value.ndim == 4 or (value.ndim == 3 and value.shape[-1] != 1):  # Video data
                self.axes[title] = self.fig.add_subplot(gs[0, video_index])
                self.axes[title].set_axis_off()
                video_index += 1
            elif value.ndim == 3 and value.shape[-1] == 1:  # Grayscale video data
                self.axes[title] = self.fig.add_subplot(gs[0, video_index])
                self.axes[title].set_axis_off()
                video_index += 1
            elif value.ndim == 2 or value.ndim == 1:  # Graph data
                if num_videos == 0:
                    self.axes[title] = self.fig.add_subplot(gs[graph_index, 0])
                else:
                    self.axes[title] = self.fig.add_subplot(gs[1 + graph_index, :])
                graph_index += 1
            else:
                raise ValueError(f"Unsupported array dimension for '{title}': {value.ndim}")
            self.axes[title].set_title(title, fontsize='small')

    def convert_rgb_to_bgr(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray) and value.shape[-1] == 3:
                dictionary[key] = value[..., ::-1]
        return dictionary

    def reshape_images(self, dictionary):
        for key, value in dictionary.items():
            if value.ndim == 4 and value.shape[1] == 3:  # Convert (timestep, 3, height, width) to (timestep, height, width, 3)
                dictionary[key] = np.transpose(value, (0, 2, 3, 1))
        return dictionary
    
    def tensor_to_numpy(self, dictionary):
        for key, value in dictionary.items():
            if torch.is_tensor(value):
                if value.is_cuda:  # GPU上にある場合、CPUに移動
                    value = value.cpu()
                dictionary[key] = value.detach().clone().numpy()
        return dictionary

    def add_graph(self, data_dict):
        data_dict = self.tensor_to_numpy(data_dict)  # テンソルをNumPyに変換
        # data_dict = self.convert_rgb_to_bgr(data_dict)
        data_dict = self.reshape_images(data_dict)
        min_timestep = self.timestep_check(data_dict)
        data_dict = self.resize_timesteps(data_dict, min_timestep)
        self.get_axes(data_dict)

        for key, value in data_dict.items():
            if value.ndim == 4 or (value.ndim == 3 and value.shape[-1] != 1):  # [timestep, height, width, (3|1)]
                if value.ndim == 3 and value.shape[-1] == 1:
                    value = value.squeeze(-1)  # Convert (timestep, height, width, 1) to (timestep, height, width)
                self.images[key] = self.axes[key].imshow(value[0], aspect="equal", cmap='gray')
            else:
                if value.ndim == 1:
                    value = value.reshape(-1, 1)
                self.axes[key].set_xlim(0, value.shape[0] - 1)  # Set x-axis limits
                min_val, max_val = np.min(value), np.max(value)
                if min_val == max_val:
                    min_val -= 1
                    max_val += 1
                self.axes[key].set_ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))  # Adaptive y-limits
                self.lines[key] = [self.axes[key].plot([], [], label=f'Dimension {i+1}')[0] for i in range(value.shape[1])]

        def update_frame(frame):
            artists = []
            for key, value in data_dict.items():
                if key in self.images:
                    self.images[key].set_array(value[frame])
                    artists.append(self.images[key])
                elif key in self.lines:
                    for i, line in enumerate(self.lines[key]):
                        if value.ndim == 1:
                            line.set_data(range(frame + 1), value[:frame + 1])
                        else:
                            line.set_data(range(frame + 1), value[:frame + 1, i])
                        artists.append(line)
            return artists

        self.animation = animation.FuncAnimation(self.fig, update_frame, frames=min_timestep, interval=100, blit=False)
        self._anim = self.animation  # Prevent garbage collection

    def save_graph(self, path=".", filename="graph", file_format="gif"):
        if file_format not in ["mp4", "gif"]:
            raise ValueError("Unsupported file format. Use 'mp4' or 'gif'.")
        
        if file_format == "mp4":
            writer = animation.FFMpegWriter(fps=10, bitrate=5000)
            self.animation.save(f"{path}/{filename}.mp4", writer=writer)
        elif file_format == "gif":
            self.animation.save(f"{path}/{filename}.gif", writer='imagemagick', fps=10)


def vis_sarnn(video, point_data, path="output.gif"):
    video = video.astype(np.uint8)
    timesteps, point_num, _ = point_data.shape
    frames = []
    
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10)[:3] for i in range(point_num)]  # Get enough colors and only take RGB
    
    colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in colors]
    
    for i in range(timesteps):
        frame = cv2.resize(video[i], (256, 256))
        
        for num in range(point_num):
            x, y = point_data[i][num]
            x = int(x * 256)  
            y = int(y * 256)
            radius = 5  # You can adjust this to make the circles bigger
            color = colors[num % len(colors)]  # Cycle through colors if needed
            cv2.circle(frame, (x, y), radius, color, -1)  # -1 fills the circle
        
        frames.append(frame)
    imageio.mimsave(path, frames, 'GIF', fps=20)




def vis_conf_sarnn(video, point_data,conf,path="output.gif"):
    video = video.astype(np.uint8)
    timesteps, point_num, _ = point_data.shape
    frames = []
    
    for i in range(timesteps):
        frame = video[i]
        for num in range(point_num):
            x, y = point_data[i][num]
            x = int(x * frame.shape[0])
            y = int(y * frame.shape[1])
            if conf[i][num]>0.5:
                cv2.circle(frame, (x, y), int(10*conf[i][num]+1), (0, 255, 0), -1)
            else:
                cv2.circle(frame, (x, y), int(10*conf[i][num]+1), (255, 0, 0), -1)
        frames.append(frame)
    imageio.mimsave(path, frames, 'GIF', fps=20)

def vis_colorful_conf_sarnn(video, point_data, conf, path="output.gif"):
    video = video.astype(np.uint8)
    timesteps, point_num, _ = point_data.shape
    frames = []
    
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10)[:3] for i in range(point_num)]  # Use only the first 3 values (RGB)
    
    colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]
    
    for i in range(timesteps):
        frame = cv2.resize(video[i], (256, 256))
        
        for num in range(point_num):
            x, y = point_data[i][num]
            x = int(x * 256)  
            y = int(y * 256)
            
            radius = int(20 * conf[i][num] + 5)  # Larger base size for circles
            color = colors[num % len(colors)]  # Cycle through colors if needed
            
            cv2.circle(frame, (x, y), radius, color, -1)  # Draw the circle with the chosen color
        
        frames.append(frame)
    
    imageio.mimsave(path, frames, 'GIF', fps=20)



def plot_tensor(tensor: torch.Tensor, save_path: str, title: str = 'Tensor Plot'):

    assert len(tensor.shape) == 2, "テンソルは2次元である必要があります (timestep, dimension)"

    # 各次元についてプロットします
    for dim in range(tensor.shape[1]):
        plt.plot(tensor[:, dim], label=f'Dimension {dim+1}')

    # タイトルとラベルを設定します
    plt.title(title)
    plt.xlabel('Timestep')
    plt.ylabel('Value')
    plt.legend()

    # 画像を指定されたパスに保存します
    plt.savefig(save_path)
    plt.clf()




def visualize_heatmap(heatmap, batch=0, channel=None, save_dir='./'):
    # Check if the heatmap is a tensor
    if isinstance(heatmap, torch.Tensor):
        # Move the tensor to CPU if it's on GPU
        if heatmap.is_cuda:
            heatmap = heatmap.cpu()
        # Detach the tensor and convert to numpy array
        heatmap = heatmap.detach().numpy()
    
    # Extract the specified batch
    batch_heatmap = heatmap[batch]
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'heatmap.png')

    # Determine the global min and max values for the heatmaps
    vmin = np.min(batch_heatmap)
    vmax = np.max(batch_heatmap)

    # If channel is None, display all channels
    if channel is None:
        num_channels = batch_heatmap.shape[0]
        fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
        for ch in range(num_channels):
            im = axes[ch].imshow(batch_heatmap[ch], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[ch].set_title(f'Channel {ch}')
            axes[ch].axis('off')
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.ax.set_ylabel('Intensity')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        # Display the specified channel
        plt.imshow(batch_heatmap[channel], cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title(f'Channel {channel}')
        plt.axis('off')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Intensity')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def visualize_images(image_tensor, batch=0, number=None, save_dir='./'):
    # Check if the input is a tensor
    if isinstance(image_tensor, torch.Tensor):
        # Move the tensor to CPU if it's on GPU
        if image_tensor.is_cuda:
            image_tensor = image_tensor.cpu()
        # Detach the tensor and convert to numpy array
        image_tensor = image_tensor.detach().numpy()
    
    # Extract the specified batch
    batch_images = image_tensor[batch]
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'images.png')

    # If number is None, display all images
    if number is None:
        num_images = batch_images.shape[0]
        if num_images == 1:
            # Handle single image case
            img = batch_images[0].transpose(1, 2, 0)  # (3, height, width) -> (height, width, 3)
            plt.imshow(img)
            plt.title('Image 0')
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
            if num_images == 1:
                axes = [axes]  # Make it iterable
            for num in range(num_images):
                img = batch_images[num].transpose(1, 2, 0)  # (3, height, width) -> (height, width, 3)
                axes[num].imshow(img)
                axes[num].set_title(f'Image {num}')
                axes[num].axis('off')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
    else:
        # Display the specified number
        img = batch_images[number].transpose(1, 2, 0)  # (3, height, width) -> (height, width, 3)
        plt.imshow(img)
        plt.title(f'Image {number}')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def vis_image(image):
    # Check if the image is a tensor
    if isinstance(image, torch.Tensor):
        # Move the tensor to the CPU if it's on the GPU
        image = image.cpu().detach().numpy()

    # Ensure the image is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array or a tensor.")

    # If the image is in the shape (3, height, width), transpose it to (height, width, 3)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    # Convert the image to BGR format for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display the image in a window
    cv2.imshow("Image", image)
    
    # Wait for 1ms to update the display
    cv2.waitKey(1)
    
def vis_video(x, path="./video.gif", fps=10):
    """
    x: numpy array or PyTorch tensor of shape (batch, timestep, height, width, 3)
    path: file path to save the gif
    fps: frames per second for the gif
    """
    # Check if x is a tensor and move to CPU if needed
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    
    # Clip the values to the range [0, 255]
    x = np.clip(x, 0, 255)

    # Convert to uint8
    if x.dtype != np.uint8:
        x = x.astype(np.uint8)
    
    # Ensure the output path ends with .gif
    if not path.lower().endswith('.gif'):
        raise ValueError("The output path must end with .gif")
    
    # Convert video frames to list of ndarrays
    frames_list = [frame for frame in x]
    
    # Save the video frames as a gif
    imageio.mimsave(path, frames_list, fps=fps)


def make_temporal_pca(data, output_dim="3d", save_path="./pca.png",data_found=None):
    # Check if data is a PyTorch tensor
    if torch.is_tensor(data):
        # Move the data to CPU if it's on GPU and convert to Numpy array
        data = data.cpu().numpy()
    
    # Ensure the data is a 3D numpy array
    if data.ndim != 3:
        raise ValueError("Data must be a 3D tensor with shape (batch, timestep, dimension).")
    
    if output_dim not in ['2d', '3d']:
        raise ValueError("Output dimension must be either '2d' or '3d'.")
    
    batch_size, timesteps, dimension = data.shape

    if data_found is None:
        data_found=torch.ones_like(torch.tensor(data[:,:,0]))

    if torch.is_tensor(data_found):
        data_found = data_found.cpu().numpy()

    indices = (data_found >= 0.999).astype(int)
    column_indices = np.arange(data_found.shape[1])+1
    weighted_cumsum = indices * column_indices[None,:]
    last_one_indices = np.argmax(weighted_cumsum, axis=1)

    data_real=[]
    for b in range(batch_size):
        data_real+=[data[b,:last_one_indices[b]+1]]
    data_reshaped = np.concatenate(data_real,axis=0)
    
    # Perform PCA on the reshaped data
    pca = PCA(n_components=2 if output_dim == '2d' else 3)
    pca_result = pca.fit_transform(data_reshaped)
    
    pca_results=[]
    idx=0
    for b in range(batch_size):
        pca_results+=[pca_result[idx:idx+last_one_indices[b]]]
        idx=idx+last_one_indices[b]+1
    
    file_ext = os.path.splitext(save_path)[1].lower()
    
    if output_dim == '2d':
        plt.figure(figsize=(10, 7))
        for i in range(batch_size):
            plt.plot(pca_results[i][:, 0], pca_results[i][:, 1], label=f'Batch {i+1}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Temporal PCA (2D)')
        plt.legend()
        plt.grid(True)
        
        if file_ext in ['.png', '.jpg', '.jpeg']:
            plt.savefig(save_path)
        elif file_ext == '.gif':
            raise ValueError("GIF is not supported for 2D output.")
        
        plt.close()
        
    elif output_dim == '3d':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(batch_size):
            ax.plot(pca_results[i][:, 0], pca_results[i][:, 1], pca_results[i][:, 2], label=f'Batch {i+1}')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        ax.set_title('Temporal PCA (3D)')
        plt.legend()
        plt.grid(True)
        
        if file_ext in ['.png', '.jpg', '.jpeg']:
            plt.savefig(save_path)
        elif file_ext == '.gif':
            def update(frame):
                ax.view_init(30, frame)
                return ax,

            ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=100)
            ani.save(save_path, writer='pillow')
        
        plt.close()

def make_several_temporal_pca(data1, data2, output_dim="3d", save_path="./pca.png"):
    # Check if inputs are PyTorch tensors and convert them to Numpy arrays if needed
    if torch.is_tensor(data1):
        data1 = data1.cpu().numpy()
    if torch.is_tensor(data2):
        data2 = data2.cpu().numpy()

    # Ensure both data1 and data2 are 3D numpy arrays
    if data1.ndim != 3 or data2.ndim != 3:
        raise ValueError("Both data1 and data2 must be 3D arrays with shape (batch, timestep, dimension).")
    
    if output_dim not in ['2d', '3d']:
        raise ValueError("Output dimension must be either '2d' or '3d'.")
    
    batch_size1, timesteps1, dimension1 = data1.shape
    batch_size2, timesteps2, dimension2 = data2.shape
    
    if dimension1 != dimension2:
        raise ValueError("Both data1 and data2 must have the same dimensionality for PCA.")
    
    # Reshape data1 to prepare it for PCA fitting
    data1_reshaped = data1.reshape(batch_size1 * timesteps1, dimension1)
    
    # Perform PCA fitting on data1
    pca = PCA(n_components=2 if output_dim == '2d' else 3)
    pca.fit(data1_reshaped)
    
    # Transform both data1 and data2 using the fitted PCA
    pca_result_data1 = pca.transform(data1_reshaped).reshape(batch_size1, timesteps1, -1)
    data2_reshaped = data2.reshape(batch_size2 * timesteps2, dimension2)
    pca_result_data2 = pca.transform(data2_reshaped).reshape(batch_size2, timesteps2, -1)
    
    # Get the file extension
    file_ext = os.path.splitext(save_path)[1].lower()
    
    if output_dim == '2d':
        plt.figure(figsize=(10, 7))
        for i in range(batch_size1):
            plt.plot(pca_result_data1[i, :, 0], pca_result_data1[i, :, 1], color='blue', label='Data1' if i == 0 else "")
        for i in range(batch_size2):
            plt.plot(pca_result_data2[i, :, 0], pca_result_data2[i, :, 1], color='red', label='Data2' if i == 0 else "")
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Temporal PCA (2D)')
        plt.legend()
        plt.grid(True)
        
        if file_ext in ['.png', '.jpg', '.jpeg']:
            plt.savefig(save_path)
        elif file_ext == '.gif':
            raise ValueError("GIF is not supported for 2D output.")
        
        plt.close()
        
    elif output_dim == '3d':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(batch_size1):
            ax.plot(pca_result_data1[i, :, 0], pca_result_data1[i, :, 1], pca_result_data1[i, :, 2], color='blue', label='Data1' if i == 0 else "")
        for i in range(batch_size2):
            ax.plot(pca_result_data2[i, :, 0], pca_result_data2[i, :, 1], pca_result_data2[i, :, 2], color='red', label='Data2' if i == 0 else "")
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        ax.set_title('Temporal PCA (3D)')
        plt.legend()
        plt.grid(True)
        
        if file_ext in ['.png', '.jpg', '.jpeg']:
            plt.savefig(save_path)
        elif file_ext == '.gif':
            def update(frame):
                ax.view_init(30, frame)
                return ax,

            ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=100)
            ani.save(save_path, writer='pillow')
        
        plt.close()
