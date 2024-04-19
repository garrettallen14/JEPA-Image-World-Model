from torch.utils.data import Dataset
import glob
import os
import json
import cv2
import torch
import random
from tqdm import tqdm
import shutil
import urllib



def relpaths_to_download(relpaths, output_dir, n_videos):
    def read_json(file_name):
        with open(file_name.replace('mp4', 'jsonl'), 'r') as json_file:
            text = json.loads('['+''.join(json_file.readlines()).replace('\n', ',')+']')

    data_path = '/'.join(relpaths[0].split('/')[:-1])
    non_defect=[]
    for vid_name in glob.glob(os.path.join(output_dir,'*.mp4')):
        try:
            vid = cv2.VideoCapture(vid_name)
            read_json(vid_name.replace('mp4', 'jsonl'))
            if vid.isOpened():
                non_defect.append(os.path.join(data_path, vid_name.split('/')[-1]))
        except:
            continue

    relpaths = set(relpaths)
    non_defect = set(non_defect)
    diff_to_download = relpaths.difference(non_defect)
    if n_videos > 1:
        print('total:', len(relpaths), '| exist:', len(non_defect), '| downloading:', len(diff_to_download))
    return diff_to_download



def load_data(data_folder, start_idx, n_videos=30, action_sequence_length=5, frame_skip=1, split_ratio=0.9, batch_size=32, num_workers=4, shuffle=True):
    # If the data_folder exists, delete it and all files in it then make a new one
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)

    os.makedirs(data_folder)

    # Open shuffled-find-cave.json
    # with open('new_new_videos.json', 'r') as f:
    with open('shuffled-find-cave.json', 'r') as f:
        cave_data = json.load(f)

    basedir = cave_data['basedir']    
    cave_data = cave_data['relpaths'][start_idx:start_idx+n_videos]

    # Download the videos and get into proper formatting
    relpaths = relpaths_to_download(cave_data, data_folder, n_videos)
    for i, relpath in enumerate(relpaths):
        url = basedir + relpath
        filename = os.path.basename(relpath)
        outpath = os.path.join(data_folder, filename)
        percent_done = 100 * i / len(relpaths)
        if n_videos > 1:
            print(f"[{percent_done:.0f}%] Downloading {outpath}")
        try:
            urllib.request.urlretrieve(url, outpath)
        except Exception as e:
            print(f"\tError downloading {url}: {e}. Moving on")
            continue

        # Also download corresponding .jsonl file
        jsonl_url = url.replace(".mp4", ".jsonl")
        jsonl_filename = filename.replace(".mp4", ".jsonl")
        jsonl_outpath = os.path.join(data_folder, jsonl_filename)
        try:
            urllib.request.urlretrieve(jsonl_url, jsonl_outpath)
        except Exception as e:
            print(f"\tError downloading {jsonl_url}: {e}. Cleaning up mp4")
            os.remove(outpath)

    # Create the dataset
    train_dataset = VideoDataset(data_folder, action_sequence_length, split='train', split_ratio=split_ratio, shuffle=shuffle, frame_skip=frame_skip)
    val_dataset = VideoDataset(data_folder, action_sequence_length, split='val', split_ratio=split_ratio, shuffle=shuffle, frame_skip=frame_skip)

    # Create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader



class VideoDataset(Dataset):
    def __init__(self, data_folder, action_sequence_length, frame_skip=1, split='train', shuffle=True, split_ratio=0.8):
        self.data_folder = data_folder
        self.action_sequence_length = action_sequence_length
        self.split = split
        self.shuffle = shuffle
        self.frame_skip = frame_skip

        # Dictionary mapping keyboard keys to their corresponding indices
        self.key_to_index = {
            "key.keyboard.w": 0,
            "key.keyboard.a": 1,
            "key.keyboard.s": 2,
            "key.keyboard.d": 3,
            "key.keyboard.space": 4,
            "key.keyboard.left.shift": 5,
            "key.keyboard.left.control": 6,
            "key.keyboard.e": 7,
            "key.keyboard.q": 8,
            "key.keyboard.escape": 9,
            "key.keyboard.f": 10,
            "key.keyboard.1": 11,
            "key.keyboard.2": 12,
            "key.keyboard.3": 13,
            "key.keyboard.4": 14,
            "key.keyboard.5": 15,
            "key.keyboard.6": 16,
            "key.keyboard.7": 17,
            "key.keyboard.8": 18,
            "key.keyboard.9": 19
        }

        # Lists to store the processed data
        self.frames_t = []
        self.action_sequences = []
        self.frames_t_plus_1 = []

        # Scaling factor for the camera
        self.original_height_px = 720
        self.camera_scaler = 360.0 / 2400.0

        # Storing the valid data locations
        self.video_title_list = []
        self.valid_data_locations = []

        # Load the valid dataset tuples
        self.load_dataset_tuples()

        # Shuffle the data if necessary
        if self.shuffle:
            random.shuffle(self.valid_data_locations)

        # Train/Validation split
        if self.split == 'train':
            self.valid_data_locations = self.valid_data_locations[:int(split_ratio * len(self.valid_data_locations))]
        elif self.split == 'val':
            self.valid_data_locations = self.valid_data_locations[int(split_ratio * len(self.valid_data_locations)):]

    def load_dataset_tuples(self):
        # Create tuples of (video_file, jsonl_file) for each video file
        video_files = glob.glob(os.path.join(self.data_folder, '*.mp4'))
        video_files = [(video_file, video_file.replace('.mp4', '.jsonl')) for video_file in video_files]
        action_data = []
        actions = []
        valid = True

        # Iterate through the video files
        for video_file, jsonl_file in tqdm(video_files):

            # Append the video title to the list
            self.video_title_list.append(video_file.split('/')[-1].split('.')[0])

            # Load the jsonl file and parse the action data
            with open(jsonl_file, 'r') as f:
                action_data = [json.loads(line) for line in f]

            # Iterate through the action data to get a list of valid indices
            for i in range(3, len(action_data) - self.frame_skip - 25):
                
                # Load the actions from t-action_sequence_length to t
                actions = []
                for j in range(self.action_sequence_length):
                    actions.append(action_data[i - self.action_sequence_length + j])
                                
                # Check to see if there are keyboard or mouse inputs in the last three actions
                valid = True
                # for action in actions[-3:]:
                #     if (action['mouse']['dx'] != 0 or action['mouse']['dy'] != 0
                #         or action['mouse']['dwheel'] != 0 or action['mouse']['buttons'] != [0]
                #         or action['keyboard']['keys'] != [] or action['keyboard']['newKeys'] != []):
                #         valid = True
                #         break

                # If the actions are valid, append the frames and actions to the lists
                if valid:
                    self.valid_data_locations.append((len(self.video_title_list)-1, i))
        
        # Clean up a little bit
        del video_files
        del action_data
        del actions
        del valid


    def transform_images(self, image):
        # Convert the image to a PyTorch tensor with size 224x224
        image = cv2.resize(image, (224, 224))
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)
        
        return image
    

    def transform_actions(self, actions):
        # Convert the keyboard actions to a one-hot vector
        action_sequence = torch.zeros((self.action_sequence_length + self.frame_skip - 1, 20))
        mouse_inputs = []
        for i, action in enumerate(actions):
            for key in action['keyboard']['keys']:
                try:
                    action_sequence[i][self.key_to_index[key]] = 1
                except:
                    pass
            for key in action['keyboard']['newKeys']:
                try:
                    action_sequence[i][self.key_to_index[key]] = 1
                except:
                    pass
        
            # Accept the mouse inputs
            mouse_input = [
                action['mouse']['x'],
                action['mouse']['y'],
                action['mouse']['dx'],
                action['mouse']['dy'],
                action['mouse']['dwheel']
            ]
            mouse_input[0] = (mouse_input[0] / (1280 / 2)) - 1
            mouse_input[1] = (mouse_input[1] / (720 / 2)) - 1
            mouse_input[2] = (mouse_input[2] / (1280 / 2))
            mouse_input[3] = (mouse_input[3] / (720 / 2))

            # One-hot encode the mouse buttons
            mouse_buttons = [0, 0, 0]
            for button in action['mouse']['buttons']:
                if button in [1, 2, 3]:
                    mouse_buttons[button - 1] = 1
            for button in action['mouse']['newButtons']:
                if button in [1, 2, 3]:
                    mouse_buttons[button - 1] += 1

            # Concatenate the mouse inputs and buttons
            mouse_inputs.append(mouse_input + mouse_buttons)
        
        # Convert mouse inputs to a tensor
        mouse_inputs = torch.tensor(mouse_inputs, dtype=torch.float32)

        # If the action sequence is less than the action_sequence_length, add action[0] to the front
        if len(actions) < self.action_sequence_length + self.frame_skip - 1:
            action_sequence = torch.cat((action_sequence[-len(actions):], action_sequence[:len(actions)]))
            mouse_inputs = torch.cat((mouse_inputs[-len(actions):], mouse_inputs[:len(actions)]))

        # Concatenate the action_sequence and mouse_inputs
        action_sequence = torch.cat((action_sequence, mouse_inputs), dim=1)

        return action_sequence
        
    def __len__(self):
        return len(self.valid_data_locations)
    
    def __getitem__(self, idx):
        # Get the video title index and frame index
        video_title_index, video_frame_index = self.valid_data_locations[idx]

        # Load in the video file and jsonl file
        video_file = os.path.join(self.data_folder, self.video_title_list[video_title_index] + '.mp4')
        jsonl_file = os.path.join(self.data_folder, self.video_title_list[video_title_index] + '.jsonl')

        # Load the jsonl file and parse the action data
        with open(jsonl_file, 'r') as f:
            action_data = [json.loads(line) for line in f]
        
        # Load the video file
        cap = cv2.VideoCapture(video_file)

        # Capture the frame_t
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_index)
        ret, frame = cap.read()
        if not ret:
            print('Error reading frame')
            return None
        frame_t = self.transform_images(frame)

        # Capture the frame_t_plus_1
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_index + self.frame_skip)
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_index + self.frame_skip)
        ret, frame = cap.read()
        if not ret:
            print(f'Error reading frame at index {video_frame_index + self.frame_skip} in video {video_file}. Skipping this sample.')
            return self.__getitem__((idx + 1) % len(self))  # Recursively call __getitem__ with the next index

        frame_t_plus_1 = self.transform_images(frame)

        # Load the actions from t-action_sequence_length to t
        actions = []
        for i in range(self.action_sequence_length + self.frame_skip - 1):
            actions.append(action_data[video_frame_index - self.action_sequence_length + i + 1])
        
        # Convert the actions to a one-hot vector
        action_sequence = self.transform_actions(actions)

        # Close the video file
        cap.release()

        return frame_t, action_sequence, frame_t_plus_1