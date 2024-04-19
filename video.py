from video_dataset import load_data
import numpy as np
from IPython.display import clear_output
import torch
import cv2
from moviepy.editor import *

def generate_avi_3_frames(encoder, predictor, action_conditioner, diffusion_model, num, path):
    data_folder = './video_datas/'
    action_sequence_length = 5
    split_ratio = 0.1

    _, test_dataloader = load_data(data_folder, start_idx=num, n_videos=1, action_sequence_length=25, split_ratio=0.25, batch_size=1, num_workers=0, shuffle=False)

    def run_dream(iterator):
        diffused_frames = []
        actions = []
        frame_t_actual = []
    
        encoder.eval()
        predictor.eval()
        action_conditioner.eval()
        diffusion_model.eval()
    
        # Load in the frame tuple
        iter_dl = iterator
        try:
            frame_t, action_sequence, _ = next(iter_dl)
        except:
            return [],[],[]
        actions.append(action_sequence[0].detach().cpu())
        frame_t_actual.append(frame_t[0])
    
        frame_t = frame_t.to("cuda")
        action_sequence = action_sequence.to("cuda")
    
        # Encode the frame_t
        z_t = encoder(frame_t)
        frame_t = frame_t.to("cpu")  # Move frame_t back to CPU
    
        diffused_frame = diffusion_model(z_t).view(-1, 3, 224, 224).detach().cpu()
        diffused_frames.append(diffused_frame)
        z_t = z_t.detach()  # Detach z_t from the computation graph

        frame_using = None
        for idx in range(150):
            try:
                frame_t, action_sequence, _ = next(iter_dl)
            except:
                return diffused_frames, actions, frame_t_actual
            if frame_using is not None:
                frame_t_actual.append(frame_using)
            if idx % 15 == 0:
                # Take the actual frame and encode it
                actions.append(action_sequence[0].detach().cpu())

                frame_using = frame_t[0].detach().cpu()

                frame_t = frame_t.to("cuda")
                action_sequence = action_sequence.to("cuda")
    
                z_t = encoder(frame_t)
                frame_t = frame_t.to("cpu")  # Move frame_t back to CPU
            else:
                # Predict the next frame
                action_sequence = action_sequence.to("cuda")
    
                z_t = action_conditioner(z_t, action_sequence)
                action_sequence = action_sequence.to("cpu")  # Move action_sequence back to CPU
    
                z_t = predictor(z_t)
    
            diffused_frame = diffusion_model(z_t).view(-1, 3, 224, 224).detach().cpu()
            diffused_frames.append(diffused_frame)
            z_t = z_t.detach()  # Detach z_t from the computation graph
    
            torch.cuda.empty_cache()

        return diffused_frames, actions, frame_t_actual

    def invert_transform(transformed_image, scale_factor=2):
        image = transformed_image.permute(1, 2, 0).numpy()
        image = image.astype(np.uint8)
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        return image

    # def generate_video(diffused_frames, frame_t_actual, output_path, fps=5):
    #     try:
    #         height, width, _ = cv2.hconcat([invert_transform(diffused_frames[0][0]), invert_transform(frame_t_actual[0])]).shape
    #     except:
    #         return False

    #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #     video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #     # for i in range(0, len(diffused_frames), 5):
    #     #     predicted_frames = diffused_frames[i:i+5]
    #     #     actual_frames = frame_t_actual[i:i+5]

    #     #     for frame, actual in zip(predicted_frames, actual_frames):
    #     #         inverted_frame = invert_transform(frame[0])
    #     #         actual_frame = invert_transform(actual)

    #     #         concatenated_frame = cv2.hconcat([inverted_frame, actual_frame])

    #     #         video_writer.write(concatenated_frame)

    #     action_names = {
    #         0: "W", 1: "A", 2: "S", 3: "D", 4: "Space", 5: "Left Shift", 6: "Left Ctrl",
    #         7: "E", 8: "Q", 9: "Esc", 10: "F", 11: "1", 12: "2", 13: "3", 14: "4",
    #         15: "5", 16: "6", 17: "7", 18: "8", 19: "9",
    #         25: "Left Click", 26: "Middle Click", 27: "Right Click"
    #     }

    #     print(f'{len(diffused_frames)=}')

    #     for i in range(0, len(diffused_frames), 5):
    #         print(i)
    #         predicted_frames = diffused_frames[i:i+5]
    #         actual_frames = frame_t_actual[i:i+5]
    #         action_frames = actions[i:i+5]

    #         print(len(predicted_frames), len(actual_frames), len(action_frames))

    #         for frame, actual, action in zip(predicted_frames, actual_frames, action_frames):
    #             inverted_frame = invert_transform(frame[0])
    #             actual_frame = invert_transform(actual)

    #             # Get the active keyboard and mouse inputs for the current frame
    #             keyboard_inputs = [action_names[j] for j in range(20) if action[-1, j] == 1]
    #             mouse_inputs = [action_names[j] for j in range(25, 28) if action[-1, j] == 1]

    #             # Create the text overlay for keyboard inputs
    #             keyboard_text = ", ".join(keyboard_inputs) if keyboard_inputs else ""

    #             # Create the text overlay for mouse inputs
    #             mouse_text = ", ".join(mouse_inputs) if mouse_inputs else ""

    #             # Create the text overlay for mouse movement
    #             mouse_movement_text = f"Mouse: ({action[-1, 20]:.2f}, {action[-1, 21]:.2f}), dx: {action[-1, 22]:.2f}, dy: {action[-1, 23]:.2f}, wheel: {action[-1, 24]:.2f}"

    #             # Add the text overlays to the predicted frame
    #             cv2.putText(inverted_frame, keyboard_text, (10, inverted_frame.shape[0] - 60),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #             cv2.putText(inverted_frame, mouse_text, (10, inverted_frame.shape[0] - 40),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #             cv2.putText(inverted_frame, mouse_movement_text, (10, inverted_frame.shape[0] - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    #             concatenated_frame = cv2.hconcat([inverted_frame, actual_frame])

    #             video_writer.write(concatenated_frame)

    #     video_writer.release()

    #     return True
    def generate_video(diffused_frames, frame_t_actual, output_path, fps=5):
        try:
            height, width, _ = cv2.hconcat([invert_transform(diffused_frames[0][0]), invert_transform(frame_t_actual[0])]).shape
        except:
            return False

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # action_names = {
        #     0: "W", 1: "A", 2: "S", 3: "D", 4: "Space", 5: "Left Shift", 6: "Left Ctrl",
        #     7: "E", 8: "Q", 9: "Esc", 10: "F", 11: "1", 12: "2", 13: "3", 14: "4",
        #     15: "5", 16: "6", 17: "7", 18: "8", 19: "9",
        #     25: "Left Click", 26: "Middle Click", 27: "Right Click"
        # }

        for i in range(len(diffused_frames)):
            frame = diffused_frames[i]
            
            if i >= len(frame_t_actual):
                # If the index is out of range for frame_t_actual, use the last available frame
                actual = frame_t_actual[-1]
            else:
                actual = frame_t_actual[i]
            
            action_index = i // 5
            
            if action_index >= len(actions):
                # If the action index is out of range, use the last available action
                action = actions[-1]
            else:
                action = actions[action_index]

            inverted_frame = invert_transform(frame[0])
            actual_frame = invert_transform(actual)

            # Get the active keyboard and mouse inputs for the current frame
            # keyboard_inputs = [action_names[j] for j in range(20) if action[-1, j] == 1]
            # mouse_inputs = [action_names[j] for j in range(25, 28) if action[-1, j] == 1]

            # Create the text overlay for keyboard inputs
            # keyboard_text = ", ".join(keyboard_inputs) if keyboard_inputs else ""

            # Create the text overlay for mouse inputs
            # mouse_text = ", ".join(mouse_inputs) if mouse_inputs else ""

            # Create the text overlay for mouse movement
            # mouse_movement_text = f"Mouse: ({action[-1, 20]:.2f}, {action[-1, 21]:.2f}), dx: {action[-1, 22]:.2f}, dy: {action[-1, 23]:.2f}, wheel: {action[-1, 24]:.2f}"

            # Add the text overlays to the predicted frame
            # cv2.putText(inverted_frame, keyboard_text, (10, inverted_frame.shape[0] - 60),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(inverted_frame, mouse_text, (10, inverted_frame.shape[0] - 40),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(inverted_frame, mouse_movement_text, (10, inverted_frame.shape[0] - 10),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            concatenated_frame = cv2.hconcat([inverted_frame, actual_frame])

            video_writer.write(concatenated_frame)

        video_writer.release()

        return True

    iterator = iter(test_dataloader)

    diffused_frames, actions, frame_t_actual = run_dream(iterator)
    output_path = f"output_video{num}.avi"
    gen_vid = generate_video(diffused_frames, frame_t_actual, output_path, fps=5)

    if gen_vid:

        in_vid = f"output_video{num}.avi"
        out_vid = f"output_video{num}.mp4"
    
        avi_video = VideoFileClip(in_vid)
        avi_video.write_videofile(out_vid)
    
    else:
        print("Error generating video")