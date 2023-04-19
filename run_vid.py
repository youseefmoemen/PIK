import torchvision
import torch
from model.ZeroPik import CLIPTextGenerator
from torchvision import transforms
from data_reader import VideoLoader
from torch.utils.data import DataLoader


def read_video(video_path='/home/youseef/GP/zerocap/zero-shot-image-to-text/example_videos/test1.mp4'):
    stream = 'video'
    video_reader = torchvision.io.VideoReader(video_path, stream)
    print(video_reader)
    fms = [frame['data'] for frame in video_reader]
    video_frames = torch.stack(fms)
    return video_frames


# def preprocess_frames(generator, frames):
# Max size for a video frame is 224*224

if __name__ == '__main__':
    test = 'A'
    if test == 'A':
        print('In A')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        frames = read_video().to(device)
        text_generator = CLIPTextGenerator(lm_model='gpt-neo')
        frame_transformation = [transforms.Resize((224, 224), antialias=True)]
        frame_transformation = transforms.Compose(frame_transformation)
        frames = [frame_transformation(frame) for frame in frames]
        frames = torch.stack(frames)
        print('run_vid.py21:', frames.shape)
        video_features = text_generator.get_img_feature(frames)
        print('run_vid.py34:', video_features.shape)
        for idx, i in enumerate(video_features):
            print('Frame: ', idx)
            captions = text_generator.run(i, cond_text="Image of a", beam_size=1)
            if idx == 3:
                break
        # preprocess_frames(text_generator, frames)
    else:
        print('In B')
        dataset = VideoLoader('/home/youseef/GP/zerocap/zero-shot-image-to-text/example_videos',
                              transform=transforms.Compose([
                                  transforms.Resize((224, 224)),
                              ]))

        dataloader = DataLoader(dataset, shuffle=True, batch_size=4, num_workers=0)
        for i_batch, batch in enumerate(dataloader):
            print(i_batch)
            print(batch.shape)
