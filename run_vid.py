import torchvision
import torch
from model.ZeroPik import CLIPTextGenerator
import torchvision.transforms as T

def read_video(video_path='/home/youseef/GP/zerocap/zero-shot-image-to-text/example_videos/testvid.mp4'):
    stream = 'video'
    video_reader = torchvision.io.VideoReader(video_path, stream)
    fms = [frame['data'] for frame in video_reader]
    video_frames = torch.stack(fms)
    return video_frames


# def preprocess_frames(generator, frames):
# Max size for a video frame is 224*224

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_generator = CLIPTextGenerator(lm_model='gpt-neo')
    frames = read_video().to(device)
    frame_transformation = [T.Resize((224, 224), antialias=True)]
    frame_transformation = T.Compose(frame_transformation)
    frames = [frame_transformation(frame) for frame in frames]
    frames = torch.stack(frames)
    print('run_vid.py21:', frames.shape)
    video_features = text_generator.get_img_feature(frames)
    captions = text_generator.run(video_features, cond_text="Image of a", beam_size=1)
    print('Feat.shape', video_features.shape)
    # preprocess_frames(text_generator, frames)
