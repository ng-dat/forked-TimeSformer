import torch
from timesformer.models.vit import TimeSformer
from timesformer.datasets import decoder as decoder
from timesformer.datasets import utils as utils
from timesformer.datasets import video_container as container

model = TimeSformer(img_size=224, 
                    num_classes=600, 
                    num_frames=8, 
                    attention_type='divided_space_time',  
                    pretrained_model='/home1/ndat/566/timesformer/forked-TimeSformer/checkpoints/TimeSformer_divST_96x4_224_K600.pyth')


def video_loading(video_path):
    '''
        Test mode only
    '''
    temporal_sample_index = 1
    spatial_sample_index = 1
    sampling_rate = utils.get_random_sampling_rate(
            0, #self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            4, #self.cfg.DATA.SAMPLING_RATE,
        )
    min_scale, max_scale, crop_size = (
                [224]*3 #[self.cfg.DATA.TEST_CROP_SIZE] * 3
                if 1>1 #if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [[256, 320][0]]*2 # else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [224] # + [self.cfg.DATA.TEST_CROP_SIZE]
            )
    
    video_container = None
    try:
        video_container = container.get_video_container(
            video_path,
            False, #self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
            "pyav", #self.cfg.DATA.DECODING_BACKEND,
        )
    except Exception as e:
        print(
            "Failed to load video from {} with error {}".format(
                video_path, e
            )
        )
        
    frames = decoder.decode(
                video_container,
                sampling_rate,
                96, #self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                1, #self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta={}, #self._video_meta[index],
                target_fps=30, #self.cfg.DATA.TARGET_FPS,
                backend="pyav", #self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
            )
    
    DATA_MEAN = [0.45, 0.45, 0.45]
    DATA_STD = [0.225, 0.225, 0.225]
    frames = utils.tensor_normalize(
                frames, DATA_MEAN, DATA_STD# self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
    
    # T H W C -> C T H W.
    frames = frames.permute(3, 0, 1, 2)
    # Perform data augmentation.
    frames = utils.spatial_sampling(
        frames,
        spatial_idx=spatial_sample_index,
        min_scale=min_scale,
        max_scale=max_scale,
        crop_size=crop_size,
        random_horizontal_flip=False, #self.cfg.DATA.RANDOM_FLIP, or trying True(default)
        inverse_uniform_sampling=True, #self.cfg.DATA.INV_UNIFORM_SAMPLE, or trying False(default)
    )
    
    NUM_FRAMES=96
    frames = torch.index_select(
         frames,
         1,
         torch.linspace(
             0, frames.shape[1] - 1, NUM_FRAMES#self.cfg.DATA.NUM_FRAMES
         ).long(),
    )
    
    return frames
    

# dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)
# pred = model(dummy_video,) # (2, 600)


print('Starting prediction', '-'*16)
kinetics_video_paths = ['/home1/ndat/566/kinetics400/k400/-bRqg_Rgmpk_000003_000013.mp4',
               '/home1/ndat/566/kinetics400/k400/-6ykAmBXIr0_000013_000023.mp4',
               '/home1/ndat/566/kinetics400/k400/-JN0MXb-zi8_000017_000027.mp4',
              ]
            # "clapping" "parkour" "hammer throw"
ucfcrime_video_paths = ['/home1/ndat/566/ucf_crime/Anomaly-Videos-Part-3/Shooting/Shooting008_x264.mp4']
for video_path in ucfcrime_video_paths:
    print('Video', video_path)
    video_frames = video_loading(video_path)
    video_frames = video_frames.unsqueeze(0)
    print('Frame shape', video_frames.shape)
    pred = model(video_frames)
    print('Pred', pred.shape, [x + 1 for x in torch.topk(pred[0], k=3)])
    
print('End prediction', '-'*16)
    
    