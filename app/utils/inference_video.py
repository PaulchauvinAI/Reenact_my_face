import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
from app.train_log.RIFE_HDv3 import Model
import _thread
import skvideo.io
from queue import Queue, Empty
from app.model.pytorch_msssim import ssim_matlab
import fire

warnings.filterwarnings("ignore")


def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor

    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system(
            'ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName)
        )

    targetNoAudio = (
        os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    )
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system(
        'ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(
            targetNoAudio, tempAudioFileName, targetVideo
        )
    )

    if (
        os.path.getsize(targetVideo) == 0
    ):  # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system(
            'ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(
                sourceVideo, tempAudioFileName
            )
        )
        os.system(
            'ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(
                targetNoAudio, tempAudioFileName, targetVideo
            )
        )
        if (
            os.path.getsize(targetVideo) == 0
        ):  # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print(
                "Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead."
            )

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")


def clear_write_buffer(write_buffer, png, vid_out):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if png:
            cv2.imwrite("vid_out/{:0>7d}.png".format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])


def build_read_buffer(read_buffer, videogen, img, montage, w):
    try:
        for frame in videogen:
            if not img is None:
                frame = cv2.imread(os.path.join(img, frame))[:, :, ::-1].copy()
            # if montage:
            #   frame = frame[:, left : left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)


def make_inference(I0, I1, n, model, scale):
    # global model
    middle = model.inference(I0, I1, scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n // 2, model, scale)
    second_half = make_inference(middle, I1, n // 2, model, scale)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


def pad_image(img, fp16, padding):
    if fp16:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)


"""
parser = argparse.ArgumentParser(description="Interpolation for a pair of images")
parser.add_argument("--video", dest="video", type=str, default=None)
parser.add_argument("--output", dest="output", type=str, default=None)
parser.add_argument("--img", dest="img", type=str, default=None)
parser.add_argument(
    "--montage", dest="montage", action="store_true", help="montage origin video"
)
parser.add_argument(
    "--model",
    dest="modelDir",
    type=str,
    default="train_log",
    help="directory with trained model files",
)

parser.add_argument("--UHD", dest="UHD", action="store_true", help="support 4k video")
parser.add_argument(
    "--scale", dest="scale", type=float, default=1.0, help="Try scale=0.5 for 4k video"
)
parser.add_argument(
    "--skip",
    dest="skip",
    action="store_true",
    help="whether to remove static frames before processing",
)
parser.add_argument("--fps", dest="fps", type=int, default=None)
parser.add_argument(
    "--png",
    dest="png",
    action="store_true",
    help="whether to vid_out png format vid_outs",
)
parser.add_argument(
    "--ext", dest="ext", type=str, default="mp4", help="vid_out video extension"
)
parser.add_argument("--exp", dest="exp", type=int, default=1)
args = parser.parse_args()
"""


def interpolate(
    video=None,
    img=None,
    output=None,
    fps=None,
    montage=False,
    modelDir="train_log",
    fp16=False,
    UHD=False,
    scale=1.0,
    skip=False,
    png=False,
    ext="mp4",
    exp=1,
):
    """modelDir:
    fp16: fp16 mode for faster and more lightweight inference on cards with Tensor Cores"
    UHD: "support 4k video"
    skip: "whether to remove static frames before processing"
    png: "whether to vid_out png format vid_outs"
    ext: "vid_out video extension"
    """
    assert not video is None or not img is None
    if skip:
        print("skip flag is abandoned, please refer to issue #207.")
    if UHD and scale == 1.0:
        scale = 0.5
    assert scale in [0.25, 0.5, 1.0, 2.0, 4.0]
    if not img is None:
        png = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    try:
        try:
            try:
                from app.model.RIFE_HDv2 import Model

                model = Model()
                model.load_model(modelDir, -1)
                print("Loaded v2.x HD model.")
            except:
                from app.train_log.RIFE_HDv3 import Model

                model = Model()
                model.load_model(modelDir, -1)
                print("Loaded v3.x HD model.")
        except:
            from app.model.RIFE_HD import Model

            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v1.x HD model")
    except:
        from app.model.RIFE import Model

        model = Model()
        model.load_model(modelDir, -1)
        print("Loaded ArXiv-RIFE model")
    model.eval()
    model.device()

    if not video is None:
        videoCapture = cv2.VideoCapture(video)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()
        if fps is None:
            fpsNotAssigned = True
            fps = fps * (2**exp)
        else:
            fpsNotAssigned = False
        videogen = skvideo.io.vreader(video)
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        video_path_wo_ext, ext = os.path.splitext(video)
        print(
            "{}.{}, {} frames in total, {}FPS to {}FPS".format(
                video_path_wo_ext, ext, tot_frame, fps, fps
            )
        )
        if png == False and fpsNotAssigned == True:
            print("The audio will be merged after interpolation process")
        else:
            print("Will not merge audio because using png or fps flag!")
    else:
        videogen = []
        for f in os.listdir(img):
            if "png" in f:
                videogen.append(f)
        tot_frame = len(videogen)
        videogen.sort(key=lambda x: int(x[:-4]))
        lastframe = cv2.imread(os.path.join(img, videogen[0]), cv2.IMREAD_UNCHANGED)[
            :, :, ::-1
        ].copy()
        videogen = videogen[1:]
    h, w, _ = lastframe.shape
    vid_out_name = None
    vid_out = None
    if png:
        if not os.path.exists("vid_out"):
            os.mkdir("vid_out")
    else:
        if output is not None:
            vid_out_name = output
        else:
            vid_out_name = "{}_{}X_{}fps.{}".format(
                video_path_wo_ext, (2**exp), int(np.round(fps)), ext
            )
        vid_out = cv2.VideoWriter(vid_out_name, fourcc, fps, (w, h))

    if montage:
        left = w // 4
        w = w // 2
    tmp = max(32, int(32 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)
    if montage:
        lastframe = lastframe[:, left : left + w]
    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(
        build_read_buffer, (read_buffer, videogen, img, montage, w)
    )
    _thread.start_new_thread(clear_write_buffer, (write_buffer, png, vid_out))

    I1 = (
        torch.from_numpy(np.transpose(lastframe, (2, 0, 1)))
        .to(device, non_blocking=True)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    I1 = pad_image(I1, fp16, padding)
    temp = None  # save lastframe when processing static frame

    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = (
            torch.from_numpy(np.transpose(frame, (2, 0, 1)))
            .to(device, non_blocking=True)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        I1 = pad_image(I1, fp16, padding)
        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:
            frame = read_buffer.get()  # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = (
                torch.from_numpy(np.transpose(frame, (2, 0, 1)))
                .to(device, non_blocking=True)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            I1 = pad_image(I1, fp16, padding)
            I1 = model.inference(I0, I1, scale)
            I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

        if ssim < 0.2:
            output = []
            for i in range((2**exp) - 1):
                output.append(I0)
            """
            output = []
            step = 1 / (2 ** exp)
            alpha = 0
            for i in range((2 ** exp) - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            """
        else:
            output = make_inference(I0, I1, 2**exp - 1, model, scale) if exp else []

        if montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
            for mid in output:
                mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
                write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
        else:
            write_buffer.put(lastframe)
            for mid in output:
                mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
                write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
        if break_flag:
            break

    if montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    else:
        write_buffer.put(lastframe)
    import time

    while not write_buffer.empty():
        time.sleep(0.1)
    pbar.close()
    if not vid_out is None:
        vid_out.release()  # to close the video file

    # move audio to new video file if appropriate
    if png == False and fpsNotAssigned == True and not video is None:
        try:
            transferAudio(video, vid_out_name)
        except:
            print("Audio transfer failed. Interpolated video will have no audio")
            targetNoAudio = (
                os.path.splitext(vid_out_name)[0]
                + "_noaudio"
                + os.path.splitext(vid_out_name)[1]
            )
            os.rename(targetNoAudio, vid_out_name)


# if __name__ == "__main__":
#    fire.Fire(interpolate)
