import os
from app.utils.alignment import crop_faces, calc_alignment_coefficients
from PIL import Image
import torch
import numpy as np
import imageio
from app.utils.inference_video import interpolate
from skimage.transform import resize
from skimage import img_as_ubyte
import warnings
from app.utils.demo import load_checkpoints, make_animation
import fire

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tiff",
]


def save_image(image: Image.Image, output_folder, image_name, image_index, ext="jpg"):
    """Only used when pasting face on image"""
    if ext == "jpeg" or ext == "jpg":
        image = image.convert("RGB")
    folder = os.path.join(output_folder, image_name)
    os.makedirs(folder, exist_ok=True)
    image.save(os.path.join(folder, f"{image_index}.{ext}"))


def paste_image(coeffs, img, orig_image):
    """Only used when pasting face on image"""
    pasted_image = orig_image.copy().convert("RGBA")
    projected = img.convert("RGBA").transform(
        orig_image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR
    )
    pasted_image.paste(projected, (0, 0), mask=projected)
    return pasted_image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for fname in sorted(os.listdir(dir)):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            fname = fname.split(".")[0]
            images.append((fname, path))
    return images


def main(source_image_path="./app/assets/macron6.jpeg", video_num=1):
    source_image = imageio.imread(source_image_path)
    gradio_funct(source_image, video_num=video_num)


def gradio_funct(source_image, video_num):

    list_video = [
        "./app/assets/bouge.mov",
        "./app/assets/bouge_2.mov"
        # "./assets/bouche_ouverte.mov",
    ]
    driving_video_path = list_video[video_num]

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    div_frame = 4  # for num==4 put exp==2 for frame interpolation
    exp = 2 if div_frame == 4 else 1
    assert exp in [1, 2]

    dataset_name = "vox"  # ['vox', 'taichi', 'ted', 'mgif']
    input_folder = "/tmp/source_images"
    output_videos = "app/output_videos"
    config_path = "app/configs/vox-256.yaml"
    checkpoint_path = "app/checkpoints/vox.pth.tar"
    predict_mode = "relative"  # ['standard', 'relative', 'avd']
    find_best_frame = False  # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result

    pixel = 256  # for vox, taichi and mgif, the resolution is 256*256
    image_size, scale, center_sigma, xy_sigma, use_fa = pixel, 1.0, 1.0, 3.0, False

    if not os.path.isdir(input_folder):
        os.mkdir(input_folder)
    if not os.path.isdir(output_videos):
        os.mkdir(output_videos)

    if dataset_name == "ted":  # for ted, the resolution is 384*384
        pixel = 384

    warnings.filterwarnings("ignore")
    reader = imageio.get_reader(driving_video_path)
    fps = reader.get_meta_data()["fps"]
    print("fps= " + str(fps) + "\n")
    driving_video = []
    try:
        for i, im in enumerate(reader):
            if i % div_frame == 0:
                driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    # Align face frames of the video
    imageio.imsave("{}/frame_{}.jpg".format(input_folder, str(1)), driving_video[0])

    files = make_dataset(input_folder)
    crops, orig_images, quads = crop_faces(
        image_size,
        files,
        scale,
        center_sigma=center_sigma,
        xy_sigma=xy_sigma,
        use_fa=use_fa,
    )

    x1, y1 = tuple(quads[0][0])
    x1, y1 = int(x1), int(y1)

    x2, y2 = tuple(quads[0][2])
    x2, y2 = int(x2), int(y2)

    driving_video = [
        resize(frame[y1:y2, x1:x2], (pixel, pixel))[..., :3] for frame in driving_video
    ]
    # Align face of the image
    imageio.imsave("{}/frame_{}.jpg".format(input_folder, str(1)), source_image)

    files = make_dataset(input_folder)
    crops, orig_images, quads = crop_faces(
        image_size,
        files,
        scale,
        center_sigma=center_sigma,
        xy_sigma=xy_sigma,
        use_fa=use_fa,
    )
    source_image = resize(np.array(crops[0]), (pixel, pixel))[..., :3]

    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
        config_path=config_path, checkpoint_path=checkpoint_path, device=device
    )

    print("creating deepfake")
    print("generating 1 frame every" + str(div_frame) + "frames")
    if predict_mode == "relative" and find_best_frame:
        from utils.demo import find_best_frame as _find

        i = _find(source_image, driving_video, device.type == "cpu")
        print("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[: (i + 1)][::-1]
        predictions_forward = make_animation(
            source_image,
            driving_forward,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            device=device,
            mode=predict_mode,
        )
        predictions_backward = make_animation(
            source_image,
            driving_backward,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            device=device,
            mode=predict_mode,
        )
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(
            source_image,
            driving_video,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            device=device,
            mode=predict_mode,
        )

    # save resulting videos
    first_video_path = output_videos + "/video_fast.mp4"
    imageio.mimsave(
        first_video_path,
        [img_as_ubyte(frame) for frame in predictions],
        fps=fps,
    )

    if div_frame > 1:
        video_interpolated_path = output_videos + "/video_interpolated.mp4"
        print("Generate final video with interpolations.")
        interpolate(
            exp=exp,
            video=first_video_path,
            fps=fps,
            output=video_interpolated_path,
        )
        return video_interpolated_path
    return first_video_path


if __name__ == "__main__":
    output_video_path = fire.Fire(main)
