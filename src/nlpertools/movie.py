from tqdm import trange, tqdm


def cv_time(file_path):
    import cv2  # pip install opencv-python-headless

    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    success, image = cap.read()
    for i in trange(frame_count):
        success, image = cap.read()
        print()
    # print(success)


def moviepy_time(file_path):
    from moviepy.editor import VideoFileClip

    video = VideoFileClip(file_path)
    fps = video.fps
    total = int(video.duration * fps)
    for idx, frame in tqdm(
        # 循环时以帧迭代
        enumerate(video.iter_frames(fps=fps)),
        total=total,
    ):
        pass


if __name__ == "__main__":

    # file_path = r"../movies/bikes.mp4"
    file_path = r"D:\[甄嬛传][76集全]\甄嬛传.EP34.WEB-DL.4K.H264.AAC-CYW.mp4"
    # cv_time(file_path)
    moviepy_time(file_path)
