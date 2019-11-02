import os
import subprocess


def render_movie(
    filename: str = None,
    fps: int = 30,
    frame_dirname: str = None,
    output_dirname: str = None,
) -> None:
    """
    Turn all the .pngs in frame_dirname into a movie with filename
    and put it in output_dirname.
    """
    movie_path = os.path.join(output_dirname, filename)

    # Prepare the arguments for the call to FFmpeg.
    input_file_format = "*.png"
    input_file_pattern = os.path.join(frame_dirname, input_file_format)
    codec = "libx264"
    command = [
        "ffmpeg",
        "-framerate", str(fps),
        "-y",
        "-pattern_type", "glob",
        "-i", input_file_pattern,
        "-c:v", codec,
        movie_path,
    ]
    print(" ".join(command))
    subprocess.call(command)
    return


def convert_to_gif(
    fps: int = 30,
    resolution: int = 640,
    filename: str = None,
    dirname: str = None,
):
    """
    Convert an mp4 video to a gif.
    """
    mp4_pathname = os.path.join(dirname, filename)
    gif_filename = filename[:-4] + ".gif"
    gif_pathname = os.path.join(dirname, gif_filename)
    create_palette_command = [
        "ffmpeg",
        "-y",
        "-i",
        mp4_pathname,
        "-vf",
        "".join([
            "fps=",
            str(fps),
            ",scale=",
            str(resolution),
            ":-1:flags=lanczos,palettegen",
        ]),
        "palette.png",
    ]

    create_gif_command = [
        "ffmpeg",
        "-y",
        "-i",
        mp4_pathname,
        "-i",
        "palette.png",
        "-filter_complex",
        "".join([
            "fps=",
            str(fps),
            ",scale=",
            str(resolution),
            ":-1:flags=lanczos[x];[x][1:v]paletteuse",
        ]),
        gif_pathname,
    ]
    print(create_palette_command)
    subprocess.call(create_palette_command)
    subprocess.call(create_gif_command)

    os.remove("palette.png")
