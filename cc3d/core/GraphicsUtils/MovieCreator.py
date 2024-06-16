# -*- coding: utf-8 -*-
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple


def makeMovie(simulationPath, frameRate, quality, enableDrawingMCS=True) -> Tuple[int, Path]:
    """
    :param simulationPath: a string path to a directory with a .cc3d file and screenshot directories
    :param frameRate: an int >= 1
    :param quality: an int 1-10 (inclusive)
    :param enableDrawingMCS: when set to true, draws the MCS of each frame onto the video
                             (recommended, but makes movie creation slower)
    :return: a tuple with: 1) the number of movies created and 2) the Path to the dir where movies
                             were generated.
    """
    # Credit to https://stackoverflow.com/q/49581846/16519580 user 'Makes' for the text overlay FFMPEG command.
    # Credit to https://superuser.com/a/939386 uer 'llogan' for the text positioning in the FFMPEG command.

    simulationPath = Path(simulationPath)
    if not simulationPath.exists():
        print(f"Error: Could not make movie inside unknown directory `{simulationPath.absolute()}`")
        # todo: make return type consistent with signature annotations
        return 0

    print("Making movie inside `", simulationPath.absolute(), "`")
    movieCount = 0
    outputPath = None

    for inputPath in sorted(simulationPath.glob('*')):
        if not inputPath.is_dir():
            continue

        # 'delete=True' removes the temporary file when it is closed.
        # So, setting 'delete=True' ensures that the tempfile stays active long enough for FFMPEG to read it.
        with tempfile.NamedTemporaryFile(delete=False, mode='+a', dir=inputPath) as tempFile:
            with tempfile.NamedTemporaryFile(delete=False, mode='+a', dir=inputPath) as textOverlayFile:
                """
                Write the names of files to be used as video frames to tempFile 
                and write the text to draw to textOverlayFile.
                """
                frameCount = 0
                duration = 1 / max(frameRate, 1)
                for p in sorted(inputPath.glob('*')):
                    fileName, fileExtension = p.stem, p.suffix
                    if fileExtension.lower() == ".png":
                        tempFile.write(f"file '{p.name}'\n")

                        # Note: frameRate has to be excluded in the FFMPEG command.
                        # Instead, we use `duration` inside the input file.
                        # This fixes a bug where the last frame appears at the beginning.
                        tempFile.write(f"duration {duration}\n")

                        if enableDrawingMCS:
                            # Try to use the MCS listed in the screenshot name
                            mcs = 0
                            try:
                                parts = fileName.split("_")
                                if len(parts) >= 2:
                                    mcs = int(parts[-1])
                            except:
                                mcs = frameCount

                            # Text will go in top right
                            textOverlayFile.write(f"{frameCount} drawtext reinit 'text=MCS {mcs}':x=w-tw-10:y=10;\n")
                        frameCount += 1

                tempFile.close()
                textOverlayFile.close()

                if frameCount > 0:
                    # Number the file name so that it does not overwrite another movie
                    fileNumber = 0
                    outputPath = Path(simulationPath).joinpath("movies")
                    outputPath.mkdir(parents=True, exist_ok=True)

                    visualizationName = inputPath.name
                    while outputPath.joinpath(f"{visualizationName}_v{fileNumber}.mp4").exists():
                        fileNumber += 1
                    outputPath = outputPath.joinpath(f"{visualizationName}_v{fileNumber}.mp4")

                    commandArgs = [
                        "ffmpeg",
                        "-n",  # never overwrite a file
                        "-f", "concat",
                        "-safe", "0",
                        "-i", tempFile.name,
                        "-crf", str(quality),  # set quality (constant rate factor, crf): 51=worst, 0=best
                        "-c:v", "libx264",  # video codec: H.264
                        "-pix_fmt", "yuv420p",
                    ]

                    if enableDrawingMCS:
                        commandArgs.append("-filter_complex")
                        commandArgs.append(f"[0:v]sendcmd=f={Path(textOverlayFile.name).name},drawtext=fontfile=PF.ttf:text='':fontcolor=white:fontsize=20")

                    commandArgs.append(str(outputPath.resolve()))
                    subprocess.run(commandArgs, cwd=inputPath)

                    if outputPath.exists():
                        movieCount += 1

                Path(textOverlayFile.name).unlink()
            Path(tempFile.name).unlink()

    print(f"Created {movieCount} movies inside `{simulationPath}` with frame rate {frameRate} and quality {quality}/51.")
    return movieCount, outputPath
