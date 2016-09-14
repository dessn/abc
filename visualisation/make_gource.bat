gource -1280x720 --start-position 0.06 --hide filenames --bloom-multiplier 1.2 --auto-skip-seconds 0.1 --seconds-per-day 0.1 -o visualisation/gource.ppm ..
ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i gource.ppm -vcodec libx264 -preset slow -pix_fmt yuv420p -threads 0 -bf 0 gource.mp4
