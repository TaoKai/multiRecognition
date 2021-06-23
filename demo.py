from multi_recognition import video_analysis

track_info = video_analysis('videos/sample2.mp4')
for k, v in track_info['frame_info'].items():
    for f in v:
        print(k, f[0], f[1], f[2], f[3])
for k, v in track_info['character_info'].items():
    print(v)