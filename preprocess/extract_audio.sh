#!/bin/bash
# change the directory path of raw videos
video_dir='xxx/video'
audio_dir='data/audio'

echo "Transfer mp4 file to wav file ..."
function getdir(){
  for element in `ls $video_dir`
    do
      dir_or_file=${video_dir}"/"$element
      if [ -d "$dir_or_file" ]; then
          echo $dir_or_file
          getdir $dir_or_file         
      elif [ "${dir_or_file##*.}"x = "mp4"x ]; then
          filename="${dir_or_file##*/}"
          outputname="${filename%.*}.wav"
          ffmpeg -i $dir_or_file -vn -ar 32000 -hide_banner -f wav ${audio_dir}"/"$outputname
      fi
    done ;
}

getdir $video_dir
echo "done";
