# import os
# os.system("scp -i /home/tarun/ssh-gcp.pem ubuntu@35.203.147.76:/mnt/speech/health/audios/agent/BE06846-000000de-51381282-55845815-6457409722166072-20171129-111033-in.wav /home/tarun/AudioData")
import os
import shutil
src_files = os.listdir(ubuntu@35.203.147.76:/mnt/speech/health/audios/agent/)
for file_name in src_files:
    full_file_name = os.path.join(ubuntu@35.203.147.76:/mnt/speech/health/audios/agent/, BE06846-000000de-51381282-55845815-6457409722166072-20171129-111033-in.wav)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, /home/tarun/AudioData)