. /home/ubuntu/miniconda3/bin/activate
. /home/ubuntu/miniconda3/etc/profile.d/conda.sh
#conda activate AgriXGEE

#conda activate AgriXGEE_Modified
#conda activate agrix_10apr # new env created on 10-04-2022
#conda activate agrix_12apr # new env created on 12-04-2022 - update pytorch and torchvision
conda activate agrix_12apr_test

nohup python /home/ubuntu/AgriX-Api/Crop_Analyser/main_model_training.py --user_id=$1 --job_id=$2 --start_date="$3" --period=$4 --label_file_path="$5" --lat_long_label_cols="$6" --crop_name="$7" > "/home/ubuntu/AgriX-Api/Job_Log/"$2".Log" 2>&1 &
echo "\$1 = $1, \$2 = $2, \$3 = $3, \$4 = $4, \$5 = $5, \$6 = $6, \$7 = $7"

# sh model_training.sh '64' '643333' '2020-08-21' '3' '/home/ubuntu/Agrix-Api/Crop_Analyser/ADANGAL.58647d69.xlsx' 'Latitude,Longitude,Crop Name' 'Paddy Ii'
# sh model_training.sh '64' '640000_test' '2020-08-21' '3' '/home/ubuntu/AgriX-Api/Crop_Analyser/ADANGAL.58647d69.xlsx' 'Latitude,Longitude,Crop Name' 'Paddy Ii'
#sh model_training.sh '64' '640110_test' '2020-08-21' '3' '/home/ubuntu/Amuthan_weed_classification_venv/codes/ModalTrainning.xlsx' 'Latitude,Longitude,Crop Name' 'Paddy Ii'