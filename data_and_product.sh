#nohup /home/ubuntu/AgriX_venv/bin/python3 /home/ubuntu/AgriX-Api/Crop_Analyser/main.py --user_id=$1 --job_id=$2 --start_date=$3 --end_date=$4 --aoi_path=$5 --product_type=$6 --product_crop_type=$7 --request_type_id=$8 --model_path=$9 --direction='DESCENDING_VVVH' --ls_mask=False --rmv_speckle=True > "/home/ubuntu/AgriX-Api/Job_Log/"$2".Log" 2>&1 &
#nohup python3 /home/ubuntu/AgriX-Api/Crop_Analyser/main.py --user_id=$1 --job_id=$2 --start_date=$3 --end_date=$4 --aoi_path=$5 --product_type=$6 --product_crop_type=$7 --request_type_id=$8 --model_path=$9 --direction='DESCENDING_VVVH' --ls_mask=False --rmv_speckle=True > "/home/ubuntu/AgriX-Api/Job_Log/"$2".Log" 2>&1 &

# initialize the shell - use '.' instead of 'source'
. /home/ubuntu/miniconda3/bin/activate
. /home/ubuntu/miniconda3/etc/profile.d/conda.sh

# Initialize the conda environment name
#conda activate AgriXGEE_Modified
#conda activate agrix_10apr
conda activate agrix_12apr # agrix_10apr - fixing pytorch issue and update the torch and torchvision version from 1.4.0 and 0.5.0 to 1.6.0 and 0.7.0


#source miniconda3/bin/activate AgriXGEE_Modified


nohup python /home/ubuntu/AgriX-Api/Crop_Analyser/main.py --user_id=$1 --job_id=$2 --start_date=$3 --end_date=$4 --aoi_path=$5 --product_type=$6 --product_crop_type=$7 --request_type_id=$8 --model_path=$9 --direction='DESCENDING_VVVH' --ls_mask=False --rmv_speckle=True > "/home/ubuntu/AgriX-Api/Job_Log/"$2".Log" 2>&1 &

#nohup /home/ubuntu/AgriX_venv/bin/python3 /home/ubuntu/AgriX-Api/Crop_Analyser/main.py --user_id=$1 --job_id=$2 --start_date=$3 --end_date=$4 --aoi_path=$5 --product_type=$6 --product_crop_type=$7 --request_type_id=$8 --model_path=$9 --direction='DESCENDING_VVVH' --ls_mask=False --rmv_speckle=True > "/home/ubuntu/AgriX-Api/Job_Log/"$2".Log" 2>&1 &

echo "\$1 = $1, \$2 = $2, \$3 = $3, \$4 = $4, \$5 = $5, \$6 = $6, \$7 = $7, \$8 = $8, \$9 = $9"
echo "Executed Succesfully"
