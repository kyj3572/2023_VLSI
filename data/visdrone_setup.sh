pip install gdown    # Python package for easy download of Google drive files

echo "========== DOWNLOAD ZIP FILES =========="
gdown --id 1a-AmQjYATzj8seXLlEm9Sx8aQmClrJka -O SOT_train_part1.zip
gdown --id 16YPyhNDQrTgW8I2HaH_HNEO-KTRA-xso -O SOT_train_part2.zip
gdown --id 18SNAOlCJtApnG2m45ud-1e_OtGYill0D -O SOT_validation.zip

echo "========== MAKING DIRECTORY =========="
mkdir part1 part2 valid

echo "========== UNZIP FILES =========="
unzip -q SOT_train_part1.zip -d part1
unzip -q SOT_train_part2.zip -d part2
unzip -q SOT_validation.zip -d valid

echo "========== REMOVE PART1 IMAGE =========="
cd part1/VisDrone2019-SOT-train/sequences
rm -rf uav0000071_00816_s uav0000084_00000_s uav0000090_00276_s uav0000090_01104_s uav0000091_01035_s
rm -rf uav0000091_01288_s uav0000091_02530_s uav0000160_00000_s uav0000169_00000_s uav0000170_00000_s
cd uav0000014_00667_s
rm img00001*.jpg img00002*.jpg
cd ../uav0000016_00000_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg
cd ../uav0000043_00377_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg
cd ../uav0000049_00435_s
rm img00001*.jpg img00002*.jpg
cd ../uav0000068_01488_s
rm img00000*.jpg img00002*.jpg img00004*.jpg img00005*.jpg  img00006*.jpg 
cd ../uav0000068_02928_s
rm img00000*.jpg img00002*.jpg img00003*.jpg
cd ../uav0000068_03768_s
rm img00001*.jpg img00002*.jpg
cd ../uav0000070_04877_s
rm img00000*.jpg img00001*.jpg
cd ../uav0000071_01536_s
rm img00002*.jpg img00003*.jpg img00004*.jpg 
cd ../uav0000071_02520_s
rm img00000*.jpg img00001*.jpg
cd ../uav0000072_02544_s
rm img00000*.jpg img00002*.jpg img00003*.jpg img00004*.jpg img00005*.jpg
cd ../uav0000072_03792_s
rm img00000*.jpg img00001*.jpg
cd ../uav0000076_00241_s
rm img0000*.jpg img00010*.jpg img00011*.jpg img00012*.jpg img00013*.jpg img00015*.jpg
cd ../uav0000080_01680_s
rm img0000*.jpg img00010*.jpg
cd ../uav0000084_00812_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00004*.jpg
cd ../uav0000085_00000_s
rm img0000*.jpg img00010*.jpg img00011*.jpg img00012*.jpg img00013*.jpg img00016*.jpg
cd ../uav0000089_00920_s
rm img0000*.jpg img00010*.jpg img00011*.jpg img00012*.jpg
cd ../uav0000091_00460_s
rm img00000*.jpg img00001*.jpg
cd ../uav0000099_02520_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg img00005*.jpg img00006*.jpg img00007*.jpg 
cd ../uav0000126_07915_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg
cd ../uav0000144_03200_s
rm img00000*.jpg img00001*.jpg img00003*.jpg
cd ../uav0000147_00000_s
rm img0000*.jpg img00010*.jpg img00011*.jpg img00012*.jpg img00013*.jpg img00014*.jpg img00015*.jpg img0002*.jpg
cd ../uav0000148_00840_s
rm img00003*.jpg img00004*.jpg img00005*.jpg img00006*.jpg img00007*.jpg img00008*.jpg 
cd ../uav0000149_00317_s
rm img0000*.jpg img0001*.jpg
cd ../uav0000159_00000_s
rm img00003*.jpg img00004*.jpg img00005*.jpg img00006*.jpg img00007*.jpg img00008*.jpg img00009*.jpg


echo "========== REMOVE PART2 IMAGE =========="
cd ../../../../part2/VisDrone2019-SOT-train/sequences
rm -rf uav0000171_00000_s uav0000173_00781_s uav0000178_00025_s uav0000198_00000_s uav0000325_01656_s
rm -rf uav0000200_00000_s uav0000204_00000_s uav0000205_00000_s uav0000209_00000_s uav0000221_10400_s
rm -rf uav0000223_00300_s uav0000226_05370_s uav0000236_00001_s uav0000237_00001_s uav0000238_01280_s
rm -rf uav0000239_11136_s uav0000300_00000_s uav0000304_00253_s uav0000342_01518_s uav0000352_00759_s
cd uav0000172_00000_s
rm img0000*.jpg img00011*.jpg img00012*.jpg img00013*.jpg img00014*.jpg
cd ../uav0000174_00000_s
rm img0000*.jpg img0001*.jpg
cd ../uav0000175_00000_s
rm img00000*.jpg img00001*.jpg img00004*.jpg
cd ../uav0000176_00000_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg img00005*.jpg img00006*.jpg
cd ../uav0000182_01075_s
rm img00000*.jpg
cd ../uav0000199_00000_s
rm img0000*.jpg img00010*.jpg img00011*.jpg img00012*.jpg img00015*.jpg img00016*.jpg
cd ../uav0000217_00001_s
rm img00000*.jpg
cd ../uav0000222_00900_s
rm img0000*.jpg
cd ../uav0000232_00960_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg
cd ../uav0000235_00001_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg img00005*.jpg img00006*.jpg img00007*.jpg
cd ../uav0000235_01032_s
rm img0000*.jpg
cd ../uav0000238_00001_s
rm img0000*.jpg img00010*.jpg 
cd ../uav0000240_00001_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg img00005*.jpg
cd ../uav0000252_00001_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg img00005*.jpg img00006*.jpg
cd ../uav0000303_00000_s
rm img0000*.jpg img00010*.jpg 
cd ../uav0000303_01250_s
rm img0000*.jpg img00010*.jpg img00011*.jpg img00012*.jpg img00015*.jpg img00016*.jpg
cd ../uav0000307_04531_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg img00005*.jpg img00006*.jpg
cd ../uav0000308_04600_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg
cd ../uav0000329_00276_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg
cd ../uav0000331_02691_s
rm img00000*.jpg img00001*.jpg img00002*.jpg
cd ../uav0000348_02415_s
rm img00000*.jpg img00001*.jpg img00002*.jpg
cd ../uav0000349_02668_s
rm img00000*.jpg img00001*.jpg img000020*.jpg img000021*.jpg img000022*.jpg img000023*.jpg img000024*.jpg


echo "========== REMOVE VALIDATION IMAGE =========="
cd ../../../../valid/VisDrone2019-SOT-val/sequences
rm -rf uav0000092_00575_s uav0000092_01150_s
cd uav0000024_00000_s
rm img0000*.jpg img00010*.jpg img00011*.jpg img00012*.jpg img00014*.jpg 
cd ../uav0000029_01102_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg 
cd ../uav0000054_00000_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg 
cd ../uav0000086_00870_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg img00004*.jpg 
cd ../uav0000115_00606_s
rm img00000*.jpg img00001*.jpg
cd ../uav0000245_00001_s
rm img00002*.jpg img00003*.jpg img00004*.jpg img00005*.jpg img00006*.jpg img00007*.jpg 
cd ../uav0000317_00000_s
rm img00000*.jpg img00001*.jpg img00002*.jpg img00003*.jpg
cd ../uav0000317_02945_s
rm img0000169.jpg img000017*.jpg img000018*.jpg img000019*.jpg img00002*.jpg img00003*.jpg img00004*.jpg img00005*.jpg img00006*.jpg 

echo "========== RENAME & MOVE =========="
cd /workspace/dataset
mkdir visdrone
python rename_move.py --src ./part1/VisDrone2019-SOT-train/sequences
python rename_move.py --src ./part2/VisDrone2019-SOT-train/sequences
python rename_move.py --src ./valid/VisDrone2019-SOT-val/sequences

echo "========== DISCARD USELESS FILES =========="
rm -rf part1 part2 valid
