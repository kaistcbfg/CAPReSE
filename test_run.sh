
# Crop collection
python caprese_main.py --header K562_example --input-file ./data/example/K562_chr9.chr22.bulk.pkl.gz --input-format pickle --chrom1 chr9 --chrom2 chr22 --model-infopath ./TipAdapterF_info.pkl.gz  --model-ptpath ./TipAdapterF_model.pt --imgproc-getcrop True

# CLIP feature generation
python ./tools/imgdir_CLIPfeature_aug.py --input-dir ./data/crops_example --num-aug 3

# Train-Test
python ./tools/train_tip_adapter_F.py --input-poswfile ./data/example/CLIPfeature/weight_pos.csv --input-negwfile ./data/example/CLIPfeature/weight_neg.csv  --input-postfile ./data/example/CLIPfeature/train_pos.csv  --input-negtfile ./data/example/CLIPfeature/train_neg.csv
python ./tools/test_tip_adapter_F.py --model-ptpath ./TipAdapterF_model.pt  --model-infopath ./TipAdapterF_info.pkl.gz --input-postest ./data/example/CLIPfeature/test_pos.csv  --input-negtest ./data/example/CLIPfeature/test_neg.csv

# Classification
python caprese_main.py --header K562_example --input-file ./data/example/K562_chr9.chr22.bulk.pkl.gz --input-format pickle --chrom1 chr9 --chrom2 chr22 --model-infopath ./TipAdapterF_info.pkl.gz  --model-ptpath ./TipAdapterF_model.pt 
python caprese_main.py --header K562_example --input-file ./data/example/K562.mcool --input-format mcool --chrom1 chr9 --chrom2 chr22 --model-infopath ./TipAdapterF_info.pkl.gz  --model-ptpath ./TipAdapterF_model.pt 


